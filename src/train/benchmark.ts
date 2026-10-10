// Runs the M0 sweep: for each (model, workers, envs-per-worker) cell, spin up a
// worker pool, step everything at once, and report aggregate throughput.
//
// Measuring workers one at a time would miss the whole point — the number that
// matters is what the machine sustains with every core busy, including memory
// bandwidth contention. So all workers are initialised first, then released
// together by a single "run" broadcast.

import { assetBase } from "../asset-url.ts";
import { markAttempt, markDone } from "./crash-log.ts";
import { CTRL_DT, DECIMATION } from "../sim/microduck.ts";
import type { BenchConfig, FromWorker, WorkerStats } from "./protocol.ts";

/** Stable identity for a sweep cell, so a crash can be attributed to it. */
export function cellId(config: BenchConfig, workers: number): string {
  return `${config.robotXml}|w${workers}|e${config.envs}|${config.memory ?? "default"}|${config.withPolicy ? "p" : "-"}`;
}

export function cellLabel(config: BenchConfig, workers: number): string {
  return `${config.robotXml.replace("robot_", "").replace(".xml", "")}, ${workers}x${config.envs}`;
}

export interface CellResult {
  robotXml: string;
  workers: number;
  envsPerWorker: number;
  withPolicy: boolean;
  /** Environments actually allocated across the pool. */
  envs: number;
  /** Aggregate control steps per second — the headline number. */
  controlStepsPerSec: number;
  /** How much faster than real time the whole pool runs. */
  realtimeFactor: number;
  bytesPerEnv: number;
  heapTotalBytes: number;
  ngeom: number;
  /** At least one worker hit the wasm heap ceiling. */
  oom: boolean;
  /** Skipped because this configuration killed the tab on a previous run. */
  skipped?: boolean;
  error?: string;
}

export interface SweepPlan {
  robotXml: string[];
  workers: number[];
  envsPerWorker: number[];
  withPolicy: boolean;
  memory?: string;
  /** Measurement window per cell. Short windows are noisy; 1.5-3 s is enough. */
  durationMs: number;
}

function spawn(): Worker {
  return new Worker(new URL("./sim-worker.ts", import.meta.url), {
    type: "module",
    name: "wicroduck-sim",
  });
}

/**
 * One request/response round trip against a worker.
 *
 * The timeout is not optional: a worker that wedges (or gets killed by the OS
 * for eating memory) never answers, and without a deadline the whole sweep
 * hangs on it instead of recording the cell as failed and moving on.
 */
function ask<T extends FromWorker["type"]>(
  worker: Worker,
  message: unknown,
  expect: T,
  timeoutMs: number,
): Promise<Extract<FromWorker, { type: T }>> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`worker did not reply with "${expect}" within ${timeoutMs} ms`));
    }, timeoutMs);
    const onMessage = (e: MessageEvent<FromWorker>) => {
      if (e.data.type === "error") {
        cleanup();
        reject(new Error(e.data.message));
      } else if (e.data.type === expect) {
        cleanup();
        resolve(e.data as Extract<FromWorker, { type: T }>);
      }
    };
    const onError = (e: ErrorEvent) => {
      cleanup();
      reject(new Error(e.message || "worker crashed"));
    };
    const cleanup = () => {
      clearTimeout(timer);
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.postMessage(message);
  });
}

/** Model compile plus per-env allocation; generous, because a big pool on a
 *  cold cache is genuinely slow. */
const INIT_TIMEOUT_MS = 120_000;

/** Measure one (model, workers, envs) cell. */
export async function runCell(
  config: BenchConfig,
  workers: number,
  durationMs: number,
): Promise<CellResult> {
  // Written down BEFORE the pool exists: if this cell takes the tab with it,
  // the missing "done" is the only evidence left.
  const id = cellId(config, workers);
  markAttempt(id, cellLabel(config, workers));
  const pool = Array.from({ length: workers }, spawn);
  const base: CellResult = {
    robotXml: config.robotXml,
    workers,
    envsPerWorker: config.envs,
    withPolicy: config.withPolicy,
    envs: 0,
    controlStepsPerSec: 0,
    realtimeFactor: 0,
    bytesPerEnv: 0,
    heapTotalBytes: 0,
    ngeom: 0,
    oom: false,
  };

  try {
    // Barrier: every worker compiles and allocates before any of them steps,
    // so the measurement window is not polluted by startup.
    const stats: WorkerStats[] = await Promise.all(
      pool.map((w) =>
        ask(w, { type: "init", baseUrl: assetBase(), config }, "ready", INIT_TIMEOUT_MS),
      ),
    ).then((rs) => rs.map((r) => r.stats));

    const results = await Promise.all(
      pool.map((w) => ask(w, { type: "run", durationMs }, "result", durationMs + 60_000)),
    );

    const physicsSteps = results.reduce((n, r) => n + r.physicsSteps, 0);
    // Divide by the LONGEST window, not the mean: workers start within a
    // message-loop tick of each other, and the slowest one bounds the wall
    // clock a real trainer would see.
    const elapsedS = Math.max(...results.map((r) => r.elapsedMs)) / 1000;
    const controlStepsPerSec = physicsSteps / DECIMATION / elapsedS;

    return {
      ...base,
      envs: stats.reduce((n, s) => n + s.envs, 0),
      controlStepsPerSec,
      realtimeFactor: controlStepsPerSec * CTRL_DT,
      bytesPerEnv: stats.reduce((n, s) => n + s.bytesPerEnv, 0) / stats.length,
      heapTotalBytes: stats.reduce((n, s) => n + s.heapTotalBytes, 0),
      ngeom: stats[0]?.ngeom ?? 0,
      oom: stats.some((s) => s.oom),
    };
  } catch (err) {
    return { ...base, error: err instanceof Error ? err.message : String(err) };
  } finally {
    for (const w of pool) w.terminate();
    markDone(id);
  }
}

export function planCells(plan: SweepPlan): BenchConfig[] {
  const cells: BenchConfig[] = [];
  for (const robotXml of plan.robotXml) {
    for (const envs of plan.envsPerWorker) {
      cells.push({ robotXml, envs, memory: plan.memory, withPolicy: plan.withPolicy });
    }
  }
  return cells;
}

export interface SweepCallbacks {
  onCell?: (result: CellResult, index: number, total: number) => void;
  shouldStop?: () => boolean;
  /** Configurations to skip because they previously killed the tab. */
  skip?: Set<string>;
}

export async function runSweep(
  plan: SweepPlan,
  { onCell, shouldStop, skip }: SweepCallbacks = {},
): Promise<CellResult[]> {
  const cells = planCells(plan);
  const total = cells.length * plan.workers.length;
  const out: CellResult[] = [];
  let i = 0;
  for (const cell of cells) {
    for (const workers of plan.workers) {
      if (shouldStop?.()) return out;
      if (skip?.has(cellId(cell, workers))) {
        const skipped: CellResult = {
          robotXml: cell.robotXml, workers, envsPerWorker: cell.envs,
          withPolicy: cell.withPolicy, envs: 0, controlStepsPerSec: 0,
          realtimeFactor: 0, bytesPerEnv: 0, heapTotalBytes: 0, ngeom: 0,
          oom: false, skipped: true,
          error: "skipped — this configuration crashed the tab before",
        };
        out.push(skipped);
        onCell?.(skipped, ++i, total);
        continue;
      }
      const result = await runCell(cell, workers, plan.durationMs);
      out.push(result);
      onCell?.(result, ++i, total);
      // A cell that could not fit its environments will not fit more of them.
      if (result.oom) break;
    }
  }
  return out;
}
