// Reactive wrapper around the M0 throughput sweep.

import { runSweep, type CellResult, type SweepPlan } from "../train/benchmark.ts";
import { clearCrashes, crashedIds, crashedList, type Attempt } from "../train/crash-log.ts";
import { EnvPool } from "../train/rollout.ts";
import type { RolloutStats } from "../train/env-protocol.ts";

/** Reference figures the sweep is judged against (see docs/training-plan.md). */
export const REFERENCE_STEPS = 4096 * 24 * 15_000; // ~1.47B control steps
export const PLAN_ESTIMATE = 40_000; // control steps/s the plan assumed

// The "-nv" twins are the training models: same dynamics, none of the visual
// meshes, so a worker pool starts fast and stays inside the wasm heap.
const MODELS = [
  "robot_walk-nv.xml",
  "robot_groundcontact-nv.xml",
  "robot_allcollisions-nv.xml",
  "robot_walk.xml",
  "robot_allcollisions.xml",
];

export class Bench {
  /** Cores the browser admits to. Chrome caps this at 8 on some platforms. */
  readonly cores = navigator.hardwareConcurrency || 4;

  models = $state<string[]>(["robot_walk-nv.xml"]);
  envsPerWorker = $state<number[]>([16, 64]);
  /** Every pool size the sweep could try. */
  allWorkerCounts = $state<number[]>([]);
  /** The subset selected in the UI. */
  workerCounts = $state<number[]>([]);
  withPolicy = $state(true);
  capMemory = $state(false);
  durationMs = $state(2000);

  running = $state(false);
  /** Configurations that killed the tab on a previous run; skipped, not retried. */
  crashed = $state<Attempt[]>(crashedList());
  progress = $state({ done: 0, total: 0 });
  results = $state<CellResult[]>([]);
  error = $state<string | null>(null);

  // ── M1: the vectorized environment ─────────────────────────────────────
  envWorkers = $state(4);
  envsPerEnvWorker = $state(16);
  envRunning = $state(false);
  envStats = $state<(RolloutStats & { stepsPerSec: number; realtimeFactor: number }) | null>(null);
  envError = $state<string | null>(null);

  readonly #pool = new EnvPool();

  #stop = false;

  constructor() {
    // 1, 2, 4, ... up to the reported core count, always including it.
    const counts: number[] = [];
    for (let n = 1; n <= this.cores; n *= 2) counts.push(n);
    if (counts[counts.length - 1] !== this.cores) counts.push(this.cores);
    this.allWorkerCounts = counts;
    this.workerCounts = counts;
  }

  get allModels(): string[] {
    return MODELS;
  }

  /** Best cell so far, by aggregate throughput. */
  get best(): CellResult | null {
    return this.results.reduce<CellResult | null>(
      (b, r) => (!r.error && (!b || r.controlStepsPerSec > b.controlStepsPerSec) ? r : b),
      null,
    );
  }

  /** Hours to replay the full reference recipe at the best measured rate. */
  get referenceHours(): number | null {
    const best = this.best;
    return best ? REFERENCE_STEPS / best.controlStepsPerSec / 3600 : null;
  }

  toggle<T>(list: T[], value: T): T[] {
    return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
  }

  stop(): void {
    this.#stop = true;
  }

  /** Let previously-crashing configurations be attempted again. */
  forgetCrashes(): void {
    clearCrashes();
    this.crashed = [];
  }

  /**
   * Run one rollout through the real vectorized env, in workers.
   *
   * The policy here is a randomly initialised MLP, so the reward is a floor,
   * not a result — what this proves is that the env steps, resets, scores and
   * terminates correctly at pool scale in a browser. Whether the reward stack
   * ranks good behaviour above bad is settled by `npm run check:env`, which
   * replays the shipped alpha_stand policy against these same modules.
   */
  async runEnv(steps = 300): Promise<void> {
    if (this.envRunning) return;
    this.envRunning = true;
    this.envError = null;
    this.envStats = null;
    try {
      await this.#pool.start({
        workers: this.envWorkers,
        envs: this.envsPerEnvWorker,
        robotXml: "robot_allcollisions-nv.xml",
        resetMix: { sit: 1, tumble: 1 },
        episodeLengthS: 6,
      });
      this.envStats = await this.#pool.rollout(steps);
    } catch (err) {
      this.envError = err instanceof Error ? err.message : String(err);
    } finally {
      await this.#pool.dispose();
      this.envRunning = false;
    }
  }

  async run(): Promise<void> {
    if (this.running) return;
    if (this.models.length === 0 || this.envsPerWorker.length === 0 || this.workerCounts.length === 0) {
      this.error = "Pick at least one model, env count and worker count.";
      return;
    }
    this.running = true;
    this.#stop = false;
    this.error = null;
    this.results = [];

    const plan: SweepPlan = {
      robotXml: this.models,
      workers: [...this.workerCounts].sort((a, b) => a - b),
      envsPerWorker: [...this.envsPerWorker].sort((a, b) => a - b),
      withPolicy: this.withPolicy,
      memory: this.capMemory ? "1M" : undefined,
      durationMs: this.durationMs,
    };

    try {
      // A cell that took the whole tab down last time is not retried — it
      // would just take it down again, and the log is the only record.
      const skip = crashedIds();
      await runSweep(plan, {
        skip,
        // Reassign rather than push: `$state` tracks the reference, and the
        // table should repaint as each cell lands.
        onCell: (result, done, total) => {
          this.results = [...this.results, result];
          this.progress = { done, total };
        },
        shouldStop: () => this.#stop,
      });
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.running = false;
    }
  }
}
