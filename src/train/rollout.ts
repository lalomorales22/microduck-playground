// Drives a pool of env workers from the main thread.
//
// This is the shape M2's PPO loop will call: configure a pool once, then ask
// every worker for the same number of steps and merge what comes back. M1
// returns aggregate statistics; M2 extends the reply with the transition
// buffers PPO needs, without changing the pool's lifecycle.

import { assetBase } from "../asset-url.ts";
import { CTRL_DT } from "../sim/microduck.ts";
import type { EnvInit, FromEnvWorker, RolloutStats, ToEnvWorker } from "./env-protocol.ts";

const INIT_TIMEOUT_MS = 120_000;
const ROLLOUT_TIMEOUT_MS = 300_000;

function ask<T extends FromEnvWorker["type"]>(
  worker: Worker,
  message: ToEnvWorker,
  expect: T,
  timeoutMs: number,
): Promise<Extract<FromEnvWorker, { type: T }>> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`env worker did not reply with "${expect}" within ${timeoutMs} ms`));
    }, timeoutMs);
    const onMessage = (e: MessageEvent<FromEnvWorker>) => {
      if (e.data.type === "error") {
        cleanup();
        reject(new Error(e.data.message));
      } else if (e.data.type === expect) {
        cleanup();
        resolve(e.data as Extract<FromEnvWorker, { type: T }>);
      }
    };
    const onError = (e: ErrorEvent) => {
      cleanup();
      reject(new Error(e.message || "env worker crashed"));
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

export interface PoolConfig extends Omit<EnvInit, "seed"> {
  workers: number;
  /** Each worker gets `baseSeed + index`, so the pool is reproducible. */
  baseSeed?: number;
}

export class EnvPool {
  #workers: Worker[] = [];
  #envs = 0;

  get totalEnvs(): number {
    return this.#envs;
  }

  get workerCount(): number {
    return this.#workers.length;
  }

  async start(config: PoolConfig): Promise<void> {
    await this.dispose();
    const baseSeed = config.baseSeed ?? 1;
    this.#workers = Array.from(
      { length: config.workers },
      () => new Worker(new URL("./env-worker.ts", import.meta.url), {
        type: "module",
        name: "wicroduck-env",
      }),
    );
    try {
      const ready = await Promise.all(
        this.#workers.map((w, i) =>
          ask(
            w,
            {
              type: "init",
              baseUrl: assetBase(),
              init: {
                robotXml: config.robotXml,
                envs: config.envs,
                resetMix: config.resetMix,
                episodeLengthS: config.episodeLengthS,
                seed: baseSeed + i,
              },
            },
            "ready",
            INIT_TIMEOUT_MS,
          ),
        ),
      );
      this.#envs = ready.reduce((n, r) => n + r.envs, 0);
    } catch (err) {
      await this.dispose();
      throw err;
    }
  }

  /**
   * Ask every worker for `steps` control steps and merge the replies.
   *
   * Workers run concurrently and each reports its own elapsed time; the pool's
   * throughput is bounded by the slowest, so aggregate rate divides by the max
   * rather than the mean.
   */
  async rollout(steps: number): Promise<RolloutStats & { stepsPerSec: number; realtimeFactor: number }> {
    if (this.#workers.length === 0) throw new Error("pool not started");
    const parts = await Promise.all(
      this.#workers.map((w) => ask(w, { type: "rollout", steps }, "rollout", ROLLOUT_TIMEOUT_MS)),
    ).then((rs) => rs.map((r) => r.stats));

    const controlSteps = parts.reduce((n, p) => n + p.controlSteps, 0);
    const elapsedMs = Math.max(...parts.map((p) => p.elapsedMs));
    const weight = (pick: (p: RolloutStats) => number) =>
      parts.reduce((n, p) => n + pick(p) * p.controlSteps, 0) / controlSteps;

    const breakdown: Record<string, number> = {};
    for (const key of Object.keys(parts[0]?.breakdown ?? {})) {
      breakdown[key] = weight((p) => p.breakdown[key] ?? 0);
    }

    const stepsPerSec = controlSteps / (elapsedMs / 1000);
    return {
      envs: this.#envs,
      steps,
      controlSteps,
      elapsedMs,
      rewardPerStep: weight((p) => p.rewardPerStep),
      episodes: parts.reduce((n, p) => n + p.episodes, 0),
      standingFraction: weight((p) => p.standingFraction),
      breakdown,
      stepsPerSec,
      realtimeFactor: stepsPerSec * CTRL_DT,
    };
  }

  async dispose(): Promise<void> {
    for (const w of this.#workers) w.terminate();
    this.#workers = [];
    this.#envs = 0;
  }
}
