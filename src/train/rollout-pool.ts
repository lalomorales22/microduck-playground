// A rollout collected across a pool of workers.
//
// Physics is the bottleneck once the SIMD kernels are in — 74% of a small-net
// iteration — and it is the one part that parallelises. The learner stays in
// one place; the environments spread out.
//
// Each worker owns a contiguous slice of the environment index space, and the
// chunks are interleaved back into the buffer's [step][env] layout so PPO sees
// exactly what a single VecEnv would have produced.

import { assetBase } from "../asset-url.ts";
import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import type { ActorCritic } from "./ac-policy.ts";
import type { RolloutBuffer } from "./ppo.ts";
import type { CollectStats, RolloutSource } from "./rollout-source.ts";
import type {
  FromRolloutWorker, RolloutChunk, RolloutInit, ToRolloutWorker,
} from "./rollout-protocol.ts";
import type { TaskSpec } from "./env/tasks.ts";
import type { WorldConfig } from "../sim/world.ts";

const INIT_TIMEOUT_MS = 120_000;
const STEP_TIMEOUT_MS = 300_000;

function ask<T extends FromRolloutWorker["type"]>(
  worker: Worker,
  message: ToRolloutWorker,
  expect: T,
  timeoutMs: number,
  transfer: Transferable[] = [],
): Promise<Extract<FromRolloutWorker, { type: T }>> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`rollout worker did not reply with "${expect}" in ${timeoutMs} ms`));
    }, timeoutMs);
    const onMessage = (e: MessageEvent<FromRolloutWorker>) => {
      if (e.data.type === "error") {
        cleanup();
        reject(new Error(e.data.message));
      } else if (e.data.type === expect) {
        cleanup();
        resolve(e.data as Extract<FromRolloutWorker, { type: T }>);
      }
    };
    const onError = (e: ErrorEvent) => {
      cleanup();
      reject(new Error(e.message || "rollout worker crashed"));
    };
    const cleanup = () => {
      clearTimeout(timer);
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.postMessage(message, transfer);
  });
}

export interface PoolOptions {
  workers: number;
  /** Environments PER WORKER. */
  envsPerWorker: number;
  robotXml: string;
  task: TaskSpec;
  world?: WorldConfig;
  hidden: readonly number[];
  baseSeed: number;
  episodeLengthS?: number;
}

export class PoolRollout implements RolloutSource {
  readonly label: string;
  readonly envs: number;
  #workers: Worker[] = [];
  readonly #envsPer: number;

  private constructor(workers: Worker[], envsPerWorker: number) {
    this.#workers = workers;
    this.#envsPer = envsPerWorker;
    this.envs = workers.length * envsPerWorker;
    this.label = `${workers.length} workers × ${envsPerWorker} envs`;
  }

  static async create(options: PoolOptions): Promise<PoolRollout> {
    const workers = Array.from(
      { length: options.workers },
      () => new Worker(new URL("./rollout-worker.ts", import.meta.url), {
        type: "module",
        name: "wicroduck-rollout",
      }),
    );
    try {
      await Promise.all(
        workers.map((w, i) => {
          const init: RolloutInit = {
            robotXml: options.robotXml,
            task: options.task,
            world: options.world,
            envs: options.envsPerWorker,
            // Distinct per worker, or every slice runs identical episodes.
            seed: options.baseSeed + i * 7919,
            episodeLengthS: options.episodeLengthS,
            hidden: options.hidden,
            obsDim: OBS_SIZE,
            actDim: NUM_JOINTS,
          };
          return ask(w, { type: "init", baseUrl: assetBase(), init }, "ready", INIT_TIMEOUT_MS);
        }),
      );
    } catch (err) {
      for (const w of workers) w.terminate();
      throw err;
    }
    return new PoolRollout(workers, options.envsPerWorker);
  }

  async syncPolicy(ac: ActorCritic): Promise<void> {
    const params = ac.flatParams();
    const { mean, var_, count } = ac.normalizer;
    // One copy per worker: structured clone would copy anyway, and a
    // transferred buffer cannot be sent twice.
    await Promise.all(
      this.#workers.map((w) =>
        ask(w, {
          type: "policy",
          params: params.slice(),
          mean: mean.slice(),
          var: var_.slice(),
          count,
        }, "synced", STEP_TIMEOUT_MS),
      ),
    );
  }

  async resetAll(): Promise<void> {
    await Promise.all(
      this.#workers.map((w) => ask(w, { type: "reset" }, "reset", STEP_TIMEOUT_MS)),
    );
  }

  async collect(buf: RolloutBuffer, steps: number): Promise<CollectStats> {
    const replies = await Promise.all(
      this.#workers.map((w) => ask(w, { type: "collect", steps }, "chunk", STEP_TIMEOUT_MS)),
    );
    const stats: CollectStats = {
      rewardSum: 0, standingSteps: 0, finishedEpisodes: 0,
      finishedReturn: 0, nonFiniteSteps: 0, steps: 0,
      taskMetricSums: {},
    };
    replies.forEach((r, wi) => this.#merge(buf, r.chunk, wi * this.#envsPer, steps, stats));
    return stats;
  }

  /** Interleave one worker's [step][sliceEnv] chunk into [step][allEnvs]. */
  #merge(
    buf: RolloutBuffer, chunk: RolloutChunk, offset: number, steps: number, stats: CollectStats,
  ): void {
    const total = buf.envs;
    const k = chunk.envs;
    for (let t = 0; t < steps; t++) {
      const dst = t * total + offset;
      const src = t * k;
      buf.obs.set(chunk.obs.subarray(src * OBS_SIZE, (src + k) * OBS_SIZE), dst * OBS_SIZE);
      buf.actions.set(chunk.actions.subarray(src * NUM_JOINTS, (src + k) * NUM_JOINTS), dst * NUM_JOINTS);
      buf.logProbs.set(chunk.logProbs.subarray(src, src + k), dst);
      buf.values.set(chunk.values.subarray(src, src + k), dst);
      buf.rewards.set(chunk.rewards.subarray(src, src + k), dst);
      buf.dones.set(chunk.dones.subarray(src, src + k), dst);
      buf.timeouts.set(chunk.timeouts.subarray(src, src + k), dst);
    }
    buf.lastValues.set(chunk.lastValues, offset);
    stats.rewardSum += chunk.rewardSum;
    stats.standingSteps += chunk.standingSteps;
    stats.finishedEpisodes += chunk.finishedEpisodes;
    stats.finishedReturn += chunk.finishedReturn;
    stats.nonFiniteSteps += chunk.nonFiniteSteps;
    stats.steps += steps * k;
    for (const [name, value] of Object.entries(chunk.taskMetricSums)) {
      stats.taskMetricSums[name] = (stats.taskMetricSums[name] ?? 0) + value;
    }
  }

  dispose(): Promise<void> {
    for (const w of this.#workers) w.terminate();
    this.#workers = [];
    return Promise.resolve();
  }
}
