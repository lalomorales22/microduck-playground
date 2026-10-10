// A rollout collected in this thread, from one VecEnv.
//
// The reference implementation: what the headless checks run, and what the
// worker-pool version has to agree with.

import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import type { ActorCritic } from "./ac-policy.ts";
import { STAND_Z } from "./env/rewards.ts";
import type { VecEnv } from "./env/vec-env.ts";
import type { RolloutBuffer } from "./ppo.ts";
import type { CollectStats, RolloutSource } from "./rollout-source.ts";

export class LocalRollout implements RolloutSource {
  readonly label = "in-thread";
  readonly #env: VecEnv;
  readonly #ac: ActorCritic;
  readonly #rng: () => number;
  readonly #episodeReturn: Float32Array;

  constructor(env: VecEnv, ac: ActorCritic, rng: () => number) {
    this.#env = env;
    this.#ac = ac;
    this.#rng = rng;
    this.#episodeReturn = new Float32Array(env.count);
  }

  get envs(): number {
    return this.#env.count;
  }

  /** Nothing to push: the policy object is already the one acting. */
  syncPolicy(): Promise<void> {
    return Promise.resolve();
  }

  resetAll(): Promise<void> {
    this.#env.resetAll();
    return Promise.resolve();
  }

  collect(buf: RolloutBuffer, steps: number): Promise<CollectStats> {
    const env = this.#env;
    const envs = env.count;
    const stats: CollectStats = {
      rewardSum: 0, standingSteps: 0, finishedEpisodes: 0,
      finishedReturn: 0, nonFiniteSteps: 0, steps: steps * envs,
      taskMetricSums: {},
    };
    const before = env.nonFiniteSteps;

    for (let t = 0; t < steps; t++) {
      const obs = env.observations;
      buf.obs.set(obs, t * envs * OBS_SIZE);
      const { actions, logProbs, values } = this.#ac.act(obs, envs, this.#rng);
      buf.actions.set(actions, t * envs * NUM_JOINTS);
      buf.logProbs.set(logProbs, t * envs);
      buf.values.set(values, t * envs);

      const { reward, done, timeout } = env.step(actions);
      for (const [name, values] of Object.entries(env.taskMetrics)) {
        stats.taskMetricSums[name] = (stats.taskMetricSums[name] ?? 0) + values.reduce((a, b) => a + b, 0);
      }
      buf.rewards.set(reward, t * envs);
      buf.dones.set(done, t * envs);
      buf.timeouts.set(timeout, t * envs);

      const z = env.trunkZ, upright = env.uprightness;
      for (let e = 0; e < envs; e++) {
        stats.rewardSum += reward[e];
        this.#episodeReturn[e] += reward[e];
        if (z[e] > STAND_Z - 0.02 && upright[e] > 0.85) stats.standingSteps++;
        if (done[e]) {
          stats.finishedEpisodes++;
          stats.finishedReturn += this.#episodeReturn[e];
          this.#episodeReturn[e] = 0;
        }
      }
    }
    buf.lastValues.set(this.#ac.value(env.observations, envs).subarray(0, envs));
    stats.nonFiniteSteps = env.nonFiniteSteps - before;
    return Promise.resolve(stats);
  }

  dispose(): Promise<void> {
    return Promise.resolve();
  }
}
