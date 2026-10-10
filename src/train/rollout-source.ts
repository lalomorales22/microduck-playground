// Where a rollout comes from.
//
// The learner does not care whether the environments live in this thread or
// across a worker pool — it needs a filled RolloutBuffer and the statistics
// that go with it. Two implementations:
//
//   LocalRollout  one VecEnv here. What the headless checks use, and the
//                 reference behaviour everything else is compared against.
//   PoolRollout   N workers, each with its own slice of environments.
//
// Inference runs wherever the environments are, never across the boundary:
// the policy has to act once per control step per environment, and a message
// round trip in that loop would dominate everything else.

import type { ActorCritic } from "./ac-policy.ts";
import type { RolloutBuffer } from "./ppo.ts";

export interface CollectStats {
  rewardSum: number;
  standingSteps: number;
  finishedEpisodes: number;
  finishedReturn: number;
  nonFiniteSteps: number;
  /** Control steps collected, across every environment. */
  steps: number;
  taskMetricSums: Record<string, number>;
}

export interface RolloutSource {
  /** Total environments across the source. */
  readonly envs: number;
  /** Human-readable, for the UI. */
  readonly label: string;

  /**
   * Push the current policy to wherever inference happens.
   *
   * Both the weights AND the observation normalizer: a worker running with
   * stale normalization produces log-probs the learner cannot reconcile, and
   * PPO's importance ratio quietly stops meaning anything.
   */
  syncPolicy(ac: ActorCritic): Promise<void>;

  /** Fill `buf` with `steps` control steps per environment. */
  collect(buf: RolloutBuffer, steps: number): Promise<CollectStats>;

  /** Reset every environment, staggered. */
  resetAll(): Promise<void>;

  dispose(): Promise<void>;
}
