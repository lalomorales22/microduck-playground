// Messages between the main thread and an env worker.

export interface EnvInit {
  robotXml: string;
  /** Environments this worker owns. */
  envs: number;
  /** Seeded per worker so a pool is reproducible as a whole. */
  seed: number;
  /** Reset distribution weights: stand / sit / tumble. */
  resetMix: { stand?: number; sit?: number; tumble?: number };
  episodeLengthS: number;
}

export type ToEnvWorker =
  | { type: "init"; baseUrl: string; init: EnvInit }
  | { type: "rollout"; steps: number }
  | { type: "dispose" };

export interface RolloutStats {
  envs: number;
  steps: number;
  controlSteps: number;
  elapsedMs: number;
  rewardPerStep: number;
  /** Episodes that ended during this rollout. */
  episodes: number;
  /** Share of steps where the duck was at height and upright. */
  standingFraction: number;
  /** Per-term reward per step, for spotting a term that dominates or is dead. */
  breakdown: Record<string, number>;
}

export type FromEnvWorker =
  | { type: "ready"; envs: number; ngeom: number }
  | { type: "rollout"; stats: RolloutStats }
  | { type: "error"; message: string };
