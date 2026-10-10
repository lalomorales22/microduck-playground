// Messages between the learner and a rollout worker.

import type { TaskSpec } from "./env/tasks.ts";
import type { WorldConfig } from "../sim/world.ts";

export interface RolloutInit {
  robotXml: string;
  task: TaskSpec;
  world?: WorldConfig;
  envs: number;
  seed: number;
  episodeLengthS?: number;
  hidden: readonly number[];
  obsDim: number;
  actDim: number;
}

export type ToRolloutWorker =
  | { type: "init"; baseUrl: string; init: RolloutInit }
  | { type: "policy"; params: Float32Array; mean: Float32Array; var: Float32Array; count: number }
  | { type: "collect"; steps: number }
  | { type: "reset" }
  | { type: "dispose" };

/** One worker's slice of a rollout, laid out [steps][envs] like the buffer. */
export interface RolloutChunk {
  envs: number;
  steps: number;
  obs: Float32Array;
  actions: Float32Array;
  logProbs: Float32Array;
  values: Float32Array;
  rewards: Float32Array;
  dones: Uint8Array;
  timeouts: Uint8Array;
  lastValues: Float32Array;
  rewardSum: number;
  standingSteps: number;
  finishedEpisodes: number;
  finishedReturn: number;
  nonFiniteSteps: number;
  taskMetricSums: Record<string, number>;
}

export type FromRolloutWorker =
  | { type: "ready"; envs: number }
  | { type: "synced" }
  | { type: "reset" }
  | { type: "chunk"; chunk: RolloutChunk }
  | { type: "error"; message: string };
