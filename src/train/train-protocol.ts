// Messages between the main thread and the training worker.

import type { IterationStats, TrainerConfig } from "./trainer.ts";
import type { TaskSpec } from "./env/tasks.ts";
import type { WorldConfig } from "../sim/world.ts";

export interface TrainInit {
  robotXml: string;
  /**
   * Rollout workers. 1 keeps the environments in the learner's own thread;
   * more spreads them out, which is what physics being ~74% of an iteration
   * calls for.
   */
  rolloutWorkers: number;
  /** What to train. "hold_pose" bootstraps, "standup" is the real one, and a
   *  motion task carries its file so every worker builds the same reward. */
  task: TaskSpec;
  world?: WorldConfig;
  config: TrainerConfig;
  /** Write a checkpoint every N iterations; 0 disables autosave. */
  autosaveEvery: number;
  checkpointName: string;
  /** Continue from the stored checkpoint of that name, if one exists. */
  resume: boolean;
  /** Read-only resume source; checkpointName is the new run's save slot. */
  resumeName?: string;
}

export type ToTrainWorker =
  | { type: "init"; baseUrl: string; init: TrainInit }
  | { type: "start"; iterations: number }
  | { type: "stop" }
  | { type: "save"; name?: string }
  | { type: "dispose" };

export type FromTrainWorker =
  | { type: "ready"; resumedAt: number; params: number; simd: boolean;
      envs: number; rolloutLabel: string; task: TaskSpec; world: WorldConfig }
  | { type: "stats"; stats: IterationStats }
  | { type: "saved"; name: string; iteration: number }
  | { type: "stopped"; iteration: number; reason: "completed" | "stopped"; checkpointName: string | null }
  | { type: "error"; message: string };
