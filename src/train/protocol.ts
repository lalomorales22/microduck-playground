// Messages between the main thread and a sim worker.

export interface BenchConfig {
  /** MJCF variant, e.g. "robot_walk.xml". */
  robotXml: string;
  /** Environments this worker owns. */
  envs: number;
  /** MuJoCo arena size per MjData ("1M"), or undefined for MuJoCo's default. */
  memory?: string;
  /**
   * Run a policy-shaped forward pass per control step. Physics-only numbers
   * flatter the real trainer, which has to infer an action for every step.
   */
  withPolicy: boolean;
}

export type ToWorker =
  | { type: "init"; baseUrl: string; config: BenchConfig }
  | { type: "run"; durationMs: number }
  | { type: "dispose" };

export interface WorkerStats {
  /** Wasm heap after compiling the model, before any MjData. */
  heapAfterCompileBytes: number;
  /** Wasm heap once every environment is allocated. */
  heapTotalBytes: number;
  /** Marginal cost of one environment. */
  bytesPerEnv: number;
  ngeom: number;
  nq: number;
  /** Environments actually allocated — fewer than asked if the heap ran out. */
  envs: number;
  /** Set when the wasm heap could not grow far enough for `config.envs`. */
  oom: boolean;
}

export type FromWorker =
  | { type: "ready"; stats: WorkerStats }
  | { type: "result"; physicsSteps: number; elapsedMs: number }
  | { type: "error"; message: string };
