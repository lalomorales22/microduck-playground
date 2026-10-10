// Reactive wrapper around the training worker.

import { assetBase } from "../asset-url.ts";
import { DEFAULT_PPO } from "../train/ppo.ts";
import { DEFAULT_TRAINER, type Checkpoint, type IterationStats, type TrainerConfig } from "../train/trainer.ts";
import type { FromTrainWorker, ToTrainWorker } from "../train/train-protocol.ts";
import { deleteCheckpoint, listCheckpoints, loadCheckpoint, opfsAvailable, type CheckpointInfo } from "../train/checkpoint-store.ts";
import { BUILTIN_TASKS, type TaskSpec } from "../train/env/tasks.ts";
import { listMotions, loadMotionFile, readMotionFile, type MotionEntry } from "../motion/motion-store.ts";
import type { MotionFile } from "../motion/format.ts";
import { defaultWorld, validateWorld, type WorldConfig } from "../sim/world.ts";
import { checkpointMetadata, validateTrainingConfig } from "../train/checkpoint-task.ts";
import type { EvaluationReport } from "../train/evaluation.ts";
import type { FromEvaluationWorker } from "../train/evaluation-protocol.ts";

const WORLD_KEY = "wicroduck.world.v1";
/** Points kept for the sparkline; older ones are dropped. */
const HISTORY = 400;
const EVALUATION_TIMEOUT_MS = 120_000;

export interface HistoryPoint {
  iteration: number;
  reward: number;
  standing: number;
}

/** The task picker's value. Motions are `motion:<id>`, where the id is the
 *  motion-store id — so a built-in and a saved file of the same name stay
 *  distinct. */
export type TaskChoice = string;
export const MOTION_TASK = "motion:";
export interface ExpectedMotionRevision { id: string; modified: number; bytes: number }

export class TrainingSession {
  task = $state<TaskChoice>("hold_pose");
  /** Saved and bundled motion targets offered beside the built-in tasks. */
  motions = $state<MotionEntry[]>([]);
  /** Environments PER rollout worker. */
  envs = $state(32);
  /** 1 keeps environments in the learner's thread; more spreads the physics
   *  out, which is where the remaining speed is. */
  rolloutWorkers = $state(1);
  stepsPerIter = $state(24);
  iterations = $state(400);
  seed = $state(7);
  useCurrentWorld = $state(false);
  worldObjectCount = $state(0);
  worldError = $state<string | null>(null);
  runWorldLabel = $state("Default floor");
  runName = $state<string | null>(null);
  runId = $state<string | null>(null);
  runTask = $state<TaskSpec | null>(null);
  completionReason = $state<"completed" | "stopped" | "error" | null>(null);
  targetIteration = $state(0);
  stopping = $state(false);
  /** Smaller than the reference 512/256/128: scalar-JS backprop is the cost. */
  hiddenPreset = $state<"small" | "reference">("small");

  status = $state<"idle" | "loading" | "running" | "stopped" | "error">("idle");
  error = $state<string | null>(null);
  iteration = $state(0);
  totalSteps = $state(0);
  paramCount = $state(0);
  totalEnvs = $state(0);
  rolloutLabel = $state("");
  resumedAt = $state(0);
  simd = $state(false);
  last = $state<IterationStats | null>(null);
  history = $state<HistoryPoint[]>([]);
  checkpoints = $state<CheckpointInfo[]>([]);
  savedAt = $state(0);
  savedName = $state<string | null>(null);
  /** Resume is always an explicitly selected checkpoint. */
  resumeFrom = $state<string | null>(null);
  resumeSummary = $state<string | null>(null);
  evaluation = $state<EvaluationReport | null>(null);
  evaluationStatus = $state<"idle" | "running" | "done" | "error">("idle");
  evaluationProgress = $state(0);
  evaluationError = $state<string | null>(null);
  evaluatingName = $state<string | null>(null);

  readonly opfs = opfsAvailable();
  #worker: Worker | null = null;
  #disposed = false;
  #requestId = 0;
  #selectionId = 0;
  #evaluationWorker: Worker | null = null;
  #cancelEvaluation: (() => void) | null = null;

  get taskInfo() {
    return BUILTIN_TASKS.find((entry) => entry.kind === this.task);
  }

  get runTaskInfo() {
    return BUILTIN_TASKS.find((entry) => entry.kind === this.runTask?.kind);
  }

  applyBudget(iterations: number): void {
    this.iterations = iterations;
  }

  #world(): WorldConfig {
    const raw = globalThis.localStorage.getItem(WORLD_KEY);
    return raw ? validateWorld(JSON.parse(raw)) : defaultWorld();
  }

  refreshWorld(): void {
    try {
      this.worldObjectCount = this.#world().objects.length;
      this.worldError = null;
    } catch (err) {
      this.worldError = err instanceof Error ? err.message : String(err);
    }
  }

  get hidden(): readonly number[] {
    return this.hiddenPreset === "reference" ? [512, 256, 128] : [128, 64];
  }

  get bestReward(): number {
    return this.history.reduce((b, p) => Math.max(b, p.reward), -Infinity);
  }

  async refreshCheckpoints(): Promise<void> {
    try {
      this.checkpoints = await listCheckpoints();
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    }
  }

  async refreshMotions(): Promise<void> {
    // A file that will not parse is listed with its error rather than
    // silently dropped, but it cannot be trained on.
    this.motions = (await listMotions()).filter((m) => !m.error);
    if (this.task.startsWith(MOTION_TASK) && !this.motions.some((m) => m.id === this.motionId)) {
      this.task = "hold_pose";
    }
  }

  /** The selected motion's store id, or null when a plain task is selected. */
  get motionId(): string | null {
    return this.task.startsWith(MOTION_TASK) ? this.task.slice(MOTION_TASK.length) : null;
  }

  /** Human name for the current task, for save names and labels. */
  get taskLabel(): string {
    const id = this.motionId;
    if (!id) return this.taskInfo?.label ?? this.task;
    return this.motions.find((m) => m.id === id)?.name ?? id;
  }

  /**
   * Resolve the picker's value into the task the workers get.
   *
   * The motion FILE travels, not its name: the learner and every rollout
   * worker have to build the same reward, and each of them looking the name up
   * in storage is a way for them to disagree.
   */
  async #task(expectedMotion?: ExpectedMotionRevision): Promise<TaskSpec> {
    const id = this.motionId;
    if (expectedMotion) {
      const { id: expectedId, modified, bytes } = expectedMotion;
      if (!id || id.startsWith("builtin:") || id !== expectedId) throw new Error("The selected motion differs from the approved motion.");
      if (!Number.isSafeInteger(modified) || modified < 0 || !Number.isSafeInteger(bytes) || bytes < 1) {
        throw new Error("The approved motion revision is invalid.");
      }
      const file = await readMotionFile(id);
      if (file.lastModified !== modified || file.size !== bytes) throw new Error("The saved motion changed after approval. Review it again before starting.");
      // Read the same immutable File whose revision was checked above.
      return { kind: "motion", motion: JSON.parse(await file.text()) as MotionFile };
    }
    if (!id) {
      const selected = BUILTIN_TASKS.find((entry) => entry.kind === this.task);
      if (!selected) throw new Error("Select a supported training task.");
      return { kind: selected.kind } as TaskSpec;
    }
    const motion = await loadMotionFile(id);
    if (!motion) throw new Error(`motion "${id}" is gone`);
    return { kind: "motion", motion };
  }

  #config(): TrainerConfig {
    return {
      ...DEFAULT_TRAINER,
      envs: this.envs,
      stepsPerIter: this.stepsPerIter,
      hidden: this.hidden,
      seed: this.seed,
      // Exploration is left to the TASK: hold-pose is destroyed by action
      // noise, stand-up cannot be discovered without it. See ExplorationHints.
      ppo: { ...DEFAULT_PPO, epochs: 4 },
    };
  }

  async selectCheckpoint(name: string): Promise<void> {
    if (this.#disposed) return;
    const selectionId = ++this.#selectionId;
    if (this.resumeFrom === name) {
      this.resumeFrom = null;
      this.resumeSummary = null;
      return;
    }
    try {
      const checkpoint = await loadCheckpoint<Checkpoint>(name);
      if (selectionId !== this.#selectionId || this.#disposed) return;
      if (!checkpoint) throw new Error("The selected checkpoint no longer exists.");
      const kind = checkpoint.taskSpec?.kind ?? checkpoint.task;
      const label = BUILTIN_TASKS.find((entry) => entry.kind === kind)?.label ?? kind ?? "Legacy task";
      const summary = `${label} · iteration ${checkpoint.iteration} · seed ${checkpoint.config.seed}`;
      this.resumeFrom = name;
      this.resumeSummary = summary;
      this.error = null;
    } catch (err) {
      if (selectionId !== this.#selectionId || this.#disposed) return;
      this.resumeFrom = null;
      this.resumeSummary = null;
      this.error = err instanceof Error ? err.message : String(err);
    }
  }

  async #resumeTask(checkpoint: Checkpoint): Promise<TaskSpec> {
    return checkpointMetadata(checkpoint).task;
  }

  async start(resume: boolean, expectedMotion?: ExpectedMotionRevision): Promise<void> {
    if (this.#disposed || this.status === "running" || this.status === "loading" || this.evaluationStatus === "running") return;
    this.#selectionId++;
    const requestId = ++this.#requestId;
    const iterations = this.iterations;
    const rolloutWorkers = this.rolloutWorkers;
    const resumeName = resume ? this.resumeFrom : null;
    this.#worker?.terminate();
    this.#worker = null;
    this.runId = crypto.randomUUID();
    this.completionReason = null;
    this.runName = null;
    this.runTask = null;
    this.savedName = null;
    this.savedAt = 0;
    this.targetIteration = !resume && Number.isInteger(iterations) && iterations >= 1 && iterations <= 5000 ? iterations : 0;
    this.iteration = 0;
    this.totalSteps = 0;
    this.resumedAt = 0;
    this.history = [];
    this.last = null;
    this.stopping = false;
    this.evaluation = null;
    this.evaluationStatus = "idle";
    this.evaluationProgress = 0;
    this.evaluationError = null;
    this.evaluatingName = null;
    this.status = "loading";
    this.error = null;
    let task: TaskSpec;
    let world: WorldConfig;
    let config: TrainerConfig;
    let checkpointName: string;
    try {
      if (!Number.isInteger(iterations) || iterations < 1 || iterations > 5000) {
        throw new Error("Iterations must be a whole number from 1 to 5,000.");
      }
      if (!resume && (!Number.isInteger(this.seed) || this.seed < 0 || this.seed > 4294967295)) {
        throw new Error("Seed must be a whole number from 0 to 4,294,967,295.");
      }
      if (resume) {
        if (!resumeName) throw new Error("Select a saved run to resume.");
        const checkpoint = await loadCheckpoint<Checkpoint>(resumeName);
        if (!checkpoint) throw new Error("The selected checkpoint no longer exists.");
        task = await this.#resumeTask(checkpoint);
        if (requestId !== this.#requestId) return;
        config = checkpoint.config;
        world = checkpoint.world ? validateWorld(checkpoint.world) : defaultWorld();
        this.iteration = checkpoint.iteration;
        this.seed = config.seed;
        this.envs = config.envs;
        this.stepsPerIter = config.stepsPerIter;
        this.totalSteps = checkpoint.totalSteps;
        this.hiddenPreset = config.hidden.length === 3 ? "reference" : "small";
        if (task.kind !== "motion") this.task = task.kind;
      } else {
        config = validateTrainingConfig(this.#config());
        world = this.useCurrentWorld ? this.#world() : defaultWorld();
        task = await this.#task(expectedMotion);
      }
      config = validateTrainingConfig(config);
      if (!Number.isInteger(rolloutWorkers) || rolloutWorkers < 1 || rolloutWorkers > 8 || config.envs * rolloutWorkers > 128) {
        throw new Error("Use 1 to 8 rollout workers and at most 128 total environments.");
      }
      if (!Number.isSafeInteger(this.iteration) || this.iteration < 0 || !Number.isSafeInteger(this.iteration + iterations)) {
        throw new Error("The checkpoint iteration plus this budget exceeds the supported range.");
      }
      checkpointName = `${task.kind}-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;
    } catch (err) {
      if (requestId !== this.#requestId) return;
      this.error = err instanceof Error ? err.message : String(err);
      this.status = "error";
      this.completionReason = "error";
      return;
    }
    if (requestId !== this.#requestId) return;
    if (!resume) {
      this.iteration = 0;
      this.totalSteps = 0;
    }
    this.history = [];
    this.last = null;
    this.savedAt = 0;
    this.savedName = null;
    this.stopping = false;
    this.runName = checkpointName;
    this.targetIteration = this.iteration + iterations;
    this.runTask = task;
    this.runWorldLabel = world.objects.length ? `World snapshot · ${world.objects.length} objects` : "Default floor";

    let worker: Worker;
    try {
      worker = new Worker(new URL("../train/train-worker.ts", import.meta.url), {
        type: "module", name: "wicroduck-train",
      });
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
      this.status = "error";
      this.completionReason = "error";
      return;
    }
    this.#worker = worker;

    worker.onmessage = (e: MessageEvent<FromTrainWorker>) => {
      if (this.#worker !== worker) return;
      const msg = e.data;
      if (msg.type === "ready") {
        this.paramCount = msg.params;
        this.simd = msg.simd;
        this.totalEnvs = msg.envs;
        this.rolloutLabel = msg.rolloutLabel;
        this.resumedAt = msg.resumedAt;
        this.iteration = msg.resumedAt;
        this.targetIteration = msg.resumedAt + iterations;
        this.runTask = msg.task;
        this.runWorldLabel = msg.world.objects.length ? `World snapshot · ${msg.world.objects.length} objects` : "Default floor";
        this.status = "running";
        this.#send({ type: "start", iterations });
      } else if (msg.type === "stats") {
        this.last = msg.stats;
        this.iteration = msg.stats.iteration;
        this.totalSteps = msg.stats.totalSteps;
        const next = [
          ...this.history,
          {
            iteration: msg.stats.iteration,
            reward: msg.stats.rewardPerStep,
            standing: msg.stats.standingFraction,
          },
        ];
        this.history = next.length > HISTORY ? next.slice(next.length - HISTORY) : next;
      } else if (msg.type === "saved") {
        this.savedAt = msg.iteration;
        this.savedName = msg.name;
        if (this.evaluation?.checkpointName === msg.name) {
          this.evaluation = null;
          this.evaluationStatus = "idle";
          this.evaluationProgress = 0;
        }
        void this.refreshCheckpoints();
      } else if (msg.type === "stopped") {
        this.status = "stopped";
        this.iteration = msg.iteration;
        this.completionReason = msg.reason;
        this.stopping = false;
        if (this.#disposed) worker.terminate();
        else if (msg.checkpointName) void this.evaluateCheckpoint(msg.checkpointName).catch(() => {});
      } else {
        this.error = msg.message;
        this.status = "error";
        this.completionReason = "error";
        this.stopping = false;
        worker.terminate();
        this.#worker = null;
      }
    };
    worker.onerror = (e) => {
      if (this.#worker !== worker) return;
      this.error = e.message || "training worker crashed";
      this.status = "error";
      this.completionReason = "error";
      this.stopping = false;
      worker.terminate();
      this.#worker = null;
    };

    this.#send({
      type: "init",
      baseUrl: assetBase(),
      init: {
        robotXml: "robot_allcollisions-nv.xml",
        rolloutWorkers,
        task,
        world,
        config,
        autosaveEvery: 25,
        checkpointName,
        resume,
        ...(resumeName ? { resumeName } : {}),
      },
    });
  }

  stop(): void {
    if (this.status === "loading") {
      this.#requestId++;
      this.#worker?.terminate();
      this.#worker = null;
      this.stopping = false;
      this.status = "stopped";
      this.completionReason = "stopped";
      return;
    }
    if (this.status !== "running") return;
    this.stopping = true;
    this.#send({ type: "stop" });
  }

  /** Save the run under a name so it can be replayed in Simulate or continued
   *  later. Returns false if the user cancelled the prompt. */
  saveAs(): boolean {
    if (!this.#worker || !this.iteration || this.stopping || this.evaluationStatus === "running") return false;
    const suggested = `${this.runTask?.kind ?? this.taskLabel}-${this.iteration}-${Date.now()}`;
    const raw = globalThis.prompt?.("Save this run as:", suggested);
    if (raw === null || raw === undefined) return false;
    // Checkpoint names become filenames in OPFS, so keep them tame.
    const name = raw.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "")
      .replace(/\.{2,}/g, "-").replace(/-+$/, "").slice(0, 64);
    if (!name) return false;
    if (this.checkpoints.some((checkpoint) => checkpoint.name === name)
      && !globalThis.confirm?.(`Replace saved run "${name}"?`)) return false;
    this.#send({ type: "save", name });
    return true;
  }

  async remove(name: string): Promise<void> {
    if (this.status === "running" || this.status === "loading" || this.evaluationStatus === "running") return;
    try {
      await deleteCheckpoint(name);
      if (this.resumeFrom === name) {
        this.resumeFrom = null;
        this.resumeSummary = null;
      }
      if (this.evaluation?.checkpointName === name) {
        this.evaluation = null;
        this.evaluationStatus = "idle";
      }
      await this.refreshCheckpoints();
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    }
  }

  #send(msg: ToTrainWorker): void {
    this.#worker?.postMessage(msg);
  }

  /** A bounded, read-only physics replay, isolated from the learner. */
  evaluateCheckpoint(name: string): Promise<void> {
    if (this.#disposed || this.status === "running" || this.status === "loading" || this.evaluationStatus === "running") {
      return Promise.reject(new Error("Stop the active run or evaluation before evaluating a checkpoint."));
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name) || name.includes("..")) {
      return Promise.reject(new Error("Invalid checkpoint name for evaluation."));
    }
    this.evaluation = null;
    this.evaluationStatus = "running";
    this.evaluationProgress = 0;
    this.evaluationError = null;
    this.evaluatingName = name;
    return new Promise((resolve, reject) => {
      let worker: Worker;
      try {
        worker = new Worker(new URL("../train/evaluation-worker.ts", import.meta.url), {
          type: "module", name: "wicroduck-evaluate",
        });
      } catch (err) {
        this.evaluationStatus = "error";
        this.evaluationError = err instanceof Error ? err.message : String(err);
        reject(err);
        return;
      }
      this.#evaluationWorker = worker;
      const finish = (error?: string, cancelled = false) => {
        if (this.#evaluationWorker !== worker) return;
        clearTimeout(timeout);
        worker.terminate();
        this.#evaluationWorker = null;
        this.#cancelEvaluation = null;
        if (cancelled) {
          this.evaluationStatus = "idle";
          this.evaluationProgress = 0;
          this.evaluatingName = null;
          resolve();
        } else if (error) {
          this.evaluationStatus = "error";
          this.evaluationError = error;
          reject(new Error(error));
        } else {
          this.evaluationStatus = "done";
          this.evaluationProgress = 1;
          resolve();
        }
      };
      const timeout = setTimeout(() => finish("Evaluation exceeded its two-minute time budget."), EVALUATION_TIMEOUT_MS);
      this.#cancelEvaluation = () => finish(undefined, true);
      worker.onmessage = (event: MessageEvent<FromEvaluationWorker>) => {
        if (this.#evaluationWorker !== worker) return;
        const msg = event.data;
        if (msg.type === "progress") {
          if (Number.isFinite(msg.progress)) this.evaluationProgress = Math.max(this.evaluationProgress, Math.min(1, Math.max(0, msg.progress)));
        } else if (msg.type === "result") {
          if (msg.report.checkpointName !== name) { finish("Evaluation returned a different checkpoint."); return; }
          this.evaluation = msg.report;
          finish();
        } else finish(msg.error || "Evaluation failed.");
      };
      worker.onerror = (event) => finish(event.message || "Evaluation worker crashed.");
      try { worker.postMessage({ type: "evaluate", name, baseUrl: assetBase() }); }
      catch (err) { finish(err instanceof Error ? err.message : String(err)); }
    });
  }

  cancelEvaluation(): void {
    this.#cancelEvaluation?.();
  }

  dispose(): void {
    this.#disposed = true;
    this.#selectionId++;
    this.cancelEvaluation();
    if (this.status === "running" || this.status === "loading") this.stop();
    else {
      this.#worker?.terminate();
      this.#worker = null;
    }
  }
}
