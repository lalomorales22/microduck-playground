import { defaultWorld, validateWorld } from "../sim/world.ts";
import { taskLabel, validateTask, type TaskSpec } from "./env/tasks.ts";
import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import type { Checkpoint, TrainerConfig } from "./trainer.ts";

export const TASK_OBSERVATION_SCHEMA = "microduck-task-v1" as const;

/** Check lengths before loading; TypedArray.set otherwise accepts truncated weights. */
export function checkpointNetwork(value: unknown) {
  if (!value || typeof value !== "object") throw new Error("invalid checkpoint");
  const ckpt = value as { iteration?: unknown; config?: { hidden?: unknown }; policy?: Record<string, unknown> };
  if (!Number.isSafeInteger(ckpt.iteration) || (ckpt.iteration as number) < 0) throw new Error("invalid checkpoint iteration");
  const p = ckpt.policy;
  if (!p || typeof p !== "object") throw new Error("checkpoint has no policy");
  const hidden = p.hidden ?? ckpt.config?.hidden;
  if (!Array.isArray(hidden) || hidden.length < 1 || hidden.length > 4 || hidden.some(n => !Number.isInteger(n) || n < 1 || n > 1024)) throw new Error("invalid checkpoint network shape");
  if (ckpt.config?.hidden !== undefined && (!Array.isArray(ckpt.config.hidden) || ckpt.config.hidden.length !== hidden.length || ckpt.config.hidden.some((n, i) => n !== hidden[i]))) throw new Error("checkpoint policy/config network shapes disagree");
  if ((p.obsDim ?? OBS_SIZE) !== OBS_SIZE || (p.actDim ?? NUM_JOINTS) !== NUM_JOINTS) throw new Error("checkpoint observation/action dimensions are incompatible");
  const vector = (a: unknown, length: number, label: string) => {
    if (!Array.isArray(a) || a.length !== length || a.some(n => typeof n !== "number" || !Number.isFinite(Math.fround(n)))) throw new Error(`invalid checkpoint ${label}`);
  };
  for (const [name, outputs] of [["actor", NUM_JOINTS], ["critic", 1]] as const) {
    const layers = p[name] as { w: unknown; b: unknown }[];
    const dims = [OBS_SIZE, ...hidden, outputs];
    if (!Array.isArray(layers) || layers.length !== dims.length - 1) throw new Error(`invalid checkpoint ${name} layers`);
    layers.forEach((layer, i) => {
      if (!layer || typeof layer !== "object") throw new Error(`invalid checkpoint ${name} layer`);
      vector(layer.w, dims[i] * dims[i + 1], `${name} weights`);
      vector(layer.b, dims[i + 1], `${name} biases`);
    });
  }
  vector(p.logStd, NUM_JOINTS, "logStd");
  const normalizer = p.normalizer as { mean: unknown; var: unknown; count: unknown };
  if (!normalizer || typeof normalizer !== "object") throw new Error("invalid checkpoint normalizer");
  vector(normalizer.mean, OBS_SIZE, "normalizer mean");
  vector(normalizer.var, OBS_SIZE, "normalizer variance");
  if ((normalizer.var as number[]).some(n => n < 0) || typeof normalizer.count !== "number" || !Number.isFinite(normalizer.count) || normalizer.count < 0) throw new Error("invalid checkpoint normalizer statistics");
  return { hidden: hidden as number[], iteration: ckpt.iteration as number, policy: p };
}

function bounded(value: unknown, name: string, min: number, max: number, integer = false): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) throw new Error(`invalid training ${name}: expected ${integer ? "integer " : ""}${min}..${max}`);
  return value;
}

/** Bound work before model/worker allocation; produced UI checkpoints fit these limits. */
export function validateTrainingConfig(value: unknown): TrainerConfig {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("invalid training config");
  const c = value as TrainerConfig;
  bounded(c.envs, "envs", 1, 128, true);
  bounded(c.stepsPerIter, "stepsPerIter", 1, 256, true);
  bounded(c.seed, "seed", 0, 0xffffffff, true);
  if (!Array.isArray(c.hidden) || c.hidden.length < 1 || c.hidden.length > 4) throw new Error("invalid training hidden layers");
  c.hidden.forEach(n => bounded(n, "hidden width", 1, 1024, true));
  if (c.envs * c.stepsPerIter > 8192 || c.hidden.reduce((a, b) => a + b, 0) > 2048) throw new Error("training batch/network exceeds the browser memory budget");
  const p = c.ppo;
  if (!p || typeof p !== "object") throw new Error("missing PPO config");
  bounded(p.epochs, "epochs", 1, 20, true);
  bounded(p.minibatches, "minibatches", 1, Math.min(32, c.envs * c.stepsPerIter), true);
  if ((c.envs * c.stepsPerIter) % p.minibatches !== 0) throw new Error("training batch must divide into complete minibatches");
  bounded(p.gamma, "gamma", 0, 1);
  bounded(p.lambda, "lambda", 0, 1);
  bounded(p.clip, "clip", 0.000001, 1);
  bounded(p.entropyCoef, "entropyCoef", 0, 1);
  bounded(p.valueCoef, "valueCoef", 0, 100);
  bounded(p.maxGradNorm, "maxGradNorm", 0.000001, 100);
  bounded(p.lr, "learning rate", 1e-10, 1);
  if (p.desiredKl !== null) bounded(p.desiredKl, "desiredKl", 1e-10, 1);
  if (c.exploration !== undefined) {
    if (!c.exploration || typeof c.exploration !== "object" || Array.isArray(c.exploration)) throw new Error("invalid exploration config");
    for (const key of Object.keys(c.exploration)) if (!["initStd", "entropyCoef", "desiredKl"].includes(key)) throw new Error("unknown exploration setting");
    if (c.exploration.initStd !== undefined) bounded(c.exploration.initStd, "exploration initStd", 1e-6, 10);
    if (c.exploration.entropyCoef !== undefined) bounded(c.exploration.entropyCoef, "exploration entropyCoef", 0, 1);
    if (c.exploration.desiredKl !== undefined) bounded(c.exploration.desiredKl, "exploration desiredKl", 1e-10, 1);
  }
  return structuredClone(c);
}

/** Resume requires complete Adam moments, unlike inference-only policy replay. */
export function validateCheckpointResume(value: unknown): Checkpoint {
  const network = checkpointNetwork(value);
  const ckpt = value as Checkpoint;
  if (ckpt.version !== 1) throw new Error("unsupported checkpoint version");
  const config = validateTrainingConfig(ckpt.config);
  if (config.hidden.length !== network.hidden.length || config.hidden.some((n, i) => n !== network.hidden[i])) throw new Error("checkpoint network/config mismatch");
  bounded(ckpt.totalSteps, "checkpoint totalSteps", 0, Number.MAX_SAFE_INTEGER, true);
  bounded(ckpt.rngState, "checkpoint rngState", 0, 0xffffffff, true);
  const optimizer = ckpt.optimizer;
  if (!optimizer || typeof optimizer !== "object") throw new Error("checkpoint lacks optimizer state");
  bounded(optimizer.t, "optimizer step", 0, Number.MAX_SAFE_INTEGER, true);
  const policy = network.policy as { actor: { w: number[]; b: number[] }[]; critic: { w: number[]; b: number[] }[] };
  const lengths = [...policy.actor, ...policy.critic].flatMap(layer => [layer.w.length, layer.b.length]);
  lengths.push(NUM_JOINTS);
  for (const name of ["m", "v"] as const) {
    const moments = optimizer[name];
    if (!Array.isArray(moments) || moments.length !== lengths.length) throw new Error(`invalid optimizer ${name} tensor count`);
    moments.forEach((values, i) => {
      if (!Array.isArray(values) || values.length !== lengths[i] || values.some(n => typeof n !== "number" || !Number.isFinite(Math.fround(n)) || (name === "v" && n < 0))) throw new Error(`invalid optimizer ${name} tensor ${i}`);
    });
  }
  return ckpt;
}

/** Legacy balance checkpoints have no task commands; never guess a missing motion. */
export function checkpointMetadata(value: unknown) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("invalid checkpoint");
  const ckpt = value as Record<string, unknown>;
  if (ckpt.version !== 1) throw new Error("unsupported checkpoint version");
  if (ckpt.observationSchema !== undefined && ckpt.observationSchema !== TASK_OBSERVATION_SCHEMA) throw new Error("unsupported checkpoint observation schema");
  let task: TaskSpec;
  if (ckpt.taskSpec !== undefined) {
    if (ckpt.observationSchema !== TASK_OBSERVATION_SCHEMA) throw new Error("task checkpoint is missing its observation schema");
    task = validateTask(ckpt.taskSpec);
    if (ckpt.task !== undefined && ckpt.task !== taskLabel(task)) throw new Error("checkpoint task metadata disagrees");
  } else if (ckpt.task === undefined || ckpt.task === "hold_pose" || ckpt.task === "standup") {
    task = { kind: ckpt.task === "standup" ? "standup" : "hold_pose" };
  } else {
    throw new Error("legacy checkpoint lacks the task specification needed for replay");
  }
  return { task, world: ckpt.world === undefined ? defaultWorld() : validateWorld(ckpt.world) };
}
