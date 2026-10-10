/// <reference lib="webworker" />
// Runs the PPO trainer off the main thread and streams statistics back.
//
// One worker, one VecEnv batch: rollout and update in the same place. A split
// arrangement (workers roll out, a central learner updates) only pays once
// inference is fast, and M0 measured inference at ~70% of the step budget — so
// M3 fixes that first and the split comes after.

import { setAssetBase } from "../asset-url.ts";
import { compileScene, loadModelAssets } from "../sim/scene.ts";
import { buildSpec, validateTask, taskLabel } from "./env/tasks.ts";
import { defaultWorld, validateWorld } from "../sim/world.ts";
import { checkpointMetadata, validateCheckpointResume, validateTrainingConfig, TASK_OBSERVATION_SCHEMA } from "./checkpoint-task.ts";
import { Trainer, type Checkpoint } from "./trainer.ts";
import { loadCheckpoint, saveCheckpoint } from "./checkpoint-store.ts";
import { loadKernels } from "./kernels/index.ts";
import { PoolRollout } from "./rollout-pool.ts";
import type { FromTrainWorker, ToTrainWorker, TrainInit } from "./train-protocol.ts";

let trainer: Trainer | null = null;
let cfg: TrainInit | null = null;
let running = false;
let saves: Promise<void> = Promise.resolve();

const post = (msg: FromTrainWorker) => self.postMessage(msg);

async function init(baseUrl: string, options: TrainInit): Promise<void> {
  setAssetBase(baseUrl);
  if (trainer || running) throw new Error("training worker is already initialized");
  const ckpt = options.resume ? await loadCheckpoint<Checkpoint>(options.resumeName ?? options.checkpointName) : null;
  if (options.resume && !ckpt) throw new Error("requested checkpoint does not exist");
  if (ckpt) validateCheckpointResume(ckpt);
  const metadata = ckpt ? checkpointMetadata(ckpt) : { task: validateTask(options.task), world: options.world === undefined ? defaultWorld() : validateWorld(options.world) };
  if (ckpt && taskLabel(validateTask(options.task)) !== taskLabel(metadata.task)) throw new Error("select the saved checkpoint's task before resuming");
  options = { ...options, task: metadata.task, world: metadata.world, config: validateTrainingConfig(ckpt?.config ?? options.config) };
  if (!Number.isInteger(options.rolloutWorkers) || options.rolloutWorkers < 1 || options.rolloutWorkers > 8) throw new Error("rollout workers must be between 1 and 8");
  if (!Number.isInteger(options.config.envs) || options.config.envs < 1 || options.config.envs > 128) throw new Error("environments per worker must be between 1 and 128");
  if (options.rolloutWorkers * options.config.envs > 128) throw new Error("at most 128 total rollout environments are supported");
  if (!Number.isInteger(options.autosaveEvery) || options.autosaveEvery < 0 || options.autosaveEvery > 5000) throw new Error("invalid autosave interval");
  cfg = options;
  const spec = buildSpec(options.task);
  // Vite turns this into a static asset URL. If it fails to load — an engine
  // without SIMD, a stripped deployment — the nets fall back to JavaScript
  // rather than the worker dying.
  const kernels = await fetch(new URL("./kernels/kernels.wasm", import.meta.url))
    .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((bytes) => loadKernels(bytes))
    .catch(() => null);
  // With more than one rollout worker the environments move out of this
  // thread entirely and this one only learns. Physics is the majority of an
  // iteration and the only part that parallelises, so this is where the
  // remaining speed is.
  let rollout: PoolRollout | undefined;
  let mujoco, model, standKey;
  if (options.rolloutWorkers > 1) {
    rollout = await PoolRollout.create({
      workers: options.rolloutWorkers,
      envsPerWorker: options.config.envs,
      robotXml: options.robotXml,
      task: options.task,
      world: options.world,
      hidden: options.config.hidden,
      baseSeed: options.config.seed,
    });
  } else {
    const assets = await loadModelAssets(() => {}, [options.robotXml]);
    ({ model, standKey } = compileScene(assets, { robotXml: options.robotXml, world: options.world }));
    mujoco = assets.mujoco;
  }
  trainer = new Trainer({
    mujoco, model, standKey, spec, config: options.config, kernels, rollout,
  });

  let resumedAt = 0;
  if (ckpt) {
    trainer.restore(ckpt);
    resumedAt = ckpt.iteration;
  }
  post({
    type: "ready",
    resumedAt,
    params: trainer.ac.actor.paramCount + trainer.ac.critic.paramCount,
    simd: trainer.usesSimd,
    envs: trainer.rollout.envs,
    rolloutLabel: trainer.rollout.label,
    task: options.task,
    world: metadata.world,
  });
}

/** `name` lets a run be saved under something memorable; the autosave keeps
 *  using the session's rolling slot so the two never fight. */
async function save(name?: string): Promise<void> {
  if (!trainer || !cfg) return;
  const target = name ?? cfg.checkpointName;
  const iteration = trainer.iteration;
  const checkpoint = {
    ...trainer.checkpoint(), taskSpec: cfg.task,
    world: cfg.world ?? defaultWorld(), observationSchema: TASK_OBSERVATION_SCHEMA,
  };
  // Preserve request order when a manual save overlaps an autosave.
  saves = saves.then(async () => {
    await saveCheckpoint(target, checkpoint);
    post({ type: "saved", name: target, iteration });
  });
  await saves;
}

/**
 * Iterate, yielding to the event loop between iterations.
 *
 * Without the yield a `stop` message would sit unread until the whole run
 * finished — the worker's queue is only drained when the task stack empties.
 */
async function run(iterations: number): Promise<void> {
  if (!trainer || !cfg) {
    post({ type: "error", message: "trainer not initialised" });
    return;
  }
  if (running) throw new Error("training is already running");
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 5000) throw new Error("iterations must be between 1 and 5000");
  if (!Number.isSafeInteger(trainer.iteration + iterations)) throw new Error("training target iteration exceeds the supported range");
  running = true;
  let completed = 0;
  for (; completed < iterations && running; completed++) {
    const stats = await trainer.iterate();
    post({ type: "stats", stats });
    if (cfg.autosaveEvery > 0 && stats.iteration % cfg.autosaveEvery === 0) {
      await save();
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  running = false;
  if (trainer.iteration > 0) await save();
  post({ type: "stopped", iteration: trainer.iteration,
    reason: completed === iterations ? "completed" : "stopped",
    checkpointName: trainer.iteration > 0 ? cfg.checkpointName : null });
}

self.onmessage = (e: MessageEvent<ToTrainWorker>) => {
  const msg = e.data;
  const fail = (err: unknown) => {
    running = false;
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  };
  try {
    if (msg.type === "init") void init(msg.baseUrl, msg.init).catch(fail);
    else if (msg.type === "start") void run(msg.iterations).catch(fail);
    else if (msg.type === "stop") running = false;
    else if (msg.type === "save") void save(msg.name).catch(fail);
    else if (msg.type === "dispose") {
      running = false;
      void trainer?.rollout.dispose();
      trainer = null;
      self.close();
    }
  } catch (err) {
    fail(err);
  }
};
