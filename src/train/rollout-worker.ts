/// <reference lib="webworker" />
// One slice of the environment pool, with its own copy of the policy.
//
// It samples actions locally rather than asking the learner, because the
// policy runs once per control step per environment and a message round trip
// there would cost more than the physics. The learner pushes weights once per
// iteration instead.

import { setAssetBase } from "../asset-url.ts";
import { compileScene, loadModelAssets } from "../sim/scene.ts";
import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import { ActorCritic } from "./ac-policy.ts";
import { VecEnv } from "./env/vec-env.ts";
import { buildSpec } from "./env/tasks.ts";
import { STAND_Z } from "./env/rewards.ts";
import { loadKernels, type Kernels } from "./kernels/index.ts";
import type { FromRolloutWorker, RolloutChunk, RolloutInit, ToRolloutWorker } from "./rollout-protocol.ts";

let env: VecEnv | null = null;
let ac: ActorCritic | null = null;
let kernels: Kernels | null = null;

const post = (msg: FromRolloutWorker, transfer: Transferable[] = []) =>
  self.postMessage(msg, transfer);

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rng: () => number = Math.random;

async function init(baseUrl: string, cfg: RolloutInit): Promise<void> {
  setAssetBase(baseUrl);
  rng = mulberry32(cfg.seed);
  const assets = await loadModelAssets(() => {}, [cfg.robotXml]);
  const { model, standKey } = compileScene(assets, { robotXml: cfg.robotXml, world: cfg.world });
  const spec = buildSpec(cfg.task, { episodeLengthS: cfg.episodeLengthS });
  env = new VecEnv({
    mujoco: assets.mujoco, model, standKey, spec, count: cfg.envs, rng,
  });
  kernels = await fetch(new URL("./kernels/kernels.wasm", import.meta.url))
    .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((bytes) => loadKernels(bytes))
    .catch(() => null);
  // initStd is irrelevant here: the learner's log-std arrives with the weights.
  ac = new ActorCritic(cfg.obsDim, cfg.actDim, rng, 1, cfg.hidden, kernels);
  post({ type: "ready", envs: env.count });
}

function collect(steps: number): void {
  if (!env || !ac) {
    post({ type: "error", message: "rollout worker not initialised" });
    return;
  }
  const envs = env.count;
  const n = steps * envs;
  const chunk: RolloutChunk = {
    envs, steps,
    obs: new Float32Array(n * OBS_SIZE),
    actions: new Float32Array(n * NUM_JOINTS),
    logProbs: new Float32Array(n),
    values: new Float32Array(n),
    rewards: new Float32Array(n),
    dones: new Uint8Array(n),
    timeouts: new Uint8Array(n),
    lastValues: new Float32Array(envs),
    rewardSum: 0, standingSteps: 0, finishedEpisodes: 0,
    finishedReturn: 0, nonFiniteSteps: 0,
    taskMetricSums: {},
  };

  const episodeReturn = new Float32Array(envs);
  const before = env.nonFiniteSteps;

  for (let t = 0; t < steps; t++) {
    const obs = env.observations;
    chunk.obs.set(obs, t * envs * OBS_SIZE);
    const { actions, logProbs, values } = ac.act(obs, envs, rng);
    chunk.actions.set(actions, t * envs * NUM_JOINTS);
    chunk.logProbs.set(logProbs, t * envs);
    chunk.values.set(values, t * envs);

    const { reward, done, timeout } = env.step(actions);
    for (const [name, values] of Object.entries(env.taskMetrics)) {
      chunk.taskMetricSums[name] = (chunk.taskMetricSums[name] ?? 0) + values.reduce((a, b) => a + b, 0);
    }
    chunk.rewards.set(reward, t * envs);
    chunk.dones.set(done, t * envs);
    chunk.timeouts.set(timeout, t * envs);

    const z = env.trunkZ, upright = env.uprightness;
    for (let e = 0; e < envs; e++) {
      chunk.rewardSum += reward[e];
      episodeReturn[e] += reward[e];
      if (z[e] > STAND_Z - 0.02 && upright[e] > 0.85) chunk.standingSteps++;
      if (done[e]) {
        chunk.finishedEpisodes++;
        chunk.finishedReturn += episodeReturn[e];
        episodeReturn[e] = 0;
      }
    }
  }
  chunk.lastValues.set(ac.value(env.observations, envs).subarray(0, envs));
  chunk.nonFiniteSteps = env.nonFiniteSteps - before;

  // Transferred, not copied: an iteration's buffers are a few hundred KB and
  // the worker has no use for them afterwards.
  post({ type: "chunk", chunk }, [
    chunk.obs.buffer, chunk.actions.buffer, chunk.logProbs.buffer,
    chunk.values.buffer, chunk.rewards.buffer, chunk.dones.buffer,
    chunk.timeouts.buffer, chunk.lastValues.buffer,
  ]);
}

self.onmessage = (e: MessageEvent<ToRolloutWorker>) => {
  const msg = e.data;
  const fail = (err: unknown) =>
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  try {
    if (msg.type === "init") {
      void init(msg.baseUrl, msg.init).catch(fail);
    } else if (msg.type === "policy") {
      if (!ac) return fail(new Error("policy before init"));
      ac.loadFlatParams(msg.params);
      ac.normalizer.mean.set(msg.mean);
      ac.normalizer.var_.set(msg.var);
      ac.normalizer.count = msg.count;
      post({ type: "synced" });
    } else if (msg.type === "collect") {
      collect(msg.steps);
    } else if (msg.type === "reset") {
      env?.resetAll();
      post({ type: "reset" });
    } else if (msg.type === "dispose") {
      env = null;
      ac = null;
      self.close();
    }
  } catch (err) {
    fail(err);
  }
};
