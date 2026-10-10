/// <reference lib="webworker" />
// Hosts a VecEnv and drives it with a policy, entirely inside the worker.
//
// Inference stays here rather than on the main thread on purpose: the policy
// has to run once per control step for every environment, and shipping
// observations out and actions back each step would put a message-loop
// round trip in the inner loop. M0 measured inference at ~70% of the step
// budget, so that barrier would dominate everything.

import { setAssetBase } from "../asset-url.ts";
import { compileScene, loadModelAssets } from "../sim/scene.ts";
import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import { VecEnv } from "./env/vec-env.ts";
import { standupSpec } from "./env/standup.ts";
import { STAND_Z } from "./env/rewards.ts";
import { Mlp } from "./mlp.ts";
import type { EnvInit, FromEnvWorker, RolloutStats, ToEnvWorker } from "./env-protocol.ts";

let env: VecEnv | null = null;
let policy: Mlp | null = null;
let actions: Float32Array | null = null;

const post = (msg: FromEnvWorker) => self.postMessage(msg);

/** Small, fast, seedable — reproducibility beats statistical purity here. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function init(baseUrl: string, cfg: EnvInit): Promise<void> {
  setAssetBase(baseUrl);
  const assets = await loadModelAssets(() => {}, [cfg.robotXml]);
  const { model, standKey } = compileScene(assets, { robotXml: cfg.robotXml });
  env = new VecEnv({
    mujoco: assets.mujoco,
    model,
    standKey,
    spec: standupSpec({
      resetMix: cfg.resetMix,
      episodeLengthS: cfg.episodeLengthS,
    }),
    count: cfg.envs,
    rng: mulberry32(cfg.seed),
  });
  policy = new Mlp(OBS_SIZE, NUM_JOINTS, cfg.envs);
  actions = new Float32Array(cfg.envs * NUM_JOINTS);
  post({ type: "ready", envs: env.count, ngeom: model.ngeom });
}

function rollout(steps: number): void {
  if (!env || !policy || !actions) {
    post({ type: "error", message: "env worker not initialised" });
    return;
  }
  const count = env.count;
  env.resetBreakdown();
  let rewardSum = 0;
  let episodes = 0;
  let standingSteps = 0;
  const start = performance.now();

  for (let t = 0; t < steps; t++) {
    const out = policy.forward(env.observations, count);
    actions.set(out);
    const { reward, done } = env.step(actions);
    const z = env.trunkZ;
    const upright = env.uprightness;
    for (let e = 0; e < count; e++) {
      rewardSum += reward[e];
      if (done[e]) episodes++;
      // "Standing" needs height AND uprightness — a face-plant at the right
      // altitude is not standing.
      if (z[e] > STAND_Z - 0.02 && upright[e] > 0.85) standingSteps++;
    }
  }

  const elapsedMs = performance.now() - start;
  const total = steps * count;
  const breakdown: Record<string, number> = {};
  for (const [k, v] of Object.entries(env.breakdown)) breakdown[k] = v / total;

  const stats: RolloutStats = {
    envs: count,
    steps,
    controlSteps: total,
    elapsedMs,
    rewardPerStep: rewardSum / total,
    episodes,
    standingFraction: standingSteps / total,
    breakdown,
  };
  post({ type: "rollout", stats });
}

self.onmessage = (e: MessageEvent<ToEnvWorker>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      void init(msg.baseUrl, msg.init).catch((err: unknown) =>
        post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
      );
    } else if (msg.type === "rollout") {
      rollout(msg.steps);
    } else if (msg.type === "dispose") {
      env = null;
      policy = null;
      actions = null;
      self.close();
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
