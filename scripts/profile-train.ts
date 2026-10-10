// Where does a training iteration actually go?
//
//   node scripts/profile-train.ts
//
// M0 measured a rollout-only loop and found inference at ~70% of the step
// budget. With a learner attached that answer changes, because every sample is
// forwarded once during rollout but forwarded AND backwarded `epochs` times
// during the update. This measures the pieces at their real shapes so M3
// optimises the thing that actually dominates rather than the thing that
// dominated a different loop.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { NUM_JOINTS, OBS_SIZE } from "../src/sim/microduck.ts";
import { loadModelFromDisk } from "../src/train/model-loader.ts";
import { holdPoseSpec } from "../src/train/env/standup.ts";
import { VecEnv } from "../src/train/env/vec-env.ts";
import { ActorCritic } from "../src/train/ac-policy.ts";
import { loadKernelsFromDisk } from "../src/train/kernels/load.node.ts";
import { DEFAULT_PPO, makeBuffer, ppoUpdate } from "../src/train/ppo.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = path.join(ROOT, "public/model/microduck");
const ROBOT_XML = "robot_allcollisions-nv.xml";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Median of repeated timings — one slow sample should not set the answer. */
function timeIt(reps: number, fn: () => void): number {
  fn(); // warm
  const samples: number[] = [];
  for (let r = 0; r < reps; r++) {
    const t = performance.now();
    fn();
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

const { mujoco, model, standKey } = await loadModelFromDisk(MODEL_DIR, ROBOT_XML, fs, path);
const kernels = await loadKernelsFromDisk();
console.log(kernels ? "kernels: SIMD wasm" : "kernels: JavaScript fallback");

const ENVS = 32;
const STEPS = 24;
const EPOCHS = DEFAULT_PPO.epochs;
const MINIBATCHES = DEFAULT_PPO.minibatches;
const MB = (ENVS * STEPS) / MINIBATCHES;

const NETS: { label: string; hidden: number[] }[] = [
  { label: "128/64 (fast)", hidden: [128, 64] },
  { label: "512/256/128 (reference)", hidden: [512, 256, 128] },
];

console.log(
  `${ENVS} envs x ${STEPS} steps = ${ENVS * STEPS} samples/iter, ` +
  `${EPOCHS} epochs x ${MINIBATCHES} minibatches of ${MB}\n`,
);

for (const { label, hidden } of NETS) {
  const rng = mulberry32(1);
  const env = new VecEnv({ mujoco, model, standKey, spec: holdPoseSpec(), count: ENVS, rng });
  const ac = new ActorCritic(OBS_SIZE, NUM_JOINTS, rng, 0.1, hidden, kernels);
  const opt = ac.makeOptimizer(DEFAULT_PPO.lr);
  const buf = makeBuffer(STEPS, ENVS, OBS_SIZE, NUM_JOINTS);
  const actions = new Float32Array(ENVS * NUM_JOINTS);

  // ── Rollout components, per control step ───────────────────────────────
  const envStep = timeIt(60, () => {
    env.step(actions);
  });
  const act = timeIt(60, () => {
    ac.act(env.observations, ENVS, rng);
  });

  // ── Update components, per minibatch ───────────────────────────────────
  const mbObs = new Float32Array(MB * OBS_SIZE);
  for (let i = 0; i < mbObs.length; i++) mbObs[i] = rng() * 2 - 1;

  const fwdSolo = timeIt(30, () => {
    ac.actor.forward(mbObs, MB);
    ac.critic.forward(mbObs, MB);
  });

  // ── The whole thing, measured end to end for a cross-check ─────────────
  for (let t = 0; t < STEPS; t++) {
    const r = ac.act(env.observations, ENVS, rng);
    buf.obs.set(env.observations, t * ENVS * OBS_SIZE);
    buf.actions.set(r.actions, t * ENVS * NUM_JOINTS);
    buf.logProbs.set(r.logProbs, t * ENVS);
    buf.values.set(r.values, t * ENVS);
    const s = env.step(r.actions);
    buf.rewards.set(s.reward, t * ENVS);
    buf.dones.set(s.done, t * ENVS);
    buf.timeouts.set(s.timeout, t * ENVS);
  }
  // The update reports its own breakdown, so this is ground truth rather than
  // a synthesis of microbenchmarks.
  let last = ppoUpdate(ac, opt, buf, DEFAULT_PPO, rng);
  const updateTotal = timeIt(8, () => {
    last = ppoUpdate(ac, opt, buf, DEFAULT_PPO, rng);
  });

  const rolloutPhysics = envStep * STEPS;
  const rolloutInfer = act * STEPS;
  const rolloutTotal = rolloutPhysics + rolloutInfer;
  const iter = rolloutTotal + updateTotal;
  const params = ac.actor.paramCount + ac.critic.paramCount;
  const pct = (x: number) => `${((x / iter) * 100).toFixed(1).padStart(5)}%`;

  console.log(`── ${label} — ${params.toLocaleString()} params ─────────────────`);
  console.log(`  rollout physics + reward   ${rolloutPhysics.toFixed(1).padStart(7)} ms  ${pct(rolloutPhysics)}`);
  console.log(`  rollout policy forward     ${rolloutInfer.toFixed(1).padStart(7)} ms  ${pct(rolloutInfer)}`);
  console.log(`  PPO update                 ${updateTotal.toFixed(1).padStart(7)} ms  ${pct(updateTotal)}`);
  console.log(`  ${"".padStart(27)}${iter.toFixed(1).padStart(7)} ms  per iteration`);
  console.log(
    `    update internals: forward ${last.fwdMs.toFixed(0)} ms, ` +
    `backward ${last.bwdMs.toFixed(0)} ms, clip+adam ${last.optMs.toFixed(0)} ms, ` +
    `other ${last.otherMs.toFixed(0)} ms  (clip fraction ${(last.clipFraction * 100).toFixed(0)}%)`,
  );
  console.log(`    isolated forward, one minibatch: ${fwdSolo.toFixed(2)} ms x ${EPOCHS * MINIBATCHES}`);
  // Everything that is a dense matmul is what a SIMD or WebGPU kernel would
  // move; physics is not.
  const matmul = rolloutInfer + last.fwdMs + last.bwdMs;
  console.log(`    dense linear algebra: ${matmul.toFixed(1)} ms = ${pct(matmul)} of the iteration\n`);
}
