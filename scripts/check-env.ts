// M1 gate: does the ported reward stack actually recognise good behaviour?
//
//   node scripts/check-env.ts [episodes]
//
// A reward function cannot be validated in isolation — "alpha_stand scores
// 8.4" means nothing without knowing what a bad policy scores. So this runs
// three controllers through the SAME VecEnv and checks the ordering:
//
//   alpha_stand  the shipped get-up policy — must win
//   zero         hold the reference pose — the do-nothing baseline
//   random       uniform noise — must lose
//
// If that ordering does not hold, the reward stack is wrong and no amount of
// PPO will fix it. It imports the real src/ modules (Node 24 strips the
// types), so this tests the code the trainer will run, not a lookalike.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web";

import { NUM_JOINTS, OBS_SIZE, TIMESTEP, DEFAULT_POSE } from "../src/sim/microduck.ts";
import { sceneXml } from "../src/sim/scene.ts";
import { VecEnv } from "../src/train/env/vec-env.ts";
import { standupSpec } from "../src/train/env/standup.ts";
import { STAND_Z } from "../src/train/env/rewards.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = path.join(ROOT, "public/model/microduck");
const ROBOT_XML = "robot_allcollisions-nv.xml"; // a fallen duck rests on its shell

ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

/** Deterministic RNG so a failing run can be reproduced exactly. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const mujoco = await loadMujoco();
const manifest = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "manifest.json"), "utf8"));
const entry = manifest.models.find((m: { file: string }) => m.file === ROBOT_XML);
if (!entry) throw new Error(`${ROBOT_XML} missing from manifest; run npm run prepare-assets`);

const vfs = new mujoco.MjVFS();
vfs.addBuffer(ROBOT_XML, new Uint8Array(fs.readFileSync(path.join(MODEL_DIR, ROBOT_XML))));
for (const mesh of entry.meshes) {
  vfs.addBuffer(`assets/${mesh}`, new Uint8Array(fs.readFileSync(path.join(MODEL_DIR, "assets", mesh))));
}
const model = mujoco.MjModel.from_xml_string(sceneXml(ROBOT_XML), vfs);
const standKey = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND");
console.log(`model ${ROBOT_XML}: nq=${model.nq} nu=${model.nu} ngeom=${model.ngeom}, timestep ${TIMESTEP}`);

const session = await ort.InferenceSession.create(
  path.join(ROOT, "public/policies/alpha_stand.onnx"),
  { executionProviders: ["wasm"] },
);

const COUNT = 8;
const EPISODES = Number(process.argv[2] ?? 3);

type Controller = {
  name: string;
  act(obs: Float32Array, out: Float32Array, rng: () => number): Promise<void> | void;
};

const controllers: Controller[] = [
  {
    // The exported policy has a FIXED batch dimension of 1 — it was traced for
    // single-robot deployment, not for vectorized rollout — so replay runs one
    // env per call. The trainer's own MLP has no such limit; this only costs
    // the gate, not M2.
    name: "alpha_stand",
    async act(obs, out) {
      for (let e = 0; e < COUNT; e++) {
        const slice = obs.slice(e * OBS_SIZE, (e + 1) * OBS_SIZE);
        const res = await session.run({ obs: new ort.Tensor("float32", slice, [1, OBS_SIZE]) });
        out.set(res.actions.data as Float32Array, e * NUM_JOINTS);
      }
    },
  },
  { name: "zero", act: (_o, out) => out.fill(0) },
  {
    name: "random",
    act: (_o, out, rng) => {
      for (let i = 0; i < out.length; i++) out[i] = (rng() * 2 - 1) * 0.5;
    },
  },
];

interface Score {
  name: string;
  rewardPerStep: number;
  standingFraction: number;
  meanZ: number;
  breakdown: Record<string, number>;
}

async function evaluate(c: Controller): Promise<Score> {
  // Same seed for every controller: they see identical reset poses, so the
  // comparison is of behaviour, not of luck.
  const rng = mulberry32(12345);
  const env = new VecEnv({ mujoco, model, standKey, spec: standupSpec(), count: COUNT, rng });
  const actions = new Float32Array(COUNT * NUM_JOINTS);
  const steps = env.maxSteps * EPISODES;

  let rewardSum = 0;
  let standingSteps = 0;
  let zSum = 0;
  env.resetBreakdown();

  for (let t = 0; t < steps; t++) {
    await c.act(env.observations, actions, rng);
    const { reward } = env.step(actions);
    for (let e = 0; e < COUNT; e++) {
      rewardSum += reward[e];
      zSum += env.trunkZ[e];
      // "Standing" = at height AND upright, so a face-plant at the right
      // altitude does not count.
      if (env.trunkZ[e] > STAND_Z - 0.02 && env.uprightness[e] > 0.85) standingSteps++;
    }
  }

  const total = steps * COUNT;
  const breakdown: Record<string, number> = {};
  for (const [k, v] of Object.entries(env.breakdown)) breakdown[k] = v / total;
  return {
    name: c.name,
    rewardPerStep: rewardSum / total,
    standingFraction: standingSteps / total,
    meanZ: zSum / total,
    breakdown,
  };
}

const scores: Score[] = [];
for (const c of controllers) scores.push(await evaluate(c));

console.log(`\n${EPISODES} episodes x ${COUNT} envs, ${controllers.length} controllers\n`);
console.log("controller      reward/step  standing%   mean z");
for (const s of scores) {
  console.log(
    `${s.name.padEnd(14)} ${s.rewardPerStep.toFixed(3).padStart(11)} ` +
    `${(s.standingFraction * 100).toFixed(1).padStart(9)} ${(s.meanZ * 100).toFixed(1).padStart(8)} cm`,
  );
}

const best = scores[0];
console.log("\nalpha_stand reward breakdown (per step):");
for (const [k, v] of Object.entries(best.breakdown).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${k.padEnd(22)} ${v.toFixed(4).padStart(9)}`);
}

// ── Gates ─────────────────────────────────────────────────────────────────
const failures: string[] = [];
const byName = Object.fromEntries(scores.map((s) => [s.name, s]));

if (byName.alpha_stand.rewardPerStep <= byName.zero.rewardPerStep) {
  failures.push("alpha_stand does not out-score the do-nothing baseline");
}
if (byName.alpha_stand.rewardPerStep <= byName.random.rewardPerStep) {
  failures.push("alpha_stand does not out-score random actions");
}
if (byName.alpha_stand.standingFraction < 0.5) {
  failures.push(
    `alpha_stand stands for only ${(byName.alpha_stand.standingFraction * 100).toFixed(1)}% of steps`,
  );
}
// Every self-negating penalty must be <= 0. A sign slip here reads as a reward
// for the violation, and the policy would farm it.
for (const name of ["height_stand_l1", "pose_stand_l1", "action_rate_l2"]) {
  if ((best.breakdown[name] ?? 0) > 0) failures.push(`${name} is positive — sign is inverted`);
}
// The reference pose must be a stable equilibrium, or the target is a lie.
if (Math.abs(DEFAULT_POSE.length - NUM_JOINTS) !== 0) failures.push("DEFAULT_POSE length mismatch");

console.log();
if (failures.length) {
  console.log("FAILED:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
console.log("PASSED: reward stack ranks alpha_stand > zero and > random, and it stands.");
