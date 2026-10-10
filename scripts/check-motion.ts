// Gate for the motion pipeline: the file format, the interpolator, and the
// task built on top of them.
//
//   node scripts/check-motion.ts
//
// The interesting gate is the last one. A motion file is only worth anything
// if a policy that PERFORMS the motion scores better than one that ignores it,
// and that is not obvious in advance: the reward stack has to notice tracking
// through a soft position actuator, a trunk that has to stay balanced, and a
// reference height nobody wrote down. So an ORACLE controller — one that reads
// the reference straight out of the sampler and asks for it — is run against
// the same do-nothing and random baselines check-env.ts uses. If the oracle
// does not win, training on the motion is pointless and the failure is here
// rather than three hours into a run.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco from "@mujoco/mujoco";

import { CTRL_DT, DEFAULT_POSE, JOINT_LIMITS, JOINT_NAMES, NUM_JOINTS } from "../src/sim/microduck.ts";
import { sceneXml } from "../src/sim/scene.ts";
import {
  MotionError, drivenIndices, motionToJson, parseMotion, type Motion,
} from "../src/motion/format.ts";
import { MotionSampler } from "../src/motion/sampler.ts";
import { BUILTIN_MOTIONS } from "../src/motion/library.ts";
import { VecEnv } from "../src/train/env/vec-env.ts";
import { motionSpec } from "../src/train/env/motion.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = path.join(ROOT, "public/model/microduck");
const ROBOT_XML = "robot_allcollisions-nv.xml";

const failures: string[] = [];
const fail = (msg: string) => failures.push(msg);

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── 1. Format: every built-in parses, and survives a round trip ───────────
const motions = new Map<string, Motion>();
for (const file of BUILTIN_MOTIONS) {
  let motion: Motion;
  try {
    motion = parseMotion(file);
  } catch (err) {
    fail(`built-in "${file.name}" does not parse: ${(err as Error).message}`);
    continue;
  }
  motions.set(motion.name, motion);
  const again = parseMotion(motionToJson(motion));
  if (again.keyframes.length !== motion.keyframes.length) {
    fail(`${motion.name}: round trip changed the keyframe count`);
    continue;
  }
  let worst = 0;
  motion.keyframes.forEach((k, i) => {
    for (let j = 0; j < NUM_JOINTS; j++) {
      worst = Math.max(worst, Math.abs(k.pose[j] - again.keyframes[i].pose[j]));
    }
  });
  // motionToFile rounds to 5 decimals, so this is a serialisation tolerance,
  // not a numerical one.
  if (worst > 1e-4) fail(`${motion.name}: round trip moved a joint by ${worst.toFixed(6)} rad`);
}
console.log(`parsed ${motions.size}/${BUILTIN_MOTIONS.length} built-in motions, round trip clean`);

// ── 2. Validation says something useful about bad files ───────────────────
const bad: { why: string; file: unknown; expect: RegExp }[] = [
  { why: "not an object", file: 42, expect: /must be a JSON object/ },
  { why: "no keyframes", file: { keyframes: [] }, expect: /non-empty/ },
  {
    why: "unknown joint",
    file: { joints: ["left_elbow"], keyframes: [{ t: 0, pose: [0] }] },
    expect: /unknown joint/,
  },
  {
    why: "pose length mismatch",
    file: { joints: ["head_yaw"], keyframes: [{ t: 0, pose: [0, 1] }] },
    expect: /2 angles but 1 joints/,
  },
  {
    why: "degrees written as radians",
    file: { joints: ["head_yaw"], keyframes: [{ t: 0, pose: [45] }] },
    expect: /outside its range/,
  },
  {
    why: "past a hinge limit",
    file: { units: "deg", joints: ["head_roll"], keyframes: [{ t: 0, pose: [60] }] },
    expect: /outside its range/,
  },
  {
    why: "does not start at zero",
    file: { joints: ["head_yaw"], keyframes: [{ t: 1, pose: [0] }] },
    expect: /first keyframe must be at t = 0/,
  },
  {
    why: "unclosed loop",
    file: { loop: true, joints: ["head_yaw"], keyframes: [{ t: 0, pose: [0] }, { t: 1, pose: [0.5] }] },
    expect: /does not repeat the first pose/,
  },
  {
    why: "typo in a field name",
    file: { keyframs: [], keyframes: [{ t: 0, pose: [] }] },
    expect: /unknown field "keyframs"/,
  },
];
for (const c of bad) {
  let message = "";
  try {
    parseMotion(c.file);
  } catch (err) {
    if (!(err instanceof MotionError)) {
      fail(`"${c.why}" threw ${(err as Error).name}, not MotionError`);
      continue;
    }
    message = err.message;
  }
  if (!message) fail(`"${c.why}" was accepted; it should not be`);
  else if (!c.expect.test(message)) fail(`"${c.why}" said "${message}", expected ${c.expect}`);
}
console.log(`${bad.length} malformed files rejected with a message that names the fix`);

// ── 3. The interpolator ───────────────────────────────────────────────────
const scratch = new Float32Array(NUM_JOINTS);
const scratch2 = new Float32Array(NUM_JOINTS);
for (const motion of motions.values()) {
  const sampler = new MotionSampler(motion);

  // Keyframes are interpolated THROUGH, not approximated.
  for (const k of motion.keyframes) {
    sampler.poseAt(k.t, scratch);
    for (let j = 0; j < NUM_JOINTS; j++) {
      if (Math.abs(scratch[j] - k.pose[j]) > 1e-5) {
        fail(`${motion.name}: t=${k.t} joint ${JOINT_NAMES[j]} sampled ${scratch[j]}, keyed ${k.pose[j]}`);
      }
    }
  }

  // The analytic velocity must agree with a central difference of the pose
  // track — the two are computed by completely separate code paths, so this
  // catches a wrong Hermite derivative, which is otherwise invisible until a
  // policy is being graded against it.
  const h = 1e-4;
  let worstVel = 0;
  for (let s = 1; s < 40; s++) {
    const t = (motion.duration * s) / 40;
    sampler.poseAt(t + h, scratch);
    sampler.poseAt(t - h, scratch2);
    const analytic = sampler.velAt(t, new Float32Array(NUM_JOINTS));
    for (let j = 0; j < NUM_JOINTS; j++) {
      worstVel = Math.max(worstVel, Math.abs((scratch[j] - scratch2[j]) / (2 * h) - analytic[j]));
    }
  }
  if (worstVel > 1e-2) fail(`${motion.name}: velocity disagrees with finite differences by ${worstVel.toFixed(4)}`);

  // A loop has to be continuous across the seam in BOTH position and velocity,
  // or the policy is asked for a step change once per cycle.
  if (motion.loop) {
    const before = sampler.velAt(motion.duration - 1e-5, new Float32Array(NUM_JOINTS));
    const after = sampler.velAt(1e-5, new Float32Array(NUM_JOINTS));
    let jump = 0;
    for (let j = 0; j < NUM_JOINTS; j++) jump = Math.max(jump, Math.abs(before[j] - after[j]));
    if (jump > 0.05) fail(`${motion.name}: velocity jumps ${jump.toFixed(3)} rad/s across the loop seam`);
  }
  console.log(
    `${motion.name.padEnd(13)} ${motion.keyframes.length} keys, ${motion.duration.toFixed(1)} s, ` +
    `${motion.loop ? "loop" : "once"}, vel err ${worstVel.toExponential(1)}`,
  );
}

// ── 4. The task: an oracle must beat doing nothing ────────────────────────
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

// JOINT_LIMITS is a hand-copied duplicate of the MJCF's ranges, and the motion
// parser rejects poses against it before any model is loaded. If the copy
// drifts, files are accepted that MuJoCo will silently clamp.
{
  const njnt = model.jnt_range.length / 2;
  JOINT_NAMES.forEach((name, j) => {
    const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name);
    if (id < 0 || id >= njnt) {
      fail(`joint "${name}" is not in the compiled model`);
      return;
    }
    const [lo, hi] = JOINT_LIMITS[j];
    const drift = Math.max(
      Math.abs(model.jnt_range[id * 2] - lo), Math.abs(model.jnt_range[id * 2 + 1] - hi),
    );
    if (drift > 1e-3) {
      fail(
        `JOINT_LIMITS[${j}] (${name}) says ${lo}..${hi}, the model says ` +
        `${model.jnt_range[id * 2]}..${model.jnt_range[id * 2 + 1]}`,
      );
    }
  });
  console.log(`JOINT_LIMITS matches the compiled model for all ${JOINT_NAMES.length} joints`);
}

const COUNT = 8;
const EPISODES = 2;

type Controller = { name: string; act(env: VecEnv, out: Float32Array, rng: () => number): void };

/** Perform the motion: ask for exactly the reference pose, as an offset from
 *  DEFAULT_POSE because that is what the actuator adds it to. */
function oracle(motion: Motion): Controller {
  const sampler = new MotionSampler(motion);
  const pose = new Float32Array(NUM_JOINTS);
  return {
    name: "oracle",
    act(env, out) {
      for (let e = 0; e < env.count; e++) {
        // One control step ahead: the action set now is what the physics runs
        // over the step that ENDS at t + CTRL_DT, which is where it is graded.
        sampler.poseAt(env.timeOf(e) + CTRL_DT, pose);
        for (let j = 0; j < NUM_JOINTS; j++) out[e * NUM_JOINTS + j] = pose[j] - DEFAULT_POSE[j];
      }
    },
  };
}

const zero: Controller = { name: "zero", act: (_e, out) => out.fill(0) };
const random: Controller = {
  name: "random",
  act: (_e, out, rng) => {
    for (let i = 0; i < out.length; i++) out[i] = (rng() * 2 - 1) * 0.5;
  },
};

interface Score {
  name: string;
  rewardPerStep: number;
  poseErr: number;
  /** Mean control steps survived before falling or timing out. */
  episodeSteps: number;
  breakdown: Record<string, number>;
}

function evaluate(motion: Motion, c: Controller): Score {
  const driven = drivenIndices(motion);
  const rng = mulberry32(4242);
  const spec = motionSpec(motion);
  const env = new VecEnv({ mujoco, model, standKey, spec, count: COUNT, rng });
  const sampler = new MotionSampler(motion);
  const ref = new Float32Array(NUM_JOINTS);
  const actions = new Float32Array(COUNT * NUM_JOINTS);
  const steps = env.maxSteps * EPISODES;
  env.resetBreakdown();

  let rewardSum = 0;
  let poseErr = 0;
  let episodes = 0;
  for (let t = 0; t < steps; t++) {
    c.act(env, actions, rng);
    const { reward, done } = env.step(actions);
    for (let e = 0; e < COUNT; e++) {
      rewardSum += reward[e];
      if (done[e]) episodes++;
      sampler.poseAt(env.timeOf(e), ref);
      const qpos = env.dataAt(e).qpos;
      for (const j of driven) {
        poseErr += Math.abs(qpos[env.joints.qpos[j]] - ref[j]);
      }
    }
  }
  const total = steps * COUNT;
  const breakdown: Record<string, number> = {};
  for (const [k, v] of Object.entries(env.breakdown)) breakdown[k] = v / total;
  return {
    name: c.name,
    rewardPerStep: rewardSum / total,
    poseErr: poseErr / (total * driven.length),
    episodeSteps: episodes > 0 ? total / episodes : steps,
    breakdown,
  };
}

// Motions the whole body is involved in. A head-only motion is a bad gate:
// thirteen of fourteen joints score identically for every controller, so the
// signal being tested is swamped.
const TASKS = ["squat", "bow"];
console.log(`\n${EPISODES} episodes x ${COUNT} envs per controller\n`);
console.log("motion        controller    reward/step   tracking   |q-qref|   survives");
for (const name of TASKS) {
  const motion = motions.get(name);
  if (!motion) {
    fail(`built-in motion "${name}" is missing`);
    continue;
  }
  const scores = [oracle(motion), zero, random].map((c) => evaluate(motion, c));
  for (const s of scores) {
    console.log(
      `${name.padEnd(13)} ${s.name.padEnd(13)} ${s.rewardPerStep.toFixed(3).padStart(11)} ` +
      `${s.breakdown.motion_pose.toFixed(3).padStart(10)} ` +
      `${s.poseErr.toFixed(4).padStart(10)} rad ` +
      `${(s.episodeSteps * CTRL_DT).toFixed(2).padStart(6)} s`,
    );
  }
  const [best, doNothing, noise] = scores;
  if (best.rewardPerStep <= doNothing.rewardPerStep) {
    fail(`${name}: performing the motion scores no better than ignoring it`);
  }
  if (best.rewardPerStep <= noise.rewardPerStep) {
    fail(`${name}: performing the motion scores no better than random actions`);
  }
  if (best.poseErr >= doNothing.poseErr) {
    fail(`${name}: the oracle does not actually track better than doing nothing`);
  }
  // Same sign discipline the standing stack is held to: a self-negating term
  // that comes out positive is a reward for the violation.
  for (const term of ["motion_pose_l1", "action_rate_l2"]) {
    if ((best.breakdown[term] ?? 0) > 0) fail(`${name}: ${term} is positive — sign is inverted`);
  }
  if (!(best.breakdown.motion_height > 0)) {
    fail(`${name}: motion_height paid nothing — the reference height table is not being built`);
  }
  // The ranking above can hold for the wrong reason — a controller that
  // happens to fall less collects more of everything. This checks the term
  // that is actually about the motion.
  if (best.breakdown.motion_pose <= doNothing.breakdown.motion_pose) {
    fail(`${name}: motion_pose does not separate performing the motion from ignoring it`);
  }
}

// Every motion is a balance problem on this robot, whatever it asks of the
// legs: with the controls held open-loop the duck topples in about a second,
// which is a measured property of the model and not a bug in any one motion.
// Reported rather than gated, because it is the reason the task exists.
console.log(
  "\nnote: open-loop playback always falls — balancing THROUGH the motion is the task.",
);

console.log();
if (failures.length) {
  console.log("FAILED:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
console.log("PASSED: motion files parse, interpolate, and reward performing the motion.");
