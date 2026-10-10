// Bounded CPU/WASM gate for every shipped joint-reference template.
// node scripts/check-motion-library.ts
// Finite open-loop playback is not a balance, gait, jump, or flip qualification.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco from "@mujoco/mujoco";
import { BUILTIN_MOTIONS, builtinMotionNames } from "../src/motion/library.ts";
import { motionToJson, parseMotion } from "../src/motion/format.ts";
import { MotionSampler } from "../src/motion/sampler.ts";
import { CTRL_DT, DECIMATION, JOINT_LIMITS, JOINT_NAMES, NUM_JOINTS, TIMESTEP } from "../src/sim/microduck.ts";
import { sceneXml } from "../src/sim/scene.ts";

const expected = [
  "march_in_place", "walk_cycle_reference", "sidestep_reference", "crouch_hold",
  "jump_preparation", "tuck_preparation", "roll_preparation", "dance_sway", "neck_stretch",
];
const names = builtinMotionNames();
assert.equal(new Set(names).size, names.length, "Motion names must be unique for library selection");
for (const name of expected) assert(names.includes(name), `Missing reference template ${name}`);
const motions = BUILTIN_MOTIONS.map(file => parseMotion(file));
{
  const crouch = motions.find(motion => motion.name === "crouch_hold")!;
  const sampler = new MotionSampler(crouch);
  const held = sampler.poseAt(1, new Float32Array(NUM_JOINTS));
  for (const t of [1.1, 1.4, 1.6, 1.9, 2.2]) {
    const pose = sampler.poseAt(t, new Float32Array(NUM_JOINTS));
    assert(pose.every((value, j) => Math.abs(value - held[j]) < 1e-6), "Crouch hold drifts between its held keys");
  }
}

for (const motion of motions) {
  assert(motion.duration > 0 && motion.duration <= 8, `${motion.name}: expected a short, nonstatic reference`);
  const copy = parseMotion(motionToJson(motion));
  assert.equal(copy.name, motion.name);
  assert.equal(copy.loop, motion.loop);
  assert.equal(copy.keyframes.length, motion.keyframes.length);
  motion.keyframes.forEach((key, k) => key.pose.forEach((angle, j) => {
    assert(Math.abs(angle - copy.keyframes[k].pose[j]) <= 1e-4, `${motion.name}: round-trip changed joint ${j}`);
  }));
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelDir = path.join(root, "public/model/microduck");
const robotXml = "robot_allcollisions-nv.xml";
const manifest = JSON.parse(fs.readFileSync(path.join(modelDir, "manifest.json"), "utf8"));
const entry = manifest.models.find((model: { file: string }) => model.file === robotXml);
assert(entry, `${robotXml} missing; prepare the existing model assets first`);
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
vfs.addBuffer(robotXml, new Uint8Array(fs.readFileSync(path.join(modelDir, robotXml))));
for (const mesh of entry.meshes) {
  vfs.addBuffer(`assets/${mesh}`, new Uint8Array(fs.readFileSync(path.join(modelDir, "assets", mesh))));
}
const model = mujoco.MjModel.from_xml_string(sceneXml(robotXml), vfs);
const data = new mujoco.MjData(model);
try {
  const standKey = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND");
  assert(standKey >= 0, "STAND keyframe missing");
  const joints = JOINT_NAMES.map(name => {
    const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name);
    assert(id >= 0, `Joint ${name} missing from model`);
    return { id, qpos: model.jnt_qposadr[id], lo: model.jnt_range[id * 2], hi: model.jnt_range[id * 2 + 1] };
  });
  const actuators = JOINT_NAMES.map((name, j) => {
    const id = [...Array(model.nu).keys()].find(index => model.actuator_trnid[index * 2] === joints[j].id);
    assert.notEqual(id, undefined, `No joint actuator for ${name}`);
    return id!;
  });
  joints.forEach((joint, j) => {
    assert(Math.abs(joint.lo - JOINT_LIMITS[j][0]) < 1e-3 && Math.abs(joint.hi - JOINT_LIMITS[j][1]) < 1e-3, `${JOINT_NAMES[j]} parser/model limit drift`);
  });
  const results = [];
  for (const motion of motions) {
    const sampler = new MotionSampler(motion);
    const pose = new Float32Array(NUM_JOINTS);
    const velocity = new Float32Array(NUM_JOINTS);
    // Include interpolated poses, not just keys: unconstrained cubic curves
    // can overshoot otherwise valid keyframes near a mechanical stop.
    const samples = Math.ceil(motion.duration / 0.001);
    let minimumMargin = Infinity;
    for (let sample = 0; sample <= samples; sample++) {
      const t = motion.duration * sample / samples;
      sampler.poseAt(t, pose);
      sampler.velAt(t, velocity);
      for (let j = 0; j < NUM_JOINTS; j++) {
        assert(Number.isFinite(pose[j]) && Number.isFinite(velocity[j]), `${motion.name}: nonfinite reference at ${t}`);
        const margin = Math.min(pose[j] - joints[j].lo, joints[j].hi - pose[j]);
        assert(margin >= -1e-5, `${motion.name}: interpolated ${JOINT_NAMES[j]} outside compiled range at ${t}`);
        minimumMargin = Math.min(minimumMargin, margin);
      }
    }
    if (motion.loop) {
      const before = sampler.velAt(motion.duration - 1e-5, new Float32Array(NUM_JOINTS));
      const after = sampler.velAt(1e-5, new Float32Array(NUM_JOINTS));
      assert(before.every((v, j) => Math.abs(v - after[j]) < 0.05), `${motion.name}: discontinuous loop velocity`);
    }

    mujoco.mj_resetDataKeyframe(model, data, standKey);
    const steps = Math.ceil(motion.duration / CTRL_DT);
    let squaredTrackingError = 0;
    let minimumUpright = 1;
    for (let step = 0; step < steps; step++) {
      sampler.poseAt(step * CTRL_DT, pose);
      for (let j = 0; j < NUM_JOINTS; j++) data.ctrl[actuators[j]] = pose[j];
      for (let substep = 0; substep < DECIMATION; substep++) mujoco.mj_step(model, data);
      assert(data.qpos.every(Number.isFinite) && data.qvel.every(Number.isFinite), `${motion.name}: nonfinite physics state`);
      assert(Math.abs(data.time - (step + 1) * DECIMATION * TIMESTEP) < 1e-8, `${motion.name}: physics time reset or failed to advance`);
      for (let j = 0; j < NUM_JOINTS; j++) squaredTrackingError += (data.qpos[joints[j].qpos] - pose[j]) ** 2;
      minimumUpright = Math.min(minimumUpright, 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2));
    }
    const result = {
      name: motion.name, keyframes: motion.keyframes.length, sampled_reference_poses: samples + 1,
      minimum_joint_limit_margin_rad: minimumMargin, physics_steps: steps * DECIMATION,
      tracking_joint_count: NUM_JOINTS,
      tracking_rmse_rad: Math.sqrt(squaredTrackingError / (steps * NUM_JOINTS)),
      minimum_upright_cosine: minimumUpright, lost_upright: minimumUpright < 0.5,
    };
    results.push(result);
    console.log(JSON.stringify(result));
  }
  console.log(JSON.stringify({
    passed: true, templates: results.length, added_templates: expected.length,
    compiled_joint_ranges_checked: joints.length,
    scope: "parse/round-trip, 1ms-grid interpolated ranges, loop velocity continuity, and one finite open-loop cycle per template; not balance or task success",
    open_loop_lost_upright: results.filter(result => result.lost_upright).map(result => result.name),
  }));
} finally {
  data.delete();
  model.delete();
  vfs.delete();
}
