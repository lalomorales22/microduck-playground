// Headless smoke test for the policy loop: drops the duck on its back and
// checks the get-up policy actually puts it back on its feet.
//
//   node scripts/check-standup.mjs [trials]
//
// It re-implements the observation layout independently of src/ on purpose:
// if the two ever disagree, that is exactly the bug this catches.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = path.join(ROOT, "public/model/microduck");

const JOINT_NAMES = [
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
];
const DEFAULT_POSE = [
  0, -0.08726646259971647, -0.457924, -0.00494, 0.452984,
  0.3490658503988659, 0.3490658503988659, 0, 0,
  0, 0.08726646259971647, 0.457924, 0.00494, -0.452984,
];
const NJ = 14, OBS = 61, DECIMATION = 4;

ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const mj = await loadMujoco();
const vfs = new mj.MjVFS();
const manifest = JSON.parse(fs.readFileSync(path.join(MODEL, "manifest.json"), "utf8"));
// The Simulate path runs the full-collision model with its visual geoms — the
// same one the viewport compiles, which is the point of this check.
const ROBOT_XML = "robot_allcollisions.xml";
const entry = manifest.models.find((m) => m.file === ROBOT_XML);
if (!entry) throw new Error(`${ROBOT_XML} missing from manifest; run npm run prepare-assets`);
vfs.addBuffer(ROBOT_XML, new Uint8Array(fs.readFileSync(path.join(MODEL, ROBOT_XML))));
for (const f of entry.meshes) {
  vfs.addBuffer(`assets/${f}`, new Uint8Array(fs.readFileSync(path.join(MODEL, "assets", f))));
}

const pose = DEFAULT_POSE.join(" ");
const xml = `<mujoco model="microduck-scene">
  <include file="${ROBOT_XML}"/>
  <option timestep="0.005"/>
  <worldbody><geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0"/></worldbody>
  <keyframe><key name="STAND" qpos="0 0 0.12 1 0 0 0 ${pose}" ctrl="${pose}"/></keyframe>
</mujoco>`;

const model = mj.MjModel.from_xml_string(xml, vfs);
const data = new mj.MjData(model);
const standKey = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY.value, "STAND");
const qposAdr = JOINT_NAMES.map((n) => model.jnt(n).qposadr);
const dofAdr = JOINT_NAMES.map((n) => model.jnt(n).dofadr);
const gyroAdr = model.sensor("imu_ang_vel").adr;
const trunkId = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, "trunk_base");

const session = await ort.InferenceSession.create(
  path.join(ROOT, "public/policies/alpha_stand.onnx"), { executionProviders: ["wasm"] },
);
console.log("policy io:", session.inputNames, "->", session.outputNames);

const obs = new Float32Array(OBS);
const lastAction = new Float32Array(NJ);

function gravity() {
  const q = data.body(trunkId).xquat;
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  return [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))];
}

function buildObs() {
  let i = 0;
  for (let a = 0; a < 3; a++) obs[i++] = data.sensordata[gyroAdr + a];
  for (const g of gravity()) obs[i++] = g;
  for (let j = 0; j < NJ; j++) obs[i++] = data.qpos[qposAdr[j]] - DEFAULT_POSE[j];
  for (let j = 0; j < NJ; j++) obs[i++] = data.qvel[dofAdr[j]];
  for (let j = 0; j < NJ; j++) obs[i++] = lastAction[j];
  for (let c = 0; c < 13; c++) obs[i++] = 0;
  return obs;
}

const stepPhysics = () => { for (let s = 0; s < DECIMATION; s++) mj.mj_step(model, data); };

let ok = 0;
const trials = Number(process.argv[2] ?? 5);
for (let t = 0; t < trials; t++) {
  mj.mj_resetDataKeyframe(model, data, standKey);
  // Same tumble the browser's knockDown() uses.
  const axis = (t / trials) * Math.PI * 2;
  const tilt = (Math.PI / 2) * 1.4;
  const s = Math.sin(tilt / 2);
  data.qpos[2] = 0.12;
  data.qpos[3] = Math.cos(tilt / 2);
  data.qpos[4] = Math.cos(axis) * s;
  data.qpos[5] = Math.sin(axis) * s;
  data.qpos[6] = 0;
  mj.mj_forward(model, data);

  // Limp fall: hold the current pose so the actuators produce ~no torque.
  for (let i = 0; i < 150; i++) {
    for (let j = 0; j < NJ; j++) data.ctrl[j] = data.qpos[qposAdr[j]];
    stepPhysics();
  }
  const gzFallen = gravity()[2];

  lastAction.fill(0);
  let upright = 0, best = gzFallen;
  for (let i = 0; i < 300; i++) {
    const out = await session.run({ obs: new ort.Tensor("float32", buildObs(), [1, OBS]) });
    const act = out.actions.data;
    lastAction.set(act);
    for (let j = 0; j < NJ; j++) data.ctrl[j] = DEFAULT_POSE[j] + act[j];
    stepPhysics();
    const gz = gravity()[2];
    best = Math.min(best, gz);
    upright = gz < -0.85 ? upright + 1 : 0;
    if (upright >= 50) break;
  }
  // Keep the policy driving for another 5 s: the demo leaves it in control once
  // upright, so it has to hold the pose, not just reach it.
  let held = true;
  if (upright >= 50) {
    for (let i = 0; i < 250; i++) {
      const out = await session.run({ obs: new ort.Tensor("float32", buildObs(), [1, OBS]) });
      const act = out.actions.data;
      lastAction.set(act);
      for (let j = 0; j < NJ; j++) data.ctrl[j] = DEFAULT_POSE[j] + act[j];
      stepPhysics();
      if (gravity()[2] > -0.85) held = false;
    }
  }
  const good = upright >= 50 && held;
  ok += good ? 1 : 0;
  console.log(
    `trial ${t + 1}: fallen gz=${gzFallen.toFixed(2)} -> best gz=${best.toFixed(2)} ` +
      `z=${(data.qpos[2] * 100).toFixed(1)}cm held=${held}  ${good ? "STOOD UP" : "FAILED"}`,
  );
}
console.log(`\n${ok}/${trials} stood up`);
process.exit(ok === trials ? 0 : 1);
