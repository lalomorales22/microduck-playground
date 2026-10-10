// Real controller + shipped ONNX exports + the viewport's clear-floor model.
// Run with Node 24: node scripts/check-policies.ts
// The default 0.25 m/s command starts these gaits in this PD-actuator scene.
// --forward 0.15 retains the known low-command diagnostic; it currently fails
// the same movement gates. Passing does not establish precise speed tracking.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web/wasm";
import { MicroduckController } from "../src/sim/controller.ts";
import { CTRL_DT, JOINT_NAMES, NUM_JOINTS, OBS_SIZE, TRUNK_BODY } from "../src/sim/microduck.ts";
import { SHIPPED_POLICIES, POLICY_REVISION } from "../src/sim/policy-catalog.ts";
import type { PolicyRunner } from "../src/sim/policy.ts";
import { sceneXml, type Simulation } from "../src/sim/scene.ts";
import type { Mujoco } from "../src/sim/mujoco.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = path.join(ROOT, "public/model/microduck");
const { values } = parseArgs({ options: {
  robot: { type: "string", default: "robot_allcollisions.xml" },
  forward: { type: "string", default: "0.25" },
  yaw: { type: "string", default: "0.5" },
} });
const ROBOT_XML = values.robot;
assert(["robot_allcollisions.xml", "robot_groundcontact.xml", "robot_walk.xml"].includes(ROBOT_XML), "unsupported robot");
const forwardCommand = Number(values.forward), yawCommand = Number(values.yaw);
assert(Number.isFinite(forwardCommand) && forwardCommand >= 0.1 && forwardCommand <= 0.25, "forward must be 0.1..0.25 m/s");
assert(Number.isFinite(yawCommand) && yawCommand >= 0.1 && yawCommand <= 1, "yaw must be 0.1..1 rad/s");
const STEPS = Math.round(5 / CTRL_DT);
const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const sourceHash = (file: string) => sha256(fs.readFileSync(path.join(ROOT, file)));
const sources = {
  controller: "src/sim/controller.ts", scene: "src/sim/scene.ts",
  microduck: "src/sim/microduck.ts", catalogue: "src/sim/policy-catalog.ts",
  test: "scripts/check-policies.ts", lockfile: "package-lock.json",
  robotXml: `public/model/microduck/${ROBOT_XML}`,
};
const sourceHashes = Object.fromEntries(Object.entries(sources).map(([key, file]) => [key, sourceHash(file)]));
const failures: string[] = [];

ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";
const mujoco = await loadMujoco() as unknown as Mujoco;
const manifest = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "manifest.json"), "utf8"));
const entry = manifest.models.find((m: { file: string }) => m.file === ROBOT_XML);
if (!entry) throw new Error(`${ROBOT_XML} missing; run npm run prepare-assets`);
const vfs = new mujoco.MjVFS();
vfs.addBuffer(ROBOT_XML, new Uint8Array(fs.readFileSync(path.join(MODEL_DIR, ROBOT_XML))));
for (const mesh of entry.meshes) {
  vfs.addBuffer(`assets/${mesh}`, new Uint8Array(fs.readFileSync(path.join(MODEL_DIR, "assets", mesh))));
}
const model = mujoco.MjModel.from_xml_string(sceneXml(ROBOT_XML), vfs);
const standKey = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND");
const trunkId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY);
const joints = JOINT_NAMES.map((name) => ({
  id: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name),
  qpos: model.jnt(name).qposadr,
}));

function freshSimulation(): Simulation {
  const data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, standKey);
  mujoco.mj_forward(model, data);
  return { mujoco, model, data, standKey };
}

function yaw(sim: Simulation): number {
  const [w, x, y, z] = sim.data.body(trunkId).xquat;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

// Check commands through the observation supplied to a real controller step,
// without reading its private fields or duplicating its clamping logic.
let commandChecks = 0;
{
  const sim = freshSimulation();
  let observed = new Float32Array(13);
  const probe: PolicyRunner = {
    label: "command probe",
    async run(obs) { observed = obs.slice(48); return new Float32Array(NUM_JOINTS); },
  };
  const controller = new MicroduckController(sim, probe);
  const near = (actual: number, expected: number) => assert(Math.abs(actual - expected) < 1e-6);
  controller.setCommand(0.1, 0.05, 0.2);
  for (const invalid of [NaN, Infinity, -Infinity]) {
    for (let axis = 0; axis < 3; axis++) {
      const command: [number, number, number] = [0.1, 0.05, 0.2];
      command[axis] = invalid;
      assert.throws(() => controller.setCommand(...command), /finite/);
      await controller.step();
      [0.1, 0.05, 0.2].forEach((v, i) => near(observed[i], v));
      commandChecks++;
    }
  }
  for (const sign of [1, -1]) {
    controller.setCommand(sign * 100, sign * 100, sign * 100);
    await controller.step();
    [0.25, 0.15, 1].forEach((v, i) => near(observed[i], sign * v));
    commandChecks++;
  }
  assert(observed.slice(3).every((v) => v === 0));
  commandChecks++;
  (sim.data as unknown as { delete(): void }).delete();
}

interface Segment {
  policy: string;
  segment: string;
  command: [number, number, number];
  seconds: number;
  finite: boolean;
  uprightFraction: number;
  standingFraction: number;
  displacementX: number;
  displacementY: number;
  signedForwardDistance: number;
  yawChange: number;
  finalHeight: number;
  maxJointLimitExcess: number;
  phaseCounts: Record<string, number>;
}
const results: Segment[] = [];
const modelHashes: Record<string, string> = {};

for (const policy of SHIPPED_POLICIES) {
  const bytes = fs.readFileSync(path.join(ROOT, "public/policies", policy.id));
  assert.equal(bytes.length, policy.bytes, `${policy.id}: size`);
  assert.equal(sha256(bytes), policy.sha256, `${policy.id}: digest`);
  modelHashes[policy.id] = sha256(bytes);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const input = session.inputMetadata[0], output = session.outputMetadata[0];
  assert(input.isTensor && input.type === "float32" && input.shape.join() === "1,61");
  assert(output.isTensor && output.type === "float32" && output.shape.join() === "1,14");
  const runner: PolicyRunner = {
    label: policy.label,
    async run(obs) {
      assert.equal(obs.length, OBS_SIZE);
      assert(obs.every(Number.isFinite), `${policy.id}: nonfinite observation`);
      const out = await session.run({ [input.name]: new ort.Tensor("float32", obs, [1, OBS_SIZE]) });
      const action = out[output.name].data as Float32Array;
      assert.equal(action.length, NUM_JOINTS);
      assert(action.every(Number.isFinite), `${policy.id}: nonfinite action`);
      return action;
    },
  };
  const sim = freshSimulation();
  const controller = new MicroduckController(sim, runner);
  const segments: { name: string; command: [number, number, number] }[] = [
    { name: "zero", command: [0, 0, 0] },
    ...(policy.commands ? [
      { name: "forward", command: [forwardCommand, 0, 0] as [number, number, number] },
      { name: "turn-left", command: [0, 0, yawCommand] as [number, number, number] },
      { name: "turn-right", command: [0, 0, -yawCommand] as [number, number, number] },
    ] : []),
  ];
  try {
    for (const segment of segments) {
      controller.setCommand(...segment.command);
      const [x0, y0] = sim.data.body(trunkId).xpos;
      const yaw0 = yaw(sim);
      let lastYaw = yaw0, accumulatedYaw = 0, upright = 0, standing = 0;
      let maxJointLimitExcess = 0, finite = true;
      const phaseCounts: Record<string, number> = {};
      for (let t = 0; t < STEPS; t++) {
        await controller.step();
        finite &&= [sim.data.qpos, sim.data.qvel, sim.data.ctrl, sim.data.sensordata]
          .every((values) => values.every(Number.isFinite));
        if (!finite) throw new Error(`${policy.id}: nonfinite physics`);
        const telemetry = controller.telemetry();
        if (telemetry.gravityZ < -0.85) {
          upright++;
          if (telemetry.height > 0.09) standing++;
        }
        phaseCounts[telemetry.phase] = (phaseCounts[telemetry.phase] ?? 0) + 1;
        const nextYaw = yaw(sim);
        accumulatedYaw += Math.atan2(Math.sin(nextYaw - lastYaw), Math.cos(nextYaw - lastYaw));
        lastYaw = nextYaw;
        for (const joint of joints) {
          const value = sim.data.qpos[joint.qpos];
          const min = model.jnt_range[2 * joint.id], max = model.jnt_range[2 * joint.id + 1];
          maxJointLimitExcess = Math.max(maxJointLimitExcess, min - value, value - max);
        }
      }
      const [x, y, z] = sim.data.body(trunkId).xpos;
      const result: Segment = {
        policy: policy.id, segment: segment.name, command: segment.command,
        seconds: STEPS * CTRL_DT, finite, uprightFraction: upright / STEPS,
        standingFraction: standing / STEPS, displacementX: x - x0, displacementY: y - y0,
        signedForwardDistance: (x - x0) * Math.cos(yaw0) + (y - y0) * Math.sin(yaw0),
        yawChange: accumulatedYaw, finalHeight: z, maxJointLimitExcess, phaseCounts,
      };
      results.push(result);
      console.log(JSON.stringify(result));
      if (result.uprightFraction < 0.9 || result.standingFraction < 0.9)
        failures.push(`${policy.id}/${segment.name}: not upright and standing for 90% of steps`);
      // MuJoCo joint limits are compliant, so report every excursion and allow
      // 0.05 rad of solver compliance, not unrestricted joint-limit violations.
      if (result.maxJointLimitExcess > 0.05)
        failures.push(`${policy.id}/${segment.name}: joint limit excess > 0.05 rad`);
      if (segment.name === "forward" && (result.displacementX <= 0.05 || result.signedForwardDistance <= 0.05))
        failures.push(`${policy.id}: forward command did not produce > 5 cm forward movement`);
      if (segment.name === "turn-left" && result.yawChange <= 0.2)
        failures.push(`${policy.id}: positive yaw command did not produce > 0.2 rad left turn`);
      if (segment.name === "turn-right" && result.yawChange >= -0.2)
        failures.push(`${policy.id}: negative yaw command did not produce > 0.2 rad right turn`);
    }
  } finally {
    (sim.data as unknown as { delete(): void }).delete();
    await session.release();
  }
}
(model as unknown as { delete(): void }).delete();
(vfs as unknown as { delete(): void }).delete();
const sourcesUnchanged = Object.entries(sources).every(([key, file]) => sourceHash(file) === sourceHashes[key]);
if (!sourcesUnchanged) failures.push("Source files changed during this run; repeat for a stable receipt");
console.log(JSON.stringify({
  status: failures.length ? "failed" : "passed", backend: "MuJoCo WASM + ONNX Runtime WASM CPU",
  robotXml: ROBOT_XML, policyRevision: POLICY_REVISION, commandChecks, results, failures,
  sourcesUnchanged, sha256: { ...modelHashes, ...sourceHashes },
}, null, 2));
if (failures.length) process.exitCode = 1;
