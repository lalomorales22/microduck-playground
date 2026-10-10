// Actual local WASM physics, with a deterministic probe policy rather than a
// trained skill. This checks the replay contract, not jump learning success.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import loadMujoco from "@mujoco/mujoco";
import { MicroduckController } from "./controller.ts";
import { CMD_OFFSET, CTRL_DT, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE } from "./microduck.ts";
import type { PolicyRunner } from "./policy.ts";
import { compileScene, type Manifest, type ModelAssets, type Simulation } from "./scene.ts";
import { acrobaticsSpec, writeAcrobaticCommand } from "../train/env/acrobatics.ts";
import { VecEnv, type StepResult } from "../train/env/vec-env.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const MODEL_DIR = path.join(ROOT, "public/model/microduck");
const ROBOTS = ["robot_allcollisions.xml", "robot_allcollisions-nv.xml"];
const ACTION_OFFSET = CMD_OFFSET - NUM_JOINTS;
const TICKS = 45;
const manifest = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "manifest.json"), "utf8")) as Manifest;
const entries = ROBOTS.map((file) => {
  const entry = manifest.models.find((item) => item.file === file);
  assert.ok(entry, `Missing local ${file}; prepare-assets is required`);
  return entry;
});
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
for (const entry of entries) vfs.addBuffer(entry.file, fs.readFileSync(path.join(MODEL_DIR, entry.file)));
for (const mesh of new Set(entries.flatMap((entry) => entry.meshes))) {
  vfs.addBuffer(`assets/${mesh}`, fs.readFileSync(path.join(MODEL_DIR, "assets", mesh)));
}
const assets = { mujoco, vfs, manifest } as unknown as ModelAssets;
after(() => vfs.delete());

function close(actual: number, expected: number, label: string, tolerance = 1e-9): void {
  assert.ok(Number.isFinite(actual), `${label} must be finite`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} != ${expected}`);
}

function closeArray(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string): void {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let i = 0; i < actual.length; i++) close(actual[i], expected[i], `${label}[${i}]`);
}

function jumpProbe() {
  const state = { resets: 0, time: 0, times: [] as number[],
    raw: [] as Float32Array[], observations: [] as Float32Array[], actions: [] as Float32Array[] };
  const policy: PolicyRunner = {
    label: "deterministic jump replay probe", taskLabel: "jump", automaticRecovery: false,
    reset() {
      state.resets++; state.time = 0;
      state.times.length = 0; state.raw.length = 0; state.observations.length = 0; state.actions.length = 0;
    },
    async run(obs) {
      assert.equal(obs.length, OBS_SIZE);
      state.raw.push(obs.slice()); state.times.push(state.time);
      const commanded = obs.slice();
      writeAcrobaticCommand("jump", state.time, commanded);
      state.observations.push(commanded);
      const action = new Float32Array(NUM_JOINTS);
      for (let j = 0; j < NUM_JOINTS; j++) {
        action[j] = 0.01 * Math.sin(state.time * 3 + j) - 0.05 * commanded[6 + j];
      }
      state.actions.push(action.slice());
      state.time += CTRL_DT;
      return action;
    },
  };
  return { policy, state };
}

for (const robotXml of ROBOTS) {
  test(`${robotXml}: fresh policy selection and reset match training state and replay deterministically`, async () => {
    const { model, standKey } = compileScene(assets, { robotXml });
    const data = new assets.mujoco.MjData(model);
    let env: VecEnv | undefined;
    try {
      assets.mujoco.mj_resetDataKeyframe(model, data, standKey);
      assets.mujoco.mj_forward(model, data);
      const sim: Simulation = { mujoco: assets.mujoco, model, data, standKey };
      const old = jumpProbe();
      const controller = new MicroduckController(sim, old.policy);
      for (let tick = 0; tick < 5; tick++) await controller.step();
      controller.setCommand(0.1, -0.1, 0.5);
      controller.push(0.5);
      controller.goLimp();
      data.qpos[2] = 0.06;
      data.qpos[3] = Math.SQRT1_2;
      data.qpos[4] = Math.SQRT1_2;
      data.qpos[5] = 0; data.qpos[6] = 0;
      data.qvel.fill(0.2); data.ctrl.fill(0.7);
      assets.mujoco.mj_forward(model, data);
      assert.ok(controller.telemetry().simTime > 0);
      assert.ok(data.xfrc_applied.some((value) => value !== 0));
      assert.ok(controller.gravityZ > -0.1, "fixture starts sideways");

      const probe = jumpProbe();
      controller.setPolicy(probe.policy);
      assert.equal(probe.state.resets, 1, "selected policy resets exactly once");
      assert.equal(controller.telemetry().simTime, 0);
      assert.equal(controller.telemetry().phase, "standing");
      assert.equal(controller.telemetry().phaseSteps, 0);
      close(data.time, 0, "MuJoCo clock after switch");
      close(controller.gravityZ, -1, "upright before first inference");
      closeArray(data.qpos, [0, 0, 0.12, 1, 0, 0, 0, ...DEFAULT_POSE], "STAND qpos");
      assert.ok(data.qvel.every((value) => value === 0), "old linear, angular and joint velocities are cleared");
      assert.ok(data.xfrc_applied.every((value) => value === 0), "old shove is cleared");
      closeArray(data.ctrl, DEFAULT_POSE, "default controls before first inference");

      env = new VecEnv({ mujoco: assets.mujoco, model, standKey,
        spec: acrobaticsSpec("jump"), count: 1, rng: () => 0.73 });
      assert.equal(env.timeOf(0), 0, "acrobatics constructor never staggers the clock");
      env.stagger();
      assert.equal(env.timeOf(0), 0, "explicit stagger still preserves acrobatics start phase");
      closeArray(data.qpos, env.dataAt(0).qpos, "replay/training initial root and joint pose");
      closeArray(data.qvel, env.dataAt(0).qvel, "replay/training initial velocities");

      const trajectory: { qpos: Float64Array; qvel: Float64Array }[] = [];
      for (let tick = 0; tick < TICKS; tick++) {
        const expectedObs = env.observations.slice();
        await controller.step();
        const action = probe.state.actions[tick];
        close(probe.state.times[tick], tick * CTRL_DT, "pre-action policy time");
        closeArray(probe.state.observations[tick], expectedObs, "task-aware replay/training observation");
        const result: StepResult = env.step(action);
        assert.equal(result.done[0], 0, "probe stays inside the first training episode");
        closeArray(data.ctrl, env.dataAt(0).ctrl, "replay/training action targets");
        closeArray(data.qpos, env.dataAt(0).qpos, "replay/training qpos after four physics substeps");
        closeArray(data.qvel, env.dataAt(0).qvel, "replay/training qvel after four physics substeps");
        close(controller.telemetry().simTime, env.timeOf(0), "post-step clocks agree");
        close(data.time, env.dataAt(0).time, "MuJoCo data clocks agree");
        trajectory.push({ qpos: Float64Array.from(data.qpos), qvel: Float64Array.from(data.qvel) });
      }
      const first = probe.state.raw[0];
      closeArray(first.slice(0, 3), [0, 0, 0], "first angular velocity observation");
      closeArray(first.slice(3, 6), [0, 0, -1], "first projected gravity");
      assert.ok(first.slice(6, CMD_OFFSET).every((value) => value === 0), "first joint offsets, velocities and previous actions are zero");
      assert.ok(first.slice(CMD_OFFSET).every((value) => value === 0), "old manual commands are cleared before task command injection");
      assert.ok(probe.state.observations.some((obs) => obs[CMD_OFFSET + 9] === 1), "comparison includes flight-phase targets");
      assert.ok(probe.state.observations.some((obs) => obs[CMD_OFFSET + 10] === 1), "comparison includes settle-phase targets");

      const observations = probe.state.observations.map((obs) => obs.slice());
      const actions = probe.state.actions.map((action) => action.slice());
      controller.setCommand(0.2, 0.1, -0.5);
      controller.push(0.5);
      controller.reset();
      assert.equal(probe.state.resets, 2);
      assert.equal(controller.telemetry().simTime, 0);
      closeArray(data.ctrl, DEFAULT_POSE, "reset default controls");
      for (let tick = 0; tick < TICKS; tick++) {
        await controller.step();
        closeArray(probe.state.observations[tick], observations[tick], "repeated task-aware observation");
        closeArray(probe.state.actions[tick], actions[tick], "repeated deterministic action");
        closeArray(data.qpos, trajectory[tick].qpos, "repeated physical trajectory qpos");
        closeArray(data.qvel, trajectory[tick].qvel, "repeated physical trajectory qvel");
      }
      assert.ok(probe.state.raw[0].slice(ACTION_OFFSET).every((value) => value === 0),
        "reset clears old action and manual command history");
      assert.equal(env.nonFiniteSteps, 0);
    } finally {
      env?.dataAt(0).delete();
      data.delete(); model.delete();
    }
  });
}
