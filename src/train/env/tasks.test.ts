import assert from "node:assert/strict";
import test from "node:test";
import { BUILTIN_TASKS, buildSpec, taskCommandWriter, validateTask } from "./tasks.ts";
import { LOCOMOTION_TARGETS, velocityTracking } from "./locomotion.ts";
import { CMD_OFFSET, DEFAULT_POSE, OBS_SIZE, NUM_JOINTS } from "../../sim/microduck.ts";
import type { EnvContext } from "./seams.ts";
import type { StepState } from "./rewards.ts";

test("all built-in tasks have a usable stable identity and finite commands", () => {
  for (const { kind } of BUILTIN_TASKS) {
    const task = { kind };
    const spec = buildSpec(task);
    assert.equal(spec.name, kind);
    assert.ok(spec.rewards.length > 0);
    const obs = new Float32Array(OBS_SIZE).fill(3);
    const write = taskCommandWriter(task);
    write(0.6, obs);
    assert.ok(obs.every(Number.isFinite));
    assert.ok(obs.slice(0, CMD_OFFSET).every(n => n === 3));
  }
});

test("unknown, malformed and unsupported task parameters fail closed", () => {
  for (const task of [null, [], {}, { kind: "teleport" }, { kind: "walk_forward", speed: NaN }, { kind: "motion", motion: {} }]) {
    assert.throws(() => validateTask(task));
  }
  for (const episodeLengthS of [NaN, Infinity, 0, -1, 61]) assert.throws(() => buildSpec({ kind: "hold_pose" }, { episodeLengthS }));
  const write = taskCommandWriter({ kind: "walk_forward" });
  assert.throws(() => write(NaN, new Float32Array(OBS_SIZE)));
  assert.throws(() => write(0, new Float32Array(2)));
});

test("locomotion reward prefers requested movement, rejects reversed direction and falling", () => {
  const qvel = new Float64Array(20), sensordata = new Float64Array(3);
  const ctx = { data: { qvel, sensordata }, model: { sensor: () => ({ adr: 0 }) } } as unknown as EnvContext;
  const s: StepState = {
    z: 0.115, quat: new Float64Array([1, 0, 0, 0]), jointPos: DEFAULT_POSE.slice(), jointTarget: DEFAULT_POSE,
    vz: 0, az: 0, action: new Float32Array(NUM_JOINTS), prevAction: new Float32Array(NUM_JOINTS),
  };
  for (const kind of Object.keys(LOCOMOTION_TARGETS) as (keyof typeof LOCOMOTION_TARGETS)[]) {
    const target = LOCOMOTION_TARGETS[kind];
    qvel[0] = target.forward;
    sensordata[2] = target.yaw;
    const correct = velocityTracking(kind, ctx, s);
    assert.equal(correct, 1);
    qvel[0] *= -1;
    sensordata[2] *= -1;
    assert.ok(velocityTracking(kind, ctx, s) < correct);
    const fallen = { ...s, quat: new Float64Array([0, 1, 0, 0]) };
    assert.equal(velocityTracking(kind, ctx, fallen), 0);
  }
});

test("balance commands remain zero and new task commands match the environment pipeline", () => {
  for (const { kind } of BUILTIN_TASKS) {
    const spec = buildSpec({ kind });
    const envObs = new Float32Array(OBS_SIZE);
    spec.obsPipeline?.apply({ time: 0.61 } as EnvContext, envObs, 0);
    const replayObs = new Float32Array(OBS_SIZE);
    taskCommandWriter({ kind })(0.61, replayObs);
    assert.deepEqual(replayObs, envObs);
  }
});
