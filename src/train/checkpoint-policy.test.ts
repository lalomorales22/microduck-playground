import assert from "node:assert/strict";
import test from "node:test";
import { ActorCritic } from "./ac-policy.ts";
import { TASK_OBSERVATION_SCHEMA } from "./checkpoint-task.ts";
import { taskCommandWriter, taskLabel, type TaskSpec } from "./env/tasks.ts";
import { CheckpointPolicy } from "../sim/policy.ts";
import { CTRL_DT, OBS_SIZE } from "../sim/microduck.ts";

function fixture(task: TaskSpec) {
  const ac = new ActorCritic(undefined, undefined, () => 0.6, 0.1, [8], null);
  return { version: 1, iteration: 5, task: taskLabel(task), taskSpec: task,
    observationSchema: TASK_OBSERVATION_SCHEMA, policy: ac.serialize() };
}

test("checkpoint replay injects the same task phase as training and resets its clock", async () => {
  const tasks: TaskSpec[] = [
    { kind: "walk_forward" }, { kind: "jump" }, { kind: "somersault" }, { kind: "backflip" },
    { kind: "motion", motion: { name: " nod ", loop: true, joints: ["head_pitch"], keyframes: [{ t: 0, pose: [0.2] }, { t: 0.5, pose: [0.4] }, { t: 1, pose: [0.2] }] } },
  ];
  for (const task of tasks) {
    const checkpoint = fixture(task);
    const policy = new CheckpointPolicy(checkpoint, "test");
    const reference = new ActorCritic(undefined, undefined, () => 0.1, 0.1, [8], null);
    reference.load(checkpoint.policy);
    const write = taskCommandWriter(task);
    let first: Float32Array | null = null;
    for (let step = 0; step < 80; step++) {
      const obs = new Float32Array(OBS_SIZE), expected = obs.slice();
      write(step * CTRL_DT, expected);
      const actual = await policy.run(obs);
      assert.deepEqual(actual, reference.actMean(expected, 1).slice());
      assert.ok(obs.every(n => n === 0), "input observation was modified");
      first ??= actual;
    }
    policy.reset();
    assert.deepEqual(await policy.run(new Float32Array(OBS_SIZE)), first);
    assert.equal(policy.automaticRecovery, false);
  }
});

test("only learned stand-up policies opt into automatic recovery", () => {
  assert.equal(new CheckpointPolicy(fixture({ kind: "standup" }), "test").automaticRecovery, true);
  assert.equal(new CheckpointPolicy(fixture({ kind: "hold_pose" }), "test").automaticRecovery, false);
  assert.equal(new CheckpointPolicy(fixture({ kind: "backflip" }), "test").experimental, true);
});

test("replay rejects malformed observations and incompatible checkpoint metadata", async () => {
  const checkpoint = fixture({ kind: "walk_forward" });
  const policy = new CheckpointPolicy(checkpoint, "test");
  assert.throws(() => policy.run(new Float32Array(2)));
  assert.throws(() => policy.run(new Float32Array(OBS_SIZE).fill(NaN)));
  assert.throws(() => new CheckpointPolicy({ ...checkpoint, version: 2 }, "test"));
});
