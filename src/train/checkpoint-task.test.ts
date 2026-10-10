import assert from "node:assert/strict";
import test from "node:test";
import { ActorCritic } from "./ac-policy.ts";
import { checkpointMetadata, checkpointNetwork, validateCheckpointResume, validateTrainingConfig, TASK_OBSERVATION_SCHEMA } from "./checkpoint-task.ts";
import { taskCommandWriter } from "./env/tasks.ts";
import { OBS_SIZE } from "../sim/microduck.ts";
import { DEFAULT_PPO } from "./ppo.ts";

const checkpoint = () => ({ version: 1, iteration: 5, task: "hold_pose", policy: new ActorCritic(undefined, undefined, () => 0.5, 0.1, [8], null).serialize() });

test("legacy hold-pose and standup checkpoints retain zero-command semantics", () => {
  for (const task of [undefined, "hold_pose", "standup"]) {
    const ckpt = { ...checkpoint(), task };
    const metadata = checkpointMetadata(ckpt);
    assert.equal(metadata.task.kind, task ?? "hold_pose");
    assert.deepEqual(metadata.world, { version: 1, objects: [] });
    assert.deepEqual(checkpointNetwork(ckpt).hidden, [8]);
  }
});

test("task checkpoints round-trip their commands and validate schema and identity", () => {
  const ckpt = { ...checkpoint(), task: "backflip", taskSpec: { kind: "backflip" }, observationSchema: TASK_OBSERVATION_SCHEMA };
  const metadata = checkpointMetadata(JSON.parse(JSON.stringify(ckpt)));
  const a = new Float32Array(OBS_SIZE), b = new Float32Array(OBS_SIZE);
  taskCommandWriter({ kind: "backflip" })(0.59, a);
  taskCommandWriter(metadata.task)(0.59, b);
  assert.deepEqual(a, b);
  assert.throws(() => checkpointMetadata({ ...ckpt, observationSchema: "unknown" }));
  assert.throws(() => checkpointMetadata({ ...ckpt, observationSchema: undefined }));
  assert.throws(() => checkpointMetadata({ ...ckpt, task: "jump" }));
  assert.throws(() => checkpointMetadata({ ...checkpoint(), task: "motion:missing" }));
  assert.throws(() => checkpointMetadata({ ...ckpt, world: { version: 100, objects: [] } }));
});

test("malformed policy shape, truncated weights, dimensions and non-finite data are refused", () => {
  for (const mutate of [
    (c: any) => { c.policy.actor[0].w.pop(); },
    (c: any) => { c.policy.actor[0].w[0] = NaN; },
    (c: any) => { c.policy.normalizer.var[0] = -1; },
    (c: any) => { c.policy.obsDim = 60; },
    (c: any) => { c.policy.hidden = [100000]; },
    (c: any) => { c.config = { hidden: [16] }; },
    (c: any) => { c.iteration = -1; },
  ]) {
    const c = checkpoint(); mutate(c);
    assert.throws(() => checkpointNetwork(c));
  }
});

function resumable() {
  const ac = new ActorCritic(undefined, undefined, () => 0.5, 0.1, [8], null);
  return { version: 1, iteration: 0, totalSteps: 0, rngState: 7, task: "hold_pose",
    config: { envs: 4, stepsPerIter: 8, hidden: [8], seed: 7, ppo: { ...DEFAULT_PPO, epochs: 1, minibatches: 2 } },
    policy: ac.serialize(), optimizer: ac.makeOptimizer(DEFAULT_PPO.lr).serialize() };
}

test("complete legacy optimizer/config checkpoints remain resumable", () => {
  const c = resumable();
  assert.equal(validateCheckpointResume(c), c);
  const copy = validateTrainingConfig(c.config);
  assert.deepEqual(copy, c.config);
  assert.notEqual(copy, c.config);
});

test("resume rejects missing, short, extra, non-finite and negative Adam moments", () => {
  for (const mutate of [
    (c: any) => { delete c.optimizer; },
    (c: any) => { c.optimizer.m.pop(); },
    (c: any) => { c.optimizer.v.push([]); },
    (c: any) => { c.optimizer.m[0].pop(); },
    (c: any) => { c.optimizer.v[0] = []; },
    (c: any) => { c.optimizer.m[0][0] = NaN; },
    (c: any) => { c.optimizer.v[0][0] = -1; },
    (c: any) => { c.optimizer.m[0][0] = 1e300; },
    (c: any) => { c.optimizer.t = -1; },
    (c: any) => { c.rngState = 2 ** 32; },
    (c: any) => { c.totalSteps = Infinity; },
  ]) {
    const c = resumable(); mutate(c);
    assert.throws(() => validateCheckpointResume(c));
  }
});

test("training config bounds reject unsafe allocation and invalid PPO parameters", () => {
  for (const mutate of [
    (c: any) => { c.envs = 1000000; },
    (c: any) => { c.stepsPerIter = 0; },
    (c: any) => { c.stepsPerIter = 1.5; },
    (c: any) => { c.hidden = [1024, 1024, 1024]; },
    (c: any) => { c.seed = -1; },
    (c: any) => { c.ppo.epochs = Infinity; },
    (c: any) => { c.ppo.minibatches = 3; },
    (c: any) => { c.ppo.lr = NaN; },
    (c: any) => { c.ppo.gamma = 2; },
    (c: any) => { c.ppo.maxGradNorm = 0; },
    (c: any) => { c.exploration = { initStd: -1 }; },
  ]) {
    const c = resumable().config; mutate(c);
    assert.throws(() => validateTrainingConfig(c));
  }
});
