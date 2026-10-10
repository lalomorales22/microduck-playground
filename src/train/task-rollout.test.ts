import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { loadModelFromDisk } from "./model-loader.ts";
import { BUILTIN_TASKS, buildSpec, taskCommandWriter } from "./env/tasks.ts";
import { VecEnv } from "./env/vec-env.ts";
import { DEFAULT_TRAINER, Trainer } from "./trainer.ts";
import { DEFAULT_PPO } from "./ppo.ts";
import { CMD_OFFSET, NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";

test("all task presets perform finite real MuJoCo rollouts and bounded PPO updates", { timeout: 120000 }, async () => {
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const loaded = await loadModelFromDisk(path.join(root, "public/model/microduck"), "robot_allcollisions-nv.xml", fs, path);
  try {
    for (const { kind } of BUILTIN_TASKS) {
      const spec = buildSpec({ kind });
      const env = new VecEnv({ ...loaded, spec, count: 2, rng: () => 0.5 });
      if (spec.staggerReset === false) assert.equal(env.timeOf(0), 0);
      for (let i = 0; i < 24; i++) {
        const step = env.step(new Float32Array(2 * NUM_JOINTS));
        assert.ok(step.reward.every(Number.isFinite), kind);
        assert.ok(step.obs.every(Number.isFinite), kind);
        const expected = new Float32Array(OBS_SIZE);
        taskCommandWriter({ kind })(env.timeOf(0), expected);
        assert.deepEqual(step.obs.slice(CMD_OFFSET, OBS_SIZE), expected.slice(CMD_OFFSET), kind);
      }
      assert.equal(env.nonFiniteSteps, 0, kind);
      for (let i = 0; i < env.count; i++) env.dataAt(i).delete();
      const trainer = new Trainer({ ...loaded, spec, config: {
        ...DEFAULT_TRAINER, envs: 2, stepsPerIter: 8, hidden: [16], seed: 7,
        ppo: { ...DEFAULT_PPO, epochs: 1, minibatches: 2 },
      }, kernels: null });
      for (let i = 0; i < 2; i++) {
        const stats = await trainer.iterate();
        assert.ok(Number.isFinite(stats.policyLoss), kind);
        assert.equal(stats.nonFiniteSteps, 0, kind);
        for (const value of Object.values(stats.taskMetrics)) assert.ok(Number.isFinite(value), kind);
      }
      assert.equal(trainer.iteration, 2);
      assert.equal(trainer.checkpoint().task, kind);
      trainer.restore(JSON.parse(JSON.stringify(trainer.checkpoint())));
      console.log(`${kind}: finite rollout and 2 PPO updates; no convergence assertion`);
      for (let i = 0; i < trainer.env!.count; i++) trainer.env!.dataAt(i).delete();
    }
  } finally {
    loaded.model.delete();
  }
});
