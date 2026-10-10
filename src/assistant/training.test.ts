import assert from "node:assert/strict";
import { test } from "node:test";
import { assistantTrainerConfig } from "./training.ts";
import type { TrainingDraft } from "./contract.ts";
import { validateTrainingConfig } from "../train/checkpoint-task.ts";
import { DEFAULT_PPO } from "../train/ppo.ts";
import { DEFAULT_TRAINER } from "../train/trainer.ts";

function draft(overrides: Partial<TrainingDraft> = {}): TrainingDraft {
  return { task: "hold_pose", iterations: 50, seed: 42, envs: 16,
    rolloutWorkers: 1, useCurrentWorld: false, hiddenPreset: "small", ...overrides };
}

for (const preset of ["small", "reference"] as const) {
  test(`${preset} supports all 128 environment counts and both uint32 seed bounds`, () => {
    const expectedHidden = preset === "small" ? [128, 64] : [512, 256, 128];
    for (let envs = 1; envs <= 128; envs++) {
      for (const seed of [0, 0xffffffff]) {
        const input = draft({ hiddenPreset: preset, envs, seed });
        const config = assistantTrainerConfig(input);
        const label = `${preset}, envs=${envs}, seed=${seed}`;
        assert.equal(config.envs, envs, label);
        assert.equal(config.seed, seed, label);
        assert.equal(config.stepsPerIter, 24, label);
        assert.equal(config.ppo.epochs, 4, label);
        assert.equal(config.ppo.minibatches, DEFAULT_PPO.minibatches, label);
        assert.equal(config.envs * config.stepsPerIter % config.ppo.minibatches, 0, label);
        assert.deepEqual(config.hidden, expectedHidden, label);
        assert.deepEqual(config.ppo, { ...DEFAULT_PPO, epochs: 4 }, label);
        assert.deepEqual(validateTrainingConfig(config), config, label);
      }
    }
  });
}

test("invalid seeds are rejected instead of being rounded, coerced or wrapped", () => {
  for (const seed of [-1, 0x100000000, 0.5, NaN, Infinity, -Infinity, "42", null, undefined]) {
    const input = { ...draft(), seed } as unknown as TrainingDraft;
    assert.throws(() => assistantTrainerConfig(input), /seed/i, `seed=${String(seed)}`);
  }
});

test("invalid environment counts are rejected instead of being rounded or coerced", () => {
  for (const envs of [-1, 0, 129, 1.5, NaN, Infinity, -Infinity, "16", null, undefined]) {
    const input = { ...draft(), envs } as unknown as TrainingDraft;
    assert.throws(() => assistantTrainerConfig(input), /env/i, `envs=${String(envs)}`);
  }
});

test("unknown runtime network presets fail instead of silently selecting a network", () => {
  for (const hiddenPreset of ["large", "SMALL", "", null, undefined, 0, [], {}]) {
    const input = { ...draft(), hiddenPreset } as unknown as TrainingDraft;
    assert.throws(() => assistantTrainerConfig(input), /hidden|preset|network/i, `preset=${String(hiddenPreset)}`);
  }
});

test("results and nested configs are detached from defaults, input and subsequent calls", () => {
  const defaultTrainer = structuredClone(DEFAULT_TRAINER);
  const defaultPpo = structuredClone(DEFAULT_PPO);
  const input = Object.freeze(draft({ hiddenPreset: "reference" }));
  const beforeInput = structuredClone(input);
  const first = assistantTrainerConfig(input);
  const second = assistantTrainerConfig(input);
  assert.notEqual(first, second);
  assert.notEqual(first.ppo, second.ppo);
  assert.notEqual(first.hidden, second.hidden);
  assert.notEqual(first.ppo, DEFAULT_PPO);
  assert.notEqual(first.ppo, DEFAULT_TRAINER.ppo);
  assert.notEqual(first.hidden, DEFAULT_TRAINER.hidden);
  assert.deepEqual(first, second);

  first.ppo.lr = 0.25;
  first.ppo.epochs = 20;
  (first.hidden as number[])[0] = 4;
  first.envs = 1;
  first.seed = 0;
  first.stepsPerIter = 256;
  first.exploration = { initStd: 10 };
  const third = assistantTrainerConfig(input);
  assert.deepEqual(second, third);
  assert.deepEqual(DEFAULT_TRAINER, defaultTrainer);
  assert.deepEqual(DEFAULT_PPO, defaultPpo);
  assert.deepEqual(input, beforeInput);
  assert.equal(third.stepsPerIter, 24);
  assert.equal(third.ppo.epochs, 4);
  assert.deepEqual(third.hidden, [512, 256, 128]);
  assert.equal(Object.hasOwn(third, "exploration"), false);
});

test("checkpoint-shaped extra fields cannot leak into fresh assistant configurations", () => {
  const input = {
    ...draft({ seed: 17, envs: 3 }), stepsPerIter: 256, hidden: [1024, 1024],
    ppo: { ...DEFAULT_PPO, epochs: 20, minibatches: 32, lr: 0.5 },
    exploration: { initStd: 9, entropyCoef: 1, desiredKl: 1 },
    resumeFrom: "existing-checkpoint", policy: { actor: "old weights" },
    optimizer: { t: 999 }, iteration: 1234,
  };
  const before = structuredClone(input);
  const config = assistantTrainerConfig(input);
  assert.deepEqual(Object.keys(config).sort(), ["envs", "hidden", "ppo", "seed", "stepsPerIter"]);
  assert.deepEqual(config, {
    ...DEFAULT_TRAINER, envs: 3, seed: 17, stepsPerIter: 24,
    hidden: [128, 64], ppo: { ...DEFAULT_PPO, epochs: 4 },
  });
  assert.equal(Object.hasOwn(config, "exploration"), false);
  assert.deepEqual(input, before);
});
