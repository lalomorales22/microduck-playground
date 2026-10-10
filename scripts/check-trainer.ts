// M2 gate: can PPO learn anything on the real robot?
//
//   node scripts/check-trainer.ts [iterations]
//
// The task is "hold the pose": start at the standing keyframe and stay there.
// Deliberately the easiest thing on the robot — a learner that cannot hold a
// stable equilibrium it was handed will certainly not discover how to reach
// one, and finding that out here costs a minute instead of an hour.
//
// Also round-trips a checkpoint, because a resume that silently differs from
// an uninterrupted run is a bug you only notice as "training got worse
// overnight".

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadModelFromDisk } from "../src/train/model-loader.ts";
import { holdPoseSpec } from "../src/train/env/standup.ts";
import { DEFAULT_TRAINER, Trainer, type TrainerConfig } from "../src/train/trainer.ts";
import { DEFAULT_PPO } from "../src/train/ppo.ts";
import { loadKernelsFromDisk } from "../src/train/kernels/load.node.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = path.join(ROOT, "public/model/microduck");
const ROBOT_XML = "robot_allcollisions-nv.xml";
// 400 iterations, ~60s. It was 150, which was not a gate but a coin flip:
// measured across seven seeds, 150 iterations put the final standing fraction
// anywhere between 19% and 100%, so an unrelated change that merely reshuffled
// which environment got which starting phase could move the result from a
// comfortable pass to a clear fail. The cause is that 150 iterations leaves
// the run unconverged and bimodal, so the fix is more iterations rather than a
// lower bar — at 400 the three worst seeds land at 43%, 61% and 100%.
const ITERS = Number(process.argv[2] ?? 400);
// Second argument is the seed. The gate runs on 7; the others exist because
// "did that change break it, or did it just reshuffle the dice?" is otherwise
// unanswerable on a task this noisy.
const SEED = Number(process.argv[3] ?? 7);

const { mujoco, model, standKey } = await loadModelFromDisk(MODEL_DIR, ROBOT_XML, fs, path);

const config: TrainerConfig = {
  ...DEFAULT_TRAINER,
  envs: 32,
  stepsPerIter: 24,
  seed: SEED,
  // Smaller than the reference 512/256/128: scalar-JS backprop is the cost
  // here, and this gate is about whether the learner works, not final quality.
  hidden: [128, 64],
  // Exploration comes from holdPoseSpec's ExplorationHints — this task is
  // destroyed by action noise, and saying so lives with the task, not here.
  ppo: { ...DEFAULT_PPO, minibatches: 4, epochs: 4 },
};

const kernels = await loadKernelsFromDisk();
const trainer = new Trainer({ mujoco, model, standKey, spec: holdPoseSpec(), config, kernels });
console.log(
  `hold-pose: ${config.envs} envs x ${config.stepsPerIter} steps, ` +
  `net ${config.hidden.join("/")}, ${trainer.ac.actor.paramCount + trainer.ac.critic.paramCount} params, ` +
  `${trainer.usesSimd ? "SIMD kernels" : "JavaScript"}, ` +
  `sigma ${trainer.exploration.initStd}, entropy ${trainer.exploration.entropyCoef}`,
);

const EVAL_STEPS = 150; // one full 3 s episode
const before = trainer.evaluate(EVAL_STEPS);
console.log(
  `before training: reward/step ${before.rewardPerStep.toFixed(3)}, ` +
  `standing ${(before.standingFraction * 100).toFixed(1)}%`,
);

let firstKl = 0;
let lastStats: ReturnType<Trainer["iterate"]> | null = null;
const t0 = Date.now();
for (let i = 0; i < ITERS; i++) {
  lastStats = await trainer.iterate();
  if (i === 0) firstKl = lastStats.initialKl;
  if (i % 10 === 0 || i === ITERS - 1) {
    const s = lastStats;
    console.log(
      `iter ${String(s.iteration).padStart(3)}  reward/step ${s.rewardPerStep.toFixed(3)}  ` +
      `standing ${(s.standingFraction * 100).toFixed(1).padStart(5)}%  ` +
      `entropy ${s.entropy.toFixed(2)}  kl ${s.approxKl.toFixed(4)}  ` +
      `lr ${s.lr.toExponential(1)}  ${(s.rolloutMs + s.updateMs).toFixed(0)}ms`,
    );
  }
}
const elapsed = (Date.now() - t0) / 1000;
const after = trainer.evaluate(EVAL_STEPS);
console.log(
  `\nafter ${ITERS} iterations (${elapsed.toFixed(1)}s, ${trainer.totalSteps.toLocaleString()} steps): ` +
  `reward/step ${before.rewardPerStep.toFixed(3)} -> ${after.rewardPerStep.toFixed(3)}, ` +
  `standing ${(before.standingFraction * 100).toFixed(1)}% -> ${(after.standingFraction * 100).toFixed(1)}%`,
);
console.log(`first-minibatch KL (must be ~0): ${firstKl.toExponential(2)}`);

// ── Checkpoint round trip ─────────────────────────────────────────────────
// A restored trainer must produce the SAME next iteration as the original
// would have. Anything less means resume silently changes the run.
const ckpt = JSON.parse(JSON.stringify(trainer.checkpoint()));
const evalAfterSave = trainer.evaluate(20);

const restored = new Trainer({ mujoco, model, standKey, spec: holdPoseSpec(), config, kernels });
restored.restore(ckpt);
const evalRestored = restored.evaluate(20);

const rewardDrift = Math.abs(evalAfterSave.rewardPerStep - evalRestored.rewardPerStep);
console.log(
  `checkpoint round trip: iteration ${ckpt.iteration}, ` +
  `eval reward ${evalAfterSave.rewardPerStep.toFixed(4)} vs ${evalRestored.rewardPerStep.toFixed(4)} ` +
  `(drift ${rewardDrift.toExponential(2)})`,
);

const failures: string[] = [];
if (!(after.rewardPerStep > before.rewardPerStep * 1.15)) {
  failures.push(
    `reward did not improve 15% (${before.rewardPerStep.toFixed(3)} -> ${after.rewardPerStep.toFixed(3)})`,
  );
}
// Not "solves balancing" — 300k steps is ~1/5000th of what the reference recipe
// spends on stand-up, so the bar is that the learner demonstrably moves the
// policy, not that it masters the task.
if (!(after.standingFraction > 0.35)) {
  failures.push(`policy holds the pose only ${(after.standingFraction * 100).toFixed(1)}% of steps`);
}
if (!Number.isFinite(lastStats?.policyLoss ?? NaN)) failures.push("policy loss went non-finite");
// The very first minibatch of the very first epoch evaluates the exact policy
// that collected the data, so its KL must be ~0. See UpdateStats.initialKl.
if (!(firstKl < 1e-3)) {
  failures.push(`first-minibatch KL ${firstKl.toExponential(2)} is not ~0; rollout and update disagree`);
}
// The evaluation is deterministic, so a correct restore must reproduce it
// exactly bar float ordering.
if (!(rewardDrift < 1e-4)) failures.push(`restored policy differs from the saved one (${rewardDrift})`);
if (restored.iteration !== trainer.iteration) failures.push("restored iteration counter mismatch");

console.log();
if (failures.length) {
  console.log("FAILED:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
console.log("PASSED: PPO learns to hold the pose, and a checkpoint restores exactly.");
