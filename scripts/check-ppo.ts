// Does the PPO implementation actually learn anything?
//
//   node scripts/check-ppo.ts
//
// A 2-D point mass that must reach a target it can see in its observation.
// Trivial, dense, and solvable in seconds — which is the point: if PPO cannot
// do this, it will not do stand-up in an hour, and debugging it here costs
// seconds instead of a wasted run. This is the gate that stands between the
// algorithm and the robot.

import { ActorCritic } from "../src/train/ac-policy.ts";
import { DEFAULT_PPO, makeBuffer, ppoUpdate, type PpoConfig } from "../src/train/ppo.ts";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Reach-a-target point mass.
 *
 * obs = [px, py, tx, ty] ; action = velocity, clipped to +-1 ; reward = the
 * distance closed this step, so the optimum is "go straight at it".
 */
const OBS = 4, ACT = 2, ENVS = 64, STEPS = 32, EPISODE = 32;

class ToyEnv {
  readonly pos = new Float32Array(ENVS * 2);
  readonly target = new Float32Array(ENVS * 2);
  readonly obs = new Float32Array(ENVS * OBS);
  readonly reward = new Float32Array(ENVS);
  readonly done = new Uint8Array(ENVS);
  readonly timeout = new Uint8Array(ENVS);
  readonly age = new Int32Array(ENVS);

  readonly rng: () => number;

  constructor(rng: () => number) {
    this.rng = rng;
    for (let e = 0; e < ENVS; e++) this.resetEnv(e);
  }

  resetEnv(e: number): void {
    for (let i = 0; i < 2; i++) {
      this.pos[e * 2 + i] = (this.rng() * 2 - 1) * 2;
      this.target[e * 2 + i] = (this.rng() * 2 - 1) * 2;
    }
    this.age[e] = 0;
    this.writeObs(e);
  }

  writeObs(e: number): void {
    const o = e * OBS;
    this.obs[o] = this.pos[e * 2];
    this.obs[o + 1] = this.pos[e * 2 + 1];
    this.obs[o + 2] = this.target[e * 2];
    this.obs[o + 3] = this.target[e * 2 + 1];
  }

  dist(e: number): number {
    const dx = this.pos[e * 2] - this.target[e * 2];
    const dy = this.pos[e * 2 + 1] - this.target[e * 2 + 1];
    return Math.hypot(dx, dy);
  }

  step(actions: Float32Array): void {
    for (let e = 0; e < ENVS; e++) {
      const before = this.dist(e);
      for (let i = 0; i < 2; i++) {
        const a = Math.min(1, Math.max(-1, actions[e * ACT + i]));
        this.pos[e * 2 + i] += a * 0.15;
      }
      this.reward[e] = before - this.dist(e);
      this.age[e]++;
      const timedOut = this.age[e] >= EPISODE;
      this.done[e] = timedOut ? 1 : 0;
      this.timeout[e] = timedOut ? 1 : 0;
      if (timedOut) this.resetEnv(e);
      else this.writeObs(e);
    }
  }
}

const rng = mulberry32(42);
// A 4-dimensional problem does not need the robot's 512/256/128 net, and at
// scalar-JS speed that net would make this gate take minutes instead of
// seconds. The algorithm under test is the same either way.
const ac = new ActorCritic(OBS, ACT, rng, 1.0, [64, 64]);
const cfg: PpoConfig = { ...DEFAULT_PPO, minibatches: 4, epochs: 5 };
const opt = ac.makeOptimizer(cfg.lr);
const env = new ToyEnv(rng);
const buf = makeBuffer(STEPS, ENVS, OBS, ACT);

/**
 * Mean final distance under the deterministic policy — the real metric.
 *
 * Stops one step short of the episode limit ON PURPOSE: the env resets an
 * environment the moment it times out, so running the full length would
 * measure a freshly randomised position and report the same number forever.
 */
function evaluate(): number {
  const evalEnv = new ToyEnv(mulberry32(999));
  for (let t = 0; t < EPISODE - 1; t++) {
    const mean = ac.actMean(evalEnv.obs, ENVS);
    evalEnv.step(mean);
  }
  let sum = 0;
  for (let e = 0; e < ENVS; e++) sum += evalEnv.dist(e);
  return sum / ENVS;
}

const before = evaluate();
console.log(`before training: mean distance to target ${before.toFixed(3)}`);

const ITERS = 60;
let last = { policyLoss: 0, valueLoss: 0, entropy: 0, approxKl: 0, clipFraction: 0, gradNorm: 0, lr: 0 };
for (let iter = 0; iter < ITERS; iter++) {
  for (let t = 0; t < STEPS; t++) {
    const { actions, logProbs, values } = ac.act(env.obs, ENVS, rng);
    buf.obs.set(env.obs, t * ENVS * OBS);
    buf.actions.set(actions, t * ENVS * ACT);
    buf.logProbs.set(logProbs, t * ENVS);
    buf.values.set(values, t * ENVS);
    env.step(actions);
    buf.rewards.set(env.reward, t * ENVS);
    buf.dones.set(env.done, t * ENVS);
    buf.timeouts.set(env.timeout, t * ENVS);
  }
  buf.lastValues.set(ac.value(env.obs, ENVS).subarray(0, ENVS));
  ac.normalizer.update(buf.obs, STEPS * ENVS);
  last = ppoUpdate(ac, opt, buf, cfg, rng);
  if (iter % 15 === 0 || iter === ITERS - 1) {
    console.log(
      `iter ${String(iter).padStart(3)}  dist ${evaluate().toFixed(3)}  ` +
      `pLoss ${last.policyLoss.toFixed(4)}  vLoss ${last.valueLoss.toFixed(4)}  ` +
      `entropy ${last.entropy.toFixed(3)}  kl ${last.approxKl.toFixed(5)}  lr ${last.lr.toExponential(1)}`,
    );
  }
}

const after = evaluate();
console.log(`\nafter ${ITERS} iterations: mean distance ${before.toFixed(3)} -> ${after.toFixed(3)}`);

const failures: string[] = [];
// The optimum is ~0: 32 steps at 0.15 covers 4.8 units, more than any start.
if (!(after < 0.35)) failures.push(`policy did not reach the target (mean distance ${after.toFixed(3)})`);
if (!(after < before * 0.25)) failures.push(`distance did not improve 4x (${before.toFixed(3)} -> ${after.toFixed(3)})`);
if (!Number.isFinite(last.policyLoss)) failures.push("policy loss is not finite");
// The entropy bonus must not run away: a log-std that explodes means the
// entropy gradient sign is wrong.
if (!(last.entropy < 6)) failures.push(`entropy exploded to ${last.entropy.toFixed(2)}`);

if (failures.length) {
  console.log("\nFAILED:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
console.log("\nPASSED: PPO solves the toy reach task.");
