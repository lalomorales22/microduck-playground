// PPO, matching the reference algorithm config in microduck_standup_env_cfg.py:
// clip 0.2, 5 epochs, 4 minibatches, gamma 0.99, lambda 0.95, entropy 0.01,
// value loss coefficient 1.0 with clipping, grad-norm clip 1.0, and the
// adaptive learning rate that targets a fixed KL.

import { gaussianLogProb } from "./distribution.ts";
import type { ActorCritic } from "./ac-policy.ts";
import type { Adam } from "./nn.ts";

export interface PpoConfig {
  gamma: number;
  lambda: number;
  clip: number;
  epochs: number;
  minibatches: number;
  entropyCoef: number;
  valueCoef: number;
  maxGradNorm: number;
  /** Target KL for the adaptive schedule; null disables it. */
  desiredKl: number | null;
  lr: number;
}

export const DEFAULT_PPO: PpoConfig = {
  gamma: 0.99,
  lambda: 0.95,
  clip: 0.2,
  epochs: 5,
  minibatches: 4,
  entropyCoef: 0.01,
  valueCoef: 1.0,
  maxGradNorm: 1.0,
  desiredKl: 0.01,
  lr: 1e-3,
};

/**
 * A rollout stored as [steps][envs], flattened row-major.
 *
 * `dones` marks episode ends; `timeouts` marks the subset of those that ended
 * because the clock ran out rather than because the episode failed. The
 * distinction matters: a timeout is not a real terminal state, so its value
 * must still be bootstrapped or the policy learns that surviving to the time
 * limit is bad.
 */
export interface RolloutBuffer {
  steps: number;
  envs: number;
  obsDim: number;
  actDim: number;
  obs: Float32Array;
  actions: Float32Array;
  logProbs: Float32Array;
  values: Float32Array;
  rewards: Float32Array;
  dones: Uint8Array;
  timeouts: Uint8Array;
  /** Value of the state after the final step, for the last GAE bootstrap. */
  lastValues: Float32Array;
  /** Filled by computeGae. */
  advantages: Float32Array;
  returns: Float32Array;
}

export function makeBuffer(steps: number, envs: number, obsDim: number, actDim: number): RolloutBuffer {
  const n = steps * envs;
  return {
    steps, envs, obsDim, actDim,
    obs: new Float32Array(n * obsDim),
    actions: new Float32Array(n * actDim),
    logProbs: new Float32Array(n),
    values: new Float32Array(n),
    rewards: new Float32Array(n),
    dones: new Uint8Array(n),
    timeouts: new Uint8Array(n),
    lastValues: new Float32Array(envs),
    advantages: new Float32Array(n),
    returns: new Float32Array(n),
  };
}

/** Generalized advantage estimation, walked backwards through the rollout. */
export function computeGae(buf: RolloutBuffer, gamma: number, lambda: number): void {
  const { steps, envs } = buf;
  for (let e = 0; e < envs; e++) {
    let lastGae = 0;
    for (let t = steps - 1; t >= 0; t--) {
      const i = t * envs + e;
      const nextValue = t === steps - 1 ? buf.lastValues[e] : buf.values[(t + 1) * envs + e];
      const done = buf.dones[i] === 1;
      // A timeout is a bootstrap point, not a terminal state — the episode was
      // cut short, so the value of what came next is still real.
      const bootstrap = !done || buf.timeouts[i] === 1;
      const nextNonTerminal = bootstrap ? 1 : 0;
      const delta = buf.rewards[i] + gamma * nextValue * nextNonTerminal - buf.values[i];
      // But a done of ANY kind breaks the GAE chain: the next step belongs to
      // a different episode.
      lastGae = delta + gamma * lambda * (done ? 0 : 1) * lastGae;
      buf.advantages[i] = lastGae;
      buf.returns[i] = lastGae + buf.values[i];
    }
  }
}

export interface UpdateStats {
  /**
   * KL of the very first minibatch, before any parameter has moved.
   *
   * PPO's importance ratio is exactly 1 there by construction, so this must be
   * ~0. Anything else means the policy being updated is not the policy that
   * collected the data — an observation normalizer shifting between rollout
   * and update is the classic cause, and it silently pins the adaptive
   * learning rate to its floor. Cheap invariant, expensive bug.
   */
  initialKl: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKl: number;
  clipFraction: number;
  gradNorm: number;
  lr: number;
  /** Where the update's time went. Measured rather than modelled: synthetic
   *  benchmarks mis-price the backward pass, which skips rows whose incoming
   *  gradient is zero — and the clip fraction decides how many those are. */
  fwdMs: number;
  bwdMs: number;
  /** Gradient clipping and the Adam step. Separate from bwdMs because both
   *  sweep every parameter once per MINIBATCH, which is 20 full passes over
   *  the net per iteration and not obviously small. */
  optMs: number;
  otherMs: number;
}

/** Fisher-Yates over a reusable index array. */
function shuffle(idx: Int32Array, rng: () => number): void {
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
}

/**
 * One PPO update over the whole buffer.
 *
 * Advantages are normalized across the full batch (not per minibatch), which
 * is what rsl_rl does.
 */
export function ppoUpdate(
  ac: ActorCritic,
  opt: Adam,
  buf: RolloutBuffer,
  cfg: PpoConfig,
  rng: () => number,
): UpdateStats {
  computeGae(buf, cfg.gamma, cfg.lambda);

  const n = buf.steps * buf.envs;
  let advMean = 0;
  for (let i = 0; i < n; i++) advMean += buf.advantages[i];
  advMean /= n;
  let advVar = 0;
  for (let i = 0; i < n; i++) advVar += (buf.advantages[i] - advMean) ** 2;
  const advStd = Math.sqrt(advVar / n) + 1e-8;

  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  const mbSize = Math.floor(n / cfg.minibatches);

  const stats: UpdateStats = {
    initialKl: 0, policyLoss: 0, valueLoss: 0, entropy: 0,
    approxKl: 0, clipFraction: 0, gradNorm: 0, lr: opt.lr,
    fwdMs: 0, bwdMs: 0, optMs: 0, otherMs: 0,
  };
  const updateStart = performance.now();
  let updates = 0;

  // Scratch, sized for one minibatch.
  const mbObs = new Float32Array(mbSize * buf.obsDim);
  const mbAct = new Float32Array(mbSize * buf.actDim);
  const gMean = new Float32Array(mbSize * buf.actDim);
  const gValue = new Float32Array(mbSize);
  const gLogStdOne = new Float32Array(buf.actDim);
  // Per-sample scratch, hoisted. Allocating these inside the sample loop cost
  // more than the matrix multiplies they feed: 192 samples x 20 minibatches is
  // ~4k allocations an iteration, and the profiler attributed over half the
  // update to it.
  const gMeanOne = new Float32Array(buf.actDim);

  for (let epoch = 0; epoch < cfg.epochs; epoch++) {
    shuffle(idx, rng);
    for (let mb = 0; mb < cfg.minibatches; mb++) {
      const start = mb * mbSize;
      // Explicit index copies rather than set(subarray(...)): each subarray is
      // a fresh TypedArray object, and two per sample adds up to thousands of
      // short-lived allocations per iteration.
      for (let k = 0; k < mbSize; k++) {
        const s = idx[start + k];
        const src = s * buf.obsDim, dst = k * buf.obsDim;
        for (let i = 0; i < buf.obsDim; i++) mbObs[dst + i] = buf.obs[src + i];
        const asrc = s * buf.actDim, adst = k * buf.actDim;
        for (let i = 0; i < buf.actDim; i++) mbAct[adst + i] = buf.actions[asrc + i];
      }

      const tFwd = performance.now();
      const normObs = ac.normalize(mbObs, mbSize);
      const mean = ac.actor.forward(normObs, mbSize);
      const value = ac.critic.forward(normObs, mbSize);
      stats.fwdMs += performance.now() - tFwd;

      ac.zeroGrad();
      gMean.fill(0);
      gValue.fill(0);

      let pLoss = 0, vLoss = 0, kl = 0, clipped = 0;
      for (let k = 0; k < mbSize; k++) {
        const s = idx[start + k];
        const off = k * buf.actDim;
        gLogStdOne.fill(0);
        gMeanOne.fill(0);
        const logp = gaussianLogProb(mbAct, mean, ac.logStd, off, gMeanOne, gLogStdOne);

        const adv = (buf.advantages[s] - advMean) / advStd;
        const ratio = Math.exp(logp - buf.logProbs[s]);
        const clipLow = 1 - cfg.clip, clipHigh = 1 + cfg.clip;
        const unclipped = ratio * adv;
        const clippedTerm = Math.min(Math.max(ratio, clipLow), clipHigh) * adv;
        const useUnclipped = unclipped <= clippedTerm;
        if (!useUnclipped) clipped++;
        pLoss += -Math.min(unclipped, clippedTerm);

        // d(-min(...))/d(logp): zero inside the clipped branch, which is what
        // stops a too-large step from being pushed further.
        const dPolicy = useUnclipped ? (-adv * ratio) / mbSize : 0;
        for (let i = 0; i < buf.actDim; i++) {
          gMean[off + i] += dPolicy * gMeanOne[i];
          ac.dLogStd[i] += dPolicy * gLogStdOne[i];
          // Entropy bonus: d(entropy)/d(logStd) is 1 per dimension.
          ac.dLogStd[i] += (-cfg.entropyCoef / mbSize) * 1;
        }

        const vClipped = buf.values[s] +
          Math.min(Math.max(value[k] - buf.values[s], -cfg.clip), cfg.clip);
        const vLossUnclipped = (value[k] - buf.returns[s]) ** 2;
        const vLossClipped = (vClipped - buf.returns[s]) ** 2;
        const useUnclippedV = vLossUnclipped >= vLossClipped;
        vLoss += Math.max(vLossUnclipped, vLossClipped);
        gValue[k] = useUnclippedV
          ? (cfg.valueCoef * 2 * (value[k] - buf.returns[s])) / mbSize
          : (cfg.valueCoef * 2 * (vClipped - buf.returns[s])) / mbSize;

        const d = buf.logProbs[s] - logp;
        kl += Math.exp(d) - 1 - d; // Schulman's low-variance KL estimator
      }

      if (epoch === 0 && mb === 0) stats.initialKl = kl / mbSize;

      const tBwd = performance.now();
      ac.actor.backward(gMean, mbSize);
      ac.critic.backward(gValue, mbSize);
      const tOpt = performance.now();
      stats.bwdMs += tOpt - tBwd;
      stats.gradNorm = ac.clipGrads(cfg.maxGradNorm);
      opt.step(ac.tensors());
      stats.optMs += performance.now() - tOpt;

      stats.policyLoss += pLoss / mbSize;
      stats.valueLoss += vLoss / mbSize;
      stats.approxKl += kl / mbSize;
      stats.clipFraction += clipped / mbSize;
      updates++;

      // Adaptive LR: rsl_rl adjusts between minibatches, not per epoch, and
      // this is what keeps a run from diverging when the policy moves too far.
      if (cfg.desiredKl !== null) {
        const meanKl = kl / mbSize;
        if (meanKl > cfg.desiredKl * 2) opt.lr = Math.max(1e-5, opt.lr / 1.5);
        else if (meanKl < cfg.desiredKl / 2 && meanKl > 0) opt.lr = Math.min(1e-2, opt.lr * 1.5);
      }
    }
  }

  stats.policyLoss /= updates;
  stats.valueLoss /= updates;
  stats.approxKl /= updates;
  stats.clipFraction /= updates;
  stats.entropy = ac.entropy;
  stats.lr = opt.lr;
  stats.otherMs = performance.now() - updateStart - stats.fwdMs - stats.bwdMs - stats.optMs;
  return stats;
}
