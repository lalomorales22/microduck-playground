// Diagonal Gaussian policy distribution.
//
// The reference config uses a scalar (state-independent) standard deviation
// with init_std 1.0, so the actor network outputs the mean and the log-std is
// a free parameter vector trained alongside it.

/** log(2*pi)/2, the per-dimension constant in a Gaussian log-density. */
const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI);

/**
 * log p(action | mean, exp(logStd)) summed over dimensions.
 *
 * When `gMean` / `gLogStd` are supplied they receive d(logProb)/d(param),
 * which is what the PPO surrogate needs. They are ACCUMULATED into, so callers
 * can sum over a batch without a temporary.
 */
export function gaussianLogProb(
  action: Float32Array,
  mean: Float32Array,
  logStd: Float32Array,
  offset: number,
  gMean?: Float32Array,
  gLogStd?: Float32Array,
  gScale = 1,
): number {
  let total = 0;
  for (let i = 0; i < logStd.length; i++) {
    const ls = logStd[i];
    const std = Math.exp(ls);
    const d = action[offset + i] - mean[offset + i];
    const z = d / std;
    total += -0.5 * z * z - ls - HALF_LOG_2PI;
    // d/d(mean) of -z^2/2 is z/std
    if (gMean) gMean[i] += (gScale * z) / std;
    // d/d(logStd) of (-z^2/2 - logStd) = z^2 - 1
    if (gLogStd) gLogStd[i] += gScale * (z * z - 1);
  }
  return total;
}

/** Differential entropy of the diagonal Gaussian, summed over dimensions. */
export function gaussianEntropy(logStd: Float32Array): number {
  let total = 0;
  for (let i = 0; i < logStd.length; i++) total += logStd[i] + 0.5 * Math.log(2 * Math.PI * Math.E);
  return total;
}

/** Box-Muller, so the sampler needs only a uniform RNG. */
export function sampleNormal(rng: () => number): number {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
