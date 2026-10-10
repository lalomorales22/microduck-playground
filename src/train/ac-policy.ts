// The actor-critic: a Gaussian policy and a value function, both plain MLPs,
// behind a running observation normalizer.
//
// Shapes and conventions follow the reference config (hidden 512/256/128, ELU,
// scalar log-std initialised at 1.0, observation normalization ON). The
// normalizer matters twice over: training is unstable without it, and AGENTS.md
// is explicit that it must be BAKED INTO the ONNX at export — an exported
// policy that expects raw observations is a silent deployment failure, because
// in-sim replay applies the normalizer anyway and hides the bug.

import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import { gaussianEntropy, gaussianLogProb, sampleNormal } from "./distribution.ts";
import { Adam, MlpNet, clipGradNorm } from "./nn.ts";
import type { Kernels } from "./kernels/index.ts";

export const HIDDEN = [512, 256, 128] as const;

/**
 * Welford running mean/variance over observations.
 *
 * Frozen at export time so the deployed policy sees the same scaling it
 * trained with.
 */
export class ObsNormalizer {
  readonly mean: Float32Array;
  readonly var_: Float32Array;
  count: number;

  readonly dim: number;

  constructor(dim: number) {
    this.dim = dim;
    this.mean = new Float32Array(dim);
    this.var_ = new Float32Array(dim).fill(1);
    this.count = 1e-4;
  }

  /** `obs` is [batch, dim]. */
  update(obs: Float32Array, batch: number): void {
    if (batch === 0) return;
    const batchMean = new Float32Array(this.dim);
    const batchVar = new Float32Array(this.dim);
    for (let n = 0; n < batch; n++) {
      const o = n * this.dim;
      for (let i = 0; i < this.dim; i++) batchMean[i] += obs[o + i];
    }
    for (let i = 0; i < this.dim; i++) batchMean[i] /= batch;
    for (let n = 0; n < batch; n++) {
      const o = n * this.dim;
      for (let i = 0; i < this.dim; i++) {
        const d = obs[o + i] - batchMean[i];
        batchVar[i] += d * d;
      }
    }
    for (let i = 0; i < this.dim; i++) batchVar[i] /= batch;

    const total = this.count + batch;
    for (let i = 0; i < this.dim; i++) {
      const delta = batchMean[i] - this.mean[i];
      const m = this.var_[i] * this.count + batchVar[i] * batch +
        (delta * delta * this.count * batch) / total;
      this.mean[i] += (delta * batch) / total;
      this.var_[i] = m / total;
    }
    this.count = total;
  }

  /** Writes the normalized copy into `out`; never mutates the input. */
  apply(obs: Float32Array, batch: number, out: Float32Array): void {
    for (let n = 0; n < batch; n++) {
      const o = n * this.dim;
      for (let i = 0; i < this.dim; i++) {
        out[o + i] = (obs[o + i] - this.mean[i]) / Math.sqrt(this.var_[i] + 1e-8);
      }
    }
  }
}

export interface ActResult {
  /** Sampled actions, [batch, actDim]. */
  actions: Float32Array;
  /** log p(action) per sample, [batch]. */
  logProbs: Float32Array;
  /** V(s) per sample, [batch]. */
  values: Float32Array;
}

export class ActorCritic {
  readonly actor: MlpNet;
  readonly critic: MlpNet;
  /** State-independent log standard deviation, one per action dimension. */
  readonly logStd: Float32Array;
  readonly dLogStd: Float32Array;
  readonly normalizer: ObsNormalizer;
  readonly obsDim: number;
  readonly actDim: number;

  #normObs = new Float32Array(0);
  #actions = new Float32Array(0);
  #logProbs = new Float32Array(0);
  #values = new Float32Array(0);

  readonly hidden: readonly number[];

  constructor(
    // Widened explicitly: NUM_JOINTS is a literal type (JOINT_NAMES is
    // `as const`), which would make actDim `14` rather than `number`.
    obsDim: number = OBS_SIZE,
    actDim: number = NUM_JOINTS,
    rng: () => number = Math.random,
    initStd = 1.0,
    /** Defaults to the reference architecture; smaller nets are for tests and
     *  for trading capacity against wall clock on a slow machine. */
    hidden: readonly number[] = HIDDEN,
    /** When present both nets run the SIMD kernels; null is the JS path. */
    kernels: Kernels | null = null,
  ) {
    this.obsDim = obsDim;
    this.actDim = actDim;
    this.hidden = hidden;
    this.actor = new MlpNet(obsDim, actDim, hidden, rng, 0.01, kernels);
    this.critic = new MlpNet(obsDim, 1, hidden, rng, 1.0, kernels);
    this.logStd = new Float32Array(actDim).fill(Math.log(initStd));
    this.dLogStd = new Float32Array(actDim);
    this.normalizer = new ObsNormalizer(obsDim);
  }

  #ensure(batch: number): void {
    if (this.#normObs.length >= batch * this.obsDim) return;
    this.#normObs = new Float32Array(batch * this.obsDim);
    this.#actions = new Float32Array(batch * this.actDim);
    this.#logProbs = new Float32Array(batch);
    this.#values = new Float32Array(batch);
  }

  /** Normalize into scratch and return the view the nets should see. */
  normalize(obs: Float32Array, batch: number): Float32Array {
    this.#ensure(batch);
    this.normalizer.apply(obs, batch, this.#normObs);
    return this.#normObs.subarray(0, batch * this.obsDim);
  }

  /** Sample actions for a rollout step. No gradients are taken here. */
  act(obs: Float32Array, batch: number, rng: () => number): ActResult {
    const normObs = this.normalize(obs, batch);
    const mean = this.actor.forward(normObs, batch);
    const value = this.critic.forward(normObs, batch);
    const a = this.#actions, lp = this.#logProbs, v = this.#values;
    for (let n = 0; n < batch; n++) {
      const off = n * this.actDim;
      for (let i = 0; i < this.actDim; i++) {
        a[off + i] = mean[off + i] + Math.exp(this.logStd[i]) * sampleNormal(rng);
      }
      lp[n] = gaussianLogProb(a, mean, this.logStd, off);
      v[n] = value[n];
    }
    return {
      actions: a.subarray(0, batch * this.actDim),
      logProbs: lp.subarray(0, batch),
      values: v.subarray(0, batch),
    };
  }

  /** Deterministic action (the distribution mean) — for evaluation and export. */
  actMean(obs: Float32Array, batch: number): Float32Array {
    return this.actor.forward(this.normalize(obs, batch), batch);
  }

  value(obs: Float32Array, batch: number): Float32Array {
    return this.critic.forward(this.normalize(obs, batch), batch);
  }

  get entropy(): number {
    return gaussianEntropy(this.logStd);
  }

  tensors(): { param: Float32Array; grad: Float32Array }[] {
    return [
      ...this.actor.tensors(),
      ...this.critic.tensors(),
      { param: this.logStd, grad: this.dLogStd },
    ];
  }

  zeroGrad(): void {
    this.actor.zeroGrad();
    this.critic.zeroGrad();
    this.dLogStd.fill(0);
  }

  makeOptimizer(lr: number): Adam {
    return new Adam(this.tensors(), lr);
  }

  clipGrads(max: number): number {
    return clipGradNorm(this.tensors(), max);
  }

  serialize(): unknown {
    return {
      obsDim: this.obsDim,
      actDim: this.actDim,
      hidden: [...this.hidden],
      actor: this.actor.layers.map((l) => ({ w: Array.from(l.w), b: Array.from(l.b) })),
      critic: this.critic.layers.map((l) => ({ w: Array.from(l.w), b: Array.from(l.b) })),
      logStd: Array.from(this.logStd),
      normalizer: {
        mean: Array.from(this.normalizer.mean),
        var: Array.from(this.normalizer.var_),
        count: this.normalizer.count,
      },
    };
  }

  load(state: ReturnType<ActorCritic["serialize"]>): void {
    const s = state as {
      actor: { w: number[]; b: number[] }[];
      critic: { w: number[]; b: number[] }[];
      logStd: number[];
      normalizer: { mean: number[]; var: number[]; count: number };
    };
    s.actor.forEach((l, i) => {
      this.actor.layers[i].w.set(l.w);
      this.actor.layers[i].b.set(l.b);
    });
    s.critic.forEach((l, i) => {
      this.critic.layers[i].w.set(l.w);
      this.critic.layers[i].b.set(l.b);
    });
    this.logStd.set(s.logStd);
    this.normalizer.mean.set(s.normalizer.mean);
    this.normalizer.var_.set(s.normalizer.var);
    this.normalizer.count = s.normalizer.count;
  }

  /** Flat parameter copy, for shipping weights to rollout workers. */
  flatParams(): Float32Array {
    const tensors = this.tensors();
    const total = tensors.reduce((n, t) => n + t.param.length, 0);
    const out = new Float32Array(total);
    let o = 0;
    for (const { param } of tensors) {
      out.set(param, o);
      o += param.length;
    }
    return out;
  }

  loadFlatParams(flat: Float32Array): void {
    let o = 0;
    for (const { param } of this.tensors()) {
      param.set(flat.subarray(o, o + param.length));
      o += param.length;
    }
  }
}
