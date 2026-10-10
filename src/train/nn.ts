// A small MLP with hand-written backprop, plus Adam.
//
// Hand-written because the architecture is fixed and tiny (512/256/128,
// ~200k parameters): a framework would cost megabytes of bundle for
// generality this never needs, and the whole thing is ~200 lines. The risk of
// hand-rolling is a silently wrong gradient, which is why
// `scripts/check-grad.ts` finite-differences every parameter tensor before any
// of this is trusted.
//
// Parameters live in flat Float32Arrays so serialization, Adam and (later) a
// WebGPU upload are all just buffer copies.

/** ELU, matching the reference config's `activation="elu"`. */
const elu = (x: number) => (x >= 0 ? x : Math.exp(x) - 1);
/** d/dx ELU, expressed via the OUTPUT so the forward pass need not be kept. */
const eluGradFromOutput = (y: number) => (y >= 0 ? 1 : y + 1);

import type { Kernels } from "./kernels/index.ts";

export interface Layer {
  /** Row-major [out, in]. */
  w: Float32Array;
  b: Float32Array;
  dw: Float32Array;
  db: Float32Array;
  inDim: number;
  outDim: number;
}

function makeLayer(
  inDim: number,
  outDim: number,
  rng: () => number,
  gain: number,
  alloc: (n: number) => Float32Array,
): Layer {
  const w = alloc(inDim * outDim);
  // Orthogonal-ish init is what rsl_rl uses; He-scaled uniform is close enough
  // for a net this small and avoids a QR decomposition here.
  const scale = gain * Math.sqrt(2 / inDim);
  for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * scale;
  return {
    w,
    b: alloc(outDim),
    dw: alloc(inDim * outDim),
    db: alloc(outDim),
    inDim,
    outDim,
  };
}

export class MlpNet {
  readonly layers: Layer[] = [];
  /** Post-activation output of each layer, for the current batch. */
  #acts: Float32Array[] = [];
  /** Gradient scratch, one per layer input. */
  #grads: Float32Array[] = [];
  #batch = 0;
  #input: Float32Array | null = null;
  /** Reusable per-block activation gradients; see backward(). */
  readonly #gyBlock = new Float32Array(8);
  #chained: Float32Array[] = [];
  #inputStage: Float32Array | null = null;
  /**
   * Whether backward() should produce the gradient with respect to its INPUT.
   * False for a net whose input is data — the only caller that needs it would
   * be one stacking another differentiable module underneath.
   */
  keepInputGrad = false;

  /** When present, parameters and activations live in the kernel heap and the
   *  SIMD kernels run instead of the loops below. */
  readonly #kernels: Kernels | null;

  constructor(
    inDim: number,
    outDim: number,
    hidden: readonly number[],
    rng: () => number = Math.random,
    /** rsl_rl scales the final layer down so the initial policy is near-zero. */
    outputGain = 0.01,
    kernels: Kernels | null = null,
  ) {
    this.#kernels = kernels;
    const dims = [inDim, ...hidden, outDim];
    for (let i = 0; i < dims.length - 1; i++) {
      const last = i === dims.length - 2;
      this.layers.push(
        makeLayer(dims[i], dims[i + 1], rng, last ? outputGain : 1, (n) => this.#alloc(n)),
      );
    }
  }

  #alloc(length: number): Float32Array {
    return this.#kernels ? this.#kernels.alloc(length) : new Float32Array(length);
  }

  get usesSimd(): boolean {
    return this.#kernels !== null;
  }

  get paramCount(): number {
    return this.layers.reduce((n, l) => n + l.w.length + l.b.length, 0);
  }

  /**
   * Size the scratch buffers for the largest batch seen so far.
   *
   * Grow-only, not resize-on-change: training alternates between the rollout
   * batch (one per environment) and the PPO minibatch every iteration, and
   * reallocating on each switch churns — fatally so against the kernel heap,
   * which is a bump allocator that never frees.
   */
  #ensureBatch(batch: number): void {
    if (batch <= this.#batch) return;
    this.#batch = batch;
    this.#acts = this.layers.map((l) => this.#alloc(batch * l.outDim));
    this.#grads = this.layers.map((l) => this.#alloc(batch * l.inDim));
    // The activation-chained gradient the backward kernel consumes; JS does
    // the chaining because it needs this layer's own outputs.
    this.#chained = this.layers.map((l) => this.#alloc(batch * l.outDim));
    // The caller's input is an ordinary JS array; the kernels can only read
    // their own heap, so it is staged here once per forward. 192x61 floats
    // against 75M multiply-accumulates — the copy does not register.
    this.#inputStage = this.#kernels ? this.#alloc(batch * this.layers[0].inDim) : null;
  }

  /**
   * `x` is [batch, inDim] row-major. Returns [batch, outDim].
   *
   * Eight samples share each weight-row load. The obvious loop — one dot
   * product at a time — re-streams the entire weight matrix once per sample,
   * which at 61x512 is 125 KB read 192 times per layer. Blocking cuts that
   * traffic 8x and hands the CPU eight independent accumulator chains to
   * overlap, and measured 2.26x faster on the real shapes.
   *
   * Each accumulator still sums over `i` in the same order as the scalar
   * version, so results are BIT-IDENTICAL — no float reassociation, and
   * therefore no numerical risk to a run.
   */
  forward(x: Float32Array, batch: number): Float32Array {
    this.#ensureBatch(batch);
    if (this.#inputStage) {
      // Staged, and remembered as the input: backward's first layer reads it
      // again and must find it inside the kernel heap too.
      this.#inputStage.set(x.subarray(0, batch * this.layers[0].inDim));
      this.#input = this.#inputStage;
    } else {
      this.#input = x;
    }
    let cur = this.#input;
    for (let li = 0; li < this.layers.length; li++) {
      const { w, b, inDim, outDim } = this.layers[li];
      const out = this.#acts[li];
      const isLast = li === this.layers.length - 1;
      if (this.#kernels) {
        this.#kernels.gemmForward(cur, w, b, out, batch, inDim, outDim, !isLast);
        cur = out;
        continue;
      }
      const n8 = batch - (batch % 8);
      for (let n = 0; n < n8; n += 8) {
        const xb = n * inDim, ob = n * outDim;
        for (let o = 0; o < outDim; o++) {
          const wo = o * inDim;
          const bias = b[o];
          let a0 = bias, a1 = bias, a2 = bias, a3 = bias;
          let a4 = bias, a5 = bias, a6 = bias, a7 = bias;
          for (let i = 0; i < inDim; i++) {
            const wv = w[wo + i];
            a0 += wv * cur[xb + i];
            a1 += wv * cur[xb + inDim + i];
            a2 += wv * cur[xb + 2 * inDim + i];
            a3 += wv * cur[xb + 3 * inDim + i];
            a4 += wv * cur[xb + 4 * inDim + i];
            a5 += wv * cur[xb + 5 * inDim + i];
            a6 += wv * cur[xb + 6 * inDim + i];
            a7 += wv * cur[xb + 7 * inDim + i];
          }
          if (isLast) {
            out[ob + o] = a0; out[ob + outDim + o] = a1;
            out[ob + 2 * outDim + o] = a2; out[ob + 3 * outDim + o] = a3;
            out[ob + 4 * outDim + o] = a4; out[ob + 5 * outDim + o] = a5;
            out[ob + 6 * outDim + o] = a6; out[ob + 7 * outDim + o] = a7;
          } else {
            out[ob + o] = elu(a0); out[ob + outDim + o] = elu(a1);
            out[ob + 2 * outDim + o] = elu(a2); out[ob + 3 * outDim + o] = elu(a3);
            out[ob + 4 * outDim + o] = elu(a4); out[ob + 5 * outDim + o] = elu(a5);
            out[ob + 6 * outDim + o] = elu(a6); out[ob + 7 * outDim + o] = elu(a7);
          }
        }
      }
      for (let n = n8; n < batch; n++) {
        const xo = n * inDim, oo = n * outDim;
        for (let o = 0; o < outDim; o++) {
          const wo = o * inDim;
          let acc = b[o];
          for (let i = 0; i < inDim; i++) acc += w[wo + i] * cur[xo + i];
          out[oo + o] = isLast ? acc : elu(acc);
        }
      }
      cur = out;
    }
    // Buffers are sized for the largest batch seen, so hand back only the rows
    // this call actually produced.
    return cur.subarray(0, batch * this.layers[this.layers.length - 1].outDim);
  }

  zeroGrad(): void {
    for (const l of this.layers) {
      l.dw.fill(0);
      l.db.fill(0);
    }
  }

  /**
   * `gOut` is d(loss)/d(output), [batch, outDim]. Accumulates into dw/db and
   * returns d(loss)/d(input). Call `forward` first — the activations it cached
   * are what this differentiates.
   */
  backward(gOut: Float32Array, batch: number): Float32Array {
    if (!this.#input) throw new Error("backward() before forward()");
    let g = gOut;
    for (let li = this.layers.length - 1; li >= 0; li--) {
      const { w, dw, db, inDim, outDim } = this.layers[li];
      const input = li === 0 ? this.#input : this.#acts[li - 1];
      const gIn = this.#grads[li];
      // Nobody wants the gradient with respect to the OBSERVATION, so the
      // first layer skips it. That is half of layer 0's backward work, and
      // layer 0 is the widest layer in this architecture (61x512).
      const wantInputGrad = li > 0 || this.keepInputGrad;
      if (wantInputGrad) gIn.fill(0);
      const isLast = li === this.layers.length - 1;
      const act = this.#acts[li];
      if (this.#kernels) {
        // The kernel takes the ALREADY activation-chained gradient, because
        // chaining needs this layer's own outputs and doing it here keeps the
        // kernel a plain GEMM.
        const chained = this.#chained[li];
        if (isLast) {
          chained.set(g.subarray(0, batch * outDim));
        } else {
          for (let j = 0; j < batch * outDim; j++) {
            chained[j] = g[j] * eluGradFromOutput(act[j]);
          }
        }
        this.#kernels.gemmBackward(
          chained, input, w, dw, db, gIn, batch, inDim, outDim, wantInputGrad,
        );
        g = gIn;
        continue;
      }

      // Eight samples per weight row, matching forward. Backward does two
      // multiply-accumulates per element against forward's one — dw and gIn —
      // so it is the more expensive half and gains most from loading w[wo+i]
      // and the dw read-modify-write once for eight samples instead of once
      // each.
      const n8 = batch - (batch % 8);
      const gy = this.#gyBlock;
      for (let n = 0; n < n8; n += 8) {
        const gb = n * outDim, xb = n * inDim;
        for (let o = 0; o < outDim; o++) {
          let any = false;
          for (let k = 0; k < 8; k++) {
            const at = gb + k * outDim + o;
            const v = isLast ? g[at] : g[at] * eluGradFromOutput(act[at]);
            gy[k] = v;
            if (v !== 0) any = true;
          }
          // Whole-block skip: PPO zeroes the policy gradient for clipped
          // samples, so 30-40% of rows really are all zero here.
          if (!any) continue;
          const y0 = gy[0], y1 = gy[1], y2 = gy[2], y3 = gy[3];
          const y4 = gy[4], y5 = gy[5], y6 = gy[6], y7 = gy[7];
          db[o] += y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7;
          const wo = o * inDim;
          const x1 = xb + inDim, x2 = xb + 2 * inDim, x3 = xb + 3 * inDim;
          const x4 = xb + 4 * inDim, x5 = xb + 5 * inDim;
          const x6 = xb + 6 * inDim, x7 = xb + 7 * inDim;
          if (wantInputGrad) {
            for (let i = 0; i < inDim; i++) {
              const wv = w[wo + i];
              dw[wo + i] +=
                y0 * input[xb + i] + y1 * input[x1 + i] +
                y2 * input[x2 + i] + y3 * input[x3 + i] +
                y4 * input[x4 + i] + y5 * input[x5 + i] +
                y6 * input[x6 + i] + y7 * input[x7 + i];
              gIn[xb + i] += y0 * wv;
              gIn[x1 + i] += y1 * wv;
              gIn[x2 + i] += y2 * wv;
              gIn[x3 + i] += y3 * wv;
              gIn[x4 + i] += y4 * wv;
              gIn[x5 + i] += y5 * wv;
              gIn[x6 + i] += y6 * wv;
              gIn[x7 + i] += y7 * wv;
            }
          } else {
            for (let i = 0; i < inDim; i++) {
              dw[wo + i] +=
                y0 * input[xb + i] + y1 * input[x1 + i] +
                y2 * input[x2 + i] + y3 * input[x3 + i] +
                y4 * input[x4 + i] + y5 * input[x5 + i] +
                y6 * input[x6 + i] + y7 * input[x7 + i];
            }
          }
        }
      }
      for (let n = n8; n < batch; n++) {
        const go = n * outDim, xo = n * inDim;
        for (let o = 0; o < outDim; o++) {
          // Chain through the activation before touching the weights.
          const gy = isLast ? g[go + o] : g[go + o] * eluGradFromOutput(act[go + o]);
          if (gy === 0) continue;
          db[o] += gy;
          const wo = o * inDim;
          if (wantInputGrad) {
            for (let i = 0; i < inDim; i++) {
              dw[wo + i] += gy * input[xo + i];
              gIn[xo + i] += gy * w[wo + i];
            }
          } else {
            for (let i = 0; i < inDim; i++) dw[wo + i] += gy * input[xo + i];
          }
        }
      }
      g = gIn;
    }
    return g;
  }

  /** Every parameter tensor paired with its gradient, for Adam and checkpoints. */
  tensors(): { param: Float32Array; grad: Float32Array }[] {
    return this.layers.flatMap((l) => [
      { param: l.w, grad: l.dw },
      { param: l.b, grad: l.db },
    ]);
  }
}

/** Global L2 norm over every gradient, then scale down if it exceeds `max`. */
export function clipGradNorm(tensors: { grad: Float32Array }[], max: number): number {
  let sumSq = 0;
  for (const { grad } of tensors) for (let i = 0; i < grad.length; i++) sumSq += grad[i] * grad[i];
  const norm = Math.sqrt(sumSq);
  if (norm > max && norm > 0) {
    const scale = max / norm;
    for (const { grad } of tensors) for (let i = 0; i < grad.length; i++) grad[i] *= scale;
  }
  return norm;
}

export class Adam {
  readonly #m: Float32Array[];
  readonly #v: Float32Array[];
  #t = 0;
  lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;

  constructor(
    tensors: { param: Float32Array }[],
    lr: number,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
  ) {
    this.#m = tensors.map((t) => new Float32Array(t.param.length));
    this.#v = tensors.map((t) => new Float32Array(t.param.length));
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
  }

  step(tensors: { param: Float32Array; grad: Float32Array }[]): void {
    this.#t++;
    const bc1 = 1 - Math.pow(this.beta1, this.#t);
    const bc2 = 1 - Math.pow(this.beta2, this.#t);
    for (let k = 0; k < tensors.length; k++) {
      const { param, grad } = tensors[k];
      const m = this.#m[k], v = this.#v[k];
      for (let i = 0; i < param.length; i++) {
        const g = grad[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
        param[i] -= (this.lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + this.eps);
      }
    }
  }

  /** Adam's moments are part of the training state — a resumed run without
   *  them takes a visible quality hit for hundreds of steps. */
  serialize(): { t: number; m: number[][]; v: number[][] } {
    return {
      t: this.#t,
      m: this.#m.map((a) => Array.from(a)),
      v: this.#v.map((a) => Array.from(a)),
    };
  }

  load(state: { t: number; m: number[][]; v: number[][] }): void {
    this.#t = state.t;
    state.m.forEach((a, i) => this.#m[i]?.set(a));
    state.v.forEach((a, i) => this.#v[i]?.set(a));
  }
}
