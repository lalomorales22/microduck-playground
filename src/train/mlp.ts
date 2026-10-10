// A batched MLP forward pass, shaped like the policies this project runs:
// 61 -> 512 -> 256 -> 128 -> 14 with ELU, ~197k parameters.
//
// For M0 the weights are random — the point is to charge the throughput
// harness the arithmetic a real rollout would pay, so the reported rate is not
// a physics-only number that the trainer can never reach. The same kernels are
// what M2's learner will need, so this is a down payment, not a throwaway.

export const HIDDEN_DIMS = [512, 256, 128] as const;

interface Layer {
  /** Row-major [out, in] — each output's weights are contiguous. */
  w: Float32Array;
  b: Float32Array;
  inDim: number;
  outDim: number;
}

export class Mlp {
  private readonly layers: Layer[] = [];
  /** Scratch per layer, sized for the batch. */
  private readonly scratch: Float32Array[] = [];

  constructor(inDim: number, outDim: number, batch: number, hidden: readonly number[] = HIDDEN_DIMS) {
    const dims = [inDim, ...hidden, outDim];
    for (let i = 0; i < dims.length - 1; i++) {
      const [din, dout] = [dims[i], dims[i + 1]];
      const w = new Float32Array(din * dout);
      // Small random weights: the values do not matter, but zeros could let a
      // JIT or a denormal path make this unrepresentatively cheap.
      const scale = Math.sqrt(2 / din);
      for (let k = 0; k < w.length; k++) w[k] = (Math.random() * 2 - 1) * scale;
      this.layers.push({ w, b: new Float32Array(dout), inDim: din, outDim: dout });
      this.scratch.push(new Float32Array(batch * dout));
    }
  }

  get paramCount(): number {
    return this.layers.reduce((n, l) => n + l.w.length + l.b.length, 0);
  }

  /**
   * `input` is [batch, inDim] row-major; returns [batch, outDim].
   * ELU on every layer but the last, matching the exported policies.
   */
  forward(input: Float32Array, batch: number): Float32Array {
    let x = input;
    for (let li = 0; li < this.layers.length; li++) {
      const { w, b, inDim, outDim } = this.layers[li];
      const out = this.scratch[li];
      const last = li === this.layers.length - 1;
      for (let n = 0; n < batch; n++) {
        const xo = n * inDim, oo = n * outDim;
        for (let o = 0; o < outDim; o++) {
          const wo = o * inDim;
          let acc = b[o];
          for (let i = 0; i < inDim; i++) acc += w[wo + i] * x[xo + i];
          out[oo + o] = last ? acc : acc >= 0 ? acc : Math.exp(acc) - 1;
        }
      }
      x = out;
    }
    return x;
  }
}
