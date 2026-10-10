// Which matrix-multiply do we want?
//
//   node scripts/bench-gemm.ts
//
// The profiler says 89% of a training iteration is dense linear algebra in
// MlpNet, so this isolates just that kernel at the real shapes. Every variant
// is checked against the naive one before it is timed — a fast wrong answer is
// worthless, and float reassociation makes "wrong" easy to introduce here.

const BATCH = 192; // one PPO minibatch at 32 envs x 24 steps / 4
const LAYERS: [number, number][] = [[61, 512], [512, 256], [256, 128], [128, 14]];

function fill(n: number, seed: number): Float32Array {
  const a = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s ^ (s >>> 15), 1 | s) + 0x6d2b79f5) >>> 0;
    a[i] = ((s >>> 8) / 8388608 - 1) * 0.1;
  }
  return a;
}

type Kernel = (x: Float32Array, w: Float32Array, b: Float32Array, out: Float32Array,
               batch: number, inDim: number, outDim: number) => void;

/** What MlpNet.forward does today: one dot product at a time. */
const naive: Kernel = (x, w, b, out, batch, inDim, outDim) => {
  for (let n = 0; n < batch; n++) {
    const xo = n * inDim, oo = n * outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let acc = b[o];
      for (let i = 0; i < inDim; i++) acc += w[wo + i] * x[xo + i];
      out[oo + o] = acc;
    }
  }
};

/**
 * Four samples per weight row.
 *
 * The naive version re-streams the whole weight matrix once per sample. At
 * 61x512 that is 125 KB read 192 times per layer. Handling four samples
 * against one weight row cuts that traffic 4x and gives the CPU four
 * independent accumulator chains to overlap.
 */
const blocked4: Kernel = (x, w, b, out, batch, inDim, outDim) => {
  const n4 = batch - (batch % 4);
  for (let n = 0; n < n4; n += 4) {
    const x0 = n * inDim, x1 = x0 + inDim, x2 = x1 + inDim, x3 = x2 + inDim;
    const o0 = n * outDim, o1 = o0 + outDim, o2 = o1 + outDim, o3 = o2 + outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      const bias = b[o];
      let a0 = bias, a1 = bias, a2 = bias, a3 = bias;
      for (let i = 0; i < inDim; i++) {
        const wv = w[wo + i];
        a0 += wv * x[x0 + i];
        a1 += wv * x[x1 + i];
        a2 += wv * x[x2 + i];
        a3 += wv * x[x3 + i];
      }
      out[o0 + o] = a0; out[o1 + o] = a1; out[o2 + o] = a2; out[o3 + o] = a3;
    }
  }
  for (let n = n4; n < batch; n++) {
    const xo = n * inDim, oo = n * outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let acc = b[o];
      for (let i = 0; i < inDim; i++) acc += w[wo + i] * x[xo + i];
      out[oo + o] = acc;
    }
  }
};

/** Eight samples per weight row. More reuse, more register pressure. */
const blocked8: Kernel = (x, w, b, out, batch, inDim, outDim) => {
  const n8 = batch - (batch % 8);
  for (let n = 0; n < n8; n += 8) {
    const xb = n * inDim, ob = n * outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      const bias = b[o];
      let a0 = bias, a1 = bias, a2 = bias, a3 = bias, a4 = bias, a5 = bias, a6 = bias, a7 = bias;
      for (let i = 0; i < inDim; i++) {
        const wv = w[wo + i];
        a0 += wv * x[xb + i];
        a1 += wv * x[xb + inDim + i];
        a2 += wv * x[xb + 2 * inDim + i];
        a3 += wv * x[xb + 3 * inDim + i];
        a4 += wv * x[xb + 4 * inDim + i];
        a5 += wv * x[xb + 5 * inDim + i];
        a6 += wv * x[xb + 6 * inDim + i];
        a7 += wv * x[xb + 7 * inDim + i];
      }
      out[ob + o] = a0; out[ob + outDim + o] = a1;
      out[ob + 2 * outDim + o] = a2; out[ob + 3 * outDim + o] = a3;
      out[ob + 4 * outDim + o] = a4; out[ob + 5 * outDim + o] = a5;
      out[ob + 6 * outDim + o] = a6; out[ob + 7 * outDim + o] = a7;
    }
  }
  for (let n = n8; n < batch; n++) {
    const xo = n * inDim, oo = n * outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let acc = b[o];
      for (let i = 0; i < inDim; i++) acc += w[wo + i] * x[xo + i];
      out[oo + o] = acc;
    }
  }
};

const KERNELS: [string, Kernel][] = [
  ["naive (current)", naive],
  ["blocked x4", blocked4],
  ["blocked x8", blocked8],
];

function timeIt(reps: number, fn: () => void): number {
  for (let i = 0; i < 5; i++) fn();
  const s: number[] = [];
  for (let r = 0; r < reps; r++) {
    const t = performance.now();
    fn();
    s.push(performance.now() - t);
  }
  s.sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

let macs = 0;
for (const [i, o] of LAYERS) macs += BATCH * i * o;
console.log(`batch ${BATCH}, layers ${LAYERS.map(([i, o]) => `${i}x${o}`).join(" ")}`);
console.log(`${(macs / 1e6).toFixed(1)}M MACs per forward pass\n`);

const inputs = LAYERS.map(([i], k) => fill(BATCH * i, 11 + k));
const weights = LAYERS.map(([i, o], k) => fill(i * o, 101 + k));
const biases = LAYERS.map(([, o], k) => fill(o, 211 + k));
const outputs = LAYERS.map(([, o]) => new Float32Array(BATCH * o));
const reference = LAYERS.map(([, o]) => new Float32Array(BATCH * o));

// Reference results, once.
LAYERS.forEach(([i, o], k) => naive(inputs[k], weights[k], biases[k], reference[k], BATCH, i, o));

let baseline = 0;
for (const [name, kernel] of KERNELS) {
  let worst = 0;
  LAYERS.forEach(([i, o], k) => {
    outputs[k].fill(0);
    kernel(inputs[k], weights[k], biases[k], outputs[k], BATCH, i, o);
    for (let j = 0; j < outputs[k].length; j++) {
      worst = Math.max(worst, Math.abs(outputs[k][j] - reference[k][j]));
    }
  });
  const ms = timeIt(40, () => {
    LAYERS.forEach(([i, o], k) => kernel(inputs[k], weights[k], biases[k], outputs[k], BATCH, i, o));
  });
  if (!baseline) baseline = ms;
  const gflops = (2 * macs) / (ms / 1000) / 1e9;
  console.log(
    `${name.padEnd(18)} ${ms.toFixed(2).padStart(7)} ms  ${gflops.toFixed(2).padStart(6)} GFLOP/s  ` +
    `${(baseline / ms).toFixed(2)}x   max abs diff ${worst.toExponential(1)}`,
  );
}
