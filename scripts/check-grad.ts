// Finite-difference check for the hand-written backprop.
//
//   node scripts/check-grad.ts
//
// Hand-rolled gradients fail silently: the loss still goes down, just to the
// wrong place, and you find out an hour into a run. This perturbs every
// parameter tensor and compares the numeric derivative against the analytic
// one before any of it is trusted.

import { Adam, MlpNet, clipGradNorm } from "../src/train/nn.ts";
import { gaussianLogProb, gaussianEntropy } from "../src/train/distribution.ts";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(7);
const IN = 6, OUT = 3, BATCH = 4;
const net = new MlpNet(IN, OUT, [8, 5], rng, 1.0);

const x = new Float32Array(BATCH * IN);
for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
const targets = new Float32Array(BATCH * OUT);
for (let i = 0; i < targets.length; i++) targets[i] = rng() * 2 - 1;

/** Mean squared error — arbitrary, but it exercises every path. */
function loss(): number {
  const y = net.forward(x, BATCH);
  let acc = 0;
  for (let i = 0; i < y.length; i++) acc += (y[i] - targets[i]) ** 2;
  return acc / y.length;
}

function analyticGrads(): void {
  const y = net.forward(x, BATCH);
  const g = new Float32Array(y.length);
  for (let i = 0; i < y.length; i++) g[i] = (2 * (y[i] - targets[i])) / y.length;
  net.zeroGrad();
  net.backward(g, BATCH);
}

analyticGrads();
const tensors = net.tensors();

// Parameters are Float32Array, so a central difference is only good to about
// 2 * 1e-7 * |loss| / EPS ~= 3e-5 absolute here. Judging individual near-zero
// entries by RELATIVE error would therefore flag precision noise as a bug. The
// robust standard check is the norm-relative error over the whole gradient
// vector, plus a per-element tolerance that admits the noise floor.
const EPS = 1e-3;
const NOISE_FLOOR = 1e-4;
let worstAbs = 0;
let worstWhere = "";
let checked = 0;
let sumDiffSq = 0;
let sumNumSq = 0;
let sumAnaSq = 0;

for (let t = 0; t < tensors.length; t++) {
  const { param, grad } = tensors[t];
  for (let i = 0; i < param.length; i++) {
    const orig = param[i];
    param[i] = orig + EPS;
    const up = loss();
    param[i] = orig - EPS;
    const down = loss();
    param[i] = orig;
    const numeric = (up - down) / (2 * EPS);
    const analytic = grad[i];
    const diff = Math.abs(numeric - analytic);
    sumDiffSq += diff * diff;
    sumNumSq += numeric * numeric;
    sumAnaSq += analytic * analytic;
    checked++;
    if (diff > worstAbs) {
      worstAbs = diff;
      worstWhere = `tensor ${t}[${i}] numeric=${numeric.toExponential(3)} analytic=${analytic.toExponential(3)}`;
    }
  }
}

const normRel = Math.sqrt(sumDiffSq) / (Math.sqrt(sumNumSq) + Math.sqrt(sumAnaSq));
console.log(`checked ${checked} parameters across ${tensors.length} tensors`);
console.log(`gradient norm-relative error: ${normRel.toExponential(2)}`);
console.log(`worst absolute error: ${worstAbs.toExponential(2)}  (${worstWhere})`);
console.log(`finite-difference noise floor at float32: ~${NOISE_FLOOR.toExponential(0)}`);

// ── Gaussian log-prob and entropy, same treatment ─────────────────────────
const mean = new Float32Array([0.3, -0.7, 1.1]);
const logStd = new Float32Array([-0.5, 0.2, 0.0]);
const action = new Float32Array([0.1, -0.4, 0.9]);
const gMean = new Float32Array(3);
const gLogStd = new Float32Array(3);
gaussianLogProb(action, mean, logStd, 0, gMean, gLogStd);

let worstDist = 0;
for (let i = 0; i < 3; i++) {
  for (const [arr, grads, label] of [
    [mean, gMean, "mean"],
    [logStd, gLogStd, "logStd"],
  ] as const) {
    const orig = arr[i];
    arr[i] = orig + EPS;
    const up = gaussianLogProb(action, mean, logStd, 0);
    arr[i] = orig - EPS;
    const down = gaussianLogProb(action, mean, logStd, 0);
    arr[i] = orig;
    const numeric = (up - down) / (2 * EPS);
    const rel = Math.abs(numeric - grads[i]) / Math.max(1e-4, Math.abs(numeric));
    if (rel > worstDist) worstDist = rel;
    void label;
  }
}
console.log(`gaussian logProb worst relative error: ${worstDist.toExponential(2)}`);
console.log(`gaussian entropy at logStd=0: ${gaussianEntropy(new Float32Array([0, 0, 0])).toFixed(6)} ` +
  `(expected ${(3 * 0.5 * Math.log(2 * Math.PI * Math.E)).toFixed(6)})`);

// ── Adam and grad clipping sanity ─────────────────────────────────────────
const before = loss();
const opt = new Adam(tensors, 1e-2);
for (let i = 0; i < 50; i++) {
  analyticGrads();
  clipGradNorm(net.tensors(), 1.0);
  opt.step(net.tensors());
}
const after = loss();
console.log(`loss after 50 Adam steps: ${before.toFixed(5)} -> ${after.toFixed(5)}`);

const failures: string[] = [];
if (!(normRel < 1e-4)) failures.push(`gradient norm-relative error ${normRel.toExponential(2)} >= 1e-4`);
if (!(worstAbs < NOISE_FLOOR)) {
  failures.push(`worst absolute gradient error ${worstAbs.toExponential(2)} exceeds the float32 noise floor`);
}
if (!(worstDist < 1e-3)) failures.push(`gaussian gradient error ${worstDist.toExponential(2)} >= 1e-3`);
if (!(after < before * 0.1)) failures.push(`Adam did not reduce the loss by 10x (${before} -> ${after})`);

if (failures.length) {
  console.log("\nFAILED:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
console.log("\nPASSED: analytic gradients match finite differences; Adam descends.");
