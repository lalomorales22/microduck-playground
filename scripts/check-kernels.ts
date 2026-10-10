// Do the SIMD kernels agree with the JavaScript reference?
//
//   node scripts/check-kernels.ts
//
// The JS path in nn.ts is the definition of correct; the wasm kernels are an
// optimisation of it. A kernel that is subtly different does not crash — it
// trains a slightly wrong policy, slowly, and you find out much later. So this
// runs both on identical inputs and compares forward outputs and every
// gradient tensor, then times them.

import { MlpNet } from "../src/train/nn.ts";
import { loadKernelsFromDisk } from "../src/train/kernels/load.node.ts";


function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const kernels = await loadKernelsFromDisk();
if (!kernels) {
  console.log("kernels.wasm unavailable — the JavaScript path is the only one; nothing to compare.");
  process.exit(0);
}

const OBS = 61, ACT = 14, BATCH = 192;
const SHAPES: readonly number[][] = [[128, 64], [512, 256, 128]];

function timeIt(reps: number, fn: () => void): number {
  for (let i = 0; i < 3; i++) fn();
  const s: number[] = [];
  for (let r = 0; r < reps; r++) {
    const t = performance.now();
    fn();
    s.push(performance.now() - t);
  }
  s.sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

const failures: string[] = [];

for (const hidden of SHAPES) {
  // Same seed for both, so the nets start with identical weights.
  const js = new MlpNet(OBS, ACT, hidden, mulberry32(3), 0.01, null);
  const wasm = new MlpNet(OBS, ACT, hidden, mulberry32(3), 0.01, kernels);

  const rng = mulberry32(9);
  const x = new Float32Array(BATCH * OBS);
  for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
  // Deliberately sparse, matching PPO: 35% of rows are zeroed by clipping and
  // the kernels have a whole-block skip that only fires on genuinely zero rows.
  const gOut = new Float32Array(BATCH * ACT);
  for (let n = 0; n < BATCH; n++) {
    if (rng() < 0.35) continue;
    for (let i = 0; i < ACT; i++) gOut[n * ACT + i] = (rng() * 2 - 1) * 1e-2;
  }

  const yJs = js.forward(x, BATCH);
  const yWasm = wasm.forward(x, BATCH);
  let fwdDiff = 0;
  for (let i = 0; i < yJs.length; i++) fwdDiff = Math.max(fwdDiff, Math.abs(yJs[i] - yWasm[i]));

  js.zeroGrad(); js.backward(gOut, BATCH);
  wasm.zeroGrad(); wasm.backward(gOut, BATCH);
  let gradDiff = 0;
  let gradScale = 0;
  const a = js.tensors(), b = wasm.tensors();
  for (let t = 0; t < a.length; t++) {
    for (let i = 0; i < a[t].grad.length; i++) {
      gradDiff = Math.max(gradDiff, Math.abs(a[t].grad[i] - b[t].grad[i]));
      gradScale = Math.max(gradScale, Math.abs(a[t].grad[i]));
    }
  }

  const msJs = timeIt(15, () => {
    js.forward(x, BATCH);
    js.zeroGrad();
    js.backward(gOut, BATCH);
  });
  const msWasm = timeIt(15, () => {
    wasm.forward(x, BATCH);
    wasm.zeroGrad();
    wasm.backward(gOut, BATCH);
  });

  console.log(
    `\n${hidden.join("/").padEnd(12)} forward max diff ${fwdDiff.toExponential(1)}, ` +
    `gradient max diff ${gradDiff.toExponential(1)} (largest gradient ${gradScale.toExponential(1)})`,
  );
  console.log(
    `${"".padEnd(12)} fwd+bwd  JS ${msJs.toFixed(2)} ms  SIMD ${msWasm.toFixed(2)} ms  ` +
    `${(msJs / msWasm).toFixed(2)}x`,
  );

  // float32 accumulation in a different instruction order gives a slightly
  // different last bit; anything beyond ~1e-5 relative is a real disagreement.
  const relGrad = gradDiff / Math.max(gradScale, 1e-12);
  if (!(fwdDiff < 1e-4)) failures.push(`${hidden.join("/")}: forward differs by ${fwdDiff}`);
  if (!(relGrad < 1e-4)) failures.push(`${hidden.join("/")}: gradients differ by ${relGrad} relative`);
  if (!(msWasm < msJs)) failures.push(`${hidden.join("/")}: SIMD is not faster (${msWasm} vs ${msJs} ms)`);
}

console.log();
if (failures.length) {
  console.log("FAILED:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
console.log("PASSED: SIMD kernels match the JavaScript reference, and are faster.");
