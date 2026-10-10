// Compiles src/train/kernels/kernels.c to a standalone SIMD wasm module.
//
//   npm run build:kernels     (requires emcc)
//
// The output is COMMITTED, so building the project does not need a C
// toolchain — only changing the kernels does.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC = path.join(ROOT, "src/train/kernels/kernels.c");
const OUT = path.join(ROOT, "src/train/kernels/kernels.wasm");

try {
  execFileSync("emcc", [
    SRC, "-O3", "-msimd128", "--no-entry",
    "-s", "STANDALONE_WASM=1",
    "-s", "ALLOW_MEMORY_GROWTH=1",
    "-s", "INITIAL_MEMORY=33554432",
    "-o", OUT,
  ], { stdio: ["ignore", "pipe", "pipe"] });
} catch (err) {
  console.error("emcc failed. Install emscripten, or leave kernels.wasm as committed.");
  console.error(err.stderr?.toString() ?? err.message);
  process.exit(1);
}
console.log(`built ${path.relative(ROOT, OUT)} (${(fs.statSync(OUT).size / 1024).toFixed(1)} KB)`);
