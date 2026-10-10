// Node-side kernel loading, for the headless checks and the profiler.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadKernels, type Kernels } from "./index.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));

export function loadKernelsFromDisk(): Promise<Kernels | null> {
  const file = path.join(HERE, "kernels.wasm");
  if (!fs.existsSync(file)) return Promise.resolve(null);
  return loadKernels(fs.readFileSync(file));
}
