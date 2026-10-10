// Loads the SIMD kernel module and hands out buffers inside its heap.
//
// The net's parameters and activations are allocated HERE rather than as plain
// JavaScript arrays, so calling a kernel costs no copying — the Float32Arrays
// JS holds are views onto the same wasm memory the kernel reads.
//
// The JS implementations in nn.ts remain the reference: they run wherever this
// does not load, and scripts/check-kernels.ts asserts the two agree.

export interface Kernels {
  gemmForward(
    x: Float32Array, w: Float32Array, b: Float32Array, out: Float32Array,
    batch: number, inDim: number, outDim: number, activate: boolean,
  ): void;
  gemmBackward(
    gy: Float32Array, input: Float32Array, w: Float32Array,
    dw: Float32Array, db: Float32Array, gIn: Float32Array,
    batch: number, inDim: number, outDim: number, wantInputGrad: boolean,
  ): void;
  /** A Float32Array view inside the kernel heap. */
  alloc(length: number): Float32Array;
  heapUsedBytes(): number;
}

interface Exports {
  memory: WebAssembly.Memory;
  gemm_forward(x: number, w: number, b: number, out: number,
               batch: number, inDim: number, outDim: number, activate: number): void;
  gemm_backward(gy: number, input: number, w: number, dw: number, db: number, gIn: number,
                batch: number, inDim: number, outDim: number, wantInputGrad: number): void;
  kalloc(bytes: number): number;
  heap_used(): number;
}

/**
 * Whether this engine can run OUR module.
 *
 * Validating the real bytes rather than a synthetic SIMD probe: the probe is
 * hand-written machine code that is easy to get wrong — mine was, and it
 * reported "no SIMD" on an engine that supports it perfectly well — whereas
 * this tests exactly the thing we are about to instantiate.
 */
export function kernelsSupported(wasmBytes: BufferSource): boolean {
  try {
    return WebAssembly.validate(wasmBytes);
  } catch {
    return false;
  }
}

let cached: Promise<Kernels | null> | null = null;

/** Resolves to null wherever SIMD or the module is unavailable — callers fall
 *  back to the JavaScript path rather than failing. */
export function loadKernels(wasmBytes: BufferSource): Promise<Kernels | null> {
  cached ??= instantiate(wasmBytes).catch((err: unknown) => {
    console.warn("[kernels] falling back to JavaScript:", err);
    return null;
  });
  return cached;
}

async function instantiate(wasmBytes: BufferSource): Promise<Kernels | null> {
  if (!kernelsSupported(wasmBytes)) return null;
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports as unknown as Exports;

  // Views are re-derived on every use: the heap can grow and detach them.
  const off = (a: Float32Array) => a.byteOffset;
  const check = (a: Float32Array) => {
    if (a.buffer !== ex.memory.buffer) {
      throw new Error("kernel called with a buffer outside its heap");
    }
  };

  return {
    alloc(length: number): Float32Array {
      const bytes = length * 4;
      const ptr = ex.kalloc(bytes);
      // kalloc only bumps a pointer; growing the memory to cover it is this
      // side's job, and a silent overrun here would be a wild pointer.
      const needed = ptr + bytes;
      if (needed > ex.memory.buffer.byteLength) {
        const pages = Math.ceil((needed - ex.memory.buffer.byteLength) / 65536);
        if (ex.memory.grow(pages) < 0) {
          throw new Error(`kernel heap could not grow to ${needed} bytes`);
        }
      }
      return new Float32Array(ex.memory.buffer, ptr, length);
    },
    heapUsedBytes: () => ex.heap_used(),
    gemmForward(x, w, b, out, batch, inDim, outDim, activate) {
      check(x); check(w); check(b); check(out);
      ex.gemm_forward(off(x), off(w), off(b), off(out), batch, inDim, outDim, activate ? 1 : 0);
    },
    gemmBackward(gy, input, w, dw, db, gIn, batch, inDim, outDim, wantInputGrad) {
      check(gy); check(input); check(w); check(dw); check(db); check(gIn);
      ex.gemm_backward(off(gy), off(input), off(w), off(dw), off(db), off(gIn),
        batch, inDim, outDim, wantInputGrad ? 1 : 0);
    },
  };
}
