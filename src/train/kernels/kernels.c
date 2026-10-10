// SIMD kernels for the MLP's forward and backward passes.
//
// These mirror the JavaScript in src/train/nn.ts exactly, including the
// eight-sample blocking and the accumulation ORDER — scripts/check-kernels.ts
// asserts the two agree, and a silently-different kernel is precisely the bug
// class that costs a training run.
//
// Build:  npm run build:kernels   (needs emcc; the .wasm is committed so this
//                                  is only required when the C changes)
//
// Everything is offsets into the module's own linear memory: JS allocates the
// net's parameters and activations here so a call costs no copying.

#include <wasm_simd128.h>
#include <math.h>

#define BLOCK 8

static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) +
         wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

static inline float elu(float x) { return x >= 0.0f ? x : expm1f(x); }

/**
 * out[batch, outDim] = act(x[batch, inDim] * wT + b), w is [outDim, inDim].
 * `activate` applies ELU; the output layer passes 0.
 */
__attribute__((export_name("gemm_forward")))
void gemm_forward(const float* x, const float* w, const float* b, float* out,
                  int batch, int inDim, int outDim, int activate) {
  int n8 = batch - (batch % BLOCK);
  for (int n = 0; n < n8; n += BLOCK) {
    const float* xb = x + (long)n * inDim;
    float* ob = out + (long)n * outDim;
    for (int o = 0; o < outDim; o++) {
      const float* wr = w + (long)o * inDim;
      v128_t a[BLOCK];
      for (int k = 0; k < BLOCK; k++) a[k] = wasm_f32x4_splat(0.0f);
      int i = 0;
      for (; i + 4 <= inDim; i += 4) {
        v128_t wv = wasm_v128_load(wr + i);
        for (int k = 0; k < BLOCK; k++) {
          a[k] = wasm_f32x4_add(a[k], wasm_f32x4_mul(wv, wasm_v128_load(xb + (long)k * inDim + i)));
        }
      }
      float bias = b[o];
      for (int k = 0; k < BLOCK; k++) {
        float s = bias + hsum(a[k]);
        for (int j = i; j < inDim; j++) s += wr[j] * xb[(long)k * inDim + j];
        ob[(long)k * outDim + o] = activate ? elu(s) : s;
      }
    }
  }
  for (int n = n8; n < batch; n++) {
    const float* xr = x + (long)n * inDim;
    float* orow = out + (long)n * outDim;
    for (int o = 0; o < outDim; o++) {
      const float* wr = w + (long)o * inDim;
      v128_t acc = wasm_f32x4_splat(0.0f);
      int i = 0;
      for (; i + 4 <= inDim; i += 4) {
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(wr + i), wasm_v128_load(xr + i)));
      }
      float s = b[o] + hsum(acc);
      for (; i < inDim; i++) s += wr[i] * xr[i];
      orow[o] = activate ? elu(s) : s;
    }
  }
}

/**
 * Accumulates dw += gyT * input and db += sum(gy); writes gIn += gy * w when
 * `wantInputGrad`. `gy` is the ALREADY activation-chained gradient, [batch,
 * outDim] — JS does that chaining because it needs the layer's own outputs.
 */
__attribute__((export_name("gemm_backward")))
void gemm_backward(const float* gy, const float* input, const float* w,
                   float* dw, float* db, float* gIn,
                   int batch, int inDim, int outDim, int wantInputGrad) {
  int n8 = batch - (batch % BLOCK);
  for (int n = 0; n < n8; n += BLOCK) {
    const float* gb = gy + (long)n * outDim;
    const float* xb = input + (long)n * inDim;
    float* gb_in = gIn + (long)n * inDim;
    for (int o = 0; o < outDim; o++) {
      float y[BLOCK];
      int any = 0;
      for (int k = 0; k < BLOCK; k++) {
        y[k] = gb[(long)k * outDim + o];
        if (y[k] != 0.0f) any = 1;
      }
      // PPO zeroes the policy gradient for clipped samples, so 30-40% of these
      // rows really are all zero.
      if (!any) continue;
      float sum = 0.0f;
      for (int k = 0; k < BLOCK; k++) sum += y[k];
      db[o] += sum;

      const float* wr = w + (long)o * inDim;
      float* dwr = dw + (long)o * inDim;
      v128_t yv[BLOCK];
      for (int k = 0; k < BLOCK; k++) yv[k] = wasm_f32x4_splat(y[k]);

      int i = 0;
      for (; i + 4 <= inDim; i += 4) {
        v128_t acc = wasm_v128_load(dwr + i);
        for (int k = 0; k < BLOCK; k++) {
          acc = wasm_f32x4_add(acc, wasm_f32x4_mul(yv[k], wasm_v128_load(xb + (long)k * inDim + i)));
        }
        wasm_v128_store(dwr + i, acc);
        if (wantInputGrad) {
          v128_t wv = wasm_v128_load(wr + i);
          for (int k = 0; k < BLOCK; k++) {
            float* dst = gb_in + (long)k * inDim + i;
            wasm_v128_store(dst, wasm_f32x4_add(wasm_v128_load(dst), wasm_f32x4_mul(yv[k], wv)));
          }
        }
      }
      for (; i < inDim; i++) {
        float acc = dwr[i];
        for (int k = 0; k < BLOCK; k++) acc += y[k] * xb[(long)k * inDim + i];
        dwr[i] = acc;
        if (wantInputGrad) {
          for (int k = 0; k < BLOCK; k++) gb_in[(long)k * inDim + i] += y[k] * wr[i];
        }
      }
    }
  }
  for (int n = n8; n < batch; n++) {
    const float* gr = gy + (long)n * outDim;
    const float* xr = input + (long)n * inDim;
    float* gr_in = gIn + (long)n * inDim;
    for (int o = 0; o < outDim; o++) {
      float v = gr[o];
      if (v == 0.0f) continue;
      db[o] += v;
      const float* wr = w + (long)o * inDim;
      float* dwr = dw + (long)o * inDim;
      for (int i = 0; i < inDim; i++) {
        dwr[i] += v * xr[i];
        if (wantInputGrad) gr_in[i] += v * wr[i];
      }
    }
  }
}

/** Bump allocator over the module's heap; JS asks for every buffer up front. */
extern unsigned char __heap_base;
static unsigned long bump = 0;

__attribute__((export_name("kalloc")))
int kalloc(int bytes) {
  if (bump == 0) bump = ((unsigned long)&__heap_base + 15) & ~15UL;
  unsigned long p = bump;
  bump = (p + (unsigned long)bytes + 15) & ~15UL;
  return (int)p;
}

__attribute__((export_name("heap_used")))
int heap_used(void) { return (int)bump; }
