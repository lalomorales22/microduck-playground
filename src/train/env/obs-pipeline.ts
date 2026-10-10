// Seam 3 implementations.

import type { EnvContext, ObsPipeline } from "./seams.ts";

/**
 * Browser-native: the policy sees exactly what the simulator knows.
 *
 * The real robot does not. Its encoders carry a per-joint bias, its IMU is
 * misaligned by a few degrees, and every term is noisy — all of which the
 * reference recipe randomizes. Those land here, and the AGENTS.md rule comes
 * with them: if an observation is remapped, any reward tracking the same
 * quantity must measure the same view, or the policy is punished for
 * correcting what it can see.
 */
export const identityObsPipeline: ObsPipeline = {
  name: "identity",
  reset(): void {
    /* stateless */
  },
  apply(_ctx: EnvContext, _obs: Float32Array, _offset: number): void {
    /* the clean observation is the observation */
  },
};
