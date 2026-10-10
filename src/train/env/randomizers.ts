// Seam 2 implementations.
//
// Browser-native ships NO randomizers — the list is empty and the env is
// deterministic given its seed. The registry exists so the sim2real set (CoM,
// head CoM, mass/inertia, armature, joint friction, encoder bias, IMU
// misalignment, velocity pushes) is a data change rather than a redesign.

import type { EnvContext, Randomizer } from "./seams.ts";

/** No perturbation at all. Browser-native default. */
export const NO_RANDOMIZERS: readonly Randomizer[] = [];

/**
 * Apply a randomizer set for one episode.
 *
 * Restore-then-apply is enforced HERE rather than trusted to each randomizer,
 * because the failure it prevents is invisible: perturbations that compound
 * across resets drift the model a little further every episode, and the run
 * just quietly gets worse. Upstream lost months to exactly this.
 */
export function applyRandomizers(
  randomizers: readonly Randomizer[],
  ctx: EnvContext,
  rng: () => number,
): void {
  for (const r of randomizers) r.restore(ctx);
  for (const r of randomizers) r.apply(ctx, rng);
}
