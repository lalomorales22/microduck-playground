// Evaluating a motion at an arbitrary time.
//
// Interpolation happens in JOINT SPACE, never in Cartesian space: these are
// hinge angles, and blending end-effector positions instead would need IK and
// would produce joint trajectories nobody authored.
//
// Cubic is the default because a linear pose track has a velocity
// discontinuity at every keyframe, and the imitation reward compares
// velocities — a policy would be punished for failing to reproduce a step
// change no actuator can produce. The tangents are the non-uniform
// Catmull-Rom ones, so unevenly spaced keyframes behave.

import { NUM_JOINTS } from "../sim/microduck.ts";
import type { Motion } from "./format.ts";

export class MotionSampler {
  readonly motion: Motion;
  readonly duration: number;
  readonly loop: boolean;

  /** Keyframe times, ascending, [0] === 0. */
  readonly #times: Float64Array;
  /** [keyframe][joint] flattened. */
  readonly #poses: Float32Array;
  /** Slope in rad/s at each keyframe, same layout. Cubic only. */
  readonly #tangents: Float32Array;
  readonly #count: number;
  readonly #cubic: boolean;

  constructor(motion: Motion) {
    this.motion = motion;
    this.duration = motion.duration;
    this.loop = motion.loop;
    this.#count = motion.keyframes.length;
    this.#cubic = motion.interp === "cubic" && this.#count > 1;

    this.#times = new Float64Array(this.#count);
    this.#poses = new Float32Array(this.#count * NUM_JOINTS);
    motion.keyframes.forEach((k, i) => {
      this.#times[i] = k.t;
      this.#poses.set(k.pose, i * NUM_JOINTS);
    });
    this.#tangents = new Float32Array(this.#count * NUM_JOINTS);
    if (this.#cubic) this.#buildTangents();
  }

  /**
   * Central differences, with the ends handled by whichever rule the motion's
   * topology implies: a loop wraps to the far side of the seam (the last
   * keyframe repeats the first, so its neighbour is index count-2 one period
   * back), an open motion uses a one-sided difference, which starts and ends
   * the track at the velocity it is actually moving rather than at zero.
   */
  #buildTangents(): void {
    const n = this.#count;
    const t = this.#times;
    const p = this.#poses;
    const period = this.duration;
    for (let i = 0; i < n; i++) {
      let prev = i - 1, next = i + 1;
      let tPrev = prev >= 0 ? t[prev] : 0, tNext = next < n ? t[next] : 0;
      if (prev < 0) {
        if (this.loop) { prev = n - 2; tPrev = t[n - 2] - period; }
        else { prev = 0; tPrev = t[0]; }
      }
      if (next >= n) {
        if (this.loop) { next = 1; tNext = t[1] + period; }
        else { next = n - 1; tNext = t[n - 1]; }
      }
      const dt = tNext - tPrev;
      for (let j = 0; j < NUM_JOINTS; j++) {
        this.#tangents[i * NUM_JOINTS + j] =
          dt > 0 ? (p[next * NUM_JOINTS + j] - p[prev * NUM_JOINTS + j]) / dt : 0;
      }
    }
  }

  /**
   * Time folded into the motion's own timeline.
   *
   * A looping motion wraps; an open one holds its final pose, which is what
   * makes "play the bow, then stand there" the default rather than a snap back
   * to the start.
   */
  localTime(t: number): number {
    if (!Number.isFinite(t)) return 0;
    if (this.duration <= 0) return 0;
    if (!this.loop) return Math.min(Math.max(t, 0), this.duration);
    const wrapped = t % this.duration;
    return wrapped < 0 ? wrapped + this.duration : wrapped;
  }

  /** Position within the motion, 0..1. Feeds the policy's phase observation. */
  phase(t: number): number {
    return this.duration > 0 ? this.localTime(t) / this.duration : 0;
  }

  /** Index of the segment containing `local`, i.e. times[i] <= local. */
  #segment(local: number): number {
    const t = this.#times;
    let lo = 0, hi = this.#count - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (t[mid] <= local) lo = mid; else hi = mid - 1;
    }
    return Math.min(lo, this.#count - 2);
  }

  /** Joint angles at time `t`, radians, policy order. */
  poseAt(t: number, out: Float32Array): Float32Array {
    if (this.#count === 1) {
      out.set(this.#poses.subarray(0, NUM_JOINTS));
      return out;
    }
    const local = this.localTime(t);
    const i = this.#segment(local);
    const t0 = this.#times[i], t1 = this.#times[i + 1];
    const h = t1 - t0;
    const u = h > 0 ? (local - t0) / h : 0;
    const a = i * NUM_JOINTS, b = (i + 1) * NUM_JOINTS;
    if (!this.#cubic) {
      for (let j = 0; j < NUM_JOINTS; j++) {
        out[j] = this.#poses[a + j] + (this.#poses[b + j] - this.#poses[a + j]) * u;
      }
      return out;
    }
    const u2 = u * u, u3 = u2 * u;
    const h00 = 2 * u3 - 3 * u2 + 1;
    const h10 = u3 - 2 * u2 + u;
    const h01 = -2 * u3 + 3 * u2;
    const h11 = u3 - u2;
    for (let j = 0; j < NUM_JOINTS; j++) {
      out[j] = h00 * this.#poses[a + j] + h10 * h * this.#tangents[a + j]
        + h01 * this.#poses[b + j] + h11 * h * this.#tangents[b + j];
    }
    return out;
  }

  /**
   * Joint velocities at time `t`, rad/s — the analytic derivative of the same
   * curve, not a finite difference, so it stays exact at any step size.
   *
   * Zero outside an open motion's span: it is being held, not moving.
   */
  velAt(t: number, out: Float32Array): Float32Array {
    out.fill(0);
    if (this.#count === 1 || this.duration <= 0) return out;
    if (!this.loop && (t <= 0 || t >= this.duration)) return out;
    const local = this.localTime(t);
    const i = this.#segment(local);
    const t0 = this.#times[i], t1 = this.#times[i + 1];
    const h = t1 - t0;
    if (h <= 0) return out;
    const u = (local - t0) / h;
    const a = i * NUM_JOINTS, b = (i + 1) * NUM_JOINTS;
    if (!this.#cubic) {
      for (let j = 0; j < NUM_JOINTS; j++) {
        out[j] = (this.#poses[b + j] - this.#poses[a + j]) / h;
      }
      return out;
    }
    const u2 = u * u;
    const d00 = 6 * u2 - 6 * u;
    const d10 = 3 * u2 - 4 * u + 1;
    const d01 = -6 * u2 + 6 * u;
    const d11 = 3 * u2 - 2 * u;
    for (let j = 0; j < NUM_JOINTS; j++) {
      out[j] = (d00 * this.#poses[a + j] + d01 * this.#poses[b + j]) / h
        + d10 * this.#tangents[a + j] + d11 * this.#tangents[b + j];
    }
    return out;
  }
}
