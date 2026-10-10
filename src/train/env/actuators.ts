// Seam 1 implementations.

import { ACTION_SCALE, DEFAULT_POSE, NUM_JOINTS } from "../../sim/microduck.ts";
import type { Actuator, EnvContext } from "./seams.ts";

/**
 * Browser-native default: hand the targets to the MJCF's own `position`
 * actuators (kp 0.55 on this model) and let MuJoCo run the PD loop.
 *
 * This is what `src/sim/controller.ts` does for the viewport, and what a
 * browser-trained policy would be tuned against — which is exactly why such a
 * policy is not sim2real-ready. The real servo is a voltage-controlled XL330
 * whose torque depends on battery sag and a load-dependent friction budget;
 * that lands here later as `BamActuator`.
 */
export class XmlPositionActuator implements Actuator {
  readonly name = "xml_position";

  reset(): void {
    /* stateless */
  }

  apply(ctx: EnvContext, targets: Float32Array): void {
    const ctrl = ctx.data.ctrl;
    for (let j = 0; j < NUM_JOINTS; j++) {
      ctrl[j] = DEFAULT_POSE[j] + targets[j] * ACTION_SCALE;
    }
  }
}

/**
 * Wraps any actuator in the control-step lag the real firmware shows.
 *
 * Built now even though browser-native uses lag 0: the reference config runs
 * `delay_min_lag=3, delay_max_lag=6` and a policy trained at one lag and
 * deployed at another transfers badly, so the plumbing should not be invented
 * under time pressure later. The lag is sampled per episode, matching mjlab's
 * DelayedActuator rather than being a fixed constant.
 */
export class DelayedActuator implements Actuator {
  readonly name: string;
  /** Ring of past target vectors, newest at `head`. */
  #buffer: Float32Array[];
  #head = 0;
  #lag = 0;

  readonly #inner: Actuator;
  readonly #minLag: number;
  readonly #maxLag: number;

  constructor(inner: Actuator, minLag: number, maxLag: number) {
    this.#inner = inner;
    this.#minLag = minLag;
    this.#maxLag = maxLag;
    this.name = `delayed(${inner.name},${minLag}-${maxLag})`;
    this.#buffer = Array.from({ length: maxLag + 1 }, () => new Float32Array(NUM_JOINTS));
  }

  reset(ctx: EnvContext, rng: () => number = Math.random): void {
    this.#lag = this.#minLag + Math.floor(rng() * (this.#maxLag - this.#minLag + 1));
    for (const slot of this.#buffer) slot.fill(0);
    this.#head = 0;
    this.#inner.reset(ctx);
  }

  apply(ctx: EnvContext, targets: Float32Array): void {
    this.#head = (this.#head + 1) % this.#buffer.length;
    this.#buffer[this.#head].set(targets);
    const tail = (this.#head - this.#lag + this.#buffer.length) % this.#buffer.length;
    this.#inner.apply(ctx, this.#buffer[tail]);
  }
}
