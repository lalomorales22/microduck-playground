// Where the trunk has to be for a given pose to have its feet on the floor.
//
// A motion file says nothing about the root — it is joint angles only — so the
// root trajectory has to be derived, and every place that shows or scores a
// motion has to derive it the SAME way. The rule: pose the model, find its
// lowest point, and offset the trunk so that point sits where it sits in the
// standing keyframe.
//
// Shared rather than reimplemented because the editor's viewport and the
// training reward both depend on it. If they disagreed, one of them would be
// lying about where the duck is supposed to be and there would be no way to
// tell which.

import type { MjData, MjModel, Mujoco } from "../sim/mujoco.ts";

export class GroundReference {
  /** Trunk height in the standing keyframe. */
  readonly standZ: number;
  /** Lowest robot geom in that same pose. */
  readonly #standLow: number;
  readonly #floorGeom: number;
  readonly #mujoco: Mujoco;
  readonly #model: MjModel;

  constructor(mujoco: Mujoco, model: MjModel, data: MjData, standKey: number) {
    this.#mujoco = mujoco;
    this.#model = model;
    this.#floorGeom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "floor");
    mujoco.mj_resetDataKeyframe(model, data, standKey);
    mujoco.mj_forward(model, data);
    this.standZ = data.qpos[2];
    this.#standLow = this.lowest(data);
  }

  /** Lowest robot geom origin, floor excluded. Valid after mj_forward. */
  lowest(data: MjData): number {
    let min = Infinity;
    for (let g = 0; g < this.#model.ngeom; g++) {
      if (g === this.#floorGeom) continue;
      const z = data.geom_xpos[g * 3 + 2];
      if (z < min) min = z;
    }
    return min;
  }

  /**
   * Trunk height for the pose currently in `data`. Call after the joints are
   * written and mj_forward has run; writing the result to qpos[2] needs a
   * second mj_forward to take effect.
   */
  heightFor(data: MjData): number {
    return this.standZ + (this.#standLow - this.lowest(data));
  }

  /** Write the joints, settle the trunk, and leave the model consistent. */
  poseAt(
    data: MjData, standKey: number, qposAdr: readonly number[], pose: Float32Array,
    vel?: Float32Array, dofAdr?: readonly number[],
  ): void {
    const { mujoco, model } = { mujoco: this.#mujoco, model: this.#model };
    mujoco.mj_resetDataKeyframe(model, data, standKey);
    data.xfrc_applied.fill(0);
    for (let j = 0; j < qposAdr.length; j++) {
      data.qpos[qposAdr[j]] = pose[j];
      data.ctrl[j] = pose[j];
      if (vel && dofAdr) data.qvel[dofAdr[j]] = vel[j];
    }
    mujoco.mj_forward(model, data);
    const height = this.heightFor(data);
    if (Number.isFinite(height)) {
      data.qpos[2] = height;
      mujoco.mj_forward(model, data);
    }
  }
}
