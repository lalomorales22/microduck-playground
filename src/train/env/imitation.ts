// Reference-motion tracking: DeepMimic-style rewards over a motion file.
//
// The shape of the task is "reproduce these joint angles at these times, on a
// robot that still has to keep its balance". That splits cleanly into two
// halves, and both are needed:
//
//   * TRACKING terms compare the joints against the reference. On their own
//     they are farmable — lying on its side, the duck has no gravity load and
//     can hit any pose asked of it — which is why they never ship alone.
//   * ROOT terms pay for the trunk being where the motion implies. The height
//     target is not a constant: a squat is SUPPOSED to lower the trunk, so a
//     fixed STAND_Z would punish the motion for being performed correctly.
//     The reference height comes from posing the model at each phase and
//     measuring, so it tracks whatever the author wrote.
//
// The policy is NOT given the reference pose. It gets the phase (two slots of
// the command block, as sin/cos so the loop seam is continuous) and has to
// carry the motion itself — the same arrangement DeepMimic uses, and the one
// that leaves a deployed policy self-contained rather than needing the motion
// file shipped alongside it.

import { CMD_OFFSET, CTRL_DT, NUM_JOINTS } from "../../sim/microduck.ts";
import { MotionSampler } from "../../motion/sampler.ts";
import { GroundReference } from "../../motion/ground.ts";
import { drivenIndices, type Motion } from "../../motion/format.ts";
import type { RewardTerm, StepState, TerminationTerm } from "./rewards.ts";
import type { EnvContext, ObsPipeline } from "./seams.ts";

const sq = (x: number) => x * x;
const gaussian = (err: number, std: number) => Math.exp(-sq(err / std));
const tiltSq = (q: StepState["quat"]) => 2 * (q[1] * q[1] + q[2] * q[2]);

/**
 * The reference, sampled once per control step and shared by every term.
 *
 * Terms run back to back for the same environment at the same instant, so a
 * one-entry memo on `t` removes all but one evaluation of the spline per step.
 * Without it the pose track is re-evaluated three or four times for numbers
 * that cannot have changed.
 */
export class MotionRef {
  readonly sampler: MotionSampler;
  readonly pose = new Float32Array(NUM_JOINTS);
  readonly vel = new Float32Array(NUM_JOINTS);

  #lastT = NaN;
  /** Root height per control step of the motion; see prepare(). */
  #heights: Float64Array | null = null;

  constructor(motion: Motion) {
    this.sampler = new MotionSampler(motion);
  }

  /** Reference joint angles at `t`, radians, policy order. */
  poseAt(t: number): Float32Array {
    if (t !== this.#lastT) {
      this.sampler.poseAt(t, this.pose);
      this.sampler.velAt(t, this.vel);
      this.#lastT = t;
    }
    return this.pose;
  }

  velocityAt(t: number): Float32Array {
    this.poseAt(t);
    return this.vel;
  }

  /**
   * Measure what the motion implies for the trunk's height, once.
   *
   * The motion file says nothing about the root — it is joint angles only — so
   * the root trajectory is derived: pose the model at each phase, find how far
   * the lowest point of the robot moved relative to the standing pose, and
   * offset the trunk by the same amount. That is the height at which the feet
   * stay on the ground through a squat or a bow.
   *
   * Uses the environment's own MjData as scratch. Safe because the only caller
   * is reset(), which overwrites the state immediately afterwards — and doing
   * it this way avoids a second MjData per worker, which at ~13 MB each is the
   * binding constraint on environment count.
   */
  prepare(ctx: EnvContext, standKey: number): void {
    if (this.#heights) return;
    const { mujoco, model, data, joints } = ctx;
    // The same rule the editor's viewport uses — see GroundReference.
    const ground = new GroundReference(mujoco, model, data, standKey);

    const steps = Math.max(1, Math.round(this.sampler.duration / CTRL_DT)) + 1;
    const heights = new Float64Array(steps);
    const scratch = new Float32Array(NUM_JOINTS);
    for (let s = 0; s < steps; s++) {
      mujoco.mj_resetDataKeyframe(model, data, standKey);
      this.sampler.poseAt(s * CTRL_DT, scratch);
      for (let j = 0; j < NUM_JOINTS; j++) data.qpos[joints.qpos[j]] = scratch[j];
      mujoco.mj_forward(model, data);
      heights[s] = ground.heightFor(data);
    }
    this.#heights = heights;
    this.#lastT = NaN; // the memo held a scratch sample, not a real one
  }

  /** Trunk height the reference pose at `t` implies, metres. */
  heightAt(t: number): number {
    const table = this.#heights;
    if (!table) return NaN;
    const index = Math.round(this.sampler.localTime(t) / CTRL_DT);
    return table[Math.min(Math.max(index, 0), table.length - 1)];
  }
}

export interface MotionRewardOptions {
  /** How closely the joints must match, radians. Larger is more forgiving. */
  poseStd?: number;
  /** How closely joint velocities must match, rad/s. */
  velStd?: number;
  /** Height tolerance around the reference root height, metres. */
  heightStd?: number;
}

/**
 * The tracking stack.
 *
 * Scored over the DRIVEN joints only, with the rest held by a separate,
 * lighter term. Averaging over all fourteen instead looks tidier and is much
 * worse: a nod drives one joint, so thirteen terms would match their reference
 * no matter what the policy did, and the one being learned would contribute a
 * fourteenth of a reward that is already mostly constant. Measured on the
 * squat, splitting them roughly quadrupled the gap between performing the
 * motion and ignoring it.
 *
 * Weights follow the house convention in rewards.ts: self-negating penalties
 * return <= 0 and take a POSITIVE weight; mjlab-style costs return >= 0 and
 * take a negative one.
 */
export function motionRewards(
  ref: MotionRef, motion: Motion, options: MotionRewardOptions = {},
): RewardTerm[] {
  const poseStd = options.poseStd ?? 0.30;
  const velStd = options.velStd ?? 5.0;
  const heightStd = options.heightStd ?? 0.03;
  const driven = drivenIndices(motion);
  const rest: number[] = [];
  for (let j = 0; j < NUM_JOINTS; j++) if (!driven.includes(j)) rest.push(j);

  const terms: RewardTerm[] = [
    // Per joint then averaged, not a Gaussian over the summed error: with the
    // sum, one badly wrong joint saturates the exponential and the other
    // thirteen stop producing gradient.
    {
      name: "motion_pose",
      weight: 5.0,
      compute: (ctx, s) => {
        const q = ref.poseAt(ctx.time);
        let acc = 0;
        for (const j of driven) acc += gaussian(s.jointPos[j] - q[j], poseStd);
        return acc / driven.length;
      },
    },
    // Self-negating, so it keeps pulling after the Gaussian above has
    // saturated. Same role height_stand_l1 plays for the standing stack.
    {
      name: "motion_pose_l1",
      weight: 2.5,
      compute: (ctx, s) => {
        const q = ref.poseAt(ctx.time);
        let acc = 0;
        for (const j of driven) acc += Math.abs(s.jointPos[j] - q[j]);
        return -acc / driven.length;
      },
    },
    // Velocity matters for TIMING: a policy that hits every pose but arrives
    // early sits at a pose error of zero for part of the cycle and is only
    // caught here.
    {
      name: "motion_vel",
      weight: 0.5,
      compute: (ctx) => {
        const qd = ref.velocityAt(ctx.time);
        const { qvel } = ctx.data;
        let acc = 0;
        for (const j of driven) acc += gaussian(qvel[ctx.joints.dof[j]] - qd[j], velStd);
        return acc / driven.length;
      },
    },
    // The root half. Without these the tracking terms are farmable by lying
    // down, where the joints carry no load and every pose is easy.
    {
      name: "motion_height",
      weight: 2.0,
      compute: (ctx, s) => {
        const target = ref.heightAt(ctx.time);
        return Number.isFinite(target) ? gaussian(s.z - target, heightStd) : 0;
      },
    },
    {
      name: "upright_linear",
      weight: 1.5,
      compute: (_c, s) => 1 - tiltSq(s.quat),
    },
    {
      name: "action_rate_l2",
      weight: -0.1,
      compute: (_c, s) => {
        let acc = 0;
        for (let j = 0; j < NUM_JOINTS; j++) acc += sq(s.action[j] - s.prevAction[j]);
        return acc;
      },
    },
  ];

  // Joints the file never mentions still have to behave: their reference is
  // DEFAULT_POSE, and letting them wander is both ugly and a way to cheat the
  // balance terms. Light weight — this is housekeeping, not the task.
  if (rest.length > 0) {
    terms.push({
      name: "pose_hold_rest",
      weight: 1.0,
      compute: (ctx, s) => {
        const q = ref.poseAt(ctx.time);
        let acc = 0;
        for (const j of rest) acc += gaussian(s.jointPos[j] - q[j], 0.4);
        return acc / rest.length;
      },
    });
  }
  return terms;
}

/**
 * End the episode once the duck is on its way over.
 *
 * Early termination is doing real work here, not just saving time: without it
 * a large share of every rollout is spent lying on the floor, where the
 * tracking terms are easy and the behaviour being reinforced is not the one
 * anybody wanted.
 */
export function motionTerminations(minUpright = 0.4): TerminationTerm[] {
  return [
    {
      name: "nan_state",
      check: (_c, s) => !Number.isFinite(s.z) || !Number.isFinite(s.quat[0]),
    },
    {
      name: "fallen",
      check: (_c, s) => 1 - tiltSq(s.quat) < minUpright,
    },
  ];
}

/**
 * Tell the policy where in the motion it is.
 *
 * sin/cos rather than a raw 0..1 ramp: a looping motion's phase jumps from 1
 * back to 0, and a network fed that discontinuity has to learn a matching
 * discontinuity in its output. Two slots of the thirteen; the rest stay zero,
 * so the observation is still the 61-float interface every other policy uses.
 */
export function motionObsPipeline(ref: MotionRef): ObsPipeline {
  return {
    name: "motion_phase",
    reset(): void {
      /* the phase comes from the episode clock, which VecEnv owns */
    },
    apply(ctx: EnvContext, obs: Float32Array, offset: number): void {
      const angle = ref.sampler.phase(ctx.time) * Math.PI * 2;
      obs[offset + CMD_OFFSET] = Math.sin(angle);
      obs[offset + CMD_OFFSET + 1] = Math.cos(angle);
    },
  };
}
