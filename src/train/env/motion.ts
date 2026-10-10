// The reference-motion task: perform an authored motion, on balance.
//
// This is the training half of plans/motion-authoring.md stage 1. The motion
// file supplies joint angles over time; everything else — where the trunk
// should be, when the episode ends, what the policy is told — is derived here.
//
// The actuator baseline is deliberately left at DEFAULT_POSE rather than being
// re-centred on the reference pose. Re-centring would make a zero action track
// the motion perfectly and the task nearly free, but the resulting policy
// would be a correction term that cannot run without the motion file beside
// it. Keeping the reference in the POLICY, driven by the phase observation, is
// what makes a trained motion a self-contained checkpoint like every other.

import { CTRL_DT, JOINT_NAMES, NUM_JOINTS } from "../../sim/microduck.ts";
import type { Motion } from "../../motion/format.ts";
import { MotionRef, motionObsPipeline, motionRewards, motionTerminations } from "./imitation.ts";
import type { EnvSpec } from "./vec-env.ts";

export interface MotionTaskOptions {
  episodeLengthS?: number;
  /** How far off vertical counts as fallen; 1 is upright, 0 is on its side. */
  minUpright?: number;
  /** Radians of joint noise added at reset, so the policy meets states either
   *  side of the reference rather than only the reference itself. */
  resetJitter?: number;
}

/** At least this long, so an episode covers a few cycles of a short loop. */
const MIN_EPISODE_S = 3.0;

/**
 * Episode length: a whole number of cycles for a loop, the motion plus a
 * settling beat for a one-shot.
 *
 * Whole cycles matter because episodes start at a random point in the clock
 * (VecEnv.stagger) — if the episode were 1.5 cycles long, the phases the
 * policy is trained on would not be uniform, and it would be systematically
 * worse at whichever part of the motion fell in the short tail.
 */
function episodeLength(motion: Motion): number {
  if (motion.duration <= 0) return MIN_EPISODE_S;
  if (!motion.loop) return motion.duration + 0.5;
  return motion.duration * Math.ceil(MIN_EPISODE_S / motion.duration);
}

export function motionSpec(motion: Motion, options: MotionTaskOptions = {}): EnvSpec {
  const ref = new MotionRef(motion);
  const jitter = options.resetJitter ?? 0.02;

  return {
    name: `motion:${motion.name}`,
    joints: JOINT_NAMES,
    rewards: motionRewards(ref, motion),
    terminations: motionTerminations(options.minUpright ?? 0.4),
    obsPipeline: motionObsPipeline(ref),
    episodeLengthS: options.episodeLengthS ?? episodeLength(motion),
    // Tracking is a stabilisation problem, not a discovery one: the reward is
    // dense and points somewhere from the first step, so this sits between
    // hold_pose's near-deterministic setting and stand-up's exploratory one.
    // The offsets a motion asks for are large (a bow moves the neck 0.6 rad),
    // so the noise has to be big enough to reach them.
    exploration: { initStd: 0.25, entropyCoef: 0.002, desiredKl: 0.02 },
    reset(ctx, rng, standKey) {
      // First call only; uses ctx.data as scratch, which the reset below
      // overwrites.
      ref.prepare(ctx, standKey);

      // ctx.time was chosen before this ran, so an episode can start anywhere
      // in the motion — reference-state initialisation. Without it the policy
      // only ever sees the motion's opening state and has to learn the whole
      // thing as a single chain from one place.
      const t = ctx.time;
      const pose = ref.poseAt(t);
      const vel = ref.velocityAt(t);

      ctx.mujoco.mj_resetDataKeyframe(ctx.model, ctx.data, standKey);
      const { qpos, qvel, ctrl } = ctx.data;
      for (let j = 0; j < NUM_JOINTS; j++) {
        const noise = jitter > 0 ? (rng() * 2 - 1) * jitter : 0;
        qpos[ctx.joints.qpos[j]] = pose[j] + noise;
        qvel[ctx.joints.dof[j]] = vel[j];
        ctrl[j] = pose[j];
      }
      // Drop the trunk to where the reference pose puts the feet on the floor,
      // or the duck starts either hovering or halfway through the ground.
      const height = ref.heightAt(t);
      if (Number.isFinite(height)) qpos[2] = height;
    },
  };
}

/** Motion clock length in control steps, for UI and diagnostics. */
export function motionSteps(motion: Motion): number {
  return Math.max(1, Math.round(motion.duration / CTRL_DT));
}
