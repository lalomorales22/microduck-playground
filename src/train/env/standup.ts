// The stand-up task: get the duck from a collapsed pose onto its feet.
//
// Reward stack and constants come from
// microduck_rl/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py. The
// reset distribution is deliberately broader than the reference's: that task
// starts from a fixed sitting keyframe, while `alpha_stand` is deployed as a
// general fall-recovery policy and the viewport demo drops it in arbitrary
// tumbled poses. Training on the wider distribution is closer to how the
// policy is actually used, and it makes the M1 replay gate meaningful.

import { DEFAULT_POSE, JOINT_NAMES } from "../../sim/microduck.ts";
import type { EnvSpec } from "./vec-env.ts";
import { holdPoseRewards, standupRewards, standupTerminations } from "./rewards.ts";
import type { EnvContext } from "./seams.ts";

/**
 * Sitting pose the reference task resets to — the measured end-state of the
 * sit policy, so the pair forms a clean sit -> stand hand-off. Indices are
 * into the 14-joint policy order.
 */
const SITTING_OVERRIDES: Record<number, number> = {
  1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
  10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0,
};

/** Drop the duck tilted past horizontal about a random ground-plane axis. */
function tumble(ctx: EnvContext, rng: () => number): void {
  const axis = rng() * Math.PI * 2;
  const tilt = (Math.PI / 2) * (1 + rng() * 0.7); // 90-153 degrees
  const s = Math.sin(tilt / 2);
  const qpos = ctx.data.qpos;
  qpos[2] = 0.12;
  qpos[3] = Math.cos(tilt / 2);
  qpos[4] = Math.cos(axis) * s;
  qpos[5] = Math.sin(axis) * s;
  qpos[6] = 0;
}

/** Seat the duck: legs folded, trunk low, upright. */
function sit(ctx: EnvContext): void {
  const qpos = ctx.data.qpos;
  qpos[2] = 0.07;
  qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;
  for (const [idx, value] of Object.entries(SITTING_OVERRIDES)) {
    qpos[ctx.joints.qpos[Number(idx)]] = value;
  }
}

/** Relative weights for how an episode starts. */
export interface ResetMix {
  /** Already standing at the reference pose — the easy case. */
  stand?: number;
  /** Folded into the seated equilibrium the sit policy converges to. */
  sit?: number;
  /** Dropped tilted past horizontal, landing however physics decides. */
  tumble?: number;
}

export interface StandupOptions {
  resetMix?: ResetMix;
  episodeLengthS?: number;
  name?: string;
}

/**
 * "Hold the pose": start standing and stay there.
 *
 * The bootstrap task — trivially learnable, and the first thing to run a new
 * learner against, because a policy that cannot hold a stable equilibrium it
 * was handed will certainly not discover how to reach one.
 */
export function holdPoseSpec(options: StandupOptions = {}): EnvSpec {
  return {
    ...standupSpec({
      name: "hold_pose",
      resetMix: { stand: 1 },
      episodeLengthS: options.episodeLengthS ?? 3.0,
      ...options,
    }),
    // No motion rewards: this task starts where it should stay, and
    // com_upward_velocity would pay it to bounce. See standupRewards().
    rewards: holdPoseRewards(),
    // Stabilisation: the duck starts where it needs to stay, so large action
    // noise destroys the behaviour before PPO can reinforce it. Small sigma
    // forces a proportionally larger KL target — see ExplorationHints.
    exploration: { initStd: 0.1, entropyCoef: 0, desiredKl: 0.05 },
  };
}

export function standupSpec(options: StandupOptions = {}): EnvSpec {
  const mix = options.resetMix ?? { sit: 1, tumble: 1 };
  const stand = mix.stand ?? 0;
  const sitW = mix.sit ?? 0;
  const tumbleW = mix.tumble ?? 0;
  const total = stand + sitW + tumbleW;
  if (total <= 0) throw new Error("resetMix must have a positive weight somewhere");
  return {
    name: options.name ?? "standup",
    // Discovery: the rise has to be stumbled into before it can be refined,
    // so this keeps the reference recipe's exploration rather than hold-pose's.
    exploration: { initStd: 1.0, entropyCoef: 0.01, desiredKl: 0.01 },
    joints: JOINT_NAMES,
    rewards: standupRewards(),
    terminations: standupTerminations(),
    // 6 s: long enough for a gentle rise plus a beat of stabilisation, which
    // is what the reference task allows.
    episodeLengthS: options.episodeLengthS ?? 6.0,
    reset(ctx, rng, standKey) {
      ctx.mujoco.mj_resetDataKeyframe(ctx.model, ctx.data, standKey);
      // ctrl starts at the reference pose so an un-acted first step is neutral.
      for (let j = 0; j < DEFAULT_POSE.length; j++) ctx.data.ctrl[j] = DEFAULT_POSE[j];
      const pick = rng() * total;
      if (pick < stand) return; // the keyframe already IS the standing pose
      if (pick < stand + sitW) sit(ctx);
      else tumble(ctx, rng);
    },
  };
}
