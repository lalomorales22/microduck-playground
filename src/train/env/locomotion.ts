import { CMD_OFFSET, DEFAULT_POSE, GYRO_SENSOR, JOINT_NAMES, NUM_JOINTS } from "../../sim/microduck.ts";
import type { EnvContext } from "./seams.ts";
import { STAND_Z, type StepState } from "./rewards.ts";
import type { EnvSpec } from "./vec-env.ts";

export type LocomotionKind = "walk_forward" | "walk_backward" | "turn_left" | "turn_right";
export const LOCOMOTION_TARGETS: Record<LocomotionKind, { forward: number; yaw: number }> = {
  walk_forward: { forward: 0.12, yaw: 0 },
  walk_backward: { forward: -0.10, yaw: 0 },
  turn_left: { forward: 0, yaw: 0.8 },
  turn_right: { forward: 0, yaw: -0.8 },
};

export function locomotionVelocity(ctx: EnvContext, s: StepState) {
  const [w, x, y, z] = s.quat;
  const hx = 1 - 2 * (y * y + z * z), hy = 2 * (x * y + w * z);
  const norm = Math.max(1e-6, Math.hypot(hx, hy));
  const vx = ctx.data.qvel[0], vy = ctx.data.qvel[1];
  return {
    forward: (hx * vx + hy * vy) / norm,
    lateral: (-hy * vx + hx * vy) / norm,
    yaw: ctx.data.sensordata[ctx.model.sensor(GYRO_SENSOR).adr + 2],
  };
}

const uprightGate = (s: StepState) => Math.max(0, Math.min(1, (1 - 2 * (s.quat[1] ** 2 + s.quat[2] ** 2) - 0.25) / 0.75));

export function velocityTracking(kind: LocomotionKind, ctx: EnvContext, s: StepState): number {
  const goal = LOCOMOTION_TARGETS[kind], v = locomotionVelocity(ctx, s);
  return uprightGate(s) * Math.exp(-(((v.forward - goal.forward) / 0.12) ** 2)
    - (v.lateral / 0.12) ** 2 - ((v.yaw - goal.yaw) / 0.7) ** 2);
}

export function writeLocomotionCommand(kind: LocomotionKind, _time: number, obs: Float32Array, offset = 0): void {
  const goal = LOCOMOTION_TARGETS[kind];
  obs[offset + CMD_OFFSET] = goal.forward;
  obs[offset + CMD_OFFSET + 1] = 0;
  obs[offset + CMD_OFFSET + 2] = goal.yaw;
}

export function locomotionSpec(kind: LocomotionKind, options: { episodeLengthS?: number } = {}): EnvSpec {
  const episodeLengthS = options.episodeLengthS ?? 5;
  if (!LOCOMOTION_TARGETS[kind]) throw new Error("unknown locomotion task");
  if (!Number.isFinite(episodeLengthS) || episodeLengthS < 0.1 || episodeLengthS > 60) throw new Error("invalid episode length");
  return {
    name: kind, joints: JOINT_NAMES, episodeLengthS,
    exploration: { initStd: 0.25, entropyCoef: 0.002, desiredKl: 0.02 },
    obsPipeline: { name: "locomotion_command", reset() {}, apply: (ctx, obs, offset) => writeLocomotionCommand(kind, ctx.time, obs, offset) },
    reset(ctx, _rng, standKey) {
      ctx.mujoco.mj_resetDataKeyframe(ctx.model, ctx.data, standKey);
      for (let j = 0; j < NUM_JOINTS; j++) ctx.data.ctrl[j] = DEFAULT_POSE[j];
    },
    rewards: [
      { name: "velocity_tracking", weight: 5, compute: (ctx, s) => velocityTracking(kind, ctx, s) },
      { name: "directed_progress", weight: 1, compute: (ctx, s) => {
        const goal = LOCOMOTION_TARGETS[kind], v = locomotionVelocity(ctx, s);
        const progress = goal.forward ? v.forward / goal.forward : v.yaw / goal.yaw;
        return uprightGate(s) * Math.max(-1, Math.min(1, progress));
      } },
      { name: "root_height", weight: 1, compute: (_c, s) => Math.exp(-(((s.z - STAND_Z) / 0.04) ** 2)) },
      { name: "upright", weight: 1.5, compute: (_c, s) => 1 - 2 * (s.quat[1] ** 2 + s.quat[2] ** 2) },
      { name: "action_rate", weight: -0.02, compute: (_c, s) => s.action.reduce((sum, x, j) => sum + Math.min(4, (x - s.prevAction[j]) ** 2), 0) },
      { name: "joint_speed", weight: -0.001, compute: (ctx) => ctx.joints.dof.reduce((sum, j) => sum + Math.min(400, ctx.data.qvel[j] ** 2), 0) },
    ],
    terminations: [
      { name: "non_finite", check: (_c, s) => !Number.isFinite(s.z) || Array.from(s.quat).some(x => !Number.isFinite(x)) },
      { name: "fallen", check: (_c, s) => s.z < 0.05 || uprightGate(s) === 0 },
    ],
    metrics: [
      { name: "forward_velocity_m_s", compute: (ctx, s) => locomotionVelocity(ctx, s).forward },
      { name: "yaw_rate_rad_s", compute: (ctx, s) => locomotionVelocity(ctx, s).yaw },
      { name: "velocity_tracking", compute: (ctx, s) => velocityTracking(kind, ctx, s) },
    ],
  };
}
