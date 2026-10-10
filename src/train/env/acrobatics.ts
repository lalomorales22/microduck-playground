// Experimental one-shot objectives, not evidence of a completed aerial trick.
// Targets depend only on the episode clock so playback can reproduce the
// command without adding privileged simulator observations to the policy.

import { CMD_OFFSET, CMD_SIZE, DEFAULT_POSE, JOINT_NAMES, NUM_JOINTS } from "../../sim/microduck.ts";
import { LEG_JOINTS, STAND_Z } from "./rewards.ts";
import type { StepState } from "./rewards.ts";
import type { EnvSpec } from "./vec-env.ts";

export type AcrobaticKind = "jump" | "somersault" | "backflip";
export const ACROBATICS_DURATION_S = 1.2;
const PREPARE_END = 0.3;
const LAUNCH_END = 0.45;
const FLIGHT_END = 0.8;
const CROUCH_Z = STAND_Z - 0.025;
const TARGET_RISE = 0.15;
const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const sq = (value: number) => value * value;
const smooth = (value: number) => value * value * (3 - 2 * value);
const gaussian = (error: number, width: number) => Math.exp(-sq(error / width));

export interface AcrobaticTarget {
  phase: "prepare" | "launch" | "flight" | "settle";
  progress: number;
  direction: -1 | 0 | 1;
  pitch: number;
  height: number;
  verticalVelocity: number;
}

export function acrobaticTarget(kind: AcrobaticKind, time: number): AcrobaticTarget {
  if (kind !== "jump" && kind !== "somersault" && kind !== "backflip") {
    throw new Error(`Unknown acrobatics task: ${String(kind)}`);
  }
  if (!Number.isFinite(time)) throw new Error("Acrobatics time must be finite");
  const t = Math.max(0, time);
  // The robot faces +X: positive right-handed pitch about +Y tips it forward.
  const direction = kind === "jump" ? 0 : kind === "somersault" ? 1 : -1;
  const progress = clamp01(t / ACROBATICS_DURATION_S);
  if (t < PREPARE_END) {
    const u = t / PREPARE_END;
    return { phase: "prepare", progress, direction, pitch: 0,
      height: STAND_Z + (CROUCH_Z - STAND_Z) * smooth(u), verticalVelocity: 0 };
  }
  if (t < LAUNCH_END) {
    const u = (t - PREPARE_END) / (LAUNCH_END - PREPARE_END);
    return { phase: "launch", progress, direction, pitch: 0,
      height: CROUCH_Z + (STAND_Z - CROUCH_Z) * smooth(u), verticalVelocity: 0 };
  }
  if (t < FLIGHT_END) {
    const u = (t - LAUNCH_END) / (FLIGHT_END - LAUNCH_END);
    // An idealised shaping target, not an imposed ballistic trajectory: its
    // initial velocity intentionally steps up at the scheduled takeoff.
    return { phase: "flight", progress, direction, pitch: direction * 2 * Math.PI * smooth(u),
      height: STAND_Z + TARGET_RISE * 4 * u * (1 - u),
      verticalVelocity: TARGET_RISE * 4 * (1 - 2 * u) / (FLIGHT_END - LAUNCH_END) };
  }
  return { phase: "settle", progress, direction, pitch: direction * 2 * Math.PI,
    height: STAND_Z, verticalVelocity: 0 };
}

/** Squared quaternion overlap: q and -q are the same physical orientation.
 * A final upright pose alone is NOT proof of a full rotation. Tracking is
 * scored throughout the directed reference, including its quarter turns. */
export function pitchTracking(quat: ArrayLike<number>, pitch: number): number {
  const norm = Math.hypot(quat[0], quat[1], quat[2], quat[3]);
  if (!Number.isFinite(norm) || norm < 1e-9 || !Number.isFinite(pitch)) return 0;
  const dot = (quat[0] * Math.cos(pitch / 2) + quat[2] * Math.sin(pitch / 2)) / norm;
  return clamp01(dot * dot);
}

/** Coupled tracking prevents a low, inverted pose collecting orientation
 * credit without also reaching the bounded root-height trajectory. */
export function acrobaticRootScore(kind: AcrobaticKind, time: number, s: StepState): number {
  if (!Number.isFinite(s.z) || !Number.isFinite(s.vz)) return 0;
  const target = acrobaticTarget(kind, time);
  const height = gaussian(s.z - target.height, 0.045);
  const pitch = pitchTracking(s.quat, target.pitch);
  // Crouching and launching are unconstrained in speed; penalising their
  // impulse would fight discovery. Descent and settling still have targets.
  const velocity = target.phase === "flight" || target.phase === "settle"
    ? gaussian(s.vz - target.verticalVelocity, 0.9) : 1;
  return height * sq(pitch) * velocity;
}

/** The 13 command slots are fixed goals, never measured height/contact state.
 * Slots: clock sin/cos, height offset, vz, pitch sin/cos, direction,
 * prepare/launch/flight/settle one-hot, progress, jump flag. */
export function writeAcrobaticCommand(
  kind: AcrobaticKind, time: number, obs: Float32Array, offset = 0,
): void {
  if (!Number.isInteger(offset) || offset < 0 || offset + CMD_OFFSET + CMD_SIZE > obs.length) {
    throw new Error("Acrobatics observation row is too short or has an invalid offset");
  }
  const t = acrobaticTarget(kind, time);
  const start = offset + CMD_OFFSET;
  obs[start] = Math.sin(2 * Math.PI * t.progress);
  obs[start + 1] = Math.cos(2 * Math.PI * t.progress);
  obs[start + 2] = t.height - STAND_Z;
  obs[start + 3] = t.verticalVelocity;
  obs[start + 4] = Math.sin(t.pitch);
  obs[start + 5] = Math.cos(t.pitch);
  obs[start + 6] = t.direction;
  obs[start + 7] = Number(t.phase === "prepare");
  obs[start + 8] = Number(t.phase === "launch");
  obs[start + 9] = Number(t.phase === "flight");
  obs[start + 10] = Number(t.phase === "settle");
  obs[start + 11] = t.progress;
  obs[start + 12] = Number(kind === "jump");
}

export function acrobaticsSpec(
  kind: AcrobaticKind, options: { episodeLengthS?: number } = {},
): EnvSpec {
  acrobaticTarget(kind, 0);
  const episodeLengthS = options.episodeLengthS ?? 3;
  if (!Number.isFinite(episodeLengthS) || episodeLengthS < ACROBATICS_DURATION_S || episodeLengthS > 60) {
    throw new Error(`Acrobatics episodeLengthS must be finite and between ${ACROBATICS_DURATION_S} and 60`);
  }
  return {
    name: kind,
    joints: JOINT_NAMES,
    episodeLengthS,
    staggerReset: false,
    exploration: { initStd: 0.35, entropyCoef: 0.005, desiredKl: 0.02 },
    obsPipeline: {
      name: `acrobatics:${kind}`,
      reset() {},
      apply: (ctx, obs, offset) => writeAcrobaticCommand(kind, ctx.time, obs, offset),
    },
    metrics: [
      { name: "root_height_m", compute: (_ctx, s) => s.z },
      { name: "pitch_tracking", compute: (ctx, s) => pitchTracking(s.quat, acrobaticTarget(kind, ctx.time).pitch) },
    ],
    rewards: [
      {
        name: "acrobatics_root_tracking", weight: 4,
        compute: (ctx, s) => acrobaticRootScore(kind, ctx.time, s) *
          (acrobaticTarget(kind, ctx.time).phase === "flight" ? 2 : 1),
      },
      {
        name: "landing_stability", weight: 2,
        compute: (ctx, s) => {
          if (ctx.time < FLIGHT_END) return 0;
          const v = ctx.data.qvel;
          const speedCost = (sq(v[0]) + sq(v[1])) / sq(0.3) +
            (sq(v[3]) + sq(v[4]) + sq(v[5])) / sq(2);
          if (!Number.isFinite(speedCost)) return 0;
          let poseError = 0;
          for (const j of LEG_JOINTS) poseError += sq(s.jointPos[j] - DEFAULT_POSE[j]);
          if (!Number.isFinite(poseError)) return 0;
          return acrobaticRootScore(kind, ctx.time, s) * Math.exp(-speedCost) *
            Math.exp(-poseError / LEG_JOINTS.length / sq(0.5));
        },
      },
      {
        name: "action_rate", weight: 0.01,
        compute: (_ctx, s) => {
          let cost = 0;
          for (let j = 0; j < NUM_JOINTS; j++) {
            const delta = s.action[j] - s.prevAction[j];
            cost += Number.isFinite(delta) ? Math.min(100, sq(delta)) : 100;
          }
          return -cost / NUM_JOINTS;
        },
      },
    ],
    terminations: [
      {
        name: "invalid_state",
        check: (ctx, s) => {
          for (const values of [ctx.data.qpos, ctx.data.qvel, s.quat, s.jointPos, s.action, s.prevAction]) {
            for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) return true;
          }
          const norm = Math.hypot(s.quat[0], s.quat[1], s.quat[2], s.quat[3]);
          return norm < 0.5 || norm > 1.5 || !Number.isFinite(s.z) || !Number.isFinite(s.vz) ||
            s.z < -0.05 || s.z > 0.75;
        },
      },
      {
        name: "failed_to_settle",
        // Inversion is intentional during a flip, never an in-flight failure.
        check: (ctx, s) => ctx.time >= ACROBATICS_DURATION_S &&
          (s.z < 0.045 || pitchTracking(s.quat, 0) < 0.5),
      },
    ],
    reset(ctx, _rng, standKey) {
      ctx.mujoco.mj_resetDataKeyframe(ctx.model, ctx.data, standKey);
      for (let j = 0; j < NUM_JOINTS; j++) ctx.data.ctrl[j] = DEFAULT_POSE[j];
    },
  };
}
