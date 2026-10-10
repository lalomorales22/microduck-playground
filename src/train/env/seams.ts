// The four seams between the browser-native trainer and a sim2real-faithful
// one. Each is an interface today with a trivial implementation, so filling it
// in later is additive rather than a rewrite. See docs/training-plan.md §3.
//
// Two contracts here are load-bearing, and both come from expensive failures
// recorded in microduck_rl/AGENTS.md:
//
//   * Randomizers RESTORE then apply. An accumulating CoM randomizer silently
//     degraded every long run for months upstream. Non-accumulation is the
//     interface's promise, not each implementation's good intentions.
//   * Nothing here may index joints by position. Backlash and roller models
//     interleave `passive_*` joints into qpos, so indices are resolved by name
//     once, in JointIndex, and passed around.

import type { MjData, MjModel, Mujoco } from "../../sim/mujoco.ts";

/** Joint addresses resolved by name — never assume qpos ordering. */
export interface JointIndex {
  /** qpos address per actuated joint, in policy order. */
  readonly qpos: readonly number[];
  /** qvel/dof address per actuated joint, in policy order. */
  readonly dof: readonly number[];
  readonly count: number;
}

export function resolveJoints(model: MjModel, names: readonly string[]): JointIndex {
  return {
    qpos: names.map((n) => model.jnt(n).qposadr),
    dof: names.map((n) => model.jnt(n).dofadr),
    count: names.length,
  };
}

/** Everything an env term is handed for one environment. */
export interface EnvContext {
  readonly mujoco: Mujoco;
  readonly model: MjModel;
  readonly data: MjData;
  readonly joints: JointIndex;
  /** Index within the vectorized batch, for per-env randomized parameters. */
  readonly envId: number;
  /**
   * Seconds since this episode started, at the instant the physics has
   * reached. Live: it advances between the actuator being applied and the
   * rewards being scored.
   *
   * A reset reads it too, and it is chosen BEFORE the pose is set, so a task
   * that starts mid-motion can pose itself for the phase it is about to be
   * scored at.
   */
  readonly time: number;
}

// ── Seam 1: actuators ────────────────────────────────────────────────────
/**
 * Turns the policy's joint-position targets into whatever the model's
 * actuators consume.
 *
 * `XmlPositionActuator` writes targets straight to `data.ctrl` and lets the
 * MJCF's `position` actuators run their own PD — that is browser-native.
 * `BamActuator` will replace it with the voltage-controlled XL330 model, which
 * is the single biggest step toward sim2real.
 */
export interface Actuator {
  readonly name: string;
  /** Fresh episode: drop any internal history (delay buffers, filters). */
  reset(ctx: EnvContext): void;
  /** Called once per control step, before the physics substeps. */
  apply(ctx: EnvContext, targets: Float32Array): void;
}

// ── Seam 2: domain randomization ─────────────────────────────────────────
/**
 * A per-environment parameter perturbation, re-sampled at reset.
 *
 * `restore` MUST return the model to its compiled defaults before `apply`
 * writes new values — otherwise successive episodes compound the perturbation
 * and the run silently drifts. `VecEnv` always calls restore-then-apply, so an
 * implementation that only implements `apply` correctly is still safe.
 */
export interface Randomizer {
  readonly name: string;
  /** Undo this randomizer's previous write. Called before every apply. */
  restore(ctx: EnvContext): void;
  apply(ctx: EnvContext, rng: () => number): void;
}

// ── Seam 3: observation pipeline ─────────────────────────────────────────
/**
 * Post-processes the clean observation the way the real robot's sensors would
 * corrupt it: per-term noise, encoder bias, IMU misalignment. Browser-native
 * ships `identityObsPipeline`; sim2real fills these in.
 *
 * Mutates in place — this runs once per env per control step.
 */
export interface ObsPipeline {
  readonly name: string;
  reset(ctx: EnvContext, rng: () => number): void;
  apply(ctx: EnvContext, obs: Float32Array, offset: number): void;
}

// ── Seam 4: the model variant is config, not code ────────────────────────
/** Which MJCF a spec runs against. Backlash/roller variants slot in here. */
export interface RobotVariant {
  /** Manifest entry, e.g. "robot_walk-nv.xml". */
  readonly xml: string;
  /** Actuated joint names, in policy order. */
  readonly joints: readonly string[];
}
