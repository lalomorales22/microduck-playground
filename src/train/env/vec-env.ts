// A batch of environments sharing one compiled MjModel inside one worker.
//
// Deliberately free of DOM and asset-loading concerns: it takes an already
// compiled model and does physics, observation, reward and reset. That is what
// lets scripts/check-env.ts run the real thing under Node instead of a
// lookalike — the M1 gate is only worth anything if it tests this code.

import {
  CTRL_DT, DECIMATION, DEFAULT_POSE, GYRO_SENSOR, NUM_JOINTS, OBS_SIZE, TRUNK_BODY,
} from "../../sim/microduck.ts";
import type { MjData, MjModel, Mujoco } from "../../sim/mujoco.ts";
import { XmlPositionActuator } from "./actuators.ts";
import { identityObsPipeline } from "./obs-pipeline.ts";
import { NO_RANDOMIZERS, applyRandomizers } from "./randomizers.ts";
import type { RewardTerm, StepState, TerminationTerm } from "./rewards.ts";
import type { Actuator, EnvContext, JointIndex, ObsPipeline, Randomizer } from "./seams.ts";
import { resolveJoints } from "./seams.ts";

/** How an episode starts. */
export type ResetPose = (
  ctx: EnvContext,
  rng: () => number,
  standKey: number,
) => void;

/**
 * Exploration settings a task wants.
 *
 * These are NOT one global tuning. A stabilisation task (hold a pose, track a
 * velocity) is destroyed by action noise — the behaviour being learned is the
 * thing the noise breaks. A discovery task (get up off the floor) cannot find
 * the behaviour at all without it. Tuning that suits one silently disables the
 * other, which is exactly what happened here: settings picked for hold-pose
 * were applied to stand-up and it never left 0% standing.
 */
export interface ExplorationHints {
  /** Initial action standard deviation, in radians of joint offset. */
  initStd: number;
  entropyCoef: number;
  /**
   * Adaptive-LR target. KL scales as (delta-mean / sigma)^2, so a task running
   * at a small sigma needs a proportionally larger target or the learning rate
   * is pinned at its floor.
   */
  desiredKl: number;
}

export interface EnvSpec {
  readonly name: string;
  /** Actuated joints, in policy order. Resolved by name — never by index. */
  readonly joints: readonly string[];
  readonly rewards: readonly RewardTerm[];
  readonly terminations: readonly TerminationTerm[];
  readonly reset: ResetPose;
  /** Episode length in seconds; the reference standup task uses 6 s. */
  readonly episodeLengthS: number;
  /** What this task needs from exploration; see ExplorationHints. */
  readonly exploration: ExplorationHints;
  /** One-shot physical manoeuvres cannot start at a future phase while standing. */
  readonly staggerReset?: boolean;
  readonly metrics?: readonly { name: string; compute(ctx: EnvContext, state: StepState): number }[];
  readonly actuator?: Actuator;
  readonly randomizers?: readonly Randomizer[];
  readonly obsPipeline?: ObsPipeline;
}

export interface StepResult {
  /** [count, OBS_SIZE] row-major. */
  obs: Float32Array;
  /** [count] total weighted reward this step. */
  reward: Float32Array;
  /** [count] 1 where the episode ended THIS step. */
  done: Uint8Array;
  /** [count] 1 where the episode ended by running out of time, not by failing. */
  timeout: Uint8Array;
}

/** Per-term reward totals over an episode, for diagnosing a reward stack. */
export type RewardBreakdown = Record<string, number>;

export class VecEnv {
  readonly count: number;
  readonly joints: JointIndex;

  readonly #mujoco: Mujoco;
  readonly #model: MjModel;
  readonly #datas: MjData[];
  readonly #standKey: number;
  readonly #spec: EnvSpec;
  readonly #actuator: Actuator;
  readonly #randomizers: readonly Randomizer[];
  readonly #obsPipeline: ObsPipeline;
  readonly #gyroAdr: number;
  readonly #trunkId: number;
  readonly #maxSteps: number;

  readonly #obs: Float32Array;
  readonly #reward: Float32Array;
  readonly #done: Uint8Array;
  readonly #timeout: Uint8Array;
  readonly #prevAction: Float32Array;
  readonly #stepCount: Int32Array;
  readonly #target = new Float32Array(DEFAULT_POSE);
  readonly #scratchJoints: Float32Array;
  /** Trunk height and uprightness per env, refreshed each step. Diagnostics
   *  read these instead of calling data.body() again: every embind accessor
   *  is a wasm boundary crossing, and one per env per step is not free. */
  readonly #zOut: Float32Array;
  readonly #uprightOut: Float32Array;
  /** Trunk vertical velocity from the previous control step, for acceleration. */
  readonly #prevVz: Float32Array;
  readonly #rng: () => number;

  /** Accumulated per-term reward since the last resetBreakdown(). */
  readonly breakdown: RewardBreakdown = {};
  /** Last measured step, before any terminal reset; values are not success rates. */
  readonly taskMetrics: Record<string, Float32Array> = {};
  /**
   * Environment-steps where the physics diverged and produced a non-finite
   * reward or observation. Counted rather than swallowed: it should be rare,
   * and a rising number means the model or the action scale is wrong.
   */
  nonFiniteSteps = 0;

  constructor(opts: {
    mujoco: Mujoco;
    model: MjModel;
    standKey: number;
    spec: EnvSpec;
    count: number;
    rng?: () => number;
  }) {
    this.#mujoco = opts.mujoco;
    this.#model = opts.model;
    this.#standKey = opts.standKey;
    this.#spec = opts.spec;
    this.count = opts.count;
    this.#rng = opts.rng ?? Math.random;

    this.#actuator = opts.spec.actuator ?? new XmlPositionActuator();
    this.#randomizers = opts.spec.randomizers ?? NO_RANDOMIZERS;
    this.#obsPipeline = opts.spec.obsPipeline ?? identityObsPipeline;

    this.joints = resolveJoints(opts.model, opts.spec.joints);
    this.#gyroAdr = opts.model.sensor(GYRO_SENSOR).adr;
    this.#trunkId = opts.mujoco.mj_name2id(
      opts.model, opts.mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY,
    );
    this.#maxSteps = Math.round(opts.spec.episodeLengthS / CTRL_DT);

    this.#datas = Array.from({ length: opts.count }, () => new opts.mujoco.MjData(opts.model));
    this.#obs = new Float32Array(opts.count * OBS_SIZE);
    this.#reward = new Float32Array(opts.count);
    this.#done = new Uint8Array(opts.count);
    this.#timeout = new Uint8Array(opts.count);
    this.#prevAction = new Float32Array(opts.count * NUM_JOINTS);
    this.#stepCount = new Int32Array(opts.count);
    this.#scratchJoints = new Float32Array(NUM_JOINTS);
    this.#zOut = new Float32Array(opts.count);
    this.#uprightOut = new Float32Array(opts.count);
    this.#prevVz = new Float32Array(opts.count);
    for (const t of opts.spec.rewards) this.breakdown[t.name] = 0;
    for (const t of opts.spec.metrics ?? []) this.taskMetrics[t.name] = new Float32Array(opts.count);

    this.resetAll();
  }

  /**
   * Desynchronize the episode clocks by resetting every environment at a
   * random point in its episode.
   *
   * Environments created (or reset) together would otherwise finish together
   * forever, so every rollout samples one narrow slice of episode time instead
   * of the state distribution. The batch stops being representative and the
   * metrics oscillate with the episode period rather than tracking learning —
   * which is exactly what it looked like the first time this was missing.
   *
   * This is a full reset, not a poke at the step counter: a task whose reset
   * pose depends on the episode clock (a reference motion starts at the pose
   * belonging to its phase) would otherwise be left posed for time 0 while its
   * clock said otherwise.
   */
  stagger(): void {
    for (let e = 0; e < this.count; e++) this.resetEnv(e, { stagger: true });
  }

  get maxSteps(): number {
    return this.#maxSteps;
  }

  #ctx(envId: number): EnvContext {
    const self = this;
    return {
      mujoco: this.#mujoco,
      model: this.#model,
      data: this.#datas[envId],
      joints: this.joints,
      envId,
      // A getter, not a snapshot: step() builds the context once and reads it
      // on both sides of the episode clock advancing.
      get time(): number {
        return self.#stepCount[envId] * CTRL_DT;
      },
    };
  }

  /**
   * Start a fresh episode in one environment.
   *
   * The episode clock is chosen BEFORE the spec poses the robot, and reaches
   * the spec as `ctx.time`, so a task that starts mid-motion can pose itself
   * to match. (Reference-state initialisation: without it a motion task only
   * ever sees the beginning of its motion, and has to learn the whole thing as
   * one chain from a single starting state.)
   */
  resetEnv(envId: number, options: { stagger?: boolean } = {}): void {
    this.#stepCount[envId] = options.stagger && this.#spec.staggerReset !== false
      ? Math.floor(this.#rng() * this.#maxSteps)
      : 0;
    const ctx = this.#ctx(envId);
    // Restore-then-apply lives in applyRandomizers, so DR cannot compound
    // across episodes no matter what an individual randomizer does.
    applyRandomizers(this.#randomizers, ctx, this.#rng);
    this.#spec.reset(ctx, this.#rng, this.#standKey);
    this.#mujoco.mj_forward(this.#model, ctx.data);
    this.#actuator.reset(ctx);
    this.#obsPipeline.reset(ctx, this.#rng);
    this.#prevAction.fill(0, envId * NUM_JOINTS, (envId + 1) * NUM_JOINTS);
    // Not carried across the reset, or the first step of a new episode reads a
    // huge acceleration from the teleport.
    this.#prevVz[envId] = 0;
    this.#writeObs(envId, new Float32Array(NUM_JOINTS));
  }

  /**
   * Reset every environment. Staggered by default — an aligned reset is only
   * wanted for evaluation, where every environment should be measured over the
   * same window.
   */
  resetAll(options: { stagger?: boolean } = {}): Float32Array {
    const stagger = options.stagger ?? true;
    for (let e = 0; e < this.count; e++) this.resetEnv(e, { stagger });
    return this.#obs;
  }

  resetBreakdown(): void {
    for (const k of Object.keys(this.breakdown)) this.breakdown[k] = 0;
  }

  /** Projected gravity in the trunk frame: world -Z rotated into the body. */
  #projectedGravity(data: MjData, out: Float32Array, offset: number): void {
    const q = data.body(this.#trunkId).xquat; // [w, x, y, z]
    const w = q[0], x = q[1], y = q[2], z = q[3];
    out[offset + 0] = -2 * (x * z - w * y);
    out[offset + 1] = -2 * (y * z + w * x);
    out[offset + 2] = -(1 - 2 * (x * x + y * y));
  }

  /**
   * Build one environment's observation.
   *
   * Layout must match the exported policies exactly — see docs/training-plan.md.
   * The command block is all zeros: the get-up policy was trained that way, and
   * the slots stay present rather than being dropped so the 61D interface holds
   * across the whole policy family.
   */
  #writeObs(envId: number, lastAction: Float32Array): void {
    const data = this.#datas[envId];
    const { qpos, qvel, sensordata } = data;
    const base = envId * OBS_SIZE;
    let i = base;
    for (let a = 0; a < 3; a++) this.#obs[i++] = sensordata[this.#gyroAdr + a];
    this.#projectedGravity(data, this.#obs, i); i += 3;
    for (let j = 0; j < NUM_JOINTS; j++) this.#obs[i++] = qpos[this.joints.qpos[j]] - DEFAULT_POSE[j];
    for (let j = 0; j < NUM_JOINTS; j++) this.#obs[i++] = qvel[this.joints.dof[j]];
    for (let j = 0; j < NUM_JOINTS; j++) this.#obs[i++] = lastAction[j];
    this.#obs.fill(0, i, base + OBS_SIZE);
    this.#obsPipeline.apply(this.#ctx(envId), this.#obs, base);
    // Same reasoning as the reward guard: a non-finite observation would reach
    // the policy AND the running normaliser, and the normaliser never
    // recovers.
    for (let k = base; k < base + OBS_SIZE; k++) {
      if (!Number.isFinite(this.#obs[k])) {
        this.#obs[k] = 0;
        this.nonFiniteSteps++;
      }
    }
  }

  /**
   * Advance every environment one control step.
   *
   * `actions` is [count, NUM_JOINTS] row-major. Environments that finish are
   * reset immediately and their observation is the FIRST of the new episode —
   * the standard vectorized-env convention, and what the PPO buffer expects.
   */
  step(actions: Float32Array): StepResult {
    const nj = NUM_JOINTS;
    for (let e = 0; e < this.count; e++) {
      const ctx = this.#ctx(e);
      const action = actions.subarray(e * nj, (e + 1) * nj);
      this.#actuator.apply(ctx, action);
      for (let s = 0; s < DECIMATION; s++) this.#mujoco.mj_step(this.#model, ctx.data);

      // Advanced BEFORE the reward, not after: `ctx.time` has to name the
      // instant the physics just reached, or every term that reads the clock —
      // a reference-motion tracker, most obviously — is graded against the
      // pose from one control step ago.
      this.#stepCount[e]++;
      const state = this.#stepState(e, action);
      for (const metric of this.#spec.metrics ?? []) {
        const value = metric.compute(ctx, state);
        this.taskMetrics[metric.name][e] = Number.isFinite(value) ? value : 0;
        if (!Number.isFinite(value)) this.nonFiniteSteps++;
      }
      let total = 0;
      for (const term of this.#spec.rewards) {
        const value = term.compute(ctx, state) * term.weight;
        this.breakdown[term.name] += value;
        total += value;
      }
      // A diverged solver produces NaN, and the nan_state termination below
      // resets the environment — but the reward for THIS step is computed
      // first, so without this guard the NaN reaches the rollout buffer,
      // poisons the advantage normalisation, and from there every gradient.
      // One exploded contact then destroys the policy permanently.
      if (!Number.isFinite(total)) {
        total = 0;
        this.nonFiniteSteps++;
      }
      this.#reward[e] = total;

      const failed = this.#spec.terminations.some((t) => t.check(ctx, state));
      const timedOut = this.#stepCount[e] >= this.#maxSteps;
      this.#done[e] = failed || timedOut ? 1 : 0;
      this.#timeout[e] = !failed && timedOut ? 1 : 0;

      if (this.#done[e]) {
        this.resetEnv(e);
      } else {
        this.#prevAction.set(action, e * nj);
        this.#writeObs(e, action);
      }
    }
    return { obs: this.#obs, reward: this.#reward, done: this.#done, timeout: this.#timeout };
  }

  #stepState(envId: number, action: Float32Array): StepState {
    const data = this.#datas[envId];
    for (let j = 0; j < NUM_JOINTS; j++) {
      this.#scratchJoints[j] = data.qpos[this.joints.qpos[j]];
    }
    const quat = data.body(this.#trunkId).xquat as Float64Array;
    this.#zOut[envId] = data.qpos[2];
    this.#uprightOut[envId] = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2]);
    // For a free joint, qvel[0..2] is linear velocity in the WORLD frame.
    const vz = data.qvel[2];
    const az = (vz - this.#prevVz[envId]) / CTRL_DT;
    this.#prevVz[envId] = vz;
    return {
      z: data.qpos[2],
      quat,
      vz,
      az,
      jointPos: this.#scratchJoints,
      jointTarget: this.#target,
      action,
      prevAction: this.#prevAction.subarray(envId * NUM_JOINTS, (envId + 1) * NUM_JOINTS),
    };
  }

  /** Episode clock for one environment, seconds — what a reward term sees as
   *  `ctx.time`. Exposed for diagnostics and for anything driving the
   *  environments from outside, such as a reference-motion oracle. */
  timeOf(envId: number): number {
    return this.#stepCount[envId] * CTRL_DT;
  }

  /** Read-only view of an environment's MuJoCo state, for diagnostics. */
  dataAt(envId: number): MjData {
    return this.#datas[envId];
  }

  /** Trunk height per env as of the last step. */
  get trunkZ(): Float32Array {
    return this.#zOut;
  }

  /** Uprightness per env as of the last step: 1 upright, 0 on its side. */
  get uprightness(): Float32Array {
    return this.#uprightOut;
  }

  get observations(): Float32Array {
    return this.#obs;
  }
}
