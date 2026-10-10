// The PPO training loop over a VecEnv.
//
// Runs rollout and update in one place. That is deliberate for M2: a
// distributed arrangement (workers roll out, a central learner updates) only
// pays off once inference is fast, and M0 measured inference at ~70% of the
// step budget — so M3 optimises that first, and the split comes after.
//
// Checkpointing is here from the start rather than bolted on: a resumed run
// that silently differs from an uninterrupted one is the kind of bug that only
// shows up as "training got worse overnight".

import { NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import type { MjModel, Mujoco } from "../sim/mujoco.ts";
import { ActorCritic, HIDDEN } from "./ac-policy.ts";
import { Adam } from "./nn.ts";
import { DEFAULT_PPO, makeBuffer, ppoUpdate, type PpoConfig, type RolloutBuffer, type UpdateStats } from "./ppo.ts";
import { VecEnv, type EnvSpec, type ExplorationHints } from "./env/vec-env.ts";
import { LocalRollout } from "./rollout-local.ts";
import type { RolloutSource } from "./rollout-source.ts";
import type { Kernels } from "./kernels/index.ts";
import { STAND_Z } from "./env/rewards.ts";
import type { TaskSpec } from "./env/tasks.ts";
import type { WorldConfig } from "../sim/world.ts";
import { validateCheckpointResume } from "./checkpoint-task.ts";

export interface TrainerConfig {
  envs: number;
  /** Control steps collected per environment per iteration. */
  stepsPerIter: number;
  ppo: PpoConfig;
  hidden: readonly number[];
  seed: number;
  /**
   * Exploration overrides. Left undefined, the TASK decides — see
   * ExplorationHints. A stabilisation task and a discovery task want opposite
   * settings, so a single global value is wrong for one of them.
   */
  exploration?: Partial<ExplorationHints>;
}

export const DEFAULT_TRAINER: Omit<TrainerConfig, "seed"> = {
  envs: 64,
  // The reference uses 24; with far fewer environments a longer rollout keeps
  // the batch big enough for 4 minibatches to mean anything.
  stepsPerIter: 32,
  ppo: DEFAULT_PPO,
  hidden: HIDDEN,
};

export interface IterationStats extends UpdateStats {
  taskMetrics: Record<string, number>;
  iteration: number;
  /** Mean reward per control step over the rollout. */
  rewardPerStep: number;
  /** Mean undiscounted return of episodes that FINISHED this iteration. */
  episodeReturn: number;
  episodes: number;
  standingFraction: number;
  /** Environment-steps where physics diverged; should stay 0. */
  nonFiniteSteps: number;
  totalSteps: number;
  rolloutMs: number;
  updateMs: number;
}

/** Everything needed to resume a run byte-identically. */
export interface Checkpoint {
  version: 1;
  iteration: number;
  totalSteps: number;
  /** Which task this was trained on, for the UI. Older files omit it. */
  task?: string;
  taskSpec?: TaskSpec;
  world?: WorldConfig;
  observationSchema?: "microduck-task-v1";
  config: TrainerConfig;
  policy: unknown;
  optimizer: ReturnType<Adam["serialize"]>;
  rngState: number;
}

/**
 * Mulberry32 whose state lives in a field rather than a closure.
 *
 * That matters for resume: VecEnv and ActorCritic capture the generator
 * FUNCTION at construction, so swapping in a fresh closure on restore would
 * leave them drawing from the old stream. Keeping one stable function over
 * mutable state means a restore actually reaches everyone holding it.
 */
class SeededRng {
  #a: number;

  constructor(seed: number) {
    this.#a = seed >>> 0;
  }

  readonly next = (): number => {
    this.#a = (this.#a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(this.#a ^ (this.#a >>> 15), 1 | this.#a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  get state(): number {
    return this.#a;
  }

  set state(value: number) {
    this.#a = value >>> 0;
  }
}

export class Trainer {
  readonly ac: ActorCritic;
  /** Present only when the environments run in this thread. The pool variant
   *  keeps them in workers, so diagnostics that need MuJoCo state are
   *  in-thread only. */
  readonly env: VecEnv | null;
  readonly rollout: RolloutSource;
  readonly config: TrainerConfig;

  /** What the task asked for, after any explicit override. Reported so a run's
   *  exploration is visible rather than implicit. */
  readonly exploration: ExplorationHints;

  #opt: Adam;
  #buf: RolloutBuffer;
  readonly #rng: SeededRng;
  #iteration = 0;
  #totalSteps = 0;
  readonly #specName: string;
  readonly #ppo: PpoConfig;

  constructor(opts: {
    /** Required unless `rollout` is supplied — those own their own models. */
    mujoco?: Mujoco;
    model?: MjModel;
    standKey?: number;
    spec: EnvSpec;
    config: TrainerConfig;
    /** SIMD kernels for the nets; null falls back to the JS path. */
    kernels?: Kernels | null;
    /** Where rollouts come from. Defaults to a VecEnv in this thread, which
     *  requires mujoco/model/standKey; a pool supplies its own. */
    rollout?: RolloutSource;
  }) {
    this.config = opts.config;
    this.#specName = opts.spec.name;
    this.#rng = new SeededRng(opts.config.seed);
    if (opts.rollout) {
      this.env = null;
    } else {
      if (!opts.mujoco || !opts.model || opts.standKey === undefined) {
        throw new Error("Trainer needs a model when no rollout source is given");
      }
      this.env = new VecEnv({
        mujoco: opts.mujoco,
        model: opts.model,
        standKey: opts.standKey,
        spec: opts.spec,
        count: opts.config.envs,
        rng: this.#rng.next,
      });
    }
    // A pool decides its own environment count; the config's is only the
    // in-thread default.
    const envs = opts.rollout?.envs ?? opts.config.envs;
    this.exploration = { ...opts.spec.exploration, ...opts.config.exploration };
    this.ac = new ActorCritic(
      OBS_SIZE,
      NUM_JOINTS,
      this.#rng.next,
      this.exploration.initStd,
      opts.config.hidden,
      opts.kernels ?? null,
    );
    this.#opt = this.ac.makeOptimizer(opts.config.ppo.lr);
    this.#ppo = {
      ...opts.config.ppo,
      entropyCoef: this.exploration.entropyCoef,
      desiredKl: this.exploration.desiredKl,
    };
    this.#buf = makeBuffer(
      opts.config.stepsPerIter,
      envs,
      this.ac.obsDim,
      this.ac.actDim,
    );
    this.rollout = opts.rollout
      ?? new LocalRollout(this.env as VecEnv, this.ac, this.#rng.next);
  }

  get iteration(): number {
    return this.#iteration;
  }

  get totalSteps(): number {
    return this.#totalSteps;
  }

  get usesSimd(): boolean {
    return this.ac.actor.usesSimd;
  }

  /** One rollout + one PPO update. */
  async iterate(): Promise<IterationStats> {
    const { stepsPerIter } = this.config;
    const buf = this.#buf;

    // The pool runs inference in its workers, so it needs this iteration's
    // weights before it starts. In-thread this is a no-op.
    await this.rollout.syncPolicy(this.ac);

    const rolloutStart = performance.now();
    const collected = await this.rollout.collect(buf, stepsPerIter);
    const rolloutMs = performance.now() - rolloutStart;

    const updateStart = performance.now();
    const stats = ppoUpdate(this.ac, this.#opt, buf, this.#ppo, this.#rng.next);
    const updateMs = performance.now() - updateStart;

    // Normalizer statistics update AFTER the update, never between rollout and
    // update. PPO's importance ratio assumes the stored log-probs came from the
    // same function the first minibatch evaluates; re-scaling the inputs in
    // between breaks that, and the resulting phantom KL drives the adaptive
    // learning rate to its floor and freezes the run.
    this.ac.normalizer.update(buf.obs, stepsPerIter * buf.envs);

    this.#iteration++;
    this.#totalSteps += collected.steps;

    return {
      ...stats,
      taskMetrics: Object.fromEntries(Object.entries(collected.taskMetricSums).map(([name, sum]) => [name, sum / collected.steps])),
      iteration: this.#iteration,
      rewardPerStep: collected.rewardSum / collected.steps,
      episodeReturn: collected.finishedEpisodes > 0
        ? collected.finishedReturn / collected.finishedEpisodes
        : 0,
      episodes: collected.finishedEpisodes,
      standingFraction: collected.standingSteps / collected.steps,
      nonFiniteSteps: collected.nonFiniteSteps,
      totalSteps: this.#totalSteps,
      rolloutMs,
      updateMs,
    };
  }

  /**
   * Deterministic-policy evaluation: no sampling noise, which is what a
   * deployed policy actually does.
   *
   * Resets every environment first, ALIGNED, so each is measured over the same
   * window — without that the measurement starts from wherever the last
   * rollout left things, which makes two evaluations incomparable and made a
   * checkpoint round-trip look broken when it was only being measured from a
   * different state.
   *
   * Training resumes from a re-staggered set afterwards: leaving the pool
   * phase-locked would poison every rollout that follows.
   */
  evaluate(steps: number): { rewardPerStep: number; standingFraction: number } {
    const env = this.env;
    if (!env) throw new Error("evaluate() needs in-thread environments");
    const envs = env.count;
    env.resetAll({ stagger: false });
    let rewardSum = 0, standingSteps = 0;
    for (let t = 0; t < steps; t++) {
      const mean = this.ac.actMean(env.observations, envs);
      const { reward } = env.step(mean);
      const z = env.trunkZ, upright = env.uprightness;
      for (let e = 0; e < envs; e++) {
        rewardSum += reward[e];
        if (z[e] > STAND_Z - 0.02 && upright[e] > 0.85) standingSteps++;
      }
    }
    const total = steps * envs;
    env.resetAll();
    return { rewardPerStep: rewardSum / total, standingFraction: standingSteps / total };
  }

  checkpoint(): Checkpoint {
    return {
      version: 1,
      iteration: this.#iteration,
      totalSteps: this.#totalSteps,
      task: this.#specName,
      config: this.config,
      policy: this.ac.serialize(),
      optimizer: this.#opt.serialize(),
      rngState: this.#rng.state,
    };
  }

  /**
   * Restore a run. Adam's moments and the normalizer statistics come back too:
   * without them a resumed run takes a visible quality dip for hundreds of
   * steps, which is easy to misread as the task getting harder.
   */
  restore(ckpt: Checkpoint): void {
    validateCheckpointResume(ckpt);
    if (ckpt.version !== 1) throw new Error(`unsupported checkpoint version ${ckpt.version}`);
    if (ckpt.task !== undefined && ckpt.task !== this.#specName) throw new Error(`checkpoint task ${ckpt.task} does not match ${this.#specName}`);
    this.ac.load(ckpt.policy as ReturnType<ActorCritic["serialize"]>);
    this.#opt.load(ckpt.optimizer);
    this.#iteration = ckpt.iteration;
    this.#totalSteps = ckpt.totalSteps;
    this.#rng.state = ckpt.rngState;
  }
}
