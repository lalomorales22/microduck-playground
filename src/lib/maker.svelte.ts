// The Motion Maker's runtime: a simulation, a viewport, and an editable draft.
//
// It boots its own MuJoCo model rather than sharing the Simulate tab's. The
// two workspaces want the robot in different states at the same time — one is
// running a policy, the other is holding a pose being edited — and a single
// MjData cannot be in both. Both are fully paused while hidden, so only one is
// ever stepping.
//
// Kinematic posing goes through GroundReference, the same rule the imitation
// reward uses to decide where the trunk should be. That is not tidiness: the
// editor exists to show you what you are about to train on, and it can only do
// that if it is deriving the root the same way.

import { CTRL_DT, DECIMATION, JOINT_NAMES, NUM_JOINTS } from "../sim/microduck.ts";
import { loadSimulation, type Simulation } from "../sim/scene.ts";
import { GroundReference } from "../motion/ground.ts";
import { MotionSampler } from "../motion/sampler.ts";
import { parseMotion } from "../motion/format.ts";
import {
  draftFromMotion, draftToFile, draftToMotion, draftDuration, draftProblem,
  newDraft, type MotionDraft,
} from "../motion/edit.ts";
import {
  deleteMotion, listMotions, loadMotion, saveMotion, type MotionEntry,
} from "../motion/motion-store.ts";
import { sanitizeName } from "../opfs.ts";
import { Viewer } from "../render/viewer.ts";

export type PlayMode = "preview" | "physics";

/** Never advance more sim time than this per frame; see Session. */
const MAX_CATCHUP_S = 0.25;

export class MakerSession {
  loadStage = $state("Starting…");
  loadProgress = $state(0);
  ready = $state(false);
  started = $state(false);
  error = $state<string | null>(null);

  draft = $state<MotionDraft>(newDraft());
  /** Playhead, seconds. */
  time = $state(0);
  playing = $state(false);
  mode = $state<PlayMode>("preview");
  /** Index of the selected keyframe, or -1 when the playhead is between keys. */
  selected = $state(0);

  library = $state<MotionEntry[]>([]);
  /** Where a save would go; blank means the draft has never been saved. */
  savedAs = $state<string | null>(null);
  notice = $state<string | null>(null);
  saveError = $state<string | null>(null);

  /** Live physics readout, so the fall in physics mode is legible. */
  trunkCm = $state(0);
  uprightPct = $state(100);

  readonly motion = $derived(draftToMotion(this.draft));
  readonly sampler = $derived(new MotionSampler(this.motion));
  readonly duration = $derived(draftDuration(this.draft));
  /** What stops this draft being a valid file, if anything. */
  readonly problem = $derived(draftProblem(this.draft));

  #viewer: Viewer | null = null;
  #sim: Simulation | null = null;
  #ground: GroundReference | null = null;
  #qposAdr: number[] = [];
  #trunkId = 0;
  #pose = new Float32Array(NUM_JOINTS);
  #frame = 0;
  #active = false;
  #resumeAt = 0;
  #disposed = false;

  async start(canvas: HTMLCanvasElement): Promise<void> {
    this.started = true;
    try {
      const viewer = new Viewer(canvas);
      this.#viewer = viewer;
      viewer.render();

      const sim = await loadSimulation((stage, done, total) => {
        this.loadStage = total ? `${stage} ${done}/${total}` : stage;
        this.loadProgress = total ? ((done ?? 0) / total) * 100 : 0;
      });
      if (this.#disposed) return;
      this.#sim = sim;
      this.#qposAdr = JOINT_NAMES.map((n) => sim.model.jnt(n).qposadr);
      this.#trunkId = sim.mujoco.mj_name2id(
        sim.model, sim.mujoco.mjtObj.mjOBJ_BODY.value, "trunk_base",
      );
      this.#ground = new GroundReference(sim.mujoco, sim.model, sim.data, sim.standKey);

      viewer.build(sim.model, { followBody: this.#trunkId });
      viewer.sync(sim.model, sim.data);
      await this.refreshLibrary();

      this.ready = true;
      this.#applyPose();
      this.#run();
    } catch (err) {
      console.error(err);
      this.error = err instanceof Error ? err.message : String(err);
    }
  }

  #run(): void {
    let last = performance.now() / 1000;
    let accumulator = 0;

    const tick = (): void => {
      if (this.#disposed) return;
      const sim = this.#sim;
      const viewer = this.#viewer;
      if (!sim || !viewer) return;

      const now = performance.now() / 1000;
      if (!this.#active) {
        last = now;
        this.#frame = requestAnimationFrame(tick);
        return;
      }
      if (this.#resumeAt) {
        last = this.#resumeAt;
        this.#resumeAt = 0;
      }
      accumulator += Math.min(now - last, MAX_CATCHUP_S);
      last = now;

      while (accumulator >= CTRL_DT) {
        accumulator -= CTRL_DT;
        if (this.playing) this.#advance();
      }
      // Posing every frame rather than only on a change: two mj_forward calls
      // is cheaper than the bookkeeping to know whether a slider moved, and
      // the frame is being drawn anyway.
      if (!this.playing || this.mode === "preview") this.#applyPose();

      this.trunkCm = Math.round(sim.data.qpos[2] * 1000) / 10;
      const q = sim.data.body(this.#trunkId).xquat;
      this.uprightPct = Math.round(Math.max(0, 1 - 2 * (q[1] * q[1] + q[2] * q[2])) * 100);

      viewer.sync(sim.model, sim.data);
      viewer.render();
      this.#frame = requestAnimationFrame(tick);
    };
    tick();
  }

  /** One control step of playback. */
  #advance(): void {
    const sim = this.#sim;
    if (!sim) return;
    if (this.mode === "physics") {
      // Open loop: the reference pose is written straight to the actuators,
      // with nothing correcting for where the duck actually ended up.
      this.sampler.poseAt(this.time, this.#pose);
      for (let j = 0; j < NUM_JOINTS; j++) sim.data.ctrl[j] = this.#pose[j];
      for (let s = 0; s < DECIMATION; s++) sim.mujoco.mj_step(sim.model, sim.data);
    }
    const next = this.time + CTRL_DT;
    if (!this.motion.loop && next >= this.duration) {
      this.time = this.duration;
      this.playing = false;
    } else {
      this.time = this.motion.loop && this.duration > 0 ? next % this.duration : next;
    }
    this.#syncSelection();
  }

  /** Place the duck at the playhead, no physics. */
  #applyPose(): void {
    const sim = this.#sim;
    const ground = this.#ground;
    if (!sim || !ground) return;
    this.sampler.poseAt(this.time, this.#pose);
    ground.poseAt(sim.data, sim.standKey, this.#qposAdr, this.#pose);
  }

  #syncSelection(): void {
    const keys = this.draft.keys;
    const index = keys.findIndex((k) => Math.abs(k.t - this.time) <= 0.011);
    this.selected = index;
  }

  /** Interpolated pose at the playhead, for a slider that is not on a key. */
  poseAtPlayhead(): Float32Array {
    return this.sampler.poseAt(this.time, new Float32Array(NUM_JOINTS));
  }

  seek(t: number): void {
    this.playing = false;
    this.time = Math.min(Math.max(t, 0), Math.max(this.duration, 0));
    this.#syncSelection();
    this.#applyPose();
  }

  play(): void {
    if (this.mode === "physics") this.seek(0);
    else if (!this.motion.loop && this.time >= this.duration) this.seek(0);
    this.playing = true;
  }

  pause(): void {
    this.playing = false;
    this.#applyPose();
  }

  setMode(mode: PlayMode): void {
    this.mode = mode;
    this.playing = false;
    this.seek(0);
  }

  /** Re-pose after an edit that changed the draft. */
  refresh(): void {
    this.#syncSelection();
    if (!this.playing || this.mode === "preview") this.#applyPose();
  }

  // ── The library ────────────────────────────────────────────────────────
  async refreshLibrary(): Promise<void> {
    this.library = (await listMotions()).filter((m) => !m.error);
  }

  async open(id: string): Promise<void> {
    this.saveError = null;
    try {
      const motion = await loadMotion(id);
      if (!motion) throw new Error(`motion "${id}" is gone`);
      this.draft = draftFromMotion(motion);
      // A built-in is a starting point, not a destination: leaving savedAs
      // unset means Save asks for a name rather than silently shadowing it.
      this.savedAs = id.includes(":") ? null : id;
      this.notice = `Opened ${motion.name}`;
      // Back to preview: opening a motion is to look at it, and dropping
      // straight into open-loop physics would just show it falling over.
      this.mode = "preview";
      this.seek(0);
    } catch (err) {
      this.saveError = err instanceof Error ? err.message : String(err);
    }
  }

  reset(): void {
    this.draft = newDraft();
    this.savedAs = null;
    this.notice = null;
    this.saveError = null;
    this.seek(0);
  }

  /**
   * Save under `name`, or under the name in the title field.
   *
   * Naming it in the Save-as prompt also RENAMES the draft. Otherwise the file
   * ends up stored as "my-wave" while still calling itself "untitled", and the
   * name that shows up in the training task picker is the one nobody chose.
   */
  async save(name?: string): Promise<boolean> {
    this.saveError = null;
    if (name?.trim()) this.draft.name = name.trim();
    const target = sanitizeName(name ?? this.savedAs ?? this.draft.name);
    if (!target) {
      this.saveError = "give the motion a name first";
      return false;
    }
    const problem = this.problem;
    if (problem) {
      this.saveError = problem;
      return false;
    }
    await saveMotion(target, { ...draftToFile(this.draft), name: this.draft.name || target });
    this.savedAs = target;
    this.notice = `Saved as ${target}`;
    await this.refreshLibrary();
    return true;
  }

  async remove(id: string): Promise<void> {
    await deleteMotion(id);
    if (this.savedAs === id) this.savedAs = null;
    await this.refreshLibrary();
  }

  /** Load a file someone pasted in — usually written by an agent against
   *  docs/motion-format.md. Parsed first, so a bad paste says why. */
  loadJson(text: string): boolean {
    this.saveError = null;
    try {
      const motion = parseMotion(text);
      this.draft = draftFromMotion(motion);
      this.savedAs = null;
      this.notice = `Loaded ${motion.name}`;
      this.mode = "preview";
      this.seek(0);
      return true;
    } catch (err) {
      this.saveError = err instanceof Error ? err.message : String(err);
      return false;
    }
  }

  json(): string {
    return JSON.stringify(draftToFile(this.draft), null, 2);
  }

  setActive(value: boolean): void {
    this.#active = value;
    if (value) this.#resumeAt = performance.now() / 1000;
  }

  destroy(): void {
    this.#disposed = true;
    cancelAnimationFrame(this.#frame);
    this.#viewer?.dispose();
    this.#viewer = null;
  }
}
