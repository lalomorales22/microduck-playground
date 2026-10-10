// The editable form of a motion.
//
// Deliberately plain data — numbers and arrays, no Float32Array, no class —
// for two reasons. Svelte's deep reactivity proxies plain objects and arrays
// but not typed arrays, so an editor built on this gets fine-grained updates
// for free. And a draft is allowed to be temporarily INVALID: an unclosed
// loop, no driven joints, a name that is not yet filled in. Motion is the
// validated shape; MotionDraft is the shape you can be halfway through.
//
// Everything here is pure, so it can be reasoned about (and tested) without a
// browser, a model, or a component tree.

import { DEFAULT_POSE, JOINT_LIMITS, JOINT_NAMES, NUM_JOINTS } from "../sim/microduck.ts";
import { MOTION_FORMAT, parseMotion, type Motion, type MotionFile } from "./format.ts";

export interface DraftKey {
  /** Seconds. keys[0].t is always 0. */
  t: number;
  /** All 14 joints, absolute radians — including the ones not driven, which
   *  sit at their DEFAULT_POSE value and are dropped on the way out. */
  pose: number[];
}

export interface MotionDraft {
  name: string;
  description: string;
  loop: boolean;
  interp: "linear" | "cubic";
  /** Per joint: is it part of this motion? Undriven joints are not written to
   *  the file at all, which is what keeps the tracking reward focused — see
   *  motionRewards(). */
  driven: boolean[];
  /** Ascending in t, at least one entry. */
  keys: DraftKey[];
}

export const DEFAULT_ANGLES: readonly number[] = Array.from(DEFAULT_POSE);

/**
 * Joints that swap when a pose is mirrored, and always by NEGATION.
 *
 * The two legs' axes are mirrored in the MJCF, which is why DEFAULT_POSE has
 * left_hip_roll at -0.087 and right_hip_roll at +0.087. Getting this wrong by
 * hand is the single most common way to author a squat that limps, so it is a
 * button rather than an instruction.
 */
export const MIRROR_PAIRS: readonly (readonly [number, number])[] = [
  [0, 9],   // hip_yaw
  [1, 10],  // hip_roll
  [2, 11],  // hip_pitch
  [3, 12],  // knee
  [4, 13],  // ankle
];
/** Head joints that mirror onto themselves: yaw and roll flip, pitch does not. */
export const SELF_MIRROR: readonly number[] = [7, 8];

export function newDraft(name = "untitled"): MotionDraft {
  return {
    name,
    description: "",
    loop: false,
    interp: "cubic",
    driven: new Array(NUM_JOINTS).fill(false),
    // Two keys, a second apart: a timeline you can immediately work in. One
    // key would be a static pose with nowhere to put a second one.
    keys: [
      { t: 0, pose: [...DEFAULT_ANGLES] },
      { t: 1, pose: [...DEFAULT_ANGLES] },
    ],
  };
}

export function draftFromMotion(motion: Motion): MotionDraft {
  const driven = new Array(NUM_JOINTS).fill(false);
  for (const name of motion.driven) {
    const index = JOINT_NAMES.indexOf(name as (typeof JOINT_NAMES)[number]);
    if (index >= 0) driven[index] = true;
  }
  return {
    name: motion.name,
    description: motion.description,
    loop: motion.loop,
    interp: motion.interp,
    driven,
    keys: motion.keyframes.map((k) => ({ t: k.t, pose: Array.from(k.pose) })),
  };
}

export function drivenList(draft: MotionDraft): number[] {
  const out: number[] = [];
  for (let j = 0; j < NUM_JOINTS; j++) if (draft.driven[j]) out.push(j);
  return out;
}

export function draftDuration(draft: MotionDraft): number {
  return draft.keys.length ? draft.keys[draft.keys.length - 1].t : 0;
}

/**
 * The draft as a Motion, WITHOUT validating it.
 *
 * The editor's viewport has to show something on every keystroke, including
 * while the loop is open or no joint is driven yet. Saving goes through
 * parseMotion instead, which is where the file is held to the rules.
 */
export function draftToMotion(draft: MotionDraft): Motion {
  const keys = draft.keys.length ? draft.keys : newDraft().keys;
  const driven = drivenList(draft);
  return {
    format: MOTION_FORMAT,
    name: draft.name || "untitled",
    description: draft.description,
    loop: draft.loop,
    interp: draft.interp,
    keyframes: keys.map((k) => ({ t: k.t, pose: Float32Array.from(k.pose) })),
    duration: keys[keys.length - 1].t,
    // Falls back to every joint so a draft with nothing driven yet still
    // previews; the file itself would be rejected, which is the point.
    driven: (driven.length ? driven : keys[0].pose.map((_, j) => j)).map((j) => JOINT_NAMES[j]),
  };
}

/** The file that gets saved: only the driven joints, radians, all keys. */
export function draftToFile(draft: MotionDraft): MotionFile {
  const driven = drivenList(draft);
  return {
    format: MOTION_FORMAT,
    name: draft.name,
    ...(draft.description ? { description: draft.description } : {}),
    loop: draft.loop,
    units: "rad",
    interp: draft.interp,
    joints: driven.map((j) => JOINT_NAMES[j]),
    keyframes: draft.keys.map((k) => ({
      t: Number(k.t.toFixed(4)),
      pose: driven.map((j) => Number(k.pose[j].toFixed(5))),
    })),
  };
}

/** null when the draft would save cleanly, otherwise what is wrong with it. */
export function draftProblem(draft: MotionDraft): string | null {
  try {
    parseMotion(draftToFile(draft));
    return null;
  } catch (err) {
    return err instanceof Error ? err.message : String(err);
  }
}

export function clampJoint(joint: number, value: number): number {
  const [lo, hi] = JOINT_LIMITS[joint];
  return Math.min(Math.max(value, lo), hi);
}

// ── Mutations ────────────────────────────────────────────────────────────
// Each mutates the draft in place and keeps its invariants: keys sorted, the
// first at t = 0, angles inside their hinge limits.

function sortKeys(draft: MotionDraft): void {
  draft.keys.sort((a, b) => a.t - b.t);
}

/** Index of the key at `t`, or -1. The tolerance is half a control step, so
 *  clicking a key selects it rather than landing a hair beside it. */
export function keyAt(draft: MotionDraft, t: number, eps = 0.011): number {
  return draft.keys.findIndex((k) => Math.abs(k.t - t) <= eps);
}

/** Insert a key at `t` holding the currently interpolated pose, or return the
 *  existing one. `pose` comes from the sampler — this module stays pure. */
export function addKey(draft: MotionDraft, t: number, pose: ArrayLike<number>): number {
  const existing = keyAt(draft, t);
  if (existing >= 0) return existing;
  draft.keys.push({ t: Math.max(0, t), pose: Array.from(pose) });
  sortKeys(draft);
  return keyAt(draft, t);
}

/** Keys 0 and the last one anchor the timeline, so only interior keys go. */
export function removeKey(draft: MotionDraft, index: number): boolean {
  if (draft.keys.length <= 2 || index <= 0 || index >= draft.keys.length) return false;
  draft.keys.splice(index, 1);
  return true;
}

/** Move a key in time. The first is pinned at 0; the rest cannot cross their
 *  neighbours, because a reordering drag is never what was meant. */
export function moveKey(draft: MotionDraft, index: number, t: number): number {
  if (index <= 0 || index >= draft.keys.length) return index;
  const lo = draft.keys[index - 1].t + 0.02;
  const hi = index + 1 < draft.keys.length ? draft.keys[index + 1].t - 0.02 : Infinity;
  draft.keys[index].t = Math.min(Math.max(t, lo), Math.max(lo, hi));
  return index;
}

/** Stretch or squash the whole motion to a new total length. */
export function setDuration(draft: MotionDraft, seconds: number): void {
  const current = draftDuration(draft);
  if (current <= 0 || seconds <= 0) return;
  const scale = seconds / current;
  for (const k of draft.keys) k.t = Number((k.t * scale).toFixed(4));
}

/** Set one joint at one key, marking it driven. Editing a joint IS how it
 *  joins the motion — a separate "add this joint" step is friction nobody
 *  would thank us for. */
export function setJoint(draft: MotionDraft, index: number, joint: number, value: number): void {
  const key = draft.keys[index];
  if (!key) return;
  key.pose[joint] = clampJoint(joint, value);
  draft.driven[joint] = true;
}

/** Drop a joint from the motion and return it to the reference angle
 *  everywhere, so the file and the preview agree about it. */
export function undriveJoint(draft: MotionDraft, joint: number): void {
  draft.driven[joint] = false;
  for (const k of draft.keys) k.pose[joint] = DEFAULT_ANGLES[joint];
}

/** Copy the left leg onto the right, negated, at one key. */
export function mirrorLegs(draft: MotionDraft, index: number, from: "left" | "right"): void {
  const key = draft.keys[index];
  if (!key) return;
  for (const [left, right] of MIRROR_PAIRS) {
    const [src, dst] = from === "left" ? [left, right] : [right, left];
    key.pose[dst] = clampJoint(dst, -key.pose[src]);
    if (draft.driven[src]) draft.driven[dst] = true;
  }
}

/** Make the last key repeat the first, which is what `loop` requires. */
export function closeLoop(draft: MotionDraft): void {
  if (draft.keys.length < 2) return;
  draft.keys[draft.keys.length - 1].pose = [...draft.keys[0].pose];
}
