// The motion file: a pose, or a sequence of poses over time, as plain JSON.
//
// This is the interchange format for stage 1 of plans/motion-authoring.md, and
// it is deliberately the SAME file a keyframe editor would produce later. It
// is also the file an AI agent is expected to write from docs/motion-format.md,
// which drives most of the design decisions here:
//
//   * Every joint is optional. "Make the duck nod" should be a two-joint file,
//     not fourteen numbers the author has to look up. Anything unlisted holds
//     its DEFAULT_POSE value.
//   * Degrees are allowed. An agent that writes 30 where 0.52 was meant is a
//     57x error that looks like a physics bug; letting the file say which unit
//     it means removes the guess.
//   * Validation is strict and the messages name the fix. The author is
//     usually not a human who can read a stack trace.
//
// parseMotion() returns a NORMALISED motion: joints in policy order, all 14
// present, radians, keyframes sorted, t starting at 0. Everything downstream
// (the sampler, the env spec, the editor) consumes only that shape, so the
// leniency lives in exactly one place.

import { DEFAULT_POSE, JOINT_LIMITS, JOINT_NAMES, NUM_JOINTS } from "../sim/microduck.ts";

export const MOTION_FORMAT = 1;

/** How the file may be written. Every field except `keyframes` is optional. */
export interface MotionFile {
  format?: number;
  name?: string;
  description?: string;
  loop?: boolean;
  units?: "rad" | "deg";
  interp?: "linear" | "cubic";
  /** Joint names this file drives; anything omitted holds DEFAULT_POSE. */
  joints?: string[];
  keyframes: { t: number; pose: number[] }[];
}

/** One keyframe of the normalised motion: all 14 joints, radians. */
export interface MotionKeyframe {
  /** Seconds from the start of the motion. The first is always 0. */
  readonly t: number;
  /** Absolute joint angles in policy order — NOT offsets from DEFAULT_POSE. */
  readonly pose: Float32Array;
}

/** A validated motion. Only parseMotion() produces one. */
export interface Motion {
  readonly format: 1;
  readonly name: string;
  readonly description: string;
  readonly loop: boolean;
  readonly interp: "linear" | "cubic";
  /** Ascending, first t is 0, at least one entry. */
  readonly keyframes: readonly MotionKeyframe[];
  /** Total length in seconds: the last keyframe's t. 0 for a static pose. */
  readonly duration: number;
  /** Which joints the file actually named, for the UI. */
  readonly driven: readonly string[];
}

export class MotionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MotionError";
  }
}

const DEG = Math.PI / 180;
/** Rounding slack on JOINT_LIMITS, so a pose written to the stated limit in
 *  degrees is not rejected by the last decimal place. */
const LIMIT_EPS = 1e-3;

function fail(message: string): never {
  throw new MotionError(message);
}

function checkKeys(obj: Record<string, unknown>, allowed: readonly string[], where: string): void {
  for (const key of Object.keys(obj)) {
    if (!allowed.includes(key)) {
      fail(`${where}: unknown field "${key}". Allowed: ${allowed.join(", ")}.`);
    }
  }
}

const FILE_KEYS = [
  "format", "name", "description", "loop", "units", "interp", "joints", "keyframes",
] as const;
const FRAME_KEYS = ["t", "pose"] as const;

/**
 * Validate and normalise a motion file.
 *
 * Accepts a JSON string or an already-parsed object, so the same path serves a
 * fetch, a file input and a literal in the library.
 */
export function parseMotion(input: string | unknown): Motion {
  let raw: unknown = input;
  if (typeof input === "string") {
    try {
      raw = JSON.parse(input);
    } catch (err) {
      fail(`not valid JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    fail("a motion file must be a JSON object");
  }
  const file = raw as Record<string, unknown>;
  checkKeys(file, FILE_KEYS, "motion");

  if (file.format !== undefined && file.format !== MOTION_FORMAT) {
    fail(`unsupported format ${String(file.format)}; this build reads format ${MOTION_FORMAT}`);
  }

  const units = file.units ?? "rad";
  if (units !== "rad" && units !== "deg") fail(`units must be "rad" or "deg", got ${JSON.stringify(units)}`);
  const scale = units === "deg" ? DEG : 1;

  const interp = file.interp ?? "cubic";
  if (interp !== "cubic" && interp !== "linear") {
    fail(`interp must be "cubic" or "linear", got ${JSON.stringify(interp)}`);
  }

  // Unlisted joints hold their reference angle, so a file only has to mention
  // what it changes.
  const driven = file.joints === undefined ? [...JOINT_NAMES] : file.joints;
  if (!Array.isArray(driven) || driven.some((j) => typeof j !== "string")) {
    fail("joints must be an array of joint names");
  }
  if (driven.length === 0) {
    fail('"joints" is empty — a motion has to drive at least one joint');
  }
  const slot: number[] = [];
  const seen = new Set<string>();
  for (const name of driven as string[]) {
    const index = JOINT_NAMES.indexOf(name as (typeof JOINT_NAMES)[number]);
    if (index < 0) {
      fail(`unknown joint "${name}". Valid names: ${JOINT_NAMES.join(", ")}.`);
    }
    if (seen.has(name)) fail(`joint "${name}" is listed twice`);
    seen.add(name);
    slot.push(index);
  }

  const frames = file.keyframes;
  if (!Array.isArray(frames) || frames.length === 0) {
    fail("keyframes must be a non-empty array; a single keyframe is a static pose");
  }

  const parsed: MotionKeyframe[] = frames.map((entry, i) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      fail(`keyframes[${i}] must be an object like { "t": 0, "pose": [...] }`);
    }
    const frame = entry as Record<string, unknown>;
    checkKeys(frame, FRAME_KEYS, `keyframes[${i}]`);
    const t = frame.t;
    if (typeof t !== "number" || !Number.isFinite(t) || t < 0) {
      fail(`keyframes[${i}].t must be a finite number of seconds >= 0`);
    }
    const pose = frame.pose;
    if (!Array.isArray(pose)) fail(`keyframes[${i}].pose must be an array`);
    if (pose.length !== slot.length) {
      fail(
        `keyframes[${i}].pose has ${pose.length} angles but ${slot.length} joints are declared` +
        (file.joints === undefined ? ` (no "joints" field means all ${NUM_JOINTS})` : ""),
      );
    }
    // Starts at the reference pose, so undriven joints are already correct.
    const full = new Float32Array(DEFAULT_POSE);
    pose.forEach((value, k) => {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        fail(`keyframes[${i}].pose[${k}] must be a finite number`);
      }
      const radians = value * scale;
      const joint = slot[k];
      const [lo, hi] = JOINT_LIMITS[joint];
      // Out of range is rejected, not clamped: MuJoCo would clamp it silently
      // and the motion would play back as something nobody wrote. This is also
      // where a degrees-for-radians mix-up surfaces, which is why the message
      // says so.
      if (radians < lo - LIMIT_EPS || radians > hi + LIMIT_EPS) {
        const show = (r: number) => (units === "deg" ? (r / DEG).toFixed(1) : r.toFixed(4));
        fail(
          `keyframes[${i}].pose[${k}] drives "${JOINT_NAMES[joint]}" to ${value} ` +
          `${units}, outside its range ${show(lo)}..${show(hi)} ${units}. ` +
          `If those numbers look like a factor of 57 out, set "units".`,
        );
      }
      full[joint] = radians;
    });
    return { t, pose: full };
  });

  parsed.sort((a, b) => a.t - b.t);
  for (let i = 1; i < parsed.length; i++) {
    if (parsed[i].t === parsed[i - 1].t) fail(`two keyframes share t = ${parsed[i].t}`);
  }
  if (parsed[0].t !== 0) {
    fail(`the first keyframe must be at t = 0, not ${parsed[0].t}`);
  }

  const loop = file.loop === true;
  const duration = parsed[parsed.length - 1].t;
  if (loop) {
    if (parsed.length < 2) fail("a looping motion needs at least two keyframes");
    // The loop period is the last keyframe's time, so the last pose has to BE
    // the first pose — otherwise the cycle contains a jump at the seam and
    // there is no non-arbitrary way to guess how long crossing it should take.
    const first = parsed[0].pose;
    const last = parsed[parsed.length - 1].pose;
    for (let j = 0; j < NUM_JOINTS; j++) {
      if (Math.abs(first[j] - last[j]) > 1e-6) {
        fail(
          `loop is true but the last keyframe (t = ${duration}) does not repeat the first pose ` +
          `— joint "${JOINT_NAMES[j]}" differs. Close the loop by repeating keyframe 0 at the end.`,
        );
      }
    }
  }
  if (parsed.length > 1 && duration <= 0) fail("the motion has no duration");

  return {
    format: MOTION_FORMAT,
    name: typeof file.name === "string" && file.name.trim() ? file.name.trim() : "untitled",
    description: typeof file.description === "string" ? file.description : "",
    loop,
    interp,
    keyframes: parsed,
    duration,
    driven: driven as string[],
  };
}

/** Serialise back to the file shape — always all 14 joints, in radians. */
export function motionToFile(motion: Motion): MotionFile {
  return {
    format: MOTION_FORMAT,
    name: motion.name,
    ...(motion.description ? { description: motion.description } : {}),
    loop: motion.loop,
    units: "rad",
    interp: motion.interp,
    joints: [...JOINT_NAMES],
    keyframes: motion.keyframes.map((k) => ({
      t: Number(k.t.toFixed(4)),
      pose: Array.from(k.pose, (v) => Number(v.toFixed(5))),
    })),
  };
}

/**
 * Policy indices of the joints the file actually drives.
 *
 * Load-bearing for the imitation reward: a two-joint nod leaves twelve joints
 * matching their reference trivially, and averaging the tracking error over
 * all fourteen buries the part being learned under a constant. The driven set
 * is scored on its own.
 */
export function drivenIndices(motion: Motion): number[] {
  return motion.driven
    .map((name) => JOINT_NAMES.indexOf(name as (typeof JOINT_NAMES)[number]))
    .filter((i) => i >= 0)
    .sort((a, b) => a - b);
}

export function motionToJson(motion: Motion): string {
  return JSON.stringify(motionToFile(motion), null, 2);
}
