// What a run is training on, in a form that survives postMessage.
//
// The learner and every rollout worker have to build the SAME environment, and
// they are in different threads, so the task cannot be an EnvSpec — it has to
// be data. For hold-pose and stand-up a tag is enough; a motion task carries
// its file, which is why the workers take a MotionFile rather than a name they
// would each have to load from storage and could disagree about.

import { parseMotion, type MotionFile } from "../../motion/format.ts";
import { motionSpec } from "./motion.ts";
import { holdPoseSpec, standupSpec } from "./standup.ts";
import type { EnvSpec } from "./vec-env.ts";
import { CMD_OFFSET, CMD_SIZE } from "../../sim/microduck.ts";
import { MotionSampler } from "../../motion/sampler.ts";
import { locomotionSpec, writeLocomotionCommand, type LocomotionKind } from "./locomotion.ts";
import { acrobaticsSpec, writeAcrobaticCommand, type AcrobaticKind } from "./acrobatics.ts";

export const BUILTIN_TASKS = [
  { kind: "hold_pose", label: "Hold the pose", experimental: false, metricLabel: "Standing" },
  { kind: "standup", label: "Stand up", experimental: false, metricLabel: "Standing" },
  { kind: "walk_forward", label: "Walk forward", experimental: false, metricLabel: "Forward velocity" },
  { kind: "walk_backward", label: "Walk backward", experimental: false, metricLabel: "Forward velocity" },
  { kind: "turn_left", label: "Turn left", experimental: false, metricLabel: "Yaw rate" },
  { kind: "turn_right", label: "Turn right", experimental: false, metricLabel: "Yaw rate" },
  { kind: "jump", label: "Jump", experimental: true, metricLabel: "Root height" },
  { kind: "somersault", label: "Forward somersault", experimental: true, metricLabel: "Pitch tracking" },
  { kind: "backflip", label: "Backflip", experimental: true, metricLabel: "Pitch tracking" },
] as const;
export type BuiltinTaskKind = (typeof BUILTIN_TASKS)[number]["kind"];

export type TaskSpec =
  | { kind: BuiltinTaskKind }
  | { kind: "motion"; motion: MotionFile };

export interface TaskOptions {
  episodeLengthS?: number;
}

export function validateTask(value: unknown): TaskSpec {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("invalid task descriptor");
  const task = value as Record<string, unknown>;
  const keys = task.kind === "motion" ? ["kind", "motion"] : ["kind"];
  if (Object.keys(task).some(key => !keys.includes(key))) throw new Error("unsupported task parameter");
  if (task.kind === "motion") {
    if (!task.motion || typeof task.motion !== "object" || Array.isArray(task.motion)) throw new Error("motion task must contain a motion object");
    parseMotion(task.motion);
    return { kind: "motion", motion: structuredClone(task.motion) as MotionFile };
  }
  if (BUILTIN_TASKS.some(item => item.kind === task.kind)) return { kind: task.kind as BuiltinTaskKind };
  throw new Error(`unknown task kind: ${String(task.kind)}`);
}

/** Shared with checkpoint replay, so the training and deployment commands agree. */
export function taskCommandWriter(value: TaskSpec): (time: number, obs: Float32Array, offset?: number) => void {
  const task = validateTask(value);
  const sampler = task.kind === "motion" ? new MotionSampler(parseMotion(task.motion)) : null;
  return (time, obs, offset = 0) => {
    if (!Number.isFinite(time) || time < 0) throw new Error("invalid task clock");
    if (!Number.isInteger(offset) || offset < 0 || obs.length < offset + CMD_OFFSET + CMD_SIZE) throw new Error("task observation row is invalid or too short");
    obs.fill(0, offset + CMD_OFFSET, offset + CMD_OFFSET + CMD_SIZE);
    if (sampler) {
      const angle = sampler.phase(time) * Math.PI * 2;
      obs[offset + CMD_OFFSET] = Math.sin(angle);
      obs[offset + CMD_OFFSET + 1] = Math.cos(angle);
    } else if (["walk_forward", "walk_backward", "turn_left", "turn_right"].includes(task.kind)) {
      writeLocomotionCommand(task.kind as LocomotionKind, time, obs, offset);
    } else if (["jump", "somersault", "backflip"].includes(task.kind)) {
      writeAcrobaticCommand(task.kind as AcrobaticKind, time, obs, offset);
    }
  };
}

export function buildSpec(task: TaskSpec, options: TaskOptions = {}): EnvSpec {
  task = validateTask(task);
  if (options.episodeLengthS !== undefined && (!Number.isFinite(options.episodeLengthS) || options.episodeLengthS < 0.1 || options.episodeLengthS > 60)) throw new Error("invalid episode length");
  switch (task.kind) {
    case "hold_pose":
      return holdPoseSpec(options);
    case "standup":
      return standupSpec(options);
    case "motion":
      // Parsed here, in the thread that will use it: validation is cheap and a
      // file that somehow arrived malformed should fail with the parser's
      // message rather than as an undefined halfway through a rollout.
      return motionSpec(parseMotion(task.motion), options);
    case "walk_forward": case "walk_backward": case "turn_left": case "turn_right":
      return locomotionSpec(task.kind, options);
    case "jump": case "somersault": case "backflip":
      return acrobaticsSpec(task.kind, options);
  }
}

export function taskLabel(task: TaskSpec): string {
  if (task.kind === "motion") return `motion:${parseMotion(task.motion).name}`;
  return task.kind;
}
