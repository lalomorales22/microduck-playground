import type { MjData as NativeData, MjModel as NativeModel } from "@mujoco/mujoco";
import { CTRL_DT, DEFAULT_POSE, JOINT_NAMES, NUM_JOINTS, OBS_SIZE, TRUNK_BODY } from "../sim/microduck.ts";
import { compileScene, type ModelAssets } from "../sim/scene.ts";
import { ActorCritic } from "./ac-policy.ts";
import { checkpointMetadata, checkpointNetwork } from "./checkpoint-task.ts";
import { buildSpec, taskLabel } from "./env/tasks.ts";
import { VecEnv } from "./env/vec-env.ts";

export const EVALUATION_SECONDS = 3;
export const EVALUATION_ROBOT = "robot_allcollisions-nv.xml";
export const MAX_EVALUATION_CHECKPOINT_BYTES = 32 * 1024 * 1024;
export const JUMP_GATE = Object.freeze({
  heightGain: 0.03, flightSeconds: 0.06, groundedSeconds: 0.08, landingSeconds: 0.5,
  upright: 0.85, flightUpright: 0.75, takeoffVelocity: 0.1, jointLimitExcess: 0.03,
  landingHorizontalSpeed: 0.25, landingVerticalSpeed: 0.15, landingAngularSpeed: 1.5,
});

export interface EvaluationSample {
  time: number;
  rootHeight: number;
  upright: number;
  verticalVelocity: number;
  horizontalSpeed: number;
  angularSpeed: number;
  /** Active contacts between a robot foot and the external world. */
  supportContacts: number;
  /** Active external contacts on any other robot body; self contacts excluded. */
  nonFootContacts: number;
  jointLimitExcess: number;
  finite: boolean;
}

export interface JumpEvaluation {
  passed: boolean;
  heightGain: number;
  flightSeconds: number;
  takeoffTime: number | null;
  landingTime: number | null;
  stableLandingSeconds: number;
  reasons: string[];
}

export interface EvaluationReport {
  version: 1;
  protocol: "upright-replay-v1";
  checkpointName: string;
  checkpointSha256: string;
  task: string;
  iteration: number;
  seconds: number;
  sampleHz: 50;
  worldObjects: number;
  verdict: "passed" | "failed" | "not-qualified";
  reasons: string[];
  metrics: {
    startHeight: number; maxHeight: number; heightGain: number; finalHeight: number;
    finalUpright: number; uprightFraction: number; maxJointLimitExcess: number; finite: boolean;
  };
  jump: JumpEvaluation | null;
}

export function validateEvaluationName(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value) || value.includes("..")) {
    throw new Error("Invalid checkpoint name for evaluation");
  }
  return value;
}

function validateSamples(samples: readonly EvaluationSample[]): void {
  if (!Array.isArray(samples) || samples.length < 2 || samples.length > 1001) throw new Error("Evaluation needs 2..1001 samples");
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (!s || typeof s.finite !== "boolean") throw new Error("Invalid evaluation sample");
    for (const key of ["time", "rootHeight", "upright", "verticalVelocity", "horizontalSpeed", "angularSpeed",
      "supportContacts", "nonFootContacts", "jointLimitExcess"] as const) {
      if (typeof s[key] !== "number" || !Number.isFinite(s[key])) throw new Error(`Invalid evaluation ${key}`);
    }
    if (Math.abs(s.time - i * CTRL_DT) > 1e-7) throw new Error("Evaluation samples must start at zero and use the 50 Hz clock");
    if (Math.abs(s.upright) > 1.000001 || s.horizontalSpeed < 0 || s.angularSpeed < 0 || s.jointLimitExcess < 0 ||
      !Number.isInteger(s.supportContacts) || s.supportContacts < 0 ||
      !Number.isInteger(s.nonFootContacts) || s.nonFootContacts < 0) throw new Error("Invalid evaluation measurement range");
  }
}

/** One deterministic upright-start trial, sampled at 50 Hz. This is not a
 * robustness certificate, and an upright endpoint is not proof of a flip. */
export function evaluateJump(samples: readonly EvaluationSample[]): JumpEvaluation {
  validateSamples(samples);
  const first = samples[0], last = samples.at(-1)!;
  const reasons: string[] = [];
  if (!first.finite || first.upright < JUMP_GATE.upright || first.rootHeight < 0.08 || first.rootHeight > 0.2 ||
    Math.abs(first.verticalVelocity) > 0.05 || first.horizontalSpeed > 0.05 || first.angularSpeed > 0.05) {
    reasons.push("The trial did not start upright and stationary at the STAND pose.");
  }
  if (last.time < EVALUATION_SECONDS - 1e-7) reasons.push("The full three-second evaluation did not complete.");
  if (samples.some(s => !s.finite)) reasons.push("A non-finite policy or simulator state was observed.");
  if (samples.some(s => s.jointLimitExcess > JUMP_GATE.jointLimitExcess)) reasons.push("A joint exceeded its physical limit by more than 0.03 radians.");

  type Flight = { start: number; end: number; landing: number | null; height: number; minUpright: number };
  const flights: Flight[] = [];
  let groundedStart: number | null = null;
  let active: Flight | null = null;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i], previous = samples[i - 1];
    const touching = s.supportContacts + s.nonFootContacts > 0;
    const grounded = s.supportContacts > 0 && s.nonFootContacts === 0 && s.upright >= JUMP_GATE.upright && s.finite;
    if (active) {
      if (touching) { active.landing = i; flights.push(active); active = null; }
      else {
        active.end = i;
        active.height = Math.max(active.height, s.rootHeight);
        active.minUpright = Math.min(active.minUpright, s.upright);
      }
    }
    // STAND begins a few millimetres above the floor. Its initial drop must
    // never count as takeoff: first require a continuous supported interval.
    if (!touching && !active && previous && groundedStart !== null &&
      previous.time - groundedStart >= JUMP_GATE.groundedSeconds - 1e-7 &&
      Math.max(previous.verticalVelocity, s.verticalVelocity) >= JUMP_GATE.takeoffVelocity) {
      active = { start: i, end: i, landing: null, height: s.rootHeight, minUpright: s.upright };
    }
    groundedStart = grounded ? groundedStart ?? s.time : null;
  }
  if (active) flights.push(active);
  const duration = (f: Flight) => samples[f.end].time - samples[f.start].time;
  const qualified = flights.filter(f => duration(f) >= JUMP_GATE.flightSeconds - 1e-7 &&
    f.height - first.rootHeight >= JUMP_GATE.heightGain - 1e-7 && f.minUpright >= JUMP_GATE.flightUpright);
  const best = [...(qualified.length ? qualified : flights)].sort((a, b) => b.height - a.height || duration(b) - duration(a))[0];
  const heightGain = best ? Math.max(0, best.height - first.rootHeight) : 0;
  const flightSeconds = best ? duration(best) : 0;
  if (!best) reasons.push("No upward takeoff after grounded support was measured.");
  if (heightGain < JUMP_GATE.heightGain - 1e-7) reasons.push("Airborne root-height gain was less than 0.03 metres.");
  if (flightSeconds < JUMP_GATE.flightSeconds - 1e-7) reasons.push("Contact-free flight lasted less than 0.06 seconds at 50 Hz sampling.");
  if (best && best.minUpright < JUMP_GATE.flightUpright) reasons.push("The robot tipped too far during the jump.");
  if (best && samples.slice(best.start).some(s => s.nonFootContacts > 0)) reasons.push("A non-foot body touched the world after takeoff.");

  let stableStart: number | null = null;
  if (best?.landing !== null && best?.landing !== undefined) {
    for (let i = best.landing; i < samples.length; i++) {
      const s = samples[i];
      const stable = s.finite && s.supportContacts > 0 && s.nonFootContacts === 0 &&
        s.upright >= JUMP_GATE.upright && s.rootHeight >= Math.max(0.08, first.rootHeight - 0.025) &&
        s.rootHeight <= first.rootHeight + 0.025 && s.horizontalSpeed <= JUMP_GATE.landingHorizontalSpeed &&
        Math.abs(s.verticalVelocity) <= JUMP_GATE.landingVerticalSpeed && s.angularSpeed <= JUMP_GATE.landingAngularSpeed;
      stableStart = stable ? stableStart ?? s.time : null;
    }
  }
  const stableLandingSeconds = stableStart === null ? 0 : last.time - stableStart;
  if (!best || best.landing === null) reasons.push("No supported landing followed the measured flight.");
  if (stableLandingSeconds < JUMP_GATE.landingSeconds - 1e-7) reasons.push("The trial did not end with 0.5 seconds of supported upright, low-speed landing.");
  return { passed: reasons.length === 0, heightGain, flightSeconds,
    takeoffTime: best ? samples[best.start].time : null,
    landingTime: best?.landing === null || best?.landing === undefined ? null : samples[best.landing].time,
    stableLandingSeconds, reasons };
}

/** No checkpoint writes, optimizer updates, sampling noise, recovery, external
 * forces or terminal resets. Physics and commands reuse the training core. */
export async function runCheckpointEvaluation(
  assets: ModelAssets, checkpoint: unknown,
  options: { checkpointName: string; checkpointSha256: string; signal?: AbortSignal; onProgress?: (progress: number) => void },
): Promise<EvaluationReport> {
  const checkpointName = validateEvaluationName(options.checkpointName);
  if (!/^[a-f0-9]{64}$/.test(options.checkpointSha256)) throw new Error("Evaluation requires the checkpoint SHA256");
  const metadata = checkpointMetadata(checkpoint), network = checkpointNetwork(checkpoint);
  if (network.hidden.reduce((sum, width) => sum + width, 0) > 2048) throw new Error("Checkpoint network exceeds the evaluation memory budget");
  if (options.signal?.aborted) throw new DOMException("Evaluation cancelled", "AbortError");
  const { model, standKey } = compileScene(assets, { robotXml: EVALUATION_ROBOT, world: metadata.world });
  let env: VecEnv | undefined;
  try {
    const spec = buildSpec(metadata.task);
    env = new VecEnv({ mujoco: assets.mujoco, model, standKey, count: 1, rng: () => 0.5,
      spec: { ...spec, episodeLengthS: 60, staggerReset: false, terminations: [],
        reset(ctx, _rng, key) {
          ctx.mujoco.mj_resetDataKeyframe(ctx.model, ctx.data, key);
          ctx.data.xfrc_applied.fill(0);
          for (let j = 0; j < NUM_JOINTS; j++) ctx.data.ctrl[j] = DEFAULT_POSE[j];
        } },
    });
    const ac = new ActorCritic(OBS_SIZE, NUM_JOINTS, () => 0.5, 1, network.hidden, null);
    ac.load(network.policy as ReturnType<ActorCritic["serialize"]>);
    const nativeModel = model as unknown as NativeModel;
    const trunk = assets.mujoco.mj_name2id(model, assets.mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY);
    const feet = ["ankle_left", "ankle_right"].map(name => assets.mujoco.mj_name2id(model, assets.mujoco.mjtObj.mjOBJ_BODY.value, name));
    if (trunk < 1 || feet.some(id => id < 1)) throw new Error("Evaluation model is missing the robot root or feet");
    const joints = JOINT_NAMES.map(name => ({
      id: assets.mujoco.mj_name2id(model, assets.mujoco.mjtObj.mjOBJ_JOINT.value, name),
      qpos: model.jnt(name).qposadr,
    }));
    const read = (time: number): EvaluationSample => {
      const data = env!.dataAt(0), nativeData = data as unknown as NativeData;
      let finite = env!.nonFiniteSteps === 0;
      for (const values of [data.qpos, data.qvel, data.ctrl, env!.observations]) {
        for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) finite = false;
      }
      const quat = data.body(trunk).xquat;
      if (Math.abs(Math.hypot(quat[0], quat[1], quat[2], quat[3]) - 1) > 0.01) finite = false;
      const number = (value: number) => { if (Number.isFinite(value)) return value; finite = false; return 0; };
      let supportContacts = 0, nonFootContacts = 0;
      const contacts = nativeData.contact;
      try {
        for (let i = 0; i < nativeData.ncon; i++) {
          const c = contacts.get(i);
          if (!c) throw new Error("Missing simulator contact record");
          try {
            if (c.efc_address < 0 || c.dist > c.includemargin) continue;
            const a = Number(model.geom_bodyid[c.geom1]), b = Number(model.geom_bodyid[c.geom2]);
            const robotA = Number(nativeModel.body_rootid[a]) === trunk, robotB = Number(nativeModel.body_rootid[b]) === trunk;
            if (robotA === robotB) continue;
            if (feet.includes(robotA ? a : b)) supportContacts++; else nonFootContacts++;
          } finally { c.delete(); }
        }
      } finally { contacts.delete(); }
      let jointLimitExcess = 0;
      for (const j of joints) jointLimitExcess = Math.max(jointLimitExcess,
        number(model.jnt_range[j.id * 2] - data.qpos[j.qpos]), number(data.qpos[j.qpos] - model.jnt_range[j.id * 2 + 1]));
      const values = {
        rootHeight: number(data.qpos[2]), upright: Math.max(-1, Math.min(1, number(1 - 2 * (quat[1] ** 2 + quat[2] ** 2)))),
        verticalVelocity: number(data.qvel[2]), horizontalSpeed: number(Math.hypot(data.qvel[0], data.qvel[1])),
        angularSpeed: number(Math.hypot(data.qvel[3], data.qvel[4], data.qvel[5])),
      };
      return { time, ...values, supportContacts, nonFootContacts, jointLimitExcess, finite };
    };
    const samples = [read(0)], steps = Math.round(EVALUATION_SECONDS / CTRL_DT);
    for (let i = 1; i <= steps; i++) {
      if (options.signal?.aborted) throw new DOMException("Evaluation cancelled", "AbortError");
      const action = ac.actMean(env.observations, 1);
      if (action.some(value => !Number.isFinite(value))) throw new Error("Checkpoint produced non-finite evaluation actions");
      env.step(action);
      samples.push(read(i * CTRL_DT));
      if (!samples.at(-1)!.finite) break;
      if (i % 10 === 0) { options.onProgress?.(i / steps); await new Promise(resolve => setTimeout(resolve, 0)); }
    }
    const first = samples[0], last = samples.at(-1)!;
    const jump = metadata.task.kind === "jump" ? evaluateJump(samples) : null;
    const maxHeight = Math.max(...samples.map(s => s.rootHeight));
    return {
      version: 1, protocol: "upright-replay-v1", checkpointName, checkpointSha256: options.checkpointSha256,
      task: taskLabel(metadata.task), iteration: network.iteration, seconds: last.time, sampleHz: 50,
      worldObjects: metadata.world.objects.length,
      verdict: jump ? jump.passed ? "passed" : "failed" : "not-qualified",
      reasons: jump?.reasons ?? ["This task has no behavioral success gate in upright-replay-v1; measured replay is not proof of a learned skill."],
      metrics: { startHeight: first.rootHeight, maxHeight, heightGain: Math.max(0, maxHeight - first.rootHeight),
        finalHeight: last.rootHeight, finalUpright: last.upright,
        uprightFraction: samples.filter(s => s.finite && s.upright >= JUMP_GATE.upright && s.rootHeight >= 0.08).length / samples.length,
        maxJointLimitExcess: Math.max(...samples.map(s => s.jointLimitExcess)), finite: samples.every(s => s.finite) },
      jump,
    };
  } finally {
    env?.dataAt(0).delete();
    model.delete();
  }
}
