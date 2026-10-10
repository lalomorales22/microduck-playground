import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import loadMujoco from "@mujoco/mujoco";
import { CTRL_DT } from "../sim/microduck.ts";
import type { Mujoco } from "../sim/mujoco.ts";
import type { ModelAssets } from "../sim/scene.ts";
import { createWorldObject } from "../sim/world.ts";
import { ActorCritic } from "./ac-policy.ts";
import { TASK_OBSERVATION_SCHEMA } from "./checkpoint-task.ts";
import { EVALUATION_ROBOT, EVALUATION_SECONDS, evaluateJump, runCheckpointEvaluation,
  validateEvaluationName, type EvaluationSample } from "./evaluation.ts";

function trace(): EvaluationSample[] {
  return Array.from({ length: Math.round(EVALUATION_SECONDS / CTRL_DT) + 1 }, (_, i) => ({
    time: i * CTRL_DT, rootHeight: 0.12, upright: 1, verticalVelocity: 0,
    horizontalSpeed: 0, angularSpeed: 0, supportContacts: 4, nonFootContacts: 0,
    jointLimitExcess: 0, finite: true,
  }));
}

function jump(): EvaluationSample[] {
  const samples = trace();
  samples[19].verticalVelocity = 0.5;
  for (let i = 20; i <= 30; i++) {
    samples[i].supportContacts = 0;
    samples[i].rootHeight += 0.04 * Math.sin(Math.PI * (i - 19) / 12);
    samples[i].verticalVelocity = i <= 25 ? 0.5 : -0.5;
  }
  return samples;
}

test("jump gate measures supported takeoff, airborne rise and a sustained final landing", () => {
  const result = evaluateJump(jump());
  assert.equal(result.passed, true);
  assert.deepEqual(result.reasons, []);
  assert(Math.abs(result.heightGain - 0.04) < 1e-9);
  assert(Math.abs(result.flightSeconds - 0.2) < 1e-9);
  assert.equal(result.takeoffTime, 0.4);
  assert.equal(result.landingTime, 0.62);
  assert.equal(result.stableLandingSeconds, 2.38);
});

test("standing upright, even for the whole trial, never qualifies as a jump", () => {
  const result = evaluateJump(trace());
  assert.equal(result.passed, false);
  assert.equal(result.flightSeconds, 0);
  assert.equal(result.heightGain, 0);
  assert.equal(result.takeoffTime, null);
  assert(result.reasons.some(reason => reason.includes("No upward takeoff")));
});

test("the unsupported STAND-keyframe drop cannot be counted as airborne takeoff", () => {
  const samples = trace();
  for (let i = 0; i < 8; i++) {
    samples[i].supportContacts = 0;
    samples[i].rootHeight -= i * 0.001;
    samples[i].verticalVelocity = -0.2;
  }
  samples[0].verticalVelocity = 0;
  const result = evaluateJump(samples);
  assert.equal(result.passed, false);
  assert.equal(result.flightSeconds, 0);
  assert.equal(result.takeoffTime, null);
});

test("rising while still supported and zero-height contact loss fail separate gates", () => {
  const stretching = jump();
  for (const s of stretching) s.supportContacts = 4;
  assert.equal(evaluateJump(stretching).passed, false);
  const noRise = jump();
  for (const s of noRise) s.rootHeight = 0.12;
  const result = evaluateJump(noRise);
  assert.equal(result.passed, false);
  assert(result.flightSeconds > 0);
  assert(result.reasons.some(reason => reason.includes("height gain")));
});

test("separate short contact-free intervals cannot be added into a qualified flight", () => {
  const samples = jump();
  for (const i of [23, 27]) samples[i].supportContacts = 4;
  const result = evaluateJump(samples);
  assert.equal(result.passed, false);
  assert(result.flightSeconds < 0.06);
});

test("falling, body-supported or ungrounded starts do not qualify", () => {
  for (const change of [
    (samples: EvaluationSample[]) => { samples[0].upright = 0; },
    (samples: EvaluationSample[]) => { samples[0].verticalVelocity = -0.3; },
    (samples: EvaluationSample[]) => { for (let i = 0; i < 20; i++) samples[i].nonFootContacts = 1; },
    (samples: EvaluationSample[]) => { for (let i = 0; i < 20; i++) samples[i].supportContacts = 0; },
  ]) {
    const samples = jump(); change(samples);
    assert.equal(evaluateJump(samples).passed, false);
  }
});

test("landing and later toppling is not a successful jump", () => {
  const samples = jump();
  samples.at(-1)!.upright = 0.2;
  samples.at(-1)!.nonFootContacts = 2;
  const result = evaluateJump(samples);
  assert.equal(result.passed, false);
  assert.equal(result.stableLandingSeconds, 0);
  assert(result.reasons.some(reason => reason.includes("non-foot")));
});

test("a body collision followed by recovery cannot hide inside an upright endpoint", () => {
  const samples = jump();
  samples[35].nonFootContacts = 1;
  const result = evaluateJump(samples);
  assert(result.stableLandingSeconds > 0.5);
  assert.equal(result.passed, false);
  assert(result.reasons.some(reason => reason.includes("non-foot")));
});

test("airborne endpoint, excess speed, low landing, and flight inversion fail", () => {
  for (const change of [
    (samples: EvaluationSample[]) => { for (let i = 31; i < samples.length; i++) samples[i].supportContacts = 0; },
    (samples: EvaluationSample[]) => { for (let i = 31; i < samples.length; i++) samples[i].horizontalSpeed = 0.3; },
    (samples: EvaluationSample[]) => { for (let i = 31; i < samples.length; i++) samples[i].angularSpeed = 2; },
    (samples: EvaluationSample[]) => { for (let i = 31; i < samples.length; i++) samples[i].rootHeight = 0.05; },
    (samples: EvaluationSample[]) => { samples[25].upright = -1; },
  ]) {
    const samples = jump(); change(samples);
    assert.equal(evaluateJump(samples).passed, false);
  }
});

test("non-finite state, joint excess and incomplete trials cannot pass", () => {
  const invalid = jump(); invalid[25].finite = false;
  assert.equal(evaluateJump(invalid).passed, false);
  const joint = jump(); joint[25].jointLimitExcess = 0.031;
  assert.equal(evaluateJump(joint).passed, false);
  assert.equal(evaluateJump(jump().slice(0, 100)).passed, false);
});

test("malformed, sparse or non-monotonic sample clocks are refused", () => {
  assert.throws(() => evaluateJump([]));
  assert.throws(() => evaluateJump(new Array(20)));
  for (const patch of [{ time: 0.19 }, { rootHeight: NaN }, { supportContacts: -1 },
    { nonFootContacts: 0.5 }, { upright: 2 }, { horizontalSpeed: -1 }]) {
    const samples = jump(); Object.assign(samples[10], patch);
    assert.throws(() => evaluateJump(samples));
  }
});

test("checkpoint names permit saved IDs but refuse paths and unbounded names", () => {
  for (const name of ["jump-123-a1b2", "named_checkpoint", "run.v2"]) assert.equal(validateEvaluationName(name), name);
  for (const name of ["", ".hidden", "..", "../saved", "a/b", "a\\b", "name\0", "a".repeat(65), null]) {
    assert.throws(() => validateEvaluationName(name));
  }
});

let assetsPromise: Promise<ModelAssets> | undefined;
function localAssets(): Promise<ModelAssets> {
  assetsPromise ??= (async () => {
    const mujoco = await loadMujoco() as unknown as Mujoco;
    const directory = fileURLToPath(new URL("../../public/model/microduck/", import.meta.url));
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, "manifest.json"), "utf8"));
    const entry = manifest.models.find((item: { file: string }) => item.file === EVALUATION_ROBOT);
    assert(entry, "Prepared local evaluation model is present");
    const vfs = new mujoco.MjVFS();
    vfs.addBuffer(EVALUATION_ROBOT, fs.readFileSync(path.join(directory, EVALUATION_ROBOT)));
    for (const mesh of entry.meshes) vfs.addBuffer(`assets/${mesh}`, fs.readFileSync(path.join(directory, "assets", mesh)));
    return { mujoco, vfs, manifest };
  })();
  return assetsPromise;
}

function checkpoint(kind: "jump" | "hold_pose" = "jump") {
  const ac = new ActorCritic(undefined, undefined, () => 0.5, 0.1, [8], null);
  const policy = ac.serialize() as { actor: { w: number[]; b: number[] }[] };
  for (const layer of policy.actor) { layer.w.fill(0); layer.b.fill(0); }
  return { version: 1, iteration: 2, task: kind, taskSpec: { kind }, observationSchema: TASK_OBSERVATION_SCHEMA,
    world: { version: 1, objects: [createWorldObject("ball", "evaluation-ball")] }, policy };
}
const sha256 = (value: string) => createHash("sha256").update(value).digest("hex");

test("actual MuJoCo deterministic partial-checkpoint replay measures no jump and preserves source bytes", async () => {
  const value = checkpoint(), before = JSON.stringify(value), progress: number[] = [];
  const report = await runCheckpointEvaluation(await localAssets(), value, {
    checkpointName: "partial-jump", checkpointSha256: sha256(before), onProgress: p => progress.push(p),
  });
  assert.equal(JSON.stringify(value), before);
  assert.equal(report.checkpointSha256, sha256(before));
  assert.equal(report.iteration, 2);
  assert.equal(report.worldObjects, 1);
  assert.equal(report.seconds, 3);
  assert.equal(report.sampleHz, 50);
  assert.equal(report.verdict, "failed");
  assert.equal(report.jump!.passed, false);
  assert.equal(report.jump!.flightSeconds, 0);
  assert.equal(report.metrics.finite, true);
  assert(report.metrics.uprightFraction >= 0 && report.metrics.uprightFraction <= 1);
  assert.equal(report.metrics.startHeight, 0.12);
  assert(Number.isFinite(report.metrics.finalHeight));
  assert(Number.isFinite(report.metrics.finalUpright));
  assert.equal(progress.at(-1), 1);
  assert(progress.every((p, i) => p >= 0 && p <= 1 && (i === 0 || p > progress[i - 1])));
});

test("other tasks remain explicitly unqualified without a behavioral success gate", async () => {
  const value = checkpoint("hold_pose"), before = JSON.stringify(value);
  const report = await runCheckpointEvaluation(await localAssets(), value, {
    checkpointName: "hold-pose", checkpointSha256: sha256(before),
  });
  assert.equal(report.verdict, "not-qualified");
  assert.equal(report.jump, null);
  assert(report.reasons[0].includes("not proof"));
  assert.equal(report.metrics.finite, true);
  assert.equal(JSON.stringify(value), before);
});

test("evaluation rejects malformed identity/checkpoints and honors cancellation before allocation", async () => {
  const assets = await localAssets(), value = checkpoint();
  const options = { checkpointName: "jump", checkpointSha256: sha256(JSON.stringify(value)) };
  await assert.rejects(runCheckpointEvaluation(assets, value, { ...options, checkpointSha256: "missing" }), /SHA256/);
  await assert.rejects(runCheckpointEvaluation(assets, { ...value, policy: {} }, options), /network shape/);
  const cancelled = new AbortController(); cancelled.abort();
  await assert.rejects(runCheckpointEvaluation(assets, value, { ...options, signal: cancelled.signal }), { name: "AbortError" });
});
