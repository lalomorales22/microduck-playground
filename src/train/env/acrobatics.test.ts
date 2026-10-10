import assert from "node:assert/strict";
import { test } from "node:test";
import { CMD_OFFSET, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE } from "../../sim/microduck.ts";
import { STAND_Z } from "./rewards.ts";
import type { StepState } from "./rewards.ts";
import type { EnvContext } from "./seams.ts";
import {
  ACROBATICS_DURATION_S, acrobaticRootScore, acrobaticTarget, acrobaticsSpec,
  pitchTracking, writeAcrobaticCommand,
} from "./acrobatics.ts";
import type { AcrobaticKind } from "./acrobatics.ts";

const kinds: AcrobaticKind[] = ["jump", "somersault", "backflip"];
const quatAt = (pitch: number) => new Float64Array([Math.cos(pitch / 2), 0, Math.sin(pitch / 2), 0]);
function state(overrides: Partial<StepState> = {}): StepState {
  return { z: STAND_Z, quat: quatAt(0), vz: 0, az: 0,
    jointPos: new Float32Array(DEFAULT_POSE), jointTarget: new Float32Array(DEFAULT_POSE),
    action: new Float32Array(NUM_JOINTS), prevAction: new Float32Array(NUM_JOINTS), ...overrides };
}
function context(time: number): EnvContext {
  return { time, data: { qpos: new Float64Array(21), qvel: new Float64Array(20) } } as unknown as EnvContext;
}

test("episode length must cover the maneuver and be finite", () => {
  for (const value of [-1, 0, 0.1, 60.01, NaN, Infinity, -Infinity]) {
    assert.throws(() => acrobaticsSpec("jump", { episodeLengthS: value }), /episodeLengthS/);
  }
  assert.equal(acrobaticsSpec("jump").episodeLengthS, 3);
  assert.equal(acrobaticsSpec("jump", { episodeLengthS: ACROBATICS_DURATION_S }).episodeLengthS, ACROBATICS_DURATION_S);
  assert.equal(acrobaticsSpec("jump", { episodeLengthS: 60 }).episodeLengthS, 60);
  for (const kind of kinds) assert.equal(acrobaticsSpec(kind).name, kind);
  assert.throws(() => acrobaticsSpec("unknown" as AcrobaticKind), /Unknown/);
});

test("targets stay finite, bounded and end upright at standing height", () => {
  for (const kind of kinds) {
    for (let i = -10; i <= 400; i++) {
      const t = acrobaticTarget(kind, i / 100);
      for (const value of [t.progress, t.direction, t.pitch, t.height, t.verticalVelocity]) assert.ok(Number.isFinite(value));
      assert.ok(t.progress >= 0 && t.progress <= 1);
      assert.ok(t.height >= STAND_Z - 0.025 - 1e-8 && t.height <= STAND_Z + 0.15 + 1e-8);
      assert.ok(Math.abs(t.verticalVelocity) < 2);
      if (kind === "jump") assert.equal(t.pitch, 0);
    }
    const end = acrobaticTarget(kind, 10);
    assert.equal(end.phase, "settle");
    assert.equal(end.height, STAND_Z);
    assert.equal(end.verticalVelocity, 0);
    assert.equal(pitchTracking(quatAt(end.pitch), 0), 1);
  }
  assert.throws(() => acrobaticTarget("jump", NaN), /finite/);
});

test("quaternion tracking is normalized and sign invariant", () => {
  for (const pitch of [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI]) {
    const q = quatAt(pitch);
    assert.ok(Math.abs(pitchTracking(q, pitch) - 1) < 1e-12);
    assert.equal(pitchTracking(q, pitch), pitchTracking(q.map((n) => -n), pitch));
    assert.equal(pitchTracking(q, pitch), pitchTracking(q.map((n) => 3 * n), pitch));
  }
  assert.equal(pitchTracking(new Float64Array(4), 0), 0);
  assert.equal(pitchTracking(new Float64Array([NaN, 0, 0, 0]), 0), 0);
  assert.ok(pitchTracking(quatAt(Math.PI), 0) < 1e-12);
});

test("directed quarter-turn tracking rejects the opposite rotation", () => {
  for (const kind of ["somersault", "backflip"] as const) {
    const target = acrobaticTarget(kind, 0.5375);
    const correct = pitchTracking(quatAt(target.pitch), target.pitch);
    const reverse = pitchTracking(quatAt(-target.pitch), target.pitch);
    assert.ok(correct > 0.999);
    assert.ok(reverse < correct - 0.6);
    assert.equal(Math.sign(target.pitch), kind === "somersault" ? 1 : -1);
  }
});

test("root reward couples pitch, bounded height and vertical velocity", () => {
  for (const kind of kinds) {
    for (let i = 0; i <= 120; i++) {
      const time = i / 100;
      const target = acrobaticTarget(kind, time);
      const ideal = state({ z: target.height, vz: target.verticalVelocity, quat: quatAt(target.pitch) });
      assert.ok(acrobaticRootScore(kind, time, ideal) > 0.999);
      const fallen = state({ z: 0.03, quat: quatAt(Math.PI), vz: 0 });
      assert.ok(acrobaticRootScore(kind, time, fallen) < 0.02);
      for (const s of [fallen, state({ z: 1e6, vz: 1e9 }), state({ z: NaN }), state({ vz: Infinity })]) {
        const score = acrobaticRootScore(kind, time, s);
        assert.ok(Number.isFinite(score) && score >= 0 && score <= 1);
      }
    }
  }
});

test("standing still and rotating backwards cannot match the flip trajectory", () => {
  for (const kind of ["somersault", "backflip"] as const) {
    let ideal = 0; let still = 0; let reverse = 0;
    for (let i = 0; i <= 32; i++) {
      const time = 0.45 + 0.35 * i / 32;
      const t = acrobaticTarget(kind, time);
      const sameHeight = state({ z: t.height, vz: t.verticalVelocity });
      ideal += acrobaticRootScore(kind, time, { ...sameHeight, quat: quatAt(t.pitch) });
      reverse += acrobaticRootScore(kind, time, { ...sameHeight, quat: quatAt(-t.pitch) });
      still += acrobaticRootScore(kind, time, state());
    }
    assert.ok(ideal > reverse * 1.5);
    assert.ok(ideal > still * 5);
  }
});

test("commands overwrite exactly one row's 13 target-only slots", () => {
  const obs = new Float32Array(OBS_SIZE * 3).fill(99);
  writeAcrobaticCommand("backflip", 0.6, obs, OBS_SIZE);
  for (let i = 0; i < obs.length; i++) {
    if (i >= OBS_SIZE + CMD_OFFSET && i < OBS_SIZE * 2) assert.ok(Number.isFinite(obs[i]) && obs[i] !== 99);
    else assert.equal(obs[i], 99);
  }
  const alone = new Float32Array(OBS_SIZE);
  writeAcrobaticCommand("backflip", 0.6, alone);
  assert.deepEqual(obs.slice(OBS_SIZE + CMD_OFFSET, OBS_SIZE * 2), alone.slice(CMD_OFFSET));
  assert.equal(alone[CMD_OFFSET + 6], -1);
  assert.equal(alone[CMD_OFFSET + 9], 1);
  for (const offset of [-1, 0.5, Infinity, OBS_SIZE]) {
    assert.throws(() => writeAcrobaticCommand("jump", 0, alone, offset), /row|offset/);
  }
  assert.throws(() => writeAcrobaticCommand("jump", 0, new Float32Array(60)), /row/);
});

test("landing reward is absent in flight and penalizes residual motion", () => {
  const spec = acrobaticsSpec("somersault");
  const landing = spec.rewards.find((r) => r.name === "landing_stability")!;
  assert.equal(landing.compute(context(0.6), state()), 0);
  assert.equal(landing.compute(context(1), state()), 1);
  for (const index of [0, 1, 2, 3, 4, 5]) {
    const ctx = context(1);
    ctx.data.qvel[index] = 5;
    const s = index === 2 ? state({ vz: 5 }) : state();
    assert.ok(landing.compute(ctx, s) < 0.01);
  }
  const fallen = state({ z: 0.03, quat: quatAt(Math.PI) });
  assert.ok(landing.compute(context(1), fallen) < 1e-8);
});

test("inversion is allowed during flight but late failure and invalid state terminate", () => {
  const spec = acrobaticsSpec("backflip");
  const inverted = state({ z: 0.2, quat: quatAt(Math.PI) });
  assert.equal(spec.terminations.some((t) => t.check(context(0.6), inverted)), false);
  assert.equal(spec.terminations.some((t) => t.check(context(1.3), inverted)), true);
  const ctx = context(0.6);
  ctx.data.qvel[4] = NaN;
  assert.equal(spec.terminations.some((t) => t.check(ctx, state())), true);
  assert.equal(spec.terminations.some((t) => t.check(context(0.6), state({ quat: new Float64Array(4) }))), true);
});

test("malformed step fields terminate without poisoning reward breakdowns", () => {
  const spec = acrobaticsSpec("backflip");
  for (const key of ["jointPos", "action", "prevAction", "quat"] as const) {
    const malformed = state();
    malformed[key][0] = NaN;
    for (const term of spec.rewards) assert.ok(Number.isFinite(term.compute(context(1), malformed)), term.name);
    assert.equal(spec.terminations.some((t) => t.check(context(1), malformed)), true);
  }
  const malformed = state({ vz: NaN });
  for (const term of spec.rewards) assert.ok(Number.isFinite(term.compute(context(1), malformed)), term.name);
  assert.equal(spec.terminations.some((t) => t.check(context(1), malformed)), true);
});

test("reset uses only the standing keyframe and joint controls; metrics do not claim success", () => {
  const spec = acrobaticsSpec("jump");
  const qpos = new Float64Array(21).fill(3);
  const qvel = new Float64Array(20).fill(4);
  const ctrl = new Float64Array(NUM_JOINTS);
  let resetKey = -1;
  const ctx = { time: 0, model: {}, data: { qpos, qvel, ctrl },
    mujoco: { mj_resetDataKeyframe(_model: unknown, _data: unknown, key: number) { resetKey = key; } },
  } as unknown as EnvContext;
  spec.reset(ctx, () => 0.5, 7);
  assert.equal(resetKey, 7);
  assert.equal(spec.staggerReset, false);
  assert.deepEqual(Array.from(ctrl), Array.from(DEFAULT_POSE));
  assert.ok(qpos.every((n) => n === 3) && qvel.every((n) => n === 4));
  assert.deepEqual(spec.metrics!.map((m) => m.name), ["root_height_m", "pitch_tracking"]);
  assert.equal(spec.metrics![0].compute(ctx, state({ z: 0.123 })), 0.123);
  assert.equal(spec.metrics![1].compute(ctx, state()), 1);
});
