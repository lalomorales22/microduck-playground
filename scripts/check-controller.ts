// Deterministic controller gate; check-world.ts separately exercises real WASM physics.
import assert from "node:assert/strict";
import { MicroduckController } from "../src/sim/controller.ts";
import { CMD_OFFSET, CTRL_DT, DECIMATION, DEFAULT_POSE, GYRO_SENSOR, JOINT_NAMES,
  NUM_JOINTS, OBS_SIZE, TRUNK_BODY } from "../src/sim/microduck.ts";
import type { PolicyRunner } from "../src/sim/policy.ts";
import type { Simulation } from "../src/sim/scene.ts";

const ACTION_OFFSET = CMD_OFFSET - NUM_JOINTS;
const close = (actual: number, expected: number, message: string) =>
  assert(Math.abs(actual - expected) < 1e-6, `${message}: ${actual} != ${expected}`);

function simulationFixture() {
  const counts = { steps: 0, resets: 0, forwards: 0, forcedSubsteps: 0 };
  const qpos = new Float64Array(7 + NUM_JOINTS);
  const qvel = new Float64Array(6 + NUM_JOINTS);
  const xquat = new Float64Array([1, 0, 0, 0]);
  const data = {
    qpos, qvel, ctrl: new Float64Array(NUM_JOINTS), sensordata: new Float64Array(3),
    xfrc_applied: new Float64Array(12),
    time: 0,
    body(id: number) { assert.equal(id, 1); return { xquat }; },
  };
  const model = {
    jnt(name: string) {
      const index = JOINT_NAMES.indexOf(name as typeof JOINT_NAMES[number]);
      assert(index >= 0, `known joint ${name}`);
      return { qposadr: 7 + index, dofadr: 6 + index };
    },
    sensor(name: string) { assert.equal(name, GYRO_SENSOR); return { adr: 0 }; },
  };
  function stand() {
    qpos.fill(0); qvel.fill(0);
    qpos[2] = 0.12; qpos[3] = 1;
    qpos.set(DEFAULT_POSE, 7);
    data.ctrl.set(DEFAULT_POSE);
    data.sensordata.fill(0);
    data.time = 0;
    xquat.set([1, 0, 0, 0]);
  }
  const mujoco = {
    mjtObj: { mjOBJ_BODY: { value: 1 } },
    mj_name2id(actualModel: unknown, _type: number, name: string) {
      assert.equal(actualModel, model); assert.equal(name, TRUNK_BODY); return 1;
    },
    mj_step(actualModel: unknown, actualData: unknown) {
      assert.equal(actualModel, model); assert.equal(actualData, data); counts.steps++;
      data.time += CTRL_DT / DECIMATION;
      if (data.xfrc_applied.some(x => x !== 0)) counts.forcedSubsteps++;
    },
    mj_resetDataKeyframe(actualModel: unknown, actualData: unknown, key: number) {
      assert.equal(actualModel, model); assert.equal(actualData, data); assert.equal(key, 0);
      counts.resets++; stand();
    },
    mj_forward(actualModel: unknown, actualData: unknown) {
      assert.equal(actualModel, model); assert.equal(actualData, data); counts.forwards++;
    },
  };
  stand();
  return {
    sim: { mujoco, model, data, standKey: 0 } as unknown as Simulation,
    data, counts,
    inverted() { xquat.set([0, 1, 0, 0]); qpos.set(xquat, 3); },
    upright() { xquat.set([1, 0, 0, 0]); qpos.set(xquat, 3); },
  };
}

function policyFixture(automaticRecovery?: boolean, value = 0.25, label = "test policy") {
  const state = { resets: 0, clock: 0, observations: [] as Float32Array[], times: [] as number[] };
  const policy: PolicyRunner = {
    label,
    ...(automaticRecovery === undefined ? {} : { automaticRecovery }),
    reset() { state.resets++; state.clock = 0; },
    async run(obs: Float32Array) {
      assert.equal(obs.length, OBS_SIZE);
      state.observations.push(obs.slice());
      state.times.push(state.clock);
      state.clock += CTRL_DT;
      return new Float32Array(NUM_JOINTS).fill(value);
    },
  };
  return { policy, state };
}

async function steps(controller: MicroduckController, count: number): Promise<void> {
  for (let i = 0; i < count; i++) await controller.step();
}

// An acrobatics/motion policy must own its complete timeline, including inversion.
{
  const fixture = simulationFixture(), { policy, state } = policyFixture(false);
  const controller = new MicroduckController(fixture.sim, policy);
  fixture.inverted();
  for (let i = 1; i <= 80; i++) {
    await controller.step();
    assert.equal(state.observations.length, i, "inverted acrobatics runs on every control tick");
    assert.equal(controller.telemetry().phase, "standing", "no automatic settle interrupts the motion");
    assert.equal(controller.telemetry().phaseSteps, i);
    close(state.times[i - 1], (i - 1) * CTRL_DT, "policy clock advances without a skipped tick");
    close(controller.telemetry().simTime, i * CTRL_DT, "simulation clock advances");
    assert.equal(state.observations[i - 1][5], 1, "the policy really sees an inverted trunk");
  }
  assert.equal(fixture.counts.steps, 80 * DECIMATION);
  assert.equal(state.resets, 0, "inversion never restarts the policy timeline");
}
console.log("PASS automaticRecovery=false: 80 inverted ticks keep policy, clock and physics running");

// Explicit true and the legacy absent flag must retain the fall/settle/get-up FSM.
for (const automaticRecovery of [true, undefined]) {
  const fixture = simulationFixture(), { policy, state } = policyFixture(automaticRecovery);
  const controller = new MicroduckController(fixture.sim, policy);
  const outcomes: boolean[] = [];
  controller.onRecoveryEnd = (success) => outcomes.push(success);
  fixture.inverted();
  await steps(controller, 14);
  assert.equal(controller.telemetry().phase, "standing");
  await controller.step();
  assert.equal(controller.telemetry().phase, "settling");
  assert.equal(state.observations.length, 15);
  const heldControl = fixture.data.ctrl.slice();
  await steps(controller, 14);
  assert.equal(controller.telemetry().phase, "settling");
  assert.deepEqual(fixture.data.ctrl, heldControl, "settling freezes the last targets");
  await controller.step();
  assert.equal(controller.telemetry().phase, "recovering");
  assert.equal(state.observations.length, 15, "all 15 settle ticks pause inference");
  await controller.step();
  assert.equal(state.observations.length, 16);
  assert(state.observations.at(-1)!.slice(ACTION_OFFSET, CMD_OFFSET).every(x => x === 0),
    "recovery starts with zero previous action");
  fixture.upright();
  await steps(controller, 49);
  assert.equal(controller.telemetry().phase, "recovering");
  await controller.step();
  assert.equal(controller.telemetry().phase, "standing");
  assert.deepEqual(outcomes, [true]);
  assert.equal(fixture.counts.steps, 81 * DECIMATION);
}
console.log("PASS automaticRecovery=true/default: debounce, frozen settle and upright recovery retained");

{
  const fixture = simulationFixture(), { policy } = policyFixture(true);
  const controller = new MicroduckController(fixture.sim, policy);
  fixture.inverted();
  await steps(controller, 14);
  fixture.upright();
  await controller.step();
  fixture.inverted();
  await steps(controller, 14);
  assert.equal(controller.telemetry().phase, "standing", "one upright tick resets consecutive fall debounce");
  await controller.step();
  assert.equal(controller.telemetry().phase, "settling");
}
{
  const fixture = simulationFixture(), { policy, state } = policyFixture(true);
  const controller = new MicroduckController(fixture.sim, policy);
  const outcomes: boolean[] = [];
  controller.onRecoveryEnd = (success) => outcomes.push(success);
  fixture.inverted();
  controller.standUp();
  assert.equal(state.resets, 1, "manual get-up starts the policy timeline over");
  await steps(controller, 299);
  assert.equal(controller.telemetry().phase, "recovering");
  assert.deepEqual(outcomes, []);
  await controller.step();
  assert.equal(controller.telemetry().phase, "standing");
  assert.deepEqual(outcomes, [false], "failed recovery reports once at 300 ticks");
  await controller.step();
  assert.deepEqual(outcomes, [false]);
}
console.log("PASS one upright tick resets fall debounce; failed recovery reports once at its 300-tick boundary");

// Switching away from a get-up attempt cannot carry fallen physics or history.
for (const phase of ["settling", "recovering", "limp"] as const) {
  const fixture = simulationFixture(), old = policyFixture(true, 0.5, "old");
  const controller = new MicroduckController(fixture.sim, old.policy);
  controller.setCommand(0.1, -0.1, 0.5);
  fixture.inverted();
  await steps(controller, 15);
  if (phase === "recovering") controller.standUp();
  if (phase === "limp") controller.goLimp();
  assert.equal(controller.telemetry().phase, phase);
  fixture.data.qvel.fill(1);
  fixture.data.sensordata.fill(2);
  controller.push();
  const replacement = policyFixture(false, -0.25, "acrobatics");
  controller.setPolicy(replacement.policy);
  assert.equal(controller.policyLabel, "acrobatics");
  assert.equal(controller.telemetry().phase, "standing");
  assert.equal(controller.telemetry().phaseSteps, 0);
  assert.equal(controller.telemetry().simTime, 0, "policy switch restarts simulation time");
  assert.equal(fixture.data.time, 0, "physics clock also restarts");
  assert.equal(controller.gravityZ, -1, "new policy starts upright even after a fallen recovery");
  assert.deepEqual(Array.from(fixture.data.qpos.slice(0, 7)), [0, 0, 0.12, 1, 0, 0, 0]);
  assert.deepEqual(Array.from(fixture.data.ctrl), Array.from(DEFAULT_POSE), "old actuator targets are cleared");
  assert(fixture.data.qvel.every(x => x === 0), "old velocities are cleared");
  assert(fixture.data.xfrc_applied.every(x => x === 0), "old shove is cleared");
  assert.equal(replacement.state.resets, 1, "new policy timeline resets once");
  await steps(controller, 40);
  const first = replacement.state.observations[0];
  assert.deepEqual(Array.from(first.slice(0, 6), x => x || 0), [0, 0, 0, 0, 0, -1], "first observation is upright and stationary");
  assert(first.slice(6, ACTION_OFFSET).every(x => x === 0), "first joint offsets and velocities are zero");
  assert(first.slice(ACTION_OFFSET, CMD_OFFSET).every(x => x === 0), "old actions are cleared");
  assert(first.slice(CMD_OFFSET).every(x => x === 0), "old commands are cleared");
  assert.equal(replacement.state.times[0], 0);
  assert.equal(replacement.state.observations.length, 40);
  assert.equal(controller.telemetry().phase, "standing");
}
{
  const fixture = simulationFixture(), old = policyFixture(true);
  const controller = new MicroduckController(fixture.sim, old.policy);
  fixture.inverted();
  await steps(controller, 14);
  controller.setPolicy(policyFixture(true).policy);
  fixture.inverted();
  await steps(controller, 14);
  assert.equal(controller.telemetry().phase, "standing", "policy switch clears prior fall debounce");
  await controller.step();
  assert.equal(controller.telemetry().phase, "settling");
}
console.log("PASS setPolicy restores upright STAND physics, clears history/forces and restarts both clocks");

{
  const fixture = simulationFixture(), { policy, state } = policyFixture(false);
  const controller = new MicroduckController(fixture.sim, policy);
  fixture.inverted();
  await steps(controller, 4);
  controller.setCommand(0.1, -0.1, 0.5);
  controller.push();
  assert(fixture.data.xfrc_applied.some(x => x !== 0));
  controller.reset();
  assert.equal(fixture.counts.resets, 1);
  assert.equal(fixture.counts.forwards, 1);
  assert.equal(state.resets, 1);
  assert.equal(controller.telemetry().simTime, 0);
  assert.equal(controller.telemetry().phase, "standing");
  assert.equal(controller.telemetry().phaseSteps, 0);
  assert.equal(controller.gravityZ, -1);
  assert(fixture.data.xfrc_applied.every(x => x === 0));
  await controller.step();
  assert.equal(state.times.at(-1), 0, "reset restarts policy time");
  assert(state.observations.at(-1)!.slice(ACTION_OFFSET, CMD_OFFSET).every(x => x === 0));
  assert(state.observations.at(-1)!.slice(CMD_OFFSET).every(x => x === 0), "reset clears commands");
  close(controller.telemetry().simTime, CTRL_DT, "clock resumes after reset");
}
console.log("PASS reset restores STAND, clears force/action history and resets policy time");

{
  const fixture = simulationFixture(), { policy } = policyFixture(false);
  const controller = new MicroduckController(fixture.sim, policy);
  controller.push(0.5);
  for (let tick = 1; tick <= 3; tick++) {
    await controller.step();
    assert.equal(fixture.counts.forcedSubsteps, tick * DECIMATION,
      "the shove remains applied during every substep of all three control ticks");
  }
  assert(fixture.data.xfrc_applied.every(x => x === 0), "force clears after the third full control step");
  await controller.step();
  assert.equal(fixture.counts.forcedSubsteps, 12, "the fourth control tick receives no shove");
  assert.equal(fixture.counts.steps, 16);
}
console.log("PASS push applies force for exactly 3 complete control ticks / 12 physics substeps");

{
  const fixture = simulationFixture(), { policy, state } = policyFixture(false);
  const controller = new MicroduckController(fixture.sim, policy);
  controller.setCommand(100, -100, 100);
  for (const bad of [NaN, Infinity, -Infinity]) {
    assert.throws(() => controller.setCommand(bad, 0, 0), /finite/);
    assert.throws(() => controller.setCommand(0, bad, 0), /finite/);
    assert.throws(() => controller.setCommand(0, 0, bad), /finite/);
  }
  await controller.step();
  const command = state.observations[0].slice(CMD_OFFSET);
  close(command[0], 0.25, "forward clamp");
  close(command[1], -0.15, "lateral clamp");
  close(command[2], 1, "yaw clamp");
  assert(command.slice(3).every(x => x === 0), "unused command channels stay zero");
}
console.log("PASS commands clamp to bounds and reject nonfinite input atomically");
console.log("CONTROLLER CHECKS PASSED");
