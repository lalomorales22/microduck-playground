import assert from "node:assert/strict";
import { test } from "node:test";
import { assertRunRequested, executeAssistantCommand, type AssistantCommandPorts } from "./commands.ts";
import type { AssistantCommand, AssistantContext } from "./contract.ts";

function context(): AssistantContext {
  return { world: { version: 1, objects: [] }, motions: [], currentMotion: null,
    training: { task: "jump", iterations: 50, seed: 7, envs: 16, rolloutWorkers: 1,
      useCurrentWorld: false, hiddenPreset: "small", status: "idle", iteration: 0, reward: null },
    checkpoints: [{ name: "jump-test", iteration: null, modified: 100, bytes: 2048 }],
    run: { runId: null, task: null, status: "idle", completionReason: null, iteration: 0,
      targetIteration: 0, totalSteps: 0, reward: null, standing: null, checkpoint: null, evaluation: null } };
}
test("start receipts require the exact live request, not a canceled or replaced run", () => {
  for (const status of ["loading", "running"]) assertRunRequested("requested", { runId: "requested", status, error: null });
  for (const status of ["idle", "stopped", "stopping"]) {
    assert.throws(() => assertRunRequested("requested", { runId: "requested", status, error: null }), /canceled/);
  }
  assert.throws(() => assertRunRequested("requested", { runId: "other", status: "loading", error: null }), /replaced/);
  assert.throws(() => assertRunRequested(null, { runId: null, status: "loading", error: null }), /canceled/);
  assert.throws(() => assertRunRequested("requested", { runId: "requested", status: "error", error: "Storage failed" }), /Storage failed/);
});
function harness() {
  const captured = context(), current = structuredClone(captured), calls: string[] = [];
  const ports: AssistantCommandPorts = {
    getContext: async () => structuredClone(current),
    startTraining: async () => { calls.push("start"); return "Loading requested, not trained."; },
    resumeTraining: async name => { calls.push(`resume:${name}`); return "Continuation requested in a new checkpoint."; },
    stopTraining: id => { calls.push(`stop:${id}`); return "Stopping requested."; },
    evaluateCheckpoint: async name => { calls.push(`evaluate:${name}`); return "Measured result."; },
    replayCheckpoint: async name => { calls.push(`replay:${name}`); return "Loaded paused."; },
  };
  const execute = (kind: AssistantCommand["kind"], target: string | null = null) =>
    executeAssistantCommand({ kind, target }, captured, ports);
  return { captured, current, calls, ports, execute };
}

test("approved start dispatches once and returns lifecycle receipt verbatim", async () => {
  const h = harness(); assert.equal(await h.execute("start_training"), "Loading requested, not trained.");
  assert.deepEqual(h.calls, ["start"]);
});
for (const field of ["iterations", "seed", "envs", "rolloutWorkers", "task", "hiddenPreset", "useCurrentWorld"] as const) {
  test(`stale ${field} refuses to start`, async () => {
    const h = harness();
    Object.assign(h.current.training, { [field]: ({ iterations: 51, seed: 8, envs: 32,
      rolloutWorkers: 2, task: "hold_pose", hiddenPreset: "reference", useCurrentWorld: true })[field] });
    await assert.rejects(h.execute("start_training")); assert.deepEqual(h.calls, []);
  });
}
test("changed current-world snapshot refuses start only when it will be used", async () => {
  const h = harness(); h.current.world.objects = [];
  h.captured.training.useCurrentWorld = h.current.training.useCurrentWorld = true;
  const { createWorldObject } = await import("../sim/world.ts");
  h.current.world.objects.push(createWorldObject("box", "new-box"));
  await assert.rejects(h.execute("start_training"), /changed/); assert.deepEqual(h.calls, []);
});
test("new completed run invalidates old start even with unchanged setup", async () => {
  const h = harness(); h.current.run!.runId = "new-run";
  await assert.rejects(h.execute("start_training"), /changed/); assert.deepEqual(h.calls, []);
});
test("motion starts require the reviewed saved revision, not only the same name", async () => {
  const h = harness();
  for (const c of [h.captured, h.current]) {
    c.training.task = "motion:saved-nod";
    c.motions = [{ id: "saved-nod", name: "Nod", bytes: 100, modified: 1234 }];
  }
  await h.execute("start_training");
  h.current.motions[0].modified = 1235;
  await assert.rejects(h.execute("start_training"), /motion changed/);
  assert.deepEqual(h.calls, ["start"]);
  delete h.captured.motions[0].modified; delete h.current.motions[0].modified;
  await assert.rejects(h.execute("start_training"), /no revision/);
});
for (const status of ["loading", "running", "stopping"]) {
  test(`${status} blocks start, evaluate and replay`, async () => {
    const h = harness(); h.current.training.status = h.current.run!.status = status;
    for (const kind of ["start_training", "resume_training", "evaluate_checkpoint", "replay_checkpoint"] as const)
      await assert.rejects(h.execute(kind, kind === "start_training" ? null : "jump-test"));
    assert.deepEqual(h.calls, []);
  });
}
test("stop targets exact active run while allowing updated progress", async () => {
  const h = harness();
  for (const c of [h.captured, h.current]) {
    c.training.status = c.run!.status = "running"; c.run!.runId = "run-1";
  }
  h.current.run!.iteration = 20;
  assert.equal(await h.execute("stop_training", "run-1"), "Stopping requested.");
  assert.deepEqual(h.calls, ["stop:run-1"]);
  h.current.run!.runId = "run-2";
  await assert.rejects(h.execute("stop_training", "run-1"));
  assert.equal(h.calls.length, 1);
});
for (const kind of ["evaluate_checkpoint", "replay_checkpoint", "resume_training"] as const) {
  test(`${kind} targets exact saved revision without mutating checkpoint data`, async () => {
    const h = harness(); await h.execute(kind, "jump-test");
    assert.deepEqual(h.calls, [{ evaluate_checkpoint: "evaluate", replay_checkpoint: "replay", resume_training: "resume" }[kind] + ":jump-test"]);
    h.current.checkpoints![0].modified = 101;
    await assert.rejects(h.execute(kind, "jump-test"), /changed/); assert.equal(h.calls.length, 1);
    h.current.checkpoints = [];
    await assert.rejects(h.execute(kind, "jump-test")); assert.equal(h.calls.length, 1);
  });
}
test("resume uses the saved policy but refuses a changed additional budget or worker count", async () => {
  for (const key of ["iterations", "rolloutWorkers"] as const) {
    const h = harness(); h.current.training[key]++;
    await assert.rejects(h.execute("resume_training", "jump-test"), /budget changed/);
    assert.deepEqual(h.calls, []);
  }
});
test("equivalent checkpoint metadata survives validation field ordering", async () => {
  const h = harness();
  h.current.checkpoints = [{ name: "jump-test", modified: 100, bytes: 2048, iteration: null }];
  await h.execute("evaluate_checkpoint", "jump-test");
  assert.deepEqual(h.calls, ["evaluate:jump-test"]);
});
test("invalid targets, unknown commands, failed context and failed workers do not report success", async () => {
  const h = harness();
  await assert.rejects(h.execute("replay_checkpoint", "../outside"));
  await assert.rejects(h.execute("shell" as AssistantCommand["kind"]));
  assert.deepEqual(h.calls, []);
  h.ports.getContext = async () => { throw new Error("Storage unavailable"); };
  await assert.rejects(h.execute("start_training"), /Storage unavailable/);
  h.ports.getContext = async () => h.current;
  h.ports.startTraining = async () => { throw new Error("Worker failed"); };
  await assert.rejects(h.execute("start_training"), /Worker failed/);
});
