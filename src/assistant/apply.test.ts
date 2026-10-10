import assert from "node:assert/strict";
import { test } from "node:test";
import { applyAssistantProposal, type AssistantApplyPorts } from "./apply.ts";
import type { AssistantContext, AssistantProposal, TrainingDraft } from "./contract.ts";
import type { MotionFile } from "../motion/format.ts";
import { DEFAULT_POSE, JOINT_NAMES } from "../sim/microduck.ts";
import { createWorldObject, type WorldConfig } from "../sim/world.ts";

const clone = <T>(value: T): T => structuredClone(value);

function training(overrides: Partial<TrainingDraft> = {}): TrainingDraft {
  return { task: "hold_pose", iterations: 50, seed: 42, envs: 16,
    rolloutWorkers: 1, useCurrentWorld: false, hiddenPreset: "small", ...overrides };
}

function motion(name = "Gentle nod"): MotionFile {
  return { format: 1, name, description: "A proposed reference", loop: false,
    units: "rad", interp: "linear", joints: [...JOINT_NAMES],
    keyframes: [{ t: 0, pose: Array.from(DEFAULT_POSE) }, { t: 0.5, pose: Array.from(DEFAULT_POSE) }] };
}

function context(): AssistantContext {
  return { world: { version: 1, objects: [] }, motions: [], currentMotion: null,
    training: { ...training(), status: "idle", iteration: 0, reward: null } };
}

function proposal(overrides: Partial<AssistantProposal> = {}): AssistantProposal {
  return { message: "Proposed changes.", world: null, motion: null, training: null, ...overrides };
}

function newWorld(): WorldConfig {
  return { version: 1, objects: [createWorldObject("box", "assistant_box")] };
}

function harness(initial = context()) {
  const current = clone(initial);
  const events: string[] = [];
  const saved = new Map<string, MotionFile>();
  const deleted: string[] = [];
  const worldCalls: WorldConfig[] = [];
  const motionCalls: MotionFile[] = [];
  const trainingCalls: TrainingDraft[] = [];
  const ports: AssistantApplyPorts = {
    async getContext() { events.push("context"); return clone(current); },
    async applyWorld(world) {
      events.push("world"); worldCalls.push(clone(world)); current.world = clone(world); return true;
    },
    async saveMotion(id, file) {
      events.push("save");
      if (saved.has(id) || current.motions.some((entry) => entry.id === id)) throw new Error("Motion already exists");
      saved.set(id, clone(file)); current.motions.push({ id, name: file.name ?? id });
    },
    async deleteMotion(id) {
      events.push("delete"); deleted.push(id); saved.delete(id);
      current.motions = current.motions.filter((entry) => entry.id !== id);
    },
    setMotion(file) { events.push("motion"); motionCalls.push(clone(file)); current.currentMotion = clone(file); },
    setTraining(draft) {
      events.push("training"); trainingCalls.push(clone(draft));
      current.training = { ...current.training, ...clone(draft) };
    },
  };
  return { current, events, saved, deleted, worldCalls, motionCalls, trainingCalls, ports,
    mutations: () => events.filter((event) => event !== "context") };
}

test("message-only proposals perform no mutations", async () => {
  const h = harness();
  const result = await applyAssistantProposal(proposal(), clone(h.current), h.ports);
  assert.deepEqual(result, { message: "No changes requested.", motionId: null });
  assert.deepEqual(h.mutations(), []);
});

test("invalid proposals and captured contexts fail before any mutation", async () => {
  for (const value of [{}, { ...proposal(), execute: "start" }, proposal({ training: training({ task: "$generated_motion" }) })]) {
    const h = harness();
    await assert.rejects(applyAssistantProposal(value, clone(h.current), h.ports));
    assert.deepEqual(h.mutations(), []);
  }
  const h = harness();
  const captured = clone(h.current);
  captured.training.seed = -1;
  await assert.rejects(applyAssistantProposal(proposal(), captured, h.ports));
  assert.deepEqual(h.mutations(), []);
});

test("busy training refuses application without saving or changing the draft", async () => {
  for (const status of ["loading", "running", "stopping"]) {
    const initial = context(); initial.training.status = status;
    const h = harness(initial);
    await assert.rejects(applyAssistantProposal(proposal({ world: newWorld(), motion: motion() }), clone(initial), h.ports), /busy|running|train|idle/i);
    assert.deepEqual(h.mutations(), []);
    assert.deepEqual(h.current, initial);
  }
});

test("stale world, training, motion library and current motion each refuse", async () => {
  const changes: ((ctx: AssistantContext) => void)[] = [
    (ctx) => { ctx.world = newWorld(); },
    (ctx) => { ctx.training.iterations = 51; },
    (ctx) => { ctx.motions.push({ id: "existing", name: "Existing" }); },
    (ctx) => { ctx.currentMotion = motion("Existing"); },
  ];
  for (const change of changes) {
    const h = harness(); const captured = clone(h.current); change(h.current);
    const changed = clone(h.current);
    await assert.rejects(applyAssistantProposal(proposal({ world: newWorld(), motion: motion() }), captured, h.ports), /stale|changed|again|match/i);
    assert.deepEqual(h.mutations(), []);
    assert.deepEqual(h.current, changed);
  }
});

test("equivalent library order and updated training diagnostics do not make a draft stale", async () => {
  const initial = context();
  initial.motions = [{ id: "one", name: "One" }, { id: "two", name: "Two" }];
  const h = harness(initial);
  h.current.motions.reverse();
  h.current.training.iteration = 50;
  h.current.training.reward = 12.5;
  await applyAssistantProposal(proposal({ world: newWorld() }), clone(initial), h.ports);
  assert.deepEqual(h.mutations(), ["world"]);
  assert.equal(h.current.training.iteration, 50);
  assert.equal(h.current.training.reward, 12.5);
});

test("context read failures leave everything untouched", async () => {
  const h = harness();
  h.ports.getContext = async () => { throw new Error("Context unavailable"); };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion() }), clone(h.current), h.ports), /Context unavailable/);
  assert.deepEqual(h.mutations(), []);
});

test("world-only proposal applies once without a save or training start", async () => {
  const h = harness(); const world = newWorld();
  const result = await applyAssistantProposal(proposal({ world }), clone(h.current), h.ports);
  assert.equal(result.motionId, null);
  assert.deepEqual(h.worldCalls, [world]);
  assert.deepEqual(h.mutations(), ["world"]);
  assert.deepEqual(h.current.training, context().training);
});

test("training-only proposal changes draft fields without starting a worker", async () => {
  const h = harness(); const draft = training({ task: "jump", iterations: 200, seed: 9 });
  await applyAssistantProposal(proposal({ training: draft }), clone(h.current), h.ports);
  assert.deepEqual(h.trainingCalls, [draft]);
  assert.deepEqual(h.mutations(), ["training"]);
  assert.equal(h.current.training.status, "idle");
  assert.equal(h.current.training.iteration, 0);
});

test("generated motion is saved exclusively before world, motion and training changes", async () => {
  const h = harness();
  h.current.motions.push({ id: "keep", name: "Keep" });
  h.saved.set("keep", motion("Keep"));
  const captured = clone(h.current);
  const raw = proposal({ world: newWorld(), motion: motion(), training: training({ task: "$generated_motion" }) });
  const untouched = clone(raw);
  const result = await applyAssistantProposal(raw, captured, h.ports);
  assert.ok(result.motionId && /^ai-.+-[0-9a-f]{8}$/i.test(result.motionId));
  assert.deepEqual(h.mutations(), ["save", "world", "motion", "training"]);
  assert.ok(h.events.indexOf("context", h.events.indexOf("save") + 1) > h.events.indexOf("save"));
  assert.equal(h.saved.size, 2);
  assert.deepEqual(h.saved.get("keep"), motion("Keep"));
  assert.equal(h.trainingCalls[0].task, `motion:${result.motionId}`);
  assert.deepEqual(h.motionCalls, [h.saved.get(result.motionId)]);
  assert.equal(h.current.training.status, "idle");
  assert.deepEqual(h.deleted, []);
  assert.deepEqual(raw, untouched);
  assert.deepEqual(captured, { ...context(), motions: [{ id: "keep", name: "Keep" }] });
});

test("motion-only proposal leaves the training draft unchanged", async () => {
  const h = harness(); const before = clone(h.current.training);
  const result = await applyAssistantProposal(proposal({ motion: motion() }), clone(h.current), h.ports);
  assert.ok(result.motionId);
  assert.deepEqual(h.mutations(), ["save", "motion"]);
  assert.deepEqual(h.current.training, before);
});

test("successful first generated motion never invokes the empty-selection restore callback", async () => {
  const h = harness();
  h.ports.restoreEmptyMotion = () => { h.events.push("emptyMotion"); h.current.currentMotion = null; };
  const result = await applyAssistantProposal(proposal({ motion: motion() }), clone(h.current), h.ports);
  assert.ok(result.motionId);
  assert.deepEqual(h.mutations(), ["save", "motion"]);
  assert.deepEqual(h.current.currentMotion, h.saved.get(result.motionId));
  assert.deepEqual(h.deleted, []);
});

test("96-character motion names still produce storage-compatible IDs", async () => {
  const h = harness();
  const result = await applyAssistantProposal(proposal({ motion: motion("A".repeat(96)) }), clone(h.current), h.ports);
  assert.ok(result.motionId && result.motionId.length <= 64);
  assert.match(result.motionId, /^ai-[a-z0-9-]+-[0-9a-f]{8}$/);
  assert.equal(h.saved.get(result.motionId)?.name, "A".repeat(96));
});

test("ID collisions fail before saving or deleting another motion", async (t) => {
  t.mock.method(globalThis.crypto, "randomUUID", () => "12345678-1234-4123-8123-123456789abc");
  const initial = context(); initial.motions.push({ id: "ai-gentle-nod-12345678", name: "Existing" });
  const h = harness(initial);
  h.saved.set(initial.motions[0].id, motion("Existing"));
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion() }), clone(initial), h.ports), /already exists/i);
  assert.deepEqual(h.mutations(), []);
  assert.deepEqual(h.current, initial);
  assert.deepEqual(h.saved.get(initial.motions[0].id), motion("Existing"));
});

test("a full 100-entry library refuses a generated motion before saving", async () => {
  const initial = context();
  initial.motions = Array.from({ length: 100 }, (_, i) => ({ id: `saved-${i}`, name: `Saved ${i}` }));
  const h = harness(initial);
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion() }), clone(initial), h.ports), /library.*full/i);
  assert.deepEqual(h.mutations(), []);
  assert.deepEqual(h.current, initial);
});

test("unchanged world and training avoid setters and do not echo provider completion claims", async () => {
  const h = harness();
  const result = await applyAssistantProposal(proposal({ message: "Training completed successfully.",
    world: clone(h.current.world), training: training() }), clone(h.current), h.ports);
  assert.deepEqual(h.mutations(), []);
  assert.match(result.message, /unchanged/i);
  assert.match(result.message, /not started/i);
  assert.doesNotMatch(result.message, /completed successfully/i);
});

test("known motion training does not resave or replace that motion", async () => {
  const initial = context(); initial.motions = [{ id: "builtin:nod", name: "nod" }];
  const h = harness(initial);
  const draft = training({ task: "motion:builtin:nod" });
  await applyAssistantProposal(proposal({ training: draft }), clone(initial), h.ports);
  assert.deepEqual(h.mutations(), ["training"]);
  assert.equal(h.trainingCalls[0].task, "motion:builtin:nod");
  assert.deepEqual(h.deleted, []);
});

test("exclusive save failure cannot delete a preexisting motion or mutate UI state", async () => {
  const h = harness(); const before = clone(h.current);
  h.ports.saveMotion = async () => { h.events.push("save"); throw new Error("Exclusive create refused"); };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), world: newWorld() }), clone(before), h.ports), /Exclusive create refused/);
  assert.deepEqual(h.mutations(), ["save"]);
  assert.deepEqual(h.current, before);
  assert.deepEqual(h.deleted, []);
});

test("post-save validation ignores only its own new library entry", async () => {
  const h = harness(); const captured = clone(h.current);
  const save = h.ports.saveMotion;
  h.ports.saveMotion = async (id, file) => {
    await save(id, file);
    h.current.motions.push({ id: "concurrent", name: "Concurrent" });
    h.saved.set("concurrent", motion("Concurrent"));
  };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), world: newWorld() }), captured, h.ports), /stale|changed|again|match/i);
  assert.deepEqual(h.mutations(), ["save", "delete"]);
  assert.equal(h.deleted.length, 1);
  assert.notEqual(h.deleted[0], "concurrent");
  assert.deepEqual(h.current.motions, [{ id: "concurrent", name: "Concurrent" }]);
  assert.ok(h.saved.has("concurrent"));
});

test("training becoming busy after save cleans up only the new motion", async () => {
  const h = harness(); const captured = clone(h.current); const save = h.ports.saveMotion;
  h.ports.saveMotion = async (id, file) => { await save(id, file); h.current.training.status = "running"; };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), training: training({ task: "$generated_motion" }) }), captured, h.ports));
  assert.deepEqual(h.mutations(), ["save", "delete"]);
  assert.equal(h.current.training.status, "running");
  assert.equal(h.current.training.task, "hold_pose");
  assert.equal(h.saved.size, 0);
});

test("a renamed own library entry is not ignored during revalidation", async () => {
  const h = harness(); const captured = clone(h.current); const save = h.ports.saveMotion;
  h.ports.saveMotion = async (id, file) => {
    await save(id, file);
    h.current.motions.find((entry) => entry.id === id)!.name = "Changed externally";
  };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), world: newWorld() }), captured, h.ports), /saved motion.*changed/i);
  assert.deepEqual(h.mutations(), ["save", "delete"]);
  assert.equal(h.saved.size, 0);
});

test("a missing newly saved entry cannot configure a dangling generated training task", async () => {
  const h = harness(); const captured = clone(h.current); const save = h.ports.saveMotion;
  h.ports.saveMotion = async (id, file) => {
    await save(id, file);
    h.saved.delete(id);
    h.current.motions = h.current.motions.filter((entry) => entry.id !== id);
  };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), world: newWorld(),
    training: training({ task: "$generated_motion" }) }), captured, h.ports), /saved motion.*missing/i);
  assert.deepEqual(h.mutations(), ["save", "delete"]);
  assert.deepEqual(h.current, captured);
  assert.equal(h.deleted.length, 1);
});

test("post-save context read failure deletes the owned new file without other changes", async () => {
  const h = harness(); const captured = clone(h.current); const get = h.ports.getContext;
  let reads = 0;
  h.ports.getContext = async () => { if (++reads > 1) throw new Error("Revalidation unavailable"); return get(); };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), world: newWorld() }), captured, h.ports), /Revalidation unavailable/);
  assert.deepEqual(h.mutations(), ["save", "delete"]);
  assert.equal(h.saved.size, 0);
});

test("invalid post-save context cleans up its own file before applying anything", async () => {
  const h = harness(); const captured = clone(h.current); const save = h.ports.saveMotion;
  h.ports.saveMotion = async (id, file) => { await save(id, file); h.current.training.reward = NaN; };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), world: newWorld() }), captured, h.ports), /nonfinite|finite/i);
  assert.deepEqual(h.mutations(), ["save", "delete"]);
  assert.equal(h.saved.size, 0);
  assert.deepEqual(h.motionCalls, []);
  assert.deepEqual(h.trainingCalls, []);
});

for (const mode of ["false", "throw"] as const) {
  test(`world application ${mode} restores the original world and cleans its new motion`, async () => {
    const h = harness(); const captured = clone(h.current); const apply = h.ports.applyWorld;
    let calls = 0;
    h.ports.applyWorld = async (world) => {
      await apply(world);
      if (++calls === 1) {
        if (mode === "throw") throw new Error("Scene rebuild failed");
        return false;
      }
      return true;
    };
    await assert.rejects(applyAssistantProposal(proposal({ world: newWorld(), motion: motion(), training: training() }), captured, h.ports), /world|scene/i);
    assert.deepEqual(h.worldCalls, [newWorld(), captured.world]);
    assert.deepEqual(h.current, captured);
    assert.equal(h.deleted.length, 1);
    assert.equal(h.saved.size, 0);
    assert.deepEqual(h.motionCalls, []);
    assert.deepEqual(h.trainingCalls, []);
  });
}

test("world rollback and file cleanup failures are both reported without skipping cleanup", async () => {
  const h = harness(); const captured = clone(h.current); const apply = h.ports.applyWorld;
  let calls = 0;
  h.ports.applyWorld = async (world) => {
    await apply(world);
    if (++calls === 1) throw new Error("Initial world failure");
    throw new Error("World rollback failure");
  };
  h.ports.deleteMotion = async (id) => { h.events.push("delete"); h.deleted.push(id); throw new Error("File cleanup failure"); };
  await assert.rejects(applyAssistantProposal(proposal({ world: newWorld(), motion: motion() }), captured, h.ports), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /Initial world failure/);
    assert.match(error.message, /World rollback failure/);
    assert.match(error.message, /File cleanup failure/);
    return true;
  });
  assert.equal(h.worldCalls.length, 2);
  assert.equal(h.deleted.length, 1);
  assert.deepEqual(h.motionCalls, []);
  assert.deepEqual(h.trainingCalls, []);
});

test("throwing training setter restores original draft, motion, world and removes its file", async () => {
  const initial = context(); initial.currentMotion = motion("Original");
  const h = harness(initial); const set = h.ports.setTraining;
  let calls = 0;
  h.ports.setTraining = (draft) => { set(draft); if (++calls === 1) throw new Error("Draft setter failed"); };
  await assert.rejects(applyAssistantProposal(proposal({ world: newWorld(), motion: motion(), training: training({ task: "$generated_motion" }) }), clone(initial), h.ports), /Draft setter failed/);
  assert.deepEqual(h.current, initial);
  assert.deepEqual(h.motionCalls.at(-1), initial.currentMotion);
  assert.deepEqual(h.trainingCalls.at(-1), training());
  assert.equal(h.worldCalls.length, 2);
  assert.equal(h.saved.size, 0);
  assert.equal(h.deleted.length, 1);
});

test("throwing motion setter restores the prior motion without applying a training draft", async () => {
  const initial = context(); initial.currentMotion = motion("Original");
  const h = harness(initial); const set = h.ports.setMotion;
  let calls = 0;
  h.ports.setMotion = (file) => { set(file); if (++calls === 1) throw new Error("Motion setter failed"); };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion(), training: training({ task: "$generated_motion" }) }), clone(initial), h.ports), /Motion setter failed/);
  assert.deepEqual(h.current, initial);
  assert.deepEqual(h.trainingCalls, []);
  assert.equal(h.deleted.length, 1);
});

for (const failingSetter of ["motion", "training"] as const) {
  test(`throwing ${failingSetter} setter restores an originally empty selection through the callback`, async () => {
    const h = harness(); const initial = clone(h.current);
    h.ports.restoreEmptyMotion = () => { h.events.push("emptyMotion"); h.current.currentMotion = null; };
    if (failingSetter === "motion") {
      const set = h.ports.setMotion;
      h.ports.setMotion = (file) => { set(file); throw new Error("Motion setter failed"); };
    } else {
      const set = h.ports.setTraining; let calls = 0;
      h.ports.setTraining = (draft) => { set(draft); if (++calls === 1) throw new Error("Training setter failed"); };
    }
    await assert.rejects(applyAssistantProposal(proposal({ world: newWorld(), motion: motion(),
      training: training({ task: "$generated_motion" }) }), initial, h.ports), (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /setter failed/i);
      assert.doesNotMatch(error.message, /rollback incomplete/i);
      return true;
    });
    assert.deepEqual(h.current, initial);
    assert.equal(h.events.filter((event) => event === "emptyMotion").length, 1);
    assert.equal(h.motionCalls.length, 1);
    assert.equal(h.worldCalls.length, 2);
    assert.equal(h.saved.size, 0);
    assert.equal(h.deleted.length, 1);
  });
}

test("throwing empty-selection restoration reports incomplete rollback and still restores world and deletes its file", async () => {
  const h = harness(); const initial = clone(h.current); const set = h.ports.setTraining;
  let calls = 0;
  h.ports.setTraining = (draft) => { set(draft); if (++calls === 1) throw new Error("Training setter failed"); };
  h.ports.restoreEmptyMotion = () => { h.events.push("emptyMotion"); throw new Error("Empty maker restore failed"); };
  await assert.rejects(applyAssistantProposal(proposal({ world: newWorld(), motion: motion(),
    training: training({ task: "$generated_motion" }) }), initial, h.ports), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /Training setter failed/);
    assert.match(error.message, /Rollback incomplete/i);
    assert.match(error.message, /Empty maker restore failed/);
    return true;
  });
  assert.equal(h.events.filter((event) => event === "emptyMotion").length, 1);
  assert.deepEqual(h.mutations().slice(-2), ["world", "delete"]);
  assert.deepEqual(h.current.world, initial.world);
  assert.deepEqual(h.current.training, initial.training);
  assert.notEqual(h.current.currentMotion, null);
  assert.equal(h.saved.size, 0);
  assert.equal(h.deleted.length, 1);
});

test("a partial motion setter with no original motion reports the rollback limitation", async () => {
  const h = harness(); const initial = clone(h.current); const set = h.ports.setMotion;
  h.ports.setMotion = (file) => { set(file); throw new Error("Motion partially applied"); };
  await assert.rejects(applyAssistantProposal(proposal({ motion: motion() }), initial, h.ports), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /Motion partially applied/);
    assert.match(error.message, /rollback|restore|partial/i);
    assert.match(error.message, /motion/i);
    return true;
  });
  assert.equal(h.deleted.length, 1);
  assert.equal(h.saved.size, 0);
  assert.deepEqual(h.current.training, initial.training);
});
