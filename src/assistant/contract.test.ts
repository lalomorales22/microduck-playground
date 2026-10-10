import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_POSE, JOINT_LIMITS, JOINT_NAMES } from "../sim/microduck.ts";
import { createWorldObject } from "../sim/world.ts";
import { BUILTIN_TASKS } from "../train/env/tasks.ts";
import { ASSISTANT_LIMITS, ASSISTANT_INSTRUCTIONS, PROPOSAL_SCHEMA, buildAssistantInstructions,
  validateAssistantRequest, validateProposal, type AssistantContext, type AssistantProposal,
  type AssistantRequest, type AssistantRunSnapshot, type TrainingDraft } from "./contract.ts";

const draft = (): TrainingDraft => ({ task: "hold_pose", iterations: 50, seed: 42,
  envs: 16, rolloutWorkers: 1, useCurrentWorld: false, hiddenPreset: "small" });
const context = (): AssistantContext => ({ world: { version: 1, objects: [] },
  motions: [{ id: "builtin:nod", name: "nod" }], currentMotion: null,
  training: { ...draft(), status: "idle", iteration: 0, reward: null } });
const request = (): AssistantRequest => ({ messages: [{ role: "user", content: "Propose a small nod" }], context: context() });
const motion = () => ({ format: 1, name: "Small nod", description: "A modest head reference", loop: false,
  units: "rad" as const, interp: "linear" as const, joints: [...JOINT_NAMES],
  keyframes: [{ t: 0, pose: Array.from(DEFAULT_POSE) },
    { t: 1, pose: Array.from(DEFAULT_POSE, (value, i) => i === 6 ? value + 0.1 : value) }] });
const proposal = (): AssistantProposal => ({ message: "Proposed a motion; nothing has been applied.",
  world: null, motion: null, training: null });
const run = (): AssistantRunSnapshot => ({ runId: "jump-run-1", task: "Jump", status: "stopped", completionReason: "completed",
  iteration: 50, targetIteration: 50, totalSteps: 19200, reward: 0.5, standing: 0.75, checkpoint: "jump-checkpoint-1",
  evaluation: { checkpointName: "jump-checkpoint-1", iteration: 50, verdict: "not-qualified", reasons: ["No measured flight"],
    heightGain: 0.01, flightSeconds: 0, stableLandingSeconds: null } });
const monitoredContext = (): AssistantContext => ({ ...context(), run: run(),
  checkpoints: [{ name: "jump-checkpoint-1", iteration: 50, modified: 1234567890, bytes: 100000 }] });

test("valid request and message-only proposal produce detached JSON data", () => {
  const input = request(), normalized = validateAssistantRequest(input);
  assert.deepEqual(normalized, input);
  normalized.messages[0].content = "changed";
  normalized.context.motions[0].name = "changed";
  assert.equal(input.messages[0].content, "Propose a small nod");
  assert.equal(input.context.motions[0].name, "nod");
  assert.deepEqual(validateProposal(proposal(), context()), proposal());
});
test("saved motion revisions round trip and reject invalid metadata", () => {
  const input = request();
  input.context.motions.push({ id: "saved-nod", name: "Nod", bytes: 100, modified: 1234 });
  assert.deepEqual(validateAssistantRequest(input), input);
  for (const patch of [{ bytes: -1 }, { modified: 0 }, { modified: Infinity }, { bytes: 0.5 }, { revisionCode: "run()" }]) {
    assert.throws(() => validateAssistantRequest({ ...input, context: { ...input.context,
      motions: [{ ...input.context.motions[1], ...patch }] } }));
  }
});

test("combined world, generated motion and bounded training round trip", () => {
  const input: AssistantProposal = { ...proposal(), world: { version: 1, objects: [createWorldObject("ball", "ball1")] },
    motion: motion(), training: { ...draft(), task: "$generated_motion", envs: 32, rolloutWorkers: 4 } };
  const checked = validateProposal(input, context());
  assert.deepEqual(checked, input);
  checked.motion!.keyframes[0].pose[0] = 0.1;
  checked.world!.objects[0].position[0] = 2;
  assert.equal(input.motion!.keyframes[0].pose[0], 0);
  assert.equal(input.world!.objects[0].position[0], 0.65);
});

test("all nine built-in tasks and exact existing motion ids are accepted", () => {
  assert.equal(BUILTIN_TASKS.length, 9);
  for (const task of [...BUILTIN_TASKS.map((item) => item.kind), "motion:builtin:nod"]) {
    assert.equal(validateProposal({ ...proposal(), training: { ...draft(), task } }, context()).training!.task, task);
  }
  for (const task of ["unknown", "motion:nod", "motion:../nod", "$generated_motion"]) {
    assert.throws(() => validateProposal({ ...proposal(), training: { ...draft(), task } }, context()), /training.task/);
  }
});

test("missing, extra and execution-related request/proposal fields are refused", () => {
  for (const field of ["endpoint", "apiKey", "model", "code", "system"]) {
    assert.throws(() => validateAssistantRequest({ ...request(), [field]: "unsafe" }), /unknown field/);
    assert.throws(() => validateProposal({ ...proposal(), [field]: "unsafe" }, context()), /unknown field/);
  }
  assert.throws(() => validateAssistantRequest({ ...request(), command: "unsafe" }), /unknown field/);
  assert.throws(() => validateProposal({ ...proposal(), command: "unsafe" }, context()), /must be an object/);
  for (const field of ["message", "world", "motion", "training"]) {
    const value = { ...proposal() } as Record<string, unknown>;
    delete value[field];
    assert.throws(() => validateProposal(value, context()), /missing field/);
  }
  assert.throws(() => validateAssistantRequest({ messages: [] }), /missing field/);
  assert.throws(() => validateProposal({ ...proposal(), message: " " }, context()), /nonempty/);
});

test("legacy contexts and four-field proposals retain their shape; explicit command null remains valid", () => {
  assert.deepEqual(validateAssistantRequest(request()), request());
  assert.deepEqual(validateProposal(proposal(), context()), proposal());
  assert.deepEqual(validateProposal({ ...proposal(), command: null }, context()), { ...proposal(), command: null });
  const empty = { ...context(), run: null, checkpoints: [] };
  assert.deepEqual(validateAssistantRequest({ ...request(), context: empty }).context, empty);
});

test("monitoring snapshots and checkpoint revisions are validated and detached", () => {
  const input = { ...request(), context: monitoredContext() };
  const checked = validateAssistantRequest(input);
  assert.deepEqual(checked, input);
  checked.context.run!.evaluation!.reasons[0] = "changed";
  checked.context.checkpoints![0].modified = 999;
  assert.equal(input.context.run!.evaluation!.reasons[0], "No measured flight");
  assert.equal(input.context.checkpoints![0].modified, 1234567890);
  for (const patch of [
    { runId: "../run" }, { task: "x".repeat(121) }, { status: "" }, { completionReason: "success" },
    { iteration: 0.1 }, { targetIteration: -1 }, { totalSteps: Infinity }, { reward: NaN },
    { standing: 1.01 }, { standing: -0.1 }, { checkpoint: "../file" }, { authorized: true },
  ]) assert.throws(() => validateAssistantRequest({ ...request(), context: { ...monitoredContext(), run: { ...run(), ...patch } } }));
  for (const patch of [
    { verdict: "success" }, { iteration: -1 }, { checkpointName: "https://example.com/key" },
    { reasons: new Array(1) }, { reasons: Array.from({ length: 17 }, () => "reason") },
    { reasons: ["x".repeat(257)] }, { heightGain: Infinity }, { flightSeconds: -1 },
    { stableLandingSeconds: 121 }, { claim: "learned" },
  ]) assert.throws(() => validateAssistantRequest({ ...request(), context: { ...monitoredContext(),
    run: { ...run(), evaluation: { ...run().evaluation!, ...patch } } } }));
  const missing = run() as unknown as Record<string, unknown>; delete missing.completionReason;
  assert.throws(() => validateAssistantRequest({ ...request(), context: { ...monitoredContext(), run: missing } }), /missing field/);
});

test("checkpoint context limits safe identities, revisions and exact fields", () => {
  const check = (checkpoints: unknown) => validateAssistantRequest({ ...request(), context: { ...context(), checkpoints } });
  assert(check([{ name: "run-1", iteration: null }]));
  for (const patch of [
    { name: "../run" }, { iteration: -1 }, { iteration: 1.5 }, { bytes: 0 }, { bytes: 1e9 + 1 },
    { modified: 0 }, { modified: Infinity }, { modified: 1.5 }, { script: "unsafe" },
  ]) assert.throws(() => check([{ name: "run-1", iteration: 1, ...patch }]));
  assert.throws(() => check([{ name: "run-1", iteration: 1 }, { name: "run-1", iteration: 2 }]), /Duplicate/);
  assert.throws(() => check(new Array(1)));
  assert.throws(() => check(Array.from({ length: 101 }, (_, i) => ({ name: `run-${i}`, iteration: i }))), /100 entries/);
});

test("command proposals bind the captured draft or exact live run/checkpoint identities", () => {
  const idle = monitoredContext();
  for (const command of [
    { kind: "start_training", target: null },
    { kind: "resume_training", target: "jump-checkpoint-1" },
    { kind: "evaluate_checkpoint", target: "jump-checkpoint-1" },
    { kind: "replay_checkpoint", target: "jump-checkpoint-1" },
  ] as const) assert.deepEqual(validateProposal({ ...proposal(), command }, idle).command, command);
  const live = { ...idle, run: { ...run(), status: "running", completionReason: null as null } };
  const stop = { kind: "stop_training", target: "jump-run-1" } as const;
  assert.deepEqual(validateProposal({ ...proposal(), command: stop }, live).command, stop);
  for (const command of [
    { kind: "start_training", target: "jump-run-1" }, { kind: "stop_training", target: null },
    { kind: "stop_training", target: "other-run" }, { kind: "evaluate_checkpoint", target: null },
    { kind: "resume_training", target: null }, { kind: "resume_training", target: "unknown" },
    { kind: "replay_checkpoint", target: "unknown" }, { kind: "exec", target: "code" },
    { kind: "start_training", target: null, approved: true },
    { kind: "start_training" },
  ]) assert.throws(() => validateProposal({ ...proposal(), command }, idle));
  assert.throws(() => validateProposal({ ...proposal(), command: stop }, idle), /matching live run/);
  assert.throws(() => validateProposal({ ...proposal(), command: stop }, context()), /matching live run/);
});

test("busy snapshots reject start/evaluate/replay while retaining matching stop", () => {
  for (const status of ["loading", "running", "stopping"]) {
    const ctx = monitoredContext(); ctx.run!.status = status; ctx.run!.completionReason = null;
    for (const kind of ["start_training", "resume_training", "evaluate_checkpoint", "replay_checkpoint"]) {
      const command = { kind, target: kind === "start_training" ? null : "jump-checkpoint-1" };
      assert.throws(() => validateProposal({ ...proposal(), command }, ctx), /busy/);
      const trainingBusy = { ...monitoredContext(), training: { ...context().training, status } };
      assert.throws(() => validateProposal({ ...proposal(), command }, trainingBusy), /busy/);
    }
    assert.equal(validateProposal({ ...proposal(), command: { kind: "stop_training", target: "jump-run-1" } }, ctx).command!.kind, "stop_training");
  }
});

test("commands cannot mix with edits or bypass captured training limits", () => {
  for (const command of [{ kind: "start_training", target: null }, { kind: "resume_training", target: "jump-checkpoint-1" }] as const) {
    for (const patch of [{ world: context().world }, { motion: motion() }, { training: draft() }])
      assert.throws(() => validateProposal({ ...proposal(), ...patch, command }, monitoredContext()), /cannot also edit/);
    for (const patch of [{ iterations: 0 }, { iterations: 5001 }, { iterations: 1.5 },
      { envs: 128, rolloutWorkers: 8 }, { rolloutWorkers: 9 }, { task: "invented" }]) {
      const ctx = monitoredContext(); Object.assign(ctx.training, patch);
      assert.throws(() => validateProposal({ ...proposal(), command }, ctx));
    }
  }
});

test("resume identifies an existing checkpoint without rewriting its task or captured budget", () => {
  const ctx = monitoredContext(); ctx.training.iterations = 75; ctx.training.rolloutWorkers = 2;
  ctx.training.task = "hold_pose";
  const before = structuredClone(ctx);
  const command = { kind: "resume_training", target: "jump-checkpoint-1" } as const;
  assert.deepEqual(validateProposal({ ...proposal(), command }, ctx).command, command);
  assert.deepEqual(ctx, before);
  for (const patch of [{ iterations: 100 }, { rolloutWorkers: 4 }, { task: "jump" }, { resume: true }])
    assert.throws(() => validateProposal({ ...proposal(), command: { ...command, ...patch } }, ctx), /unknown field/);
  const instructions = buildAssistantInstructions(ctx);
  for (const phrase of ["resume_training", "fresh random run", "ADDITIONAL iteration budget", "original is preserved",
    "exact checkpoint name", "do not confuse additional iterations", "does not adopt fresh draft task"])
    assert(instructions.includes(phrase), phrase);
});

test("message roles, final sender, dense history and size limits are enforced", () => {
  for (const role of ["system", "developer", "tool", "function"]) {
    assert.throws(() => validateAssistantRequest({ ...request(), messages: [{ role, content: "hi" }] }), /message.role/);
  }
  assert.throws(() => validateAssistantRequest({ ...request(), messages: [{ role: "assistant", content: "hi" }] }), /last message/);
  for (const messages of [[], new Array(1), Array.from({ length: 21 }, () => ({ role: "user", content: "hi" }))]) {
    assert.throws(() => validateAssistantRequest({ ...request(), messages }));
  }
  assert.throws(() => validateAssistantRequest({ ...request(), messages: [{ role: "user", content: "x".repeat(8001) }] }), /8000/);
  assert.throws(() => validateAssistantRequest({ ...request(), messages: Array.from({ length: 4 }, () => ({ role: "user", content: "x".repeat(8000) })) }), /24000/);
  assert.throws(() => validateAssistantRequest({ ...request(), messages: [{ role: "user", content: "a\0b" }] }), /NUL/);
});

test("context contains bounded, safe, unique motion identities and explicit status", () => {
  for (const id of ["../nod", "/tmp/nod", "C:\\nod", "https://example.com/nod", "bad\nname", ".hidden", "a..b"]) {
    const input = request(); input.context.motions[0].id = id;
    assert.throws(() => validateAssistantRequest(input), /safe name/);
  }
  const duplicated = request(); duplicated.context.motions.push({ ...duplicated.context.motions[0] });
  assert.throws(() => validateAssistantRequest(duplicated), /Duplicate/);
  const many = request(); many.context.motions = Array.from({ length: 101 }, (_, i) => ({ id: `motion${i}`, name: `motion${i}` }));
  assert.throws(() => validateAssistantRequest(many), /100 entries/);
  for (const patch of [{ reward: Infinity }, { iteration: -1 }, { status: "" }, { reward: "1" }]) {
    const input = request(); Object.assign(input.context.training, patch);
    assert.throws(() => validateAssistantRequest(input));
  }
});

test("training uses finite integer budgets, exact fields and total environment cap", () => {
  for (const patch of [
    { iterations: 0 }, { iterations: 5001 }, { iterations: 1.5 }, { seed: -1 }, { seed: 2 ** 32 },
    { envs: 0 }, { envs: 129 }, { envs: NaN }, { envs: 32, rolloutWorkers: 5 },
    { rolloutWorkers: 0 }, { rolloutWorkers: 9 }, { rolloutWorkers: 1.5 },
    { hiddenPreset: "huge" }, { useCurrentWorld: 1 }, { rewardCode: "return 1" },
  ]) assert.throws(() => validateProposal({ ...proposal(), training: { ...draft(), ...patch } }, context()));
  assert.equal(validateProposal({ ...proposal(), training: { ...draft(), seed: 0xffffffff, envs: 16, rolloutWorkers: 8 } }, context()).training!.seed, 0xffffffff);
});

test("generated motions require explicit canonical radians, linear interpolation and 14 joints", () => {
  for (const patch of [{ units: "deg" }, { interp: "cubic" }, { format: 2 }, { loop: "true" },
    { name: "../bad" }, { joints: ["head_pitch"] }, { joints: [...JOINT_NAMES].reverse() }, { eval: "x" }]) {
    assert.throws(() => validateProposal({ ...proposal(), motion: { ...motion(), ...patch } }, context()));
  }
  for (const field of ["format", "name", "description", "loop", "units", "interp", "joints", "keyframes"]) {
    const file = { ...motion() } as Record<string, unknown>; delete file[field];
    assert.throws(() => validateProposal({ ...proposal(), motion: file }, context()), /missing field/);
  }
});

test("generated frame bounds, time order, dense poses and exact named limits are enforced", () => {
  const variants = [
    (m: ReturnType<typeof motion>) => { m.keyframes = m.keyframes.slice(0, 1); },
    (m: ReturnType<typeof motion>) => { m.keyframes = Array.from({ length: 33 }, (_, i) => ({ t: i / 2, pose: [...m.keyframes[0].pose] })); },
    (m: ReturnType<typeof motion>) => { m.keyframes[0].t = 0.1; },
    (m: ReturnType<typeof motion>) => { m.keyframes[1].t = 20.01; },
    (m: ReturnType<typeof motion>) => { m.keyframes[1].t = 0.01; },
    (m: ReturnType<typeof motion>) => { m.keyframes[1].t = 0; },
    (m: ReturnType<typeof motion>) => { m.keyframes.reverse(); },
    (m: ReturnType<typeof motion>) => { m.keyframes[0].pose.pop(); },
    (m: ReturnType<typeof motion>) => { m.keyframes[0].pose[0] = JOINT_LIMITS[0][1] + 0.0001; },
    (m: ReturnType<typeof motion>) => { m.keyframes[0].pose[0] = Infinity; },
    (m: ReturnType<typeof motion>) => { m.keyframes[0].pose = new Array(14); },
  ];
  for (const mutate of variants) { const file = motion(); mutate(file); assert.throws(() => validateProposal({ ...proposal(), motion: file }, context())); }
  const atBoundary = motion(); atBoundary.keyframes[1].t = 20; atBoundary.keyframes[0].pose[0] = JOINT_LIMITS[0][1];
  assert(validateProposal({ ...proposal(), motion: atBoundary }, context()).motion);
});

test("loop closure comes from the real motion parser", () => {
  const file = motion(); file.loop = true;
  assert.throws(() => validateProposal({ ...proposal(), motion: file }, context()), /repeat the first pose/);
  file.keyframes[1].pose = [...file.keyframes[0].pose];
  assert.equal(validateProposal({ ...proposal(), motion: file }, context()).motion!.loop, true);
});

test("existing partial-joint context retains its driven set and cubic interpolation", () => {
  const input = request();
  input.context.currentMotion = { name: "nod", joints: ["head_pitch"], interp: "cubic",
    keyframes: [{ t: 0, pose: [0.3] }, { t: 1, pose: [0.4] }] };
  assert.deepEqual(validateAssistantRequest(input).context.currentMotion, input.context.currentMotion);
  for (const patch of [{ name: 42 }, { description: 42 }, { loop: "false" }]) {
    assert.throws(() => validateAssistantRequest({ ...input, context: { ...input.context, currentMotion: { ...input.context.currentMotion, ...patch } } }));
  }
});

test("world proposals reuse geometry validation instead of accepting arbitrary MJCF", () => {
  for (const object of [
    { ...createWorldObject("ball", "ball"), size: [0.1, 0.2, 0.1] },
    { ...createWorldObject("ramp", "ramp"), dynamic: true },
    { ...createWorldObject("box", "box"), position: [0, 0, -1] },
    { ...createWorldObject("box", "box"), xml: "<include file='x'/>" },
  ]) assert.throws(() => validateProposal({ ...proposal(), world: { version: 1, objects: [object] } }, context()));
  assert.throws(() => validateProposal({ ...proposal(), world: { version: 1, objects: [], script: "x" } }, context()));
});

test("JSON-only budgets reject cycles, accessors, functions, symbols and giant responses", () => {
  const cyclic = request() as unknown as Record<string, unknown>; cyclic.self = cyclic;
  assert.throws(() => validateAssistantRequest(cyclic), /cycle/);
  let called = false;
  const getter = { ...proposal(), get message() { called = true; return "bad"; } };
  assert.throws(() => validateProposal(getter, context()), /accessors/);
  assert.equal(called, false);
  for (const extra of [() => 1, Symbol("x"), undefined, BigInt(1), new Date()]) {
    assert.throws(() => validateAssistantRequest({ ...request(), extra }));
  }
  assert.throws(() => validateProposal({ ...proposal(), message: "x".repeat(ASSISTANT_LIMITS.proposalBytes + 1) }, context()), /size limit/);
  const proto = Object.create(Array.prototype); proto.toJSON = () => { called = true; return []; };
  const messages = Object.setPrototypeOf([], proto);
  assert.throws(() => validateAssistantRequest({ ...request(), messages }), /array prototype/);
  assert.equal(called, false);
});

test("schema requires every object property and every nullable proposal field", () => {
  function inspect(value: unknown): void {
    if (!value || typeof value !== "object") return;
    const schema = value as Record<string, any>;
    if (schema.type === "object") {
      assert.equal(schema.additionalProperties, false);
      assert.deepEqual([...schema.required].sort(), Object.keys(schema.properties).sort());
    }
    for (const child of Object.values(schema)) {
      if (Array.isArray(child)) child.forEach(inspect); else inspect(child);
    }
  }
  inspect(PROPOSAL_SCHEMA);
  assert.deepEqual(PROPOSAL_SCHEMA.required, ["message", "world", "motion", "training", "command"]);
});

test("instructions distinguish proposals, CPU training, untrusted context and unproven flips", () => {
  const prompt = buildAssistantInstructions(context());
  assert(prompt.startsWith(ASSISTANT_INSTRUCTIONS));
  for (const phrase of ["Nothing you propose has been applied", "never starts training", "CPU/WebAssembly",
    "experimental, not proven skills", "untrusted data", "$generated_motion", "Joint limits", '"builtin:nod"',
    "pending user-approved command", "Reward is not success", "built-in jump training task", "worker acknowledgements"]) {
    assert(prompt.includes(phrase), phrase);
  }
});

test("instructions propose ready commands immediately and stage edits only when configuration must change", () => {
  const prompt = buildAssistantInstructions(monitoredContext());
  for (const phrase of ["ONLY when the requested action requires configuration changes",
    "return the pending command immediately in this first response", "do not ask the user to repeat the request",
    "evaluating or replaying a known checkpoint", "stopping the matching live run",
    "resuming a known checkpoint with the current additional budget", "starting with the current draft",
    "Local command approval still happens after this response"])
    assert(prompt.includes(phrase), phrase);
  assert(!prompt.includes("First propose edits for the user to apply, then request a command in a separate response."));
});
