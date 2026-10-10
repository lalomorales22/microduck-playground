import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";
import { after, beforeEach, test } from "node:test";
import { compileModule } from "svelte/compiler";
import ts from "typescript";

// Compile the real rune class without a DOM, keeping worker/storage boundaries
// under test control. No production test hooks or checkpoint writes are needed.
const sourceUrl = new URL("./training.svelte.ts", import.meta.url);
const source = stripTypeScriptTypes(await readFile(sourceUrl, "utf8"));
let code = compileModule(source, { filename: "training.svelte.js", generate: "server" }).js.code;
const ast = ts.createSourceFile("training.js", code, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
const replacements = ast.statements.filter(ts.isImportDeclaration).map((node) => {
  const specifier = node.moduleSpecifier.text;
  const replacement = specifier.startsWith(".")
    ? `const { ${node.importClause.namedBindings.elements.map((item) => item.name.text).join(", ")} } = globalThis[Symbol.for("training-session-test")];`
    : `import * as $ from ${JSON.stringify(import.meta.resolve(specifier))};`;
  return { start: node.getStart(ast), end: node.end, replacement };
});
for (const { start, end, replacement } of replacements.reverse()) code = code.slice(0, start) + replacement + code.slice(end);
code = code.replaceAll("import.meta.url", JSON.stringify(sourceUrl.href));

let checkpoints, loader, motionLoader, motionFileReader, deleted;
const world = () => ({ version: 1, objects: [] });
const config = { envs: 16, stepsPerIter: 24, hidden: [128, 64], seed: 7, ppo: {} };
globalThis[Symbol.for("training-session-test")] = {
  assetBase: () => "http://127.0.0.1:8767/assets/",
  DEFAULT_PPO: {}, DEFAULT_TRAINER: config,
  deleteCheckpoint: async (name) => { deleted.push(name); },
  listCheckpoints: async () => [...checkpoints.keys()].map((name) => ({ name, bytes: 100, modified: 1 })),
  loadCheckpoint: async (name) => loader ? loader(name) : checkpoints.get(name) ?? null,
  opfsAvailable: () => true,
  BUILTIN_TASKS: [{ kind: "hold_pose", label: "Hold pose" }, { kind: "jump", label: "Jump" }],
  listMotions: async () => [], loadMotionFile: async (name) => motionLoader ? motionLoader(name) : null,
  readMotionFile: async (name) => motionFileReader(name),
  defaultWorld: world, validateWorld: (value) => structuredClone(value),
  checkpointMetadata: (checkpoint) => ({ task: checkpoint.taskSpec }),
  validateTrainingConfig: (value) => structuredClone(value),
};
const { TrainingSession } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);
const originalWorker = globalThis.Worker;
class FakeWorker {
  static instances = [];
  static failConstruction = false;
  constructor(url, options) {
    if (FakeWorker.failConstruction) throw new Error("Worker denied");
    this.url = String(url); this.name = options.name;
    this.posted = []; this.terminated = false;
    FakeWorker.instances.push(this);
  }
  postMessage(message) { this.posted.push(message); }
  terminate() { this.terminated = true; }
  emit(data) { this.onmessage?.({ data }); }
  crash(message) { this.onerror?.({ message }); }
}
globalThis.Worker = FakeWorker;
beforeEach(() => {
  checkpoints = new Map(); loader = null; motionLoader = null; motionFileReader = null; deleted = [];
  FakeWorker.instances = []; FakeWorker.failConstruction = false;
});
after(() => { globalThis.Worker = originalWorker; delete globalThis[Symbol.for("training-session-test")]; });
const ready = (worker, resumedAt = 0) => worker.emit({ type: "ready", resumedAt, params: 100,
  simd: false, envs: 16, rolloutLabel: "test", task: { kind: "jump" }, world: world() });
const report = (name) => ({ checkpointName: name, verdict: "failed", reasons: ["No takeoff"] });

test("start captures budget, creates unique run identity, and waits for stop acknowledgement", async () => {
  const s = new TrainingSession(); s.task = "jump"; s.iterations = 2;
  await s.start(false);
  const worker = FakeWorker.instances[0], firstId = s.runId;
  assert.equal(s.status, "loading"); assert.match(firstId, /^[a-f0-9-]{36}$/);
  s.iterations = 999;
  ready(worker);
  assert.deepEqual(worker.posted.at(-1), { type: "start", iterations: 2 });
  assert.equal(s.targetIteration, 2);
  s.stop();
  assert.equal(s.status, "running"); assert.equal(s.stopping, true); assert.equal(s.completionReason, null);
  worker.emit({ type: "stopped", iteration: 0, reason: "stopped", checkpointName: null });
  assert.equal(s.status, "stopped"); assert.equal(s.completionReason, "stopped");
  assert.equal(FakeWorker.instances.length, 1);
  await s.start(false);
  assert.notEqual(s.runId, firstId);
  s.dispose();
});

test("autosaves do not trigger evaluation; final completion triggers exactly one read-only worker", async () => {
  const s = new TrainingSession(); s.iterations = 2;
  await s.start(false);
  const worker = FakeWorker.instances[0]; ready(worker);
  worker.emit({ type: "saved", name: "run-one", iteration: 1 });
  assert.equal(s.evaluationStatus, "idle"); assert.equal(FakeWorker.instances.length, 1);
  worker.emit({ type: "stopped", iteration: 2, reason: "completed", checkpointName: "run-one" });
  const evaluationWorker = FakeWorker.instances[1];
  assert.equal(s.completionReason, "completed"); assert.equal(s.evaluationStatus, "running");
  assert.deepEqual(evaluationWorker.posted[0], { type: "evaluate", name: "run-one", baseUrl: "http://127.0.0.1:8767/assets/" });
  assert.equal(s.evaluation, null);
  evaluationWorker.emit({ type: "progress", progress: 0.6 });
  evaluationWorker.emit({ type: "progress", progress: 0.1 });
  assert.equal(s.evaluationProgress, 0.6);
  evaluationWorker.emit({ type: "result", report: report("run-one") });
  assert.equal(s.evaluationStatus, "done"); assert.equal(s.evaluation.verdict, "failed");
  assert.equal(evaluationWorker.terminated, true); assert.equal(worker.posted.length, 2);
  s.dispose();
});

test("resume forks the checkpoint and targets saved iteration plus captured budget", async () => {
  const original = { taskSpec: { kind: "jump" }, config, world: world(), iteration: 40, totalSteps: 15360 };
  checkpoints.set("original", original);
  const before = JSON.stringify(original);
  const s = new TrainingSession(); s.resumeFrom = "original"; s.iterations = 3;
  await s.start(true);
  const worker = FakeWorker.instances[0], init = worker.posted[0].init;
  assert.equal(init.resumeName, "original"); assert.notEqual(init.checkpointName, "original");
  assert.equal(init.resume, true); assert.equal(s.targetIteration, 43);
  ready(worker, 40);
  assert.equal(s.targetIteration, 43); assert.equal(s.resumedAt, 40);
  assert.equal(JSON.stringify(original), before); assert.deepEqual(deleted, []);
  s.dispose(); worker.emit({ type: "stopped", iteration: 40, reason: "stopped", checkpointName: init.checkpointName });
});

test("stopping while checkpoint loading prevents late worker creation", async () => {
  let release;
  loader = () => new Promise((resolve) => { release = resolve; });
  const s = new TrainingSession(); s.resumeFrom = "slow";
  const pending = s.start(true);
  assert.equal(s.status, "loading"); s.stop();
  release({ taskSpec: { kind: "jump" }, config, iteration: 8, totalSteps: 20 });
  await pending;
  assert.equal(FakeWorker.instances.length, 0); assert.equal(s.completionReason, "stopped");
});

test("worker error and constructor failure never report completion or start evaluation", async () => {
  const s = new TrainingSession();
  await s.start(false); const worker = FakeWorker.instances[0]; ready(worker);
  worker.emit({ type: "error", message: "disk full" });
  assert.equal(s.status, "error"); assert.equal(s.completionReason, "error");
  assert.equal(worker.terminated, true); assert.equal(s.evaluationStatus, "idle");
  FakeWorker.failConstruction = true;
  await s.start(false);
  assert.equal(s.status, "error"); assert.equal(s.error, "Worker denied"); s.dispose();
});

test("evaluation blocks start/delete, rejects concurrent evaluations, and ignores cancelled results", async () => {
  const s = new TrainingSession();
  const pending = s.evaluateCheckpoint("saved");
  const worker = FakeWorker.instances[0];
  await s.start(false); await s.remove("saved");
  assert.equal(s.status, "idle"); assert.deepEqual(deleted, []); assert.equal(FakeWorker.instances.length, 1);
  await assert.rejects(s.evaluateCheckpoint("other"), /active run or evaluation/);
  s.cancelEvaluation(); await pending;
  worker.emit({ type: "result", report: report("saved") });
  assert.equal(s.evaluationStatus, "idle"); assert.equal(s.evaluation, null);
  assert.equal(worker.terminated, true); s.dispose();
});

test("evaluation errors and checkpoint mismatches reject without a qualification", async () => {
  const s = new TrainingSession();
  await assert.rejects(s.evaluateCheckpoint("../bad"), /Invalid checkpoint/);
  const pending = s.evaluateCheckpoint("saved");
  const failure = assert.rejects(pending, /different checkpoint/);
  FakeWorker.instances[0].emit({ type: "result", report: report("other") });
  await failure;
  assert.equal(s.evaluationStatus, "error"); assert.equal(s.evaluation, null);
  const again = s.evaluateCheckpoint("saved"); const crashed = assert.rejects(again, /physics failed/);
  FakeWorker.instances[1].crash("physics failed"); await crashed;
  assert.equal(s.evaluationStatus, "error"); s.dispose();
});

test("evaluation has a two-minute bound and timeout terminates its worker", async (t) => {
  let timeout, delay;
  t.mock.method(globalThis, "setTimeout", (callback, ms) => { timeout = callback; delay = ms; return 1; });
  t.mock.method(globalThis, "clearTimeout", () => {});
  const s = new TrainingSession();
  const pending = s.evaluateCheckpoint("saved"); const failed = assert.rejects(pending, /two-minute/);
  assert.equal(delay, 120_000); timeout(); await failed;
  assert.equal(FakeWorker.instances[0].terminated, true); assert.equal(s.evaluationStatus, "error"); s.dispose();
});

test("disposed training saves/stops without starting evaluation", async () => {
  const s = new TrainingSession(); await s.start(false);
  const worker = FakeWorker.instances[0]; ready(worker); s.dispose();
  worker.emit({ type: "saved", name: "saved", iteration: 1 });
  worker.emit({ type: "stopped", iteration: 1, reason: "stopped", checkpointName: "saved" });
  assert.equal(worker.terminated, true); assert.equal(FakeWorker.instances.length, 1);
  assert.equal(s.evaluationStatus, "idle");
});

test("late events from a replaced training worker cannot alter the new run", async () => {
  const s = new TrainingSession(); await s.start(false);
  const old = FakeWorker.instances[0]; s.stop();
  await s.start(false); const current = FakeWorker.instances[1]; ready(current);
  old.emit({ type: "error", message: "stale failure" });
  old.emit({ type: "stopped", reason: "completed", iteration: 99, checkpointName: "old" });
  assert.equal(s.status, "running"); assert.equal(s.iteration, 0); assert.equal(s.evaluationStatus, "idle");
  s.dispose(); current.emit({ type: "stopped", reason: "stopped", iteration: 0, checkpointName: null });
});

test("invalid and pending starts clear previous run receipts and stale metrics", async () => {
  const s = new TrainingSession();
  s.runName = "old"; s.runTask = { kind: "jump" }; s.savedName = "old"; s.savedAt = 4;
  s.iteration = 4; s.last = { rewardPerStep: 9 }; s.iterations = 0;
  await s.start(false);
  assert.equal(s.status, "error"); assert.equal(s.runName, null); assert.equal(s.runTask, null);
  assert.equal(s.savedName, null); assert.equal(s.savedAt, 0); assert.equal(s.last, null);
  assert.equal(s.iteration, 0); assert.equal(s.targetIteration, 0);
  s.dispose();
});

test("save-as produces bounded replayable checkpoint names", async (t) => {
  const s = new TrainingSession(); await s.start(false);
  const worker = FakeWorker.instances[0]; ready(worker); s.iteration = 1;
  const originalPrompt = globalThis.prompt;
  globalThis.prompt = () => `..${"long".repeat(30)}..suffix`;
  t.after(() => { globalThis.prompt = originalPrompt; });
  assert.equal(s.saveAs(), true);
  const name = worker.posted.at(-1).name;
  assert.equal(name.length, 64); assert.match(name, /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/);
  assert.equal(name.includes(".."), false);
  s.dispose(); worker.emit({ type: "stopped", reason: "stopped", iteration: 0, checkpointName: null });
});

test("resume bounds saved environment count times requested workers before allocating", async () => {
  checkpoints.set("large", { taskSpec: { kind: "jump" }, config: { ...config, envs: 64 }, iteration: 2, totalSteps: 100 });
  const s = new TrainingSession(); s.resumeFrom = "large"; s.rolloutWorkers = 4;
  await s.start(true);
  assert.equal(s.status, "error"); assert.match(s.error, /128 total environments/);
  assert.equal(FakeWorker.instances.length, 0); s.dispose();
});

test("resume rejects non-finite and overflowing iteration targets before allocating", async () => {
  for (const iteration of [Infinity, NaN, -1, 1.5, Number.MAX_SAFE_INTEGER]) {
    checkpoints.set("bad", { taskSpec: { kind: "jump" }, config, iteration, totalSteps: 100 });
    const s = new TrainingSession(); s.resumeFrom = "bad";
    await s.start(true);
    assert.equal(s.status, "error"); assert.match(s.error, /checkpoint iteration/);
    assert.equal(FakeWorker.instances.length, 0); s.dispose();
  }
});

test("saving over an evaluated checkpoint invalidates its old qualification", async () => {
  const s = new TrainingSession(); await s.start(false);
  const worker = FakeWorker.instances[0]; ready(worker);
  worker.emit({ type: "stopped", iteration: 0, reason: "stopped", checkpointName: null });
  const pending = s.evaluateCheckpoint("saved");
  FakeWorker.instances[1].emit({ type: "result", report: { ...report("saved"), verdict: "passed" } });
  await pending;
  worker.emit({ type: "saved", name: "saved", iteration: 1 });
  assert.equal(s.evaluation, null); assert.equal(s.evaluationStatus, "idle");
  s.dispose();
});

test("new training snapshots config, task choice and scene before motion IO", async (t) => {
  const originalStorage = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  let storedWorld = JSON.stringify({ version: 1, objects: [{ id: "reviewed-scene" }] });
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: () => storedWorld } });
  t.after(() => {
    if (originalStorage) Object.defineProperty(globalThis, "localStorage", originalStorage);
    else delete globalThis.localStorage;
  });
  let release, loadedId;
  motionLoader = (id) => { loadedId = id; return new Promise((resolve) => { release = resolve; }); };
  const s = new TrainingSession(); s.task = "motion:reviewed-motion"; s.useCurrentWorld = true;
  s.envs = 16; s.seed = 12; s.stepsPerIter = 24; s.hiddenPreset = "small";
  const pending = s.start(false);
  s.envs = 128; s.seed = 99; s.stepsPerIter = 96; s.hiddenPreset = "reference";
  s.task = "jump"; s.useCurrentWorld = false;
  storedWorld = JSON.stringify({ version: 1, objects: [{ id: "later-scene" }] });
  release({ name: "Reviewed motion", keyframes: [] }); await pending;
  const init = FakeWorker.instances[0].posted[0].init;
  assert.equal(loadedId, "reviewed-motion"); assert.equal(init.task.kind, "motion");
  assert.equal(init.task.motion.name, "Reviewed motion");
  assert.equal(init.config.envs, 16); assert.equal(init.config.seed, 12);
  assert.equal(init.config.stepsPerIter, 24); assert.deepEqual(init.config.hidden, [128, 64]);
  assert.deepEqual(init.world.objects, [{ id: "reviewed-scene" }]);
  s.dispose();
});

test("late checkpoint selection and failure cannot replace a later selection", async () => {
  for (const fail of [false, true]) {
    let release, reject;
    const checkpoint = { taskSpec: { kind: "jump" }, config, iteration: 3 };
    loader = (name) => name === "slow" ? new Promise((resolve, fail) => { release = resolve; reject = fail; }) : checkpoint;
    const s = new TrainingSession(); const pending = s.selectCheckpoint("slow");
    await s.selectCheckpoint("latest"); const summary = s.resumeSummary;
    if (fail) reject(new Error("stale load failure")); else release(checkpoint);
    await pending;
    assert.equal(s.resumeFrom, "latest"); assert.equal(s.resumeSummary, summary); assert.equal(s.error, null);
    s.dispose();
  }
});

test("toggling selection invalidates a previous pending selection", async () => {
  let release;
  loader = () => new Promise((resolve) => { release = resolve; });
  const s = new TrainingSession(); s.resumeFrom = "selected"; s.resumeSummary = "old summary";
  const pending = s.selectCheckpoint("slow"); await s.selectCheckpoint("selected");
  release({ taskSpec: { kind: "jump" }, config, iteration: 3 }); await pending;
  assert.equal(s.resumeFrom, null); assert.equal(s.resumeSummary, null); s.dispose();
});

test("new runs and disposal invalidate pending checkpoint selections", async () => {
  for (const action of ["start", "dispose"]) {
    let release;
    loader = () => new Promise((resolve) => { release = resolve; });
    const s = new TrainingSession(); const pending = s.selectCheckpoint("slow");
    if (action === "start") await s.start(false); else s.dispose();
    release({ taskSpec: { kind: "jump" }, config, iteration: 3 }); await pending;
    assert.equal(s.resumeFrom, null); assert.equal(s.resumeSummary, null); assert.equal(s.error, null);
    s.dispose();
  }
});

test("assistant motion revision changes and task mismatches refuse worker creation", async () => {
  const file = new File([JSON.stringify({ name: "Updated motion", keyframes: [] })], "motion.json", { lastModified: 200 });
  motionFileReader = async () => file;
  for (const expected of [
    { id: "approved", modified: 100, bytes: file.size },
    { id: "approved", modified: 200, bytes: file.size + 1 },
    { id: "different-motion", modified: 200, bytes: file.size },
  ]) {
    const s = new TrainingSession(); s.task = "motion:approved";
    await s.start(false, expected);
    assert.equal(s.status, "error"); assert.match(s.error, /changed after approval|differs from the approved/);
    assert.equal(FakeWorker.instances.length, 0); s.dispose();
  }
});

test("assistant motion reads the approved immutable file snapshot exactly once", async () => {
  const approved = new File([JSON.stringify({ name: "Approved snapshot", keyframes: [] })], "motion.json", { lastModified: 100 });
  let current = approved, reads = 0, textReads = 0;
  const readApprovedText = approved.text.bind(approved);
  approved.text = async () => {
    textReads++;
    current = new File([JSON.stringify({ name: "Edited later", keyframes: [] })], "motion.json", { lastModified: 200 });
    return readApprovedText();
  };
  motionFileReader = async (id) => { reads++; assert.equal(id, "approved"); return current; };
  motionLoader = async () => { throw new Error("Revision-checked start must not re-read by name"); };
  const s = new TrainingSession(); s.task = "motion:approved";
  await s.start(false, { id: "approved", modified: approved.lastModified, bytes: approved.size });
  assert.equal(s.status, "loading"); assert.equal(reads, 1); assert.equal(textReads, 1);
  assert.equal(FakeWorker.instances[0].posted[0].init.task.motion.name, "Approved snapshot");
  assert.equal(current.lastModified, 200); s.dispose();
});
