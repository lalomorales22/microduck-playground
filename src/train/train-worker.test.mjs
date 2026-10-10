import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";
import { test } from "node:test";
import ts from "typescript";

const sourceUrl = new URL("./train-worker.ts", import.meta.url);
const source = stripTypeScriptTypes(await readFile(sourceUrl, "utf8"));
const ast = ts.createSourceFile("worker.js", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
let code = source;
for (const node of ast.statements.filter(ts.isImportDeclaration).reverse()) {
  const names = node.importClause.namedBindings.elements.map((item) => item.name.text).join(", ");
  code = code.slice(0, node.getStart(ast)) + `const { ${names} } = globalThis[Symbol.for("train-worker-test")];` + code.slice(node.end);
}
code = code.replaceAll("import.meta.url", JSON.stringify(sourceUrl.href));

async function fixture(t, options = {}) {
  const messages = [], writes = [], loads = [], waiting = [];
  const world = { version: 1, objects: [] };
  const config = { envs: 1, stepsPerIter: 24, hidden: [128, 64], seed: 7, ppo: {} };
  class Trainer {
    constructor() {
      this.iteration = 0; this.ac = { actor: { paramCount: 1 }, critic: { paramCount: 1 } };
      this.usesSimd = false; this.rollout = { envs: 1, label: "test", dispose: async () => {} };
    }
    async iterate() {
      if (options.iterationError) throw new Error(options.iterationError);
      return { iteration: ++this.iteration, totalSteps: this.iteration * 24 };
    }
    restore(checkpoint) { this.iteration = checkpoint.iteration; }
    checkpoint() { return { iteration: this.iteration, config }; }
  }
  const ports = {
    setAssetBase: () => {}, compileScene: () => ({ model: {}, standKey: 0 }),
    loadModelAssets: async () => ({ mujoco: {} }), buildSpec: (task) => task,
    validateTask: (task) => task, taskLabel: (task) => task.kind,
    defaultWorld: () => world, validateWorld: (value) => value,
    checkpointMetadata: (checkpoint) => ({ task: checkpoint.taskSpec, world }),
    validateCheckpointResume: () => {}, validateTrainingConfig: (value) => value,
    TASK_OBSERVATION_SCHEMA: "test", Trainer,
    loadCheckpoint: async (name) => { loads.push(name); return options.checkpoint ?? null; },
    saveCheckpoint: async (name, checkpoint) => {
      writes.push({ name, checkpoint: structuredClone(checkpoint) });
      await options.save?.(name, checkpoint, writes.length);
    },
    loadKernels: () => null, PoolRollout: { create: async () => { throw new Error("unexpected pool"); } },
  };
  const prior = { self: globalThis.self, fetch: globalThis.fetch };
  const self = {
    onmessage: null,
    postMessage(message) {
      messages.push(message);
      options.onPost?.(message, (data) => self.onmessage({ data }));
      for (const waiter of [...waiting]) if (waiter.type === message.type) {
        waiting.splice(waiting.indexOf(waiter), 1); clearTimeout(waiter.timeout); waiter.resolve(message);
      }
    },
    close() {},
  };
  globalThis.self = self;
  globalThis.fetch = async () => ({ ok: false, status: 404 });
  globalThis[Symbol.for("train-worker-test")] = ports;
  t.after(() => {
    for (const waiter of waiting) clearTimeout(waiter.timeout);
    globalThis.self = prior.self; globalThis.fetch = prior.fetch;
    delete globalThis[Symbol.for("train-worker-test")];
  });
  await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}#${crypto.randomUUID()}`);
  const send = (data) => self.onmessage({ data });
  const wait = (type) => {
    const existing = messages.find((message) => message.type === type);
    if (existing) return Promise.resolve(existing);
    return new Promise((resolve, reject) => {
      const waiter = { type, resolve, timeout: setTimeout(() => reject(new Error(`Worker did not send ${type}`)), 2000) };
      waiting.push(waiter);
    });
  };
  send({ type: "init", baseUrl: "http://127.0.0.1/assets/", init: {
    robotXml: "robot.xml", rolloutWorkers: 1, task: { kind: "jump" }, world, config,
    autosaveEvery: 0, checkpointName: "new-run", resume: !!options.checkpoint,
    ...(options.checkpoint ? { resumeName: "original" } : {}),
  } });
  await wait("ready");
  return { send, wait, messages, writes, loads };
}

test("worker saves its final checkpoint before acknowledging completed budget", async (t) => {
  const f = await fixture(t);
  f.send({ type: "start", iterations: 2 });
  const stopped = await f.wait("stopped");
  assert.deepEqual(stopped, { type: "stopped", iteration: 2, reason: "completed", checkpointName: "new-run" });
  assert.equal(f.writes.length, 1); assert.equal(f.writes[0].checkpoint.iteration, 2);
  assert.ok(f.messages.findIndex((m) => m.type === "saved") < f.messages.indexOf(stopped));
});

test("worker acknowledges requested stop distinctly from full-budget completion", async (t) => {
  const f = await fixture(t, { onPost(message, send) { if (message.type === "stats") send({ type: "stop" }); } });
  f.send({ type: "start", iterations: 3 });
  assert.deepEqual(await f.wait("stopped"), { type: "stopped", iteration: 1, reason: "stopped", checkpointName: "new-run" });
  assert.equal(f.writes[0].checkpoint.iteration, 1);
});

test("worker resumes from the original but only saves its fork", async (t) => {
  const original = { iteration: 7, config: { envs: 1 }, taskSpec: { kind: "jump" } };
  const f = await fixture(t, { checkpoint: original });
  f.send({ type: "start", iterations: 2 });
  const stopped = await f.wait("stopped");
  assert.equal(stopped.iteration, 9); assert.deepEqual(f.loads, ["original"]);
  assert.deepEqual(f.writes.map((write) => write.name), ["new-run"]);
  assert.equal(original.iteration, 7);
});

test("checkpoint write failure cannot produce a completed or stopped success", async (t) => {
  const f = await fixture(t, { save: async () => { throw new Error("storage full"); } });
  f.send({ type: "start", iterations: 1 });
  assert.equal((await f.wait("error")).message, "storage full");
  assert.equal(f.messages.some((message) => message.type === "stopped" || message.type === "saved"), false);
});

test("training failure cannot produce a success receipt or checkpoint", async (t) => {
  const f = await fixture(t, { iterationError: "non-finite update" });
  f.send({ type: "start", iterations: 1 });
  assert.equal((await f.wait("error")).message, "non-finite update");
  assert.equal(f.writes.length, 0); assert.equal(f.messages.some((message) => message.type === "stopped"), false);
});

test("overlapping save requests serialize writes and report their snapshot iterations", async (t) => {
  let release;
  const delayed = new Promise((resolve) => { release = resolve; });
  const f = await fixture(t, { save: async (_name, _checkpoint, count) => { if (count === 1) await delayed; } });
  f.send({ type: "save", name: "manual" });
  f.send({ type: "start", iterations: 1 });
  await f.wait("stats");
  assert.equal(f.writes.length, 1); release();
  await f.wait("stopped");
  assert.deepEqual(f.writes.map((write) => [write.name, write.checkpoint.iteration]), [["manual", 0], ["new-run", 1]]);
  assert.deepEqual(f.messages.filter((message) => message.type === "saved").map((message) => message.iteration), [0, 1]);
});

test("worker rejects overflowing resume budgets before iterating", async (t) => {
  const f = await fixture(t, { checkpoint: { iteration: Number.MAX_SAFE_INTEGER, config: { envs: 1 }, taskSpec: { kind: "jump" } } });
  f.send({ type: "start", iterations: 1 });
  assert.match((await f.wait("error")).message, /target iteration/);
  assert.equal(f.writes.length, 0); assert.equal(f.messages.some((message) => message.type === "stats"), false);
});
