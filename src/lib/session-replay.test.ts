// Exercise the real Session lifecycle without a browser or asset/network loads.
// Svelte's server transform removes reactive plumbing; dependency imports alone
// are replaced through the TypeScript AST, leaving private methods intact.
import assert from "node:assert/strict";
import fs from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { test } from "node:test";
import { compileModule } from "svelte/compiler";
import ts from "typescript";
import * as microduck from "../sim/microduck.ts";
import * as worldModule from "../sim/world.ts";
import { SHIPPED_POLICIES } from "../sim/policy-catalog.ts";
import type { Session as SessionInstance } from "./session.svelte.ts";
import type { WorldConfig } from "../sim/world.ts";

const clone = <T>(value: T): T => structuredClone(value);
const source = stripTypeScriptTypes(fs.readFileSync(new URL("./session.svelte.ts", import.meta.url), "utf8"));
const compiled = compileModule(source, { filename: "session.svelte.js", generate: "server" }).js.code;
const parsed = ts.createSourceFile("session.js", compiled, ts.ScriptTarget.ESNext, true, ts.ScriptKind.JS);
const dependency = (module: string, name?: string): ts.Expression => {
  const moduleValue = ts.factory.createElementAccessExpression(ts.factory.createIdentifier("dependencies"), ts.factory.createStringLiteral(module));
  return name ? ts.factory.createElementAccessExpression(moduleValue, ts.factory.createStringLiteral(name)) : moduleValue;
};
const local = (name: string, initializer: ts.Expression): ts.VariableStatement => ts.factory.createVariableStatement(undefined,
  ts.factory.createVariableDeclarationList([ts.factory.createVariableDeclaration(name, undefined, undefined, initializer)], ts.NodeFlags.Const));
const statements = parsed.statements.flatMap((statement): ts.Statement[] => {
  if (ts.isImportDeclaration(statement)) {
    const module = (statement.moduleSpecifier as ts.StringLiteral).text;
    const clause = statement.importClause;
    assert.ok(clause, "Session imports must have bindings");
    const imports: ts.Statement[] = [];
    if (clause.name) imports.push(local(clause.name.text, dependency(module, "default")));
    const bindings = clause.namedBindings;
    if (bindings && ts.isNamespaceImport(bindings)) imports.push(local(bindings.name.text, dependency(module)));
    else if (bindings) for (const binding of bindings.elements) {
      imports.push(local(binding.name.text, dependency(module, binding.propertyName?.text ?? binding.name.text)));
    }
    return imports;
  }
  if (ts.isVariableStatement(statement)) return [ts.factory.updateVariableStatement(statement,
    statement.modifiers?.filter((modifier) => modifier.kind !== ts.SyntaxKind.ExportKeyword), statement.declarationList)];
  if (ts.isClassDeclaration(statement)) return [ts.factory.updateClassDeclaration(statement,
    statement.modifiers?.filter((modifier) => modifier.kind !== ts.SyntaxKind.ExportKeyword),
    statement.name, statement.typeParameters, statement.heritageClauses, statement.members)];
  return [statement];
});
const body = ts.createPrinter().printFile(ts.factory.updateSourceFile(parsed, statements));
const createSessionClass = new Function("dependencies", "localStorage", "requestAnimationFrame", "cancelAnimationFrame",
  "setTimeout", "clearTimeout", "performance", `${body}\nreturn Session;`) as (...args: unknown[]) => new () => SessionInstance;

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((accept, refuse) => { resolve = accept; reject = refuse; });
  return { promise, resolve, reject };
}
async function flush(): Promise<void> { await Promise.resolve(); await Promise.resolve(); }
function world(id: string): WorldConfig {
  return { version: 1, objects: [worldModule.createWorldObject("platform", id)] };
}
function checkpoint(savedWorld: WorldConfig) {
  return { valid: true, world: clone(savedWorld), taskLabel: "jump", iteration: 120, experimental: true };
}

function harness(initialWorld = world("current")) {
  const events: string[] = [];
  const checkpoints = new Map<string, unknown>();
  const timers = new Map<number, () => void>();
  const frames = new Map<number, () => void>();
  const storage = new Map([["wicroduck.world.v1", JSON.stringify(initialWorld)]]);
  const controls = {
    compileFailure: false, renderFailures: 0, renderCount: 0, renderFailureAt: null as number | null,
    timerDelays: [] as number[], stepGate: null as Promise<void> | null,
    loadGate: null as Promise<void> | null, loaded: [] as string[], cancelledTimers: [] as number[],
  };
  class Model {
    deleted = false;
    readonly world: WorldConfig;
    constructor(world: WorldConfig) { this.world = world; }
    jnt() { return { dofadr: 0 }; }
    delete() { this.deleted = true; events.push("model.delete"); }
  }
  class Data {
    deleted = false;
    time = 0;
    qpos = new Float64Array(21);
    qvel = new Float64Array(20);
    ctrl = new Float64Array(microduck.NUM_JOINTS);
    readonly model: Model;
    constructor(model: Model) { this.model = model; datas.push(this); }
    delete() { this.deleted = true; events.push("data.delete"); }
  }
  const models: Model[] = [], datas: Data[] = [];
  function stand(data: Data) {
    data.qpos.fill(0); data.qvel.fill(0); data.time = 0;
    data.qpos[2] = 0.12; data.qpos[3] = 1;
    data.qpos.set(microduck.DEFAULT_POSE, 7); data.ctrl.set(microduck.DEFAULT_POSE);
  }
  const mujoco = { MjData: Data, mjtObj: { mjOBJ_BODY: { value: 1 } },
    mj_name2id: () => 1, mj_forward() {},
    mj_resetDataKeyframe(_model: Model, data: Data) { stand(data); } };
  class Policy {
    resets = 0;
    readonly label: string;
    constructor(label: string) { this.label = label; }
    reset() { this.resets++; }
    static async load(_url: string, label: string) { events.push("shipped.load"); return new Policy(label); }
  }
  class CheckpointPolicy extends Policy {
    readonly world: WorldConfig;
    readonly taskLabel = "jump";
    readonly iteration = 120;
    readonly experimental = true;
    constructor(raw: unknown, name: string) {
      super(name);
      events.push("checkpoint.construct");
      const value = raw as ReturnType<typeof checkpoint>;
      if (!value?.valid) throw new Error("Invalid checkpoint network");
      this.world = worldModule.validateWorld(value.world);
    }
  }
  class Controller {
    resets = 0;
    knocks = 0;
    onRecoveryEnd: ((succeeded: boolean) => void) | null = null;
    readonly sim: { model: Model; data: Data };
    policy: Policy;
    constructor(sim: { model: Model; data: Data }, policy: Policy) {
      this.sim = sim; this.policy = policy;
      events.push("controller.construct"); controllers.push(this);
    }
    reset() { events.push("controller.reset"); this.resets++; stand(this.sim.data); this.policy.reset(); }
    setPolicy(policy: Policy) { events.push("controller.policy"); this.policy = policy; this.reset(); }
    setCommand() {}
    knockDown() { this.knocks++; this.sim.data.qpos[3] = Math.SQRT1_2; this.sim.data.qpos[4] = Math.SQRT1_2; }
    push() {}
    standUp() {}
    async step() { events.push("step.start"); await controls.stepGate; this.sim.data.time += microduck.CTRL_DT; events.push("step.end"); }
    telemetry() { return { phase: "standing", gravityZ: -(1 - 2 * this.sim.data.qpos[4] ** 2),
      height: this.sim.data.qpos[2], simTime: this.sim.data.time, phaseSteps: 0 }; }
  }
  const controllers: Controller[] = [];
  class Viewer {
    model: Model | null = null;
    disposed = false;
    constructor(_canvas: unknown) { viewers.push(this); }
    build(model: Model) { events.push("viewer.build"); this.model = model; }
    sync() {}
    render() {
      events.push("viewer.render");
      controls.renderCount++;
      if (controls.renderFailureAt === controls.renderCount) {
        controls.renderFailureAt = null;
        throw new Error("Post-commit render failed");
      }
      if (controls.renderFailures > 0) { controls.renderFailures--; throw new Error("Candidate render failed"); }
    }
    setCollisionVisible() {}
    setFollowEnabled() {}
    frameWorld() {}
    dispose() { this.disposed = true; }
  }
  const viewers: Viewer[] = [];
  const dependencies: Record<string, unknown> = {
    "svelte/internal/server": {},
    "../sim/scene.ts": {
      async loadModelAssets() { return { mujoco }; },
      compileScene(_assets: unknown, options: { world: WorldConfig }) {
        events.push("scene.compile");
        if (controls.compileFailure) throw new Error("Candidate compile failed");
        const model = new Model(clone(options.world)); models.push(model);
        return { model, standKey: 0 };
      },
    },
    "../sim/policy.ts": { Policy, CheckpointPolicy },
    "../sim/policy-catalog.ts": { SHIPPED_POLICIES },
    "../train/checkpoint-store.ts": {
      async listCheckpoints() { return [...checkpoints.keys()].map((name) => ({ name })); },
      async loadCheckpoint(name: string) { controls.loaded.push(name); await controls.loadGate; return clone(checkpoints.get(name) ?? null); },
    },
    "../sim/controller.ts": { MicroduckController: Controller },
    "../sim/microduck.ts": microduck,
    "../sim/world.ts": worldModule,
    "../render/viewer.ts": { Viewer },
    "../asset-url.ts": { assetUrl: (value: string) => value },
  };
  let serial = 0;
  const Session = createSessionClass(dependencies,
    { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => storage.set(key, value) },
    (callback: () => void) => { const id = ++serial; frames.set(id, callback); return id; },
    (id: number) => frames.delete(id),
    (callback: () => void, delay: number) => { const id = ++serial; controls.timerDelays.push(delay); timers.set(id, callback); return id; },
    (id: number | undefined) => { if (id !== undefined) { controls.cancelledTimers.push(id); timers.delete(id); } },
    { now: () => 0 });
  const session = new Session();
  return { session, events, controls, checkpoints, timers, models, datas, controllers, viewers, storage,
    async start() { await session.start({} as HTMLCanvasElement); assert.equal(session.error, null); assert.equal(session.ready, true); },
    fireNextTimer() {
      const next = timers.entries().next().value;
      assert.ok(next, "a repeat timer must be pending");
      timers.delete(next[0]); next[1]();
    },
    async destroy() { session.destroy(); await flush(); } };
}

test("checkpoint replay restores its saved scene, starts upright and paused, and cancels repeated knockdown", async () => {
  const h = harness(), savedWorld = world("saved");
  h.checkpoints.set("jump-run", checkpoint(savedWorld));
  try {
    await h.start();
    const old = h.controllers[0];
    h.session.setAutoRepeat(true); await flush();
    old.onRecoveryEnd?.(true);
    assert.equal(h.timers.size, 1);
    assert.ok(old.sim.data.qpos[4] > 0);
    old.sim.data.time = 2;
    h.session.forward = 0.2;
    assert.equal(await h.session.replayCheckpoint("jump-run"), true);
    assert.equal(h.session.selectedCheckpointName, "jump-run");
    assert.equal(h.session.activePolicy, "ckpt:jump-run");
    assert.deepEqual(h.session.world, savedWorld);
    assert.deepEqual(h.session.checkpointWorld, savedWorld);
    assert.equal(h.session.checkpointSceneMatches, true);
    assert.equal(h.session.paused, true);
    assert.equal(h.session.autoRepeat, false);
    assert.equal(h.timers.size, 0);
    assert.equal(h.controls.cancelledTimers.length, 1);
    assert.equal(h.session.simTime, 0);
    assert.equal(h.session.uprightPct, 100);
    assert.equal(h.session.forward, 0);
    assert.equal(h.models.length, 2);
    assert.equal(old.sim.model.deleted, true);
    assert.equal(old.sim.data.deleted, true);
    const next = h.controllers.at(-1)!;
    assert.equal(next.policy.label, "jump-run", "replacement controller receives candidate policy");
    assert.equal(next.policy.resets, 1);
    assert.equal(next.sim.data.qpos[3], 1);
    assert.ok(next.sim.data.qvel.every((value) => value === 0));
    assert.deepEqual(JSON.parse(h.storage.get("wicroduck.world.v1")!), savedWorld);
  } finally { await h.destroy(); }
});

test("checkpoint with equal world skips compilation while resetting and pausing the existing controller", async () => {
  const initial = world("same"), h = harness(initial);
  h.checkpoints.set("same-world", checkpoint(initial));
  try {
    await h.start(); const old = h.controllers[0];
    old.knockDown(); old.sim.data.time = 5;
    assert.equal(await h.session.selectPolicy("ckpt:same-world"), true);
    assert.equal(h.models.length, 1);
    assert.equal(h.controllers.length, 1);
    assert.equal(old.resets, 1);
    assert.equal(old.sim.data.qpos[3], 1);
    assert.equal(old.sim.data.time, 0);
    assert.equal(h.session.simTime, 0);
    assert.equal(h.session.paused, true);
    assert.equal(h.session.selectedCheckpointName, "same-world");
  } finally { await h.destroy(); }
});

for (const paused of [false, true]) {
  test(`shipped selection preserves world and paused=${paused}`, async () => {
    const h = harness(), savedWorld = world("saved");
    h.checkpoints.set("jump-run", checkpoint(savedWorld));
    try {
      await h.start(); await h.session.replayCheckpoint("jump-run");
      h.session.setPaused(paused);
      const count = h.models.length;
      assert.equal(await h.session.selectPolicy("alpha_stand.onnx"), true);
      assert.deepEqual(h.session.world, savedWorld);
      assert.equal(h.session.paused, paused);
      assert.equal(h.models.length, count);
      assert.equal(h.session.selectedCheckpointName, null);
      assert.equal(h.session.checkpointWorld, null);
      assert.equal(h.session.checkpointLabel, "");
    } finally { await h.destroy(); }
  });
}

for (const problem of ["missing", "invalid", "compile", "render"] as const) {
  test(`${problem} checkpoint replay preserves old scene, policy, pause and repeat timer`, async () => {
    const initial = world("old"), h = harness(initial);
    if (problem !== "missing") h.checkpoints.set("broken", problem === "invalid" ? { valid: false } : checkpoint(world("next")));
    try {
      await h.start();
      h.session.setPaused(true);
      h.session.setAutoRepeat(true); await flush();
      const old = h.controllers[0], policy = old.policy;
      old.onRecoveryEnd?.(true);
      const oldData = Float64Array.from(old.sim.data.qpos);
      h.controls.compileFailure = problem === "compile";
      h.controls.renderFailures = problem === "render" ? 1 : 0;
      assert.equal(await h.session.replayCheckpoint("broken"), false);
      assert.ok(h.session.policyError);
      assert.equal(h.session.activePolicy, "alpha_stand.onnx");
      assert.equal(h.session.selectedCheckpointName, null);
      assert.deepEqual(h.session.world, initial);
      assert.equal(h.session.paused, true);
      assert.equal(h.session.autoRepeat, true);
      assert.equal(h.timers.size, 1);
      assert.equal(old.policy, policy);
      assert.deepEqual(old.sim.data.qpos, oldData);
      assert.equal(old.sim.model.deleted, false);
      assert.equal(old.sim.data.deleted, false);
      assert.equal(h.viewers[0].model, old.sim.model);
      assert.equal(h.session.policyBusy, false);
      if (problem === "render") {
        assert.equal(h.models[1].deleted, true);
        assert.equal(h.datas[1].deleted, true);
      }
    } finally { await h.destroy(); }
  });
}

for (const sameWorld of [true, false]) {
  test(`post-commit render failure honestly reports loaded selection with sameWorld=${sameWorld}`, async () => {
    const initial = world("old"), h = harness(initial);
    const savedWorld = sameWorld ? initial : world("new");
    h.checkpoints.set("jump", checkpoint(savedWorld));
    try {
      await h.start();
      h.controls.renderFailureAt = h.controls.renderCount + (sameWorld ? 1 : 2);
      assert.equal(await h.session.replayCheckpoint("jump"), true);
      assert.equal(h.session.activePolicy, "ckpt:jump");
      assert.equal(h.session.selectedCheckpointName, "jump");
      assert.deepEqual(h.session.world, savedWorld);
      assert.equal(h.session.paused, true);
      assert.match(h.session.policyError ?? "", /Policy loaded, but rendering failed/);
      assert.equal(h.controllers.at(-1)!.policy.label, "jump");
      assert.equal(h.models.length, sameWorld ? 1 : 2);
      assert.equal(h.models[0].deleted, !sameWorld);
      assert.equal(h.session.policyBusy, false);
    } finally { await h.destroy(); }
  });
}

test("unsafe checkpoint names never reach storage or scene construction", async () => {
  const h = harness();
  try {
    await h.start();
    for (const name of ["", "../jump", "a..b", "/tmp/file", "x".repeat(65), ".", "bad name"]) {
      assert.equal(await h.session.replayCheckpoint(name), false);
      assert.equal(await h.session.selectPolicy(`ckpt:${name}`), false);
    }
    assert.deepEqual(h.controls.loaded, []);
    assert.equal(h.models.length, 1);
    assert.equal(h.session.selectedCheckpointName, null);
  } finally { await h.destroy(); }
});

test("not-ready and busy sessions refuse replay without storage reads", async () => {
  const h = harness();
  try {
    assert.equal(await h.session.replayCheckpoint("jump"), false);
    await h.start();
    for (const key of ["policyBusy", "worldBusy"] as const) {
      h.session[key] = true;
      assert.equal(await h.session.replayCheckpoint("jump"), false);
      h.session[key] = false;
    }
    assert.deepEqual(h.controls.loaded, []);
    assert.equal(h.models.length, 1);
  } finally { await h.destroy(); }
});

test("replay waits for an in-flight step before loading and refuses a concurrent selection", async () => {
  const h = harness(), gate = deferred<void>();
  h.checkpoints.set("jump", checkpoint(world("next")));
  try {
    await h.start(); h.session.setPaused(true); h.controls.stepGate = gate.promise;
    const step = h.session.singleStep();
    const replay = h.session.replayCheckpoint("jump");
    await flush();
    assert.equal(h.session.policyBusy, true);
    assert.deepEqual(h.controls.loaded, []);
    assert.equal(h.models[0].deleted, false);
    assert.equal(await h.session.selectPolicy("alpha_stand.onnx"), false);
    gate.resolve(); await step;
    assert.equal(await replay, true);
    assert.deepEqual(h.controls.loaded, ["jump"]);
    assert.equal(h.session.simTime, 0);
    assert.equal(h.session.paused, true);
  } finally { gate.resolve(); await h.destroy(); }
});

test("deferred checkpoint load leaves old scene intact until successful commit", async () => {
  const h = harness(), gate = deferred<void>();
  h.checkpoints.set("jump", checkpoint(world("next")));
  try {
    await h.start(); h.controls.loadGate = gate.promise;
    const replay = h.session.replayCheckpoint("jump"); await flush();
    assert.deepEqual(h.controls.loaded, ["jump"]);
    assert.equal(h.session.policyBusy, true);
    assert.equal(h.session.activePolicy, "alpha_stand.onnx");
    assert.equal(h.models.length, 1);
    assert.equal(h.models[0].deleted, false);
    assert.equal(await h.session.replayCheckpoint("jump"), false);
    gate.resolve(); assert.equal(await replay, true);
    assert.equal(h.session.policyBusy, false);
    assert.equal(h.models.length, 2);
  } finally { gate.resolve(); await h.destroy(); }
});

for (const succeeds of [false, true]) {
  test(`repeat timer reschedules during deferred load and ${succeeds ? "cancels on success" : "survives failure"}`, async () => {
    const h = harness(), gate = deferred<void>();
    h.checkpoints.set("jump", succeeds ? checkpoint(world("next")) : { valid: false });
    try {
      await h.start();
      h.session.setAutoRepeat(true); await flush();
      const old = h.controllers[0];
      old.onRecoveryEnd?.(true);
      assert.equal(h.timers.size, 1);
      const knocks = old.knocks;
      h.controls.loadGate = gate.promise;
      const replay = h.session.replayCheckpoint("jump"); await flush();
      assert.equal(h.session.policyBusy, true);
      h.fireNextTimer(); await flush();
      assert.equal(h.timers.size, 1, "busy load must reschedule rather than consume repeat");
      assert.equal(old.knocks, knocks, "busy load cannot knock down the current simulation");
      gate.resolve();
      assert.equal(await replay, succeeds);
      assert.equal(h.session.autoRepeat, !succeeds);
      assert.equal(h.timers.size, succeeds ? 0 : 1);
      assert.ok(h.controls.timerDelays.every((delay) => delay === 1200));
      if (!succeeds) {
        h.fireNextTimer(); await flush();
        assert.equal(old.knocks, knocks + 1, "failed load's preserved repeat still executes");
      }
    } finally { gate.resolve(); await h.destroy(); }
  });
}

test("disposal during checkpoint loading refuses commit and never builds a replacement", async () => {
  const h = harness(), gate = deferred<void>();
  h.checkpoints.set("jump", checkpoint(world("next")));
  try {
    await h.start(); h.controls.loadGate = gate.promise;
    const replay = h.session.replayCheckpoint("jump"); await flush();
    h.session.destroy(); await flush();
    gate.resolve(); assert.equal(await replay, false);
    assert.equal(h.models.length, 1);
    assert.equal(h.models[0].deleted, true);
    assert.equal(h.session.activePolicy, "alpha_stand.onnx");
    assert.equal(h.session.policyBusy, false);
    assert.equal(await h.session.selectPolicy("alpha_stand.onnx"), false);
  } finally { gate.resolve(); }
});
