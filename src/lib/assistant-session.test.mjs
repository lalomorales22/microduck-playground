import assert from "node:assert/strict";
import { after, test } from "node:test";
import { AssistantSession } from "./assistant.svelte.ts";

// Exercise request/command transitions without a DOM; browser gates cover
// Svelte reactivity and rendering separately.
const previousState = globalThis.$state;
globalThis.$state = Object.assign(value => value, { snapshot: structuredClone });
after(() => { if (previousState === undefined) delete globalThis.$state; else globalThis.$state = previousState; });

const context = () => ({
  world: { version: 1, objects: [] }, motions: [], currentMotion: null,
  training: { task: "jump", iterations: 2, seed: 7, envs: 1, rolloutWorkers: 1,
    useCurrentWorld: false, hiddenPreset: "small", status: "idle", iteration: 0, reward: null },
  checkpoints: [],
});
const proposal = () => ({ message: "Start a two-iteration test, not a qualified jumping skill.",
  world: null, motion: null, training: null, command: { kind: "start_training", target: null } });
function sessionWithCommand() {
  const session = new AssistantSession();
  session.messages = [{ id: "command-1", role: "assistant", content: "Start a short run.",
    proposal: proposal(), context: context(), proposalStatus: "pending" }];
  return session;
}

test("commands require explicit execution and cannot fall through data Apply", async () => {
  const session = sessionWithCommand();
  let calls = 0;
  await session.apply("command-1", async () => { calls++; return "wrong path"; });
  assert.equal(calls, 0);
  assert.equal(session.messages[0].proposalStatus, "pending");
  await session.execute("command-1", async (command, captured) => {
    calls++;
    assert.deepEqual(command, proposal().command);
    assert.equal(captured.training.iterations, 2);
    captured.training.seed = 999;
    return "Run requested; loading environment.";
  });
  assert.equal(calls, 1);
  assert.equal(session.messages[0].context.training.seed, 7);
  assert.equal(session.messages[0].applicationResult, "Run requested; loading environment.");
  assert.equal(session.messages[0].proposalStatus, "applied");
  await session.execute("command-1", async () => { calls++; return "again"; });
  assert.equal(calls, 1);
});

test("concurrent execution dispatches at most once", async () => {
  const session = sessionWithCommand();
  let finish, calls = 0;
  const pending = session.execute("command-1", () => { calls++; return new Promise(resolve => { finish = resolve; }); });
  assert.equal(session.applying, true);
  await session.execute("command-1", async () => { calls++; return "duplicate"; });
  finish("Run requested.");
  await pending;
  assert.equal(calls, 1);
  assert.equal(session.applying, false);
});

test("resume executes once with the captured checkpoint and additional budget", async () => {
  const session = sessionWithCommand();
  const entry = session.messages[0];
  entry.proposal.command = { kind: "resume_training", target: "jump-saved-1" };
  entry.context.checkpoints = [{ name: "jump-saved-1", iteration: 10 }];
  entry.context.training.iterations = 5;
  entry.context.training.rolloutWorkers = 2;
  let calls = 0;
  const execute = async (command, captured) => {
    calls++;
    assert.equal(command.target, "jump-saved-1");
    assert.equal(captured.training.iterations, 5);
    assert.equal(captured.training.rolloutWorkers, 2);
    assert.equal(captured.checkpoints[0].iteration, 10);
    return "Resume requested into a new checkpoint; original preserved.";
  };
  await session.execute(entry.id, execute);
  await session.execute(entry.id, execute);
  assert.equal(calls, 1);
  assert.equal(entry.proposalStatus, "applied");
});

test("failed commands remain terminal rather than redispatching uncertain work", async () => {
  const session = sessionWithCommand();
  let calls = 0;
  const execute = async () => { calls++; throw new Error("The captured run changed."); };
  await session.execute("command-1", execute);
  await session.execute("command-1", execute);
  assert.equal(calls, 1);
  assert.equal(session.messages[0].proposalStatus, "failed");
  assert.equal(session.messages[0].applicationError, "The captured run changed.");
  assert.equal(session.applying, false);
});

test("data proposals still use Apply and cannot execute as commands", async () => {
  const session = sessionWithCommand();
  session.messages[0].proposal.command = null;
  session.messages[0].proposal.world = { version: 1, objects: [] };
  let calls = 0;
  await session.execute("command-1", async () => { calls++; return "wrong path"; });
  await session.apply("command-1", async () => { calls++; return "Scene applied."; });
  assert.equal(calls, 1);
  assert.equal(session.messages[0].applicationResult, "Scene applied.");
});

test("receiving a command only stages it and supersedes older pending proposals", async t => {
  const session = sessionWithCommand();
  session.consent = true;
  session.configured = true;
  session.draft = "Start the short run";
  let requests = 0;
  t.mock.method(globalThis, "fetch", async (url, options) => {
    requests++;
    assert.equal(url, "/api/assistant/chat");
    assert.equal(options.headers["X-Microduck-Assistant"], "1");
    return new Response(JSON.stringify(proposal()), { status: 200 });
  });
  await session.send(async () => context());
  assert.equal(requests, 1);
  assert.equal(session.messages[0].proposalStatus, "superseded");
  assert.equal(session.messages.at(-1).proposalStatus, "pending");
  assert.deepEqual(session.messages.at(-1).proposal.command, proposal().command);
  assert.equal(session.applying, false);
});

test("context timeout clears pending state and permits retry without fetching", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let resolveContext, requests = 0;
  t.mock.method(globalThis, "fetch", async () => { requests++; throw new Error("Unexpected fetch"); });
  const session = new AssistantSession();
  session.consent = true; session.configured = true; session.draft = "Review run";
  const pending = session.send(() => new Promise(resolve => { resolveContext = resolve; }));
  await Promise.resolve();
  t.mock.timers.tick(120000);
  await pending;
  assert.equal(session.busy, false);
  assert.equal(session.retryAvailable, true);
  assert.match(session.error, /timed out/);
  resolveContext(context());
  await Promise.resolve();
  assert.equal(requests, 0);
});

test("cancel ignores late context and never fetches", async t => {
  let resolveContext, requests = 0;
  t.mock.method(globalThis, "fetch", async () => { requests++; throw new Error("Unexpected fetch"); });
  const session = new AssistantSession();
  session.consent = true; session.configured = true; session.draft = "Review run";
  const pending = session.send(() => new Promise(resolve => { resolveContext = resolve; }));
  await Promise.resolve();
  session.cancel();
  await pending;
  resolveContext(context());
  await Promise.resolve();
  assert.equal(session.busy, false);
  assert.equal(session.notice, "Request canceled.");
  assert.equal(session.retryAvailable, true);
  assert.equal(requests, 0);
});

test("nested provider errors are bounded and retryable", async t => {
  const session = new AssistantSession();
  session.consent = true; session.configured = true; session.draft = "Review run";
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({ error: { code: "quota", message: "x".repeat(800) } }), { status: 429 }));
  await session.send(async () => context());
  assert.equal(session.error, "x".repeat(500));
  assert.equal(session.retryAvailable, true);
  assert.equal(session.busy, false);
});
