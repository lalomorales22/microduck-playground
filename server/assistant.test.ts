import assert from "node:assert/strict";
import { test } from "node:test";
import { createServer, request as httpRequest, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdir, mkdtemp, realpath, rm, symlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import { createAssistantMiddleware, assistantPlugin } from "./assistant.ts";
import { PROPOSAL_SCHEMA } from "../src/assistant/contract.ts";

// All credentials below are inert test fixtures; no test uses process.env or a real AI endpoint.
const KEY = "sk-test-only-not-a-real-credential";
const PROPOSAL = { message: "Proposed a clear floor for review.", world: { version: 1, objects: [] }, motion: null, training: null };
const INPUT = { messages: [{ role: "user", content: "Clear the floor." }], context: {
  world: { version: 1, objects: [] }, motions: [], currentMotion: null,
  training: { task: "standup", iterations: 10, seed: 1, envs: 2, rolloutWorkers: 1,
    useCurrentWorld: false, hiddenPreset: "small", status: "idle", iteration: 0, reward: null },
} };
const monitoredInput = () => ({ ...INPUT, context: { ...INPUT.context,
  run: { runId: "jump-run-1", task: "Jump", status: "stopped", completionReason: "completed" as string | null,
    iteration: 50, targetIteration: 50, totalSteps: 19200, reward: 0.5, standing: 0.75,
    checkpoint: "jump-checkpoint-1", evaluation: null },
  checkpoints: [{ name: "jump-checkpoint-1", iteration: 50, modified: 1234567890, bytes: 100000 }],
} });
function reply(proposal: unknown = PROPOSAL): Response {
  return Response.json({ status: "completed", output: [
    { type: "reasoning", summary: [{ type: "summary_text", text: "not user-visible" }] },
    { type: "message", status: "completed", role: "assistant", content: [{ type: "output_text", text: JSON.stringify(proposal) }] },
  ] });
}
type Settings = { method?: string; headers?: Record<string, string>; body?: string; route?: string };
type Options = NonNullable<Parameters<typeof createAssistantMiddleware>[0]>;
async function fixture(options: Options = {}, configuration = `OPENAI_API_KEY=${KEY}\nOPENAI_MODEL=gpt-5.4-mini\n`) {
  const root = await realpath(await mkdtemp(path.join(tmpdir(), "microduck-assistant-test-")));
  const envfile = path.join(root, "server.env");
  await writeFile(envfile, configuration, { mode: 0o600 });
  const env: NodeJS.ProcessEnv = { MICRODUCK_ENV_FILE: envfile, ...options.env };
  let calls = 0;
  const upstream = options.fetchImpl ?? (async () => reply()) as typeof fetch;
  const middleware = createAssistantMiddleware({ ...options, appRoot: root, env,
    fetchImpl: (async (...args: Parameters<typeof fetch>) => { calls++; return upstream(...args); }) as typeof fetch });
  const server = createServer((request, response) => middleware(request, response, () => {
    response.statusCode = 404; response.end("not found");
  }));
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert(address && typeof address !== "string");
  const url = `http://127.0.0.1:${address.port}`;
  const headers = { Origin: url, "X-Microduck-Assistant": "1", "Content-Type": "application/json" };
  return { root, envfile, env, url, headers, server, calls: () => calls,
    async send(settings: Settings = {}) {
      return new Promise<{ status: number; headers: Headers; text: string; json: any }>((resolve, reject) => {
        const request = httpRequest(url + (settings.route ?? "/api/assistant/chat"), {
          method: settings.method ?? "POST", headers: { ...headers, ...settings.headers },
        }, response => {
          const chunks: Buffer[] = [];
          response.on("data", chunk => chunks.push(chunk));
          response.on("error", reject);
          response.on("end", () => {
            const text = Buffer.concat(chunks).toString();
            const received = new Headers();
            for (const [name, value] of Object.entries(response.headers)) {
              if (value !== undefined) received.set(name, Array.isArray(value) ? value.join(", ") : value);
            }
            try { resolve({ status: response.statusCode!, headers: received, text, json: JSON.parse(text) }); }
            catch (error) { reject(error); }
          });
        });
        request.on("error", reject);
        request.end((settings.method ?? "POST") === "POST" ? settings.body ?? JSON.stringify(INPUT) : undefined);
      });
    },
    async close() {
      server.closeAllConnections();
      await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
      await rm(root, { recursive: true, force: true });
    },
  };
}

test("blank configuration reports disabled and rejects chat without an upstream call", async t => {
  const f = await fixture({}, "OPENAI_API_KEY=\nOPENAI_MODEL=gpt-5.4-mini\n"); t.after(() => f.close());
  const status = await f.send({ method: "GET", route: "/api/assistant/status" });
  assert.deepEqual(status.json, { configured: false, model: "gpt-5.4-mini" });
  assert.equal(status.headers.get("cache-control"), "no-store");
  const result = await f.send();
  assert.equal(result.status, 503); assert.equal(result.json.error.code, "not_configured");
  assert.equal(f.calls(), 0);
});

test("valid strict Responses request returns only a revalidated completed proposal", async t => {
  const f = await fixture({ fetchImpl: (async (url, init) => {
    assert.equal(url, "https://api.openai.com/v1/responses");
    assert.equal(init?.redirect, "error"); assert.equal(init?.method, "POST");
    assert.equal((init!.headers as Record<string, string>).Authorization, `Bearer ${KEY}`);
    const body = JSON.parse(init!.body as string);
    assert.equal(body.store, false); assert.equal(body.model, "gpt-5.4-mini");
    assert.equal(body.max_output_tokens, 4096);
    assert.deepEqual(body.input, INPUT.messages);
    assert.deepEqual(body.text.format, { type: "json_schema", name: "microduck_proposal", strict: true, schema: PROPOSAL_SCHEMA });
    assert.equal(typeof body.instructions, "string");
    assert(!JSON.stringify(body).includes(KEY));
    return reply();
  }) as typeof fetch }); t.after(() => f.close());
  assert.deepEqual((await f.send({ method: "GET", route: "/api/assistant/status" })).json,
    { configured: true, model: "gpt-5.4-mini" });
  const result = await f.send();
  assert.equal(result.status, 200); assert.deepEqual(result.json, PROPOSAL);
  assert.equal(result.headers.get("access-control-allow-origin"), null);
  assert(!result.text.includes("not user-visible")); assert.equal(f.calls(), 1);
});

test("envfile is reread on every request, with explicit server environment fallback", async t => {
  const f = await fixture({}, "OPENAI_API_KEY=\n"); t.after(() => f.close());
  const status = () => f.send({ method: "GET", route: "/api/assistant/status" });
  assert.equal((await status()).json.configured, false);
  await writeFile(f.envfile, `OPENAI_API_KEY=${KEY}\n`);
  assert.equal((await status()).json.configured, true);
  await writeFile(f.envfile, "OPENAI_API_KEY=\n");
  assert.equal((await status()).json.configured, false);
  f.env.OPENAI_API_KEY = KEY;
  assert.equal((await status()).json.configured, true);
  assert.equal(f.calls(), 0);
});

test("allowlisted gpt-6-luna is reported and uses bounded low reasoning", async t => {
  const f = await fixture({ fetchImpl: (async (_url, init) => {
    const body = JSON.parse(init!.body as string);
    assert.equal(body.model, "gpt-6-luna");
    assert.deepEqual(body.reasoning, { effort: "low" });
    assert.equal(body.max_output_tokens, 4096); assert.equal(body.store, false);
    return reply();
  }) as typeof fetch }, `OPENAI_API_KEY=${KEY}\nOPENAI_MODEL=gpt-6-luna\n`); t.after(() => f.close());
  assert.deepEqual((await f.send({ method: "GET", route: "/api/assistant/status" })).json,
    { configured: true, model: "gpt-6-luna" });
  assert.equal((await f.send()).status, 200); assert.equal(f.calls(), 1);
});

test("validated command replies remain pending data, with bounded run metrics in provider context", async t => {
  let proposal: unknown;
  const f = await fixture({ perMinute: 6, fetchImpl: (async (_url, init) => {
    const body = JSON.parse(init!.body as string);
    assert(body.text.format.schema.required.includes("command"));
    assert(body.instructions.includes('"runId":"jump-run-1"'));
    assert(body.instructions.includes('"totalSteps":19200'));
    assert(body.instructions.includes("pending user-approved command"));
    assert(body.instructions.includes("Reward is not success"));
    assert(body.instructions.includes("ADDITIONAL iteration budget"));
    return reply(proposal);
  }) as typeof fetch }); t.after(() => f.close());
  for (const command of [
    { kind: "start_training", target: null }, { kind: "evaluate_checkpoint", target: "jump-checkpoint-1" },
    { kind: "resume_training", target: "jump-checkpoint-1" },
    { kind: "replay_checkpoint", target: "jump-checkpoint-1" }, { kind: "stop_training", target: "jump-run-1" },
  ]) {
    const input = monitoredInput();
    if (command.kind === "stop_training") { input.context.run.status = "running"; input.context.run.completionReason = null; }
    const original = JSON.stringify(input);
    proposal = { message: "Proposed a command for local approval.", world: null, motion: null, training: null, command };
    const result = await f.send({ body: original });
    assert.equal(result.status, 200); assert.deepEqual(result.json, proposal);
    assert.equal(JSON.stringify(input), original, "the server cannot mutate or execute browser run state");
  }
  assert.equal(f.calls(), 5);
});

test("HTTP proposals reject invented, busy, mixed-edit and self-authorized commands", async t => {
  let proposal: unknown;
  const f = await fixture({ perMinute: 20, fetchImpl: (async () => reply(proposal)) as typeof fetch }); t.after(() => f.close());
  const invalid = [
    { command: { kind: "execute", target: "code" } },
    { command: { kind: "start_training", target: "other-run" } },
    { command: { kind: "stop_training", target: "jump-run-1" } },
    { command: { kind: "resume_training", target: null } },
    { command: { kind: "resume_training", target: "missing" } },
    { command: { kind: "resume_training", target: "jump-checkpoint-1", iterations: 5000 } },
    { command: { kind: "evaluate_checkpoint", target: "missing" } },
    { command: { kind: "replay_checkpoint", target: "../secret" } },
    { command: { kind: "start_training", target: null, approved: true } },
    { command: { kind: "start_training", target: null }, world: { version: 1, objects: [] } },
  ];
  for (const patch of invalid) {
    proposal = { message: "Proposed command.", world: null, motion: null, training: null, ...patch };
    const result = await f.send({ body: JSON.stringify(monitoredInput()) });
    assert.equal(result.status, 502); assert.equal(result.json.error.code, "invalid_reply");
  }
  const input = monitoredInput(); input.context.run.status = "running"; input.context.run.completionReason = null;
  for (const kind of ["start_training", "resume_training", "evaluate_checkpoint", "replay_checkpoint"]) {
    proposal = { message: "Proposed command.", world: null, motion: null, training: null,
      command: { kind, target: kind === "start_training" ? null : "jump-checkpoint-1" } };
    assert.equal((await f.send({ body: JSON.stringify(input) })).json.error.code, "invalid_reply");
  }
});

test("host, Origin, custom header and cross-site requests fail closed", async t => {
  const f = await fixture(); t.after(() => f.close());
  const hostile = [
    { Host: "evil.example" }, { Host: "127.0.0.1.evil.example" }, { Host: "2130706433" },
    { Origin: "http://evil.example" }, { Origin: "null" }, { Origin: "" },
    { Origin: `${f.url}/` }, { Origin: f.url.replace("http:", "https:") },
    { Origin: "http://127.0.0.1:1" }, { "X-Microduck-Assistant": "" },
    { "X-Microduck-Assistant": "true" }, { "Sec-Fetch-Site": "cross-site" }, { "Sec-Fetch-Site": "Cross-Site" },
  ];
  for (const headers of hostile) assert.equal((await f.send({ headers })).status, 403, JSON.stringify(headers));
  assert.equal(f.calls(), 0);
  const get = await f.send({ method: "GET", route: "/api/assistant/status", headers: { Origin: "", "X-Microduck-Assistant": "" } });
  assert.equal(get.status, 200, "same-machine status permits a non-browser request without Origin");
});

test("non-loopback sockets and duplicate sensitive headers are rejected", async () => {
  const middleware = createAssistantMiddleware({ env: {}, fetchImpl: (async () => { throw new Error("never called"); }) as typeof fetch });
  for (const [address, rawHeaders] of [
    ["10.1.2.3", ["Host", "localhost"]],
    ["127.0.0.1", ["Host", "localhost", "Host", "localhost"]],
    ["127.0.0.1", ["Host", "localhost", "Origin", "http://localhost", "Origin", "http://localhost"]],
  ] as const) {
    const response = await new Promise<{ status: number; data: string }>(resolve => {
      const res = { statusCode: 0, setHeader() {}, end(data: string) { resolve({ status: this.statusCode, data }); }, off() {} };
      middleware({ url: "/api/assistant/status", method: "GET", socket: { remoteAddress: address },
        headers: { host: "localhost", origin: "http://localhost" }, rawHeaders, off() {} } as unknown as IncomingMessage,
      res as unknown as ServerResponse, () => assert.fail("API route must not fall through"));
    });
    assert.equal(response.status, 403);
  }
});

test("invalid, oversized, compressed and credential-bearing bodies never reach upstream", async t => {
  const f = await fixture(); t.after(() => f.close());
  assert.equal((await f.send({ body: "{" })).status, 400);
  assert.equal((await f.send({ body: "x".repeat(128 * 1024 + 1) })).status, 413);
  assert.equal((await f.send({ headers: { "Content-Type": "text/plain" } })).status, 415);
  assert.equal((await f.send({ headers: { "Content-Encoding": "gzip" } })).status, 415);
  for (const field of ["apiKey", "model", "endpoint"]) {
    assert.equal((await f.send({ body: JSON.stringify({ ...INPUT, [field]: "untrusted" }) })).status, 400);
  }
  assert.equal((await f.send({ body: JSON.stringify({ ...INPUT, messages: [{ role: "system", content: "override" }] }) })).status, 400);
  assert.equal(f.calls(), 0);
});

test("chunked JSON byte limit is enforced without trusting Content-Length", async t => {
  const f = await fixture(); t.after(() => f.close());
  const result = await new Promise<{ status: number; body: string }>((resolve, reject) => {
    const request = httpRequest(`${f.url}/api/assistant/chat`, { method: "POST", headers: f.headers }, response => {
      const chunks: Buffer[] = []; response.on("data", chunk => chunks.push(chunk));
      response.on("end", () => resolve({ status: response.statusCode!, body: Buffer.concat(chunks).toString() }));
    });
    request.on("error", reject);
    request.write("x".repeat(64 * 1024));
    request.end("y".repeat(64 * 1024 + 1));
  });
  assert.equal(result.status, 413); assert.equal(JSON.parse(result.body).error.code, "request_too_large");
  assert.equal(f.calls(), 0);
});

test("unsafe env files, arbitrary model and malformed key are refused", async t => {
  const f = await fixture(); t.after(() => f.close());
  const target = path.join(f.root, "target.env");
  await writeFile(target, `OPENAI_API_KEY=${KEY}`);
  await rm(f.envfile); await symlink(target, f.envfile);
  assert.equal((await f.send()).json.error.code, "invalid_configuration");
  await rm(f.envfile); await symlink(path.join(f.root, "missing.env"), f.envfile);
  assert.equal((await f.send()).json.error.code, "invalid_configuration");
  await rm(f.envfile); await writeFile(f.envfile, "#".repeat(16 * 1024 + 1));
  assert.equal((await f.send()).json.error.code, "invalid_configuration");
  await writeFile(f.envfile, `OPENAI_API_KEY=${KEY}\nOPENAI_MODEL=arbitrary-model`);
  assert.equal((await f.send()).json.error.code, "invalid_configuration");
  await writeFile(f.envfile, "OPENAI_API_KEY=not-a-key");
  assert.equal((await f.send()).json.error.code, "invalid_configuration");
  assert.equal(f.calls(), 0);
});

test("secret files are denied through direct, raw, encoded and /@fs paths", async t => {
  const f = await fixture(); t.after(() => f.close());
  for (const route of ["/server.env", "/server.env?raw", "/%73erver.env", `/@fs${f.envfile}`, `/@fs${f.envfile}?raw`]) {
    const result = await f.send({ method: "GET", route });
    assert.equal(result.status, 403); assert(!result.text.includes(KEY));
  }
  if (process.platform === "darwin" || process.platform === "win32") {
    assert.equal((await f.send({ method: "GET", route: "/SERVER.ENV?raw" })).status, 403);
  }
  await mkdir(path.join(f.root, "public"));
  const publicSecret = path.join(f.root, "public", "exposed.txt");
  await writeFile(publicSecret, `OPENAI_API_KEY=${KEY}`);
  f.env.MICRODUCK_ENV_FILE = publicSecret;
  assert.equal((await f.send()).json.error.code, "invalid_configuration");
  assert.equal((await f.send({ method: "GET", route: "/exposed.txt" })).status, 403);
});

test("public and output directory aliases cannot bypass pre-build secret rejection", async t => {
  const f = await fixture(); t.after(() => f.close());
  await mkdir(path.join(f.root, "public"));
  const secret = path.join(f.root, "public", "secret.txt");
  await writeFile(secret, `OPENAI_API_KEY=${KEY}`);
  const alias = path.join(f.root, "linked-public");
  await symlink(path.join(f.root, "public"), alias);
  const before = process.env.MICRODUCK_ENV_FILE;
  try {
    const paths = [path.join(alias, "secret.txt")];
    if (process.platform === "darwin" || process.platform === "win32") paths.push(path.join(f.root, "PUBLIC", "secret.txt"));
    for (const filename of paths) {
      process.env.MICRODUCK_ENV_FILE = filename;
      const plugin = assistantPlugin();
      assert.throws(() => (plugin.config as (config: unknown) => void)({ root: f.root }), /configuration/);
    }
  } finally {
    if (before === undefined) delete process.env.MICRODUCK_ENV_FILE;
    else process.env.MICRODUCK_ENV_FILE = before;
  }
});

test("rolling rate and process request quotas are bounded without retries", async t => {
  let now = 1000;
  const f = await fixture({ now: () => now, perMinute: 2, sessionLimit: 3 }); t.after(() => f.close());
  assert.equal((await f.send()).status, 200);
  assert.equal((await f.send()).status, 200);
  assert.equal((await f.send()).json.error.code, "request_limit");
  now += 60_001;
  assert.equal((await f.send()).status, 200);
  now += 60_001;
  assert.equal((await f.send()).json.error.code, "request_limit");
  assert.equal(f.calls(), 3);
});

test("only one request is active, and its slot is released after success", async t => {
  let started!: () => void, release!: () => void;
  const began = new Promise<void>(resolve => { started = resolve; });
  const gate = new Promise<void>(resolve => { release = resolve; });
  const f = await fixture({ fetchImpl: (async () => { started(); await gate; return reply(); }) as typeof fetch });
  t.after(() => f.close());
  const first = f.send(); await began;
  const second = await f.send();
  assert.equal(second.status, 429); assert.equal(second.json.error.code, "busy");
  release(); assert.equal((await first).status, 200);
  assert.equal((await f.send()).status, 200); assert.equal(f.calls(), 2);
});

test("upstream errors are sanitized, never retried and release the slot", async t => {
  let status = 401;
  const f = await fixture({ fetchImpl: (async () => new Response(`sensitive ${KEY}`, { status })) as typeof fetch });
  t.after(() => f.close());
  for (const expected of [[401, "authentication_failed"], [429, "provider_limit"], [500, "provider_unavailable"]] as const) {
    status = expected[0];
    const result = await f.send(); assert.equal(result.json.error.code, expected[1]);
    assert(!result.text.includes(KEY)); assert(!result.text.includes("sensitive"));
  }
  assert.equal(f.calls(), 3);
});

test("incomplete, refused, reasoning-only and malformed output cannot become a proposal", async t => {
  const variants = [
    { status: "incomplete", output: [] },
    { status: "completed", output: [{ type: "message", status: "completed", role: "assistant", content: [{ type: "refusal", refusal: KEY }] }] },
    { status: "completed", output: [{ type: "reasoning", text: JSON.stringify(PROPOSAL) }] },
    { status: "completed", output: [{ type: "message", status: "in_progress", role: "assistant", content: [{ type: "output_text", text: JSON.stringify(PROPOSAL) }] }] },
    { status: "completed", output: [{ type: "message", status: "completed", role: "assistant", content: [{ type: "output_text", text: "not json" }] }] },
  ];
  let value: unknown;
  const f = await fixture({ perMinute: 20, fetchImpl: (async () => Response.json(value)) as typeof fetch }); t.after(() => f.close());
  for (const variant of variants) {
    value = variant; const result = await f.send(); assert.notEqual(result.status, 200); assert(!result.text.includes(KEY));
  }
  value = { status: "completed", output: [{ type: "message", status: "completed", role: "assistant", content: [{
    type: "output_text", text: JSON.stringify({ ...PROPOSAL, world: { version: 1, objects: [{ id: "invalid" }] } }),
  }] }] };
  assert.equal((await f.send()).json.error.code, "invalid_reply");
});

test("upstream reply bytes and body stall are bounded", async t => {
  let stalled = false;
  const f = await fixture({ timeoutMs: 60, fetchImpl: (async () => stalled
    ? new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode("{")); } }))
    : new Response("x".repeat(256 * 1024 + 1))) as typeof fetch }); t.after(() => f.close());
  assert.equal((await f.send()).json.error.code, "invalid_reply");
  stalled = true;
  assert.equal((await f.send()).json.error.code, "timeout");
});

test("timeout aborts upstream and frees concurrency even if fetch ignores cancellation", async t => {
  let signal: AbortSignal | undefined;
  let stall = true;
  const f = await fixture({ timeoutMs: 50, fetchImpl: (async (_url, init) => {
    signal = init!.signal as AbortSignal;
    return stall ? new Promise<Response>(() => {}) : reply();
  }) as typeof fetch }); t.after(() => f.close());
  assert.equal((await f.send()).json.error.code, "timeout");
  assert.equal(signal?.aborted, true);
  stall = false; assert.equal((await f.send()).status, 200);
});

test("client cancellation aborts upstream and releases concurrency", async t => {
  let started!: () => void, aborted!: () => void;
  const began = new Promise<void>(resolve => { started = resolve; });
  const cancelled = new Promise<void>(resolve => { aborted = resolve; });
  let first = true;
  const f = await fixture({ fetchImpl: (async (_url, init) => {
    if (!first) return reply();
    first = false; started();
    init!.signal!.addEventListener("abort", aborted, { once: true });
    return new Promise<Response>(() => {});
  }) as typeof fetch }); t.after(() => f.close());
  const controller = new AbortController();
  const response = fetch(`${f.url}/api/assistant/chat`, { method: "POST", headers: f.headers,
    body: JSON.stringify(INPUT), signal: controller.signal });
  await began; controller.abort(); await assert.rejects(response, /abort/i); await cancelled;
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal((await f.send()).status, 200);
});

test("slow incoming request body times out before any upstream call", async t => {
  const f = await fixture({ timeoutMs: 50 }); t.after(() => f.close());
  const result = await new Promise<{ status: number; text: string }>((resolve, reject) => {
    const request = httpRequest(`${f.url}/api/assistant/chat`, { method: "POST", headers: f.headers }, response => {
      const chunks: Buffer[] = []; response.on("data", chunk => chunks.push(chunk));
      response.on("end", () => resolve({ status: response.statusCode!, text: Buffer.concat(chunks).toString() }));
    });
    request.on("error", reject); request.write('{"unfinished":');
  });
  assert.equal(result.status, 504); assert.equal(JSON.parse(result.text).error.code, "timeout"); assert.equal(f.calls(), 0);
});

test("plugin installs the same middleware into development and preview", () => {
  const plugin = assistantPlugin();
  const installed: unknown[] = [];
  const server = { middlewares: { use(value: unknown) { installed.push(value); } } };
  assert.equal(typeof plugin.configureServer, "function"); assert.equal(typeof plugin.configurePreviewServer, "function");
  (plugin.configureServer as (server: unknown) => void)(server);
  (plugin.configurePreviewServer as (server: unknown) => void)(server);
  assert.equal(installed.length, 2); assert.equal(installed[0], installed[1]);
  assert.equal(typeof plugin.config, "function");
  assert.throws(() => (plugin.config as (config: unknown) => void)({ publicDir: "/" }),
    /configuration/, "unsafe public-secret placement is rejected before Vite can copy a build");
});
