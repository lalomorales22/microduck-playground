import { constants, realpathSync } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as util from "node:util";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";
import { buildAssistantInstructions, PROPOSAL_SCHEMA, validateAssistantRequest,
  validateProposal, type AssistantContext, type AssistantProposal } from "../src/assistant/contract.ts";

const DEFAULT_MODEL = "gpt-5.4-mini";
const MODELS = new Set([DEFAULT_MODEL, "gpt-6-luna"]);
const ENDPOINT = "https://api.openai.com/v1/responses";
const APP_ROOT = fileURLToPath(new URL("../", import.meta.url));
const ENV_LIMIT = 16 * 1024;
const REQUEST_LIMIT = 128 * 1024;
const REPLY_LIMIT = 256 * 1024;
const OUTPUT_TOKENS = 4096;
const DEFAULT_TIMEOUT_MS = 45_000;
const parseEnv = (util as unknown as { parseEnv(input: string): Record<string, string> }).parseEnv;

type Next = (error?: unknown) => void;
type Middleware = (request: IncomingMessage, response: ServerResponse, next: Next) => void;
interface AssistantOptions {
  appRoot?: string;
  env?: NodeJS.ProcessEnv;
  fetchImpl?: typeof fetch;
  now?: () => number;
  timeoutMs?: number;
  perMinute?: number;
  sessionLimit?: number;
}
class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) { super(message); this.status = status; this.code = code; }
}
const denied = () => new ApiError(403, "local_only", "The assistant is available only from this local application.");
const configurationError = () => new ApiError(503, "invalid_configuration", "The local assistant configuration is invalid or cannot be read safely.");
const missingKey = () => new ApiError(503, "not_configured", "Add OPENAI_API_KEY to the local environment file before using the assistant.");
const invalidReply = () => new ApiError(502, "invalid_reply", "The assistant returned an invalid proposal. Nothing was applied.");

function envPath(root: string, env: NodeJS.ProcessEnv): string {
  const explicit = env.MICRODUCK_ENV_FILE;
  if (explicit && !path.isAbsolute(explicit)) throw configurationError();
  return explicit || path.join(root, ".env");
}
function within(filename: string, directory: string): boolean {
  if (process.platform === "darwin" || process.platform === "win32") {
    filename = filename.toLowerCase(); directory = directory.toLowerCase();
  }
  const relative = path.relative(directory, filename);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}
function canonicalPath(filename: string): string {
  try { return realpathSync(filename); }
  catch {
    try { return path.join(realpathSync(path.dirname(filename)), path.basename(filename)); }
    catch { return filename; }
  }
}

async function localConfiguration(root: string, env: NodeJS.ProcessEnv): Promise<{ key: string; model: string }> {
  const filename = path.resolve(envPath(root, env));
  // Vite copies public files into builds; fs.deny cannot make such a location safe.
  if (within(filename, path.join(root, "public")) || within(filename, path.join(root, "dist"))) throw configurationError();
  let parsed: Record<string, string> = {};
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    const entry = await lstat(filename);
    if (!entry.isFile() || entry.isSymbolicLink()) throw configurationError();
    if (await realpath(filename) !== filename) throw configurationError();
    handle = await open(filename, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size > ENV_LIMIT) throw configurationError();
    const bytes = Buffer.alloc(ENV_LIMIT + 1);
    let size = 0;
    while (size < bytes.length) {
      const result = await handle.read(bytes, size, bytes.length - size, size);
      if (result.bytesRead === 0) break;
      size += result.bytesRead;
    }
    if (size > ENV_LIMIT) throw configurationError();
    parsed = parseEnv(new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(0, size)));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT" || env.MICRODUCK_ENV_FILE) throw configurationError();
  } finally {
    await handle?.close();
  }
  const key = (parsed.OPENAI_API_KEY || env.OPENAI_API_KEY || "").trim();
  const model = (parsed.OPENAI_MODEL || env.OPENAI_MODEL || DEFAULT_MODEL).trim();
  if (!MODELS.has(model)) throw new ApiError(503, "invalid_configuration", "OPENAI_MODEL must be gpt-5.4-mini or gpt-6-luna.");
  if (key && !/^sk-[A-Za-z0-9_-]{16,509}$/.test(key))
    throw new ApiError(503, "invalid_configuration", "OPENAI_API_KEY is not a supported key format. Update the local configuration.");
  return { key, model };
}

function header(request: IncomingMessage, name: string): string | undefined {
  let count = 0;
  for (let i = 0; i < request.rawHeaders.length; i += 2) if (request.rawHeaders[i].toLowerCase() === name) count++;
  const value = request.headers[name];
  if (count > 1 || Array.isArray(value)) throw denied();
  return value;
}
function loopback(address: string | undefined): boolean {
  if (!address) return false;
  if (address === "::1") return true;
  const ipv4 = address.startsWith("::ffff:") ? address.slice(7) : address;
  return /^127(?:\.\d{1,3}){3}$/.test(ipv4) && ipv4.split(".").every(part => Number(part) <= 255);
}
function requireLocal(request: IncomingMessage): void {
  if (!loopback(request.socket.remoteAddress)) throw denied();
  const host = header(request, "host");
  const match = host?.match(/^(localhost|127(?:\.\d{1,3}){3}|\[::1\])(?::([0-9]{1,5}))?$/i);
  if (!match || (match[1].startsWith("127.") && !loopback(match[1])) ||
      (match[2] && (Number(match[2]) < 1 || Number(match[2]) > 65535))) throw denied();
  const scheme = (request.socket as { encrypted?: boolean }).encrypted ? "https" : "http";
  const expected = new URL(`${scheme}://${host}`).origin;
  const origin = header(request, "origin");
  if (header(request, "sec-fetch-site")?.toLowerCase() === "cross-site") throw denied();
  if (origin) {
    let normalized: URL;
    try { normalized = new URL(origin); } catch { throw denied(); }
    if (normalized.origin !== expected || origin !== normalized.origin) throw denied();
  }
  if (request.method === "POST" && (!origin || header(request, "x-microduck-assistant") !== "1")) throw denied();
}

function json(response: ServerResponse, status: number, value: unknown): void {
  if (response.destroyed || response.writableEnded) return;
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  response.end(JSON.stringify(value));
}
function signalReason(signal: AbortSignal): ApiError {
  return signal.reason instanceof ApiError ? signal.reason : new ApiError(499, "cancelled", "The assistant request was cancelled.");
}
function abortable<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(signalReason(signal));
  return new Promise((resolve, reject) => {
    const abort = () => reject(signalReason(signal));
    signal.addEventListener("abort", abort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
  });
}
function readRequest(request: IncomingMessage, signal: AbortSignal): Promise<unknown> {
  const type = header(request, "content-type");
  if (!type || !/^application\/json(?:\s*;\s*charset=utf-8)?$/i.test(type) ||
      (header(request, "content-encoding") ?? "identity") !== "identity")
    throw new ApiError(415, "json_required", "Send an uncompressed JSON request.");
  const length = header(request, "content-length");
  if (length && (!/^\d+$/.test(length) || Number(length) > REQUEST_LIMIT))
    throw new ApiError(413, "request_too_large", "The assistant request is too large.");
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    const cleanup = () => {
      request.off("data", data); request.off("end", end); request.off("error", error);
      signal.removeEventListener("abort", abort);
    };
    const fail = (reason: unknown) => { cleanup(); request.pause(); reject(reason); };
    const data = (chunk: Buffer) => {
      size += chunk.length;
      if (size > REQUEST_LIMIT) { fail(new ApiError(413, "request_too_large", "The assistant request is too large.")); return; }
      chunks.push(chunk);
    };
    const end = () => {
      cleanup();
      try { resolve(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)))); }
      catch { reject(new ApiError(400, "invalid_request", "The assistant request must contain valid JSON.")); }
    };
    const error = () => fail(new ApiError(400, "invalid_request", "The assistant request was interrupted."));
    const abort = () => fail(signalReason(signal));
    request.on("data", data); request.on("end", end); request.on("error", error);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}
async function readReply(response: Response, signal: AbortSignal): Promise<unknown> {
  if (!response.body || Number(response.headers.get("content-length") ?? 0) > REPLY_LIMIT) throw invalidReply();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const result = await abortable(reader.read(), signal);
      if (result.done) break;
      size += result.value.length;
      if (size > REPLY_LIMIT) throw invalidReply();
      chunks.push(result.value);
    }
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)));
  } catch (error) {
    void reader.cancel().catch(() => {});
    if (signal.aborted) throw signalReason(signal);
    throw error instanceof ApiError ? error : invalidReply();
  } finally { reader.releaseLock(); }
}
function proposalFromReply(value: unknown, context: AssistantContext): AssistantProposal {
  if (typeof value !== "object" || value === null) throw invalidReply();
  const reply = value as Record<string, unknown>;
  if (reply.status === "incomplete") throw new ApiError(502, "incomplete_reply", "The assistant response was incomplete. Nothing was applied.");
  if (reply.status !== "completed" || !Array.isArray(reply.output)) throw invalidReply();
  const text: string[] = [];
  for (const raw of reply.output) {
    if (typeof raw !== "object" || raw === null) throw invalidReply();
    const item = raw as Record<string, unknown>;
    if (item.type !== "message") continue; // Reasoning is not a proposal or user-visible text.
    if (item.role !== "assistant" || item.status !== "completed" || !Array.isArray(item.content)) throw invalidReply();
    for (const rawPart of item.content) {
      if (typeof rawPart !== "object" || rawPart === null) throw invalidReply();
      const part = rawPart as Record<string, unknown>;
      if (part.type === "refusal") throw new ApiError(422, "refused", "The assistant declined this request. Nothing was applied.");
      if (part.type !== "output_text" || typeof part.text !== "string") throw invalidReply();
      text.push(part.text);
    }
  }
  if (text.length !== 1) throw invalidReply();
  try { return validateProposal(JSON.parse(text[0]), context); } catch { throw invalidReply(); }
}

/** Middleware options are server-owned test hooks, never accepted from an HTTP body. */
export function createAssistantMiddleware(options: AssistantOptions = {}): Middleware {
  const root = path.resolve(options.appRoot ?? APP_ROOT), env = options.env ?? process.env;
  const fetchImpl = options.fetchImpl ?? fetch, now = options.now ?? Date.now;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const perMinute = options.perMinute ?? 4, sessionLimit = options.sessionLimit ?? 30;
  let active = false, count = 0;
  let recent: number[] = [];
  return (request, response, next) => {
    let pathname: string;
    try { pathname = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname); }
    catch { json(response, 400, { error: { code: "invalid_path", message: "Invalid request path." } }); return; }
    // Cover dev /@fs routes and direct static/preview paths, including ?raw requests.
    try {
      const secret = path.resolve(envPath(root, env));
      const candidates = [path.resolve(root, `.${pathname}`), path.resolve(root, "public", `.${pathname}`),
        path.resolve(root, "dist", `.${pathname}`)];
      if (pathname.startsWith("/@fs/")) candidates.push(path.resolve(pathname.slice(4)));
      const identity = (filename: string) => process.platform === "darwin" || process.platform === "win32"
        ? filename.toLowerCase() : filename;
      const sameFile = (candidate: string) => identity(candidate) === identity(secret) ||
        identity(canonicalPath(candidate)) === identity(canonicalPath(secret));
      if (candidates.some(sameFile)) { json(response, 403, { error: { code: "forbidden", message: "This file is not served." } }); return; }
    } catch { /* Invalid configuration is reported by API routes, never exposed. */ }
    if (!pathname.startsWith("/api/assistant/")) { next(); return; }
    void (async () => {
      let controller: AbortController | undefined;
      let timer: ReturnType<typeof setTimeout> | undefined;
      let release = false;
      const disconnected = () => { if (!response.writableEnded) controller?.abort(); };
      try {
        requireLocal(request);
        if (pathname === "/api/assistant/status" && request.method === "GET") {
          let configured = false, model = DEFAULT_MODEL;
          try { const configuration = await localConfiguration(root, env); configured = !!configuration.key; model = configuration.model; }
          catch { /* Status never leaks configuration values. */ }
          json(response, 200, { configured, model }); return;
        }
        if (pathname !== "/api/assistant/chat") throw new ApiError(404, "not_found", "Assistant route not found.");
        if (request.method !== "POST") throw new ApiError(405, "method_not_allowed", "This endpoint requires POST.");
        if (active) throw new ApiError(429, "busy", "Another assistant request is already running.");
        active = true; release = true;
        controller = new AbortController();
        request.once("aborted", disconnected); response.once("close", disconnected);
        timer = setTimeout(() => controller!.abort(new ApiError(504, "timeout", "The assistant request timed out. Nothing was applied.")), timeoutMs);
        const raw = await readRequest(request, controller.signal);
        let input: ReturnType<typeof validateAssistantRequest>;
        try { input = validateAssistantRequest(raw); }
        catch { throw new ApiError(400, "invalid_request", "The assistant request does not match the supported playground format."); }
        const configuration = await localConfiguration(root, env);
        if (!configuration.key) throw missingKey();
        const time = now();
        recent = recent.filter(timestamp => time - timestamp < 60_000);
        if (recent.length >= perMinute || count >= sessionLimit)
          throw new ApiError(429, "request_limit", "The local assistant request limit has been reached. Wait a minute, or restart the server after its 30-request session limit.");
        recent.push(time); count++;
        const upstream = await abortable(fetchImpl(ENDPOINT, {
          method: "POST", redirect: "error", signal: controller.signal,
          headers: { "Authorization": `Bearer ${configuration.key}`, "Content-Type": "application/json" },
          body: JSON.stringify({ model: configuration.model, store: false, max_output_tokens: OUTPUT_TOKENS,
            ...(configuration.model === "gpt-6-luna" ? { reasoning: { effort: "low" } } : {}),
            instructions: buildAssistantInstructions(input.context), input: input.messages,
            text: { format: { type: "json_schema", name: "microduck_proposal", strict: true, schema: PROPOSAL_SCHEMA } } }),
        }), controller.signal);
        if (!upstream.ok) {
          void upstream.body?.cancel().catch(() => {});
          if (upstream.status === 401 || upstream.status === 403)
            throw new ApiError(502, "authentication_failed", "The AI provider rejected the configured credentials or access.");
          if (upstream.status === 429) throw new ApiError(429, "provider_limit", "The AI provider rate or account limit was reached.");
          throw new ApiError(502, "provider_unavailable", "The AI provider could not complete the request.");
        }
        const proposal = proposalFromReply(await readReply(upstream, controller.signal), input.context);
        if (controller.signal.aborted) throw signalReason(controller.signal);
        json(response, 200, proposal);
      } catch (error) {
        const failure = controller?.signal.aborted ? signalReason(controller.signal) :
          error instanceof ApiError ? error : new ApiError(502, "assistant_unavailable", "The assistant could not complete the request. Nothing was applied.");
        response.setHeader("Connection", "close");
        json(response, failure.status, { error: { code: failure.code, message: failure.message } });
      } finally {
        if (timer) clearTimeout(timer);
        request.off("aborted", disconnected); response.off("close", disconnected);
        if (release) active = false;
      }
    })();
  };
}

/** Server-only in both Vite development and preview; never bundled into the client. */
export function assistantPlugin(): Plugin {
  const root = APP_ROOT, env = process.env;
  const middleware = createAssistantMiddleware({ appRoot: root, env });
  return {
    name: "microduck-local-assistant",
    config(config) {
      const filename = path.resolve(envPath(root, env));
      const configuredRoot = path.resolve(config.root ?? root);
      const publicDir = path.resolve(configuredRoot, config.publicDir || "public");
      const outDir = path.resolve(configuredRoot, config.build?.outDir ?? "dist");
      if ([filename, canonicalPath(filename)].some(secret => [publicDir, outDir].some(directory =>
        within(secret, directory) || within(secret, canonicalPath(directory))))) throw configurationError();
      return { server: { fs: { deny: [...(config.server?.fs?.deny ?? [".env", ".env.*", "*.{crt,pem}", "**/.git/**"]), envPath(root, env)] } } };
    },
    configureServer(server) { server.middlewares.use(middleware); },
    configurePreviewServer(server) { server.middlewares.use(middleware); },
  };
}
