import { execFileSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";

const MAX_BYTES = 2 * 1024 * 1024;
const SUBMODULE_PIN = "5bbe9637294d0c794edb185e284cfb1a77c6a0b4";
const SUBMODULE_URL = "https://github.com/pollen-robotics/microduck_rl";
const WASM = "src/train/kernels/kernels.wasm";
const REQUIRED = [
  ".env.example", ".gitignore", ".gitmodules", "README.md", "UPSTREAM.md", "NOTICE.md", "SECURITY.md",
  "package.json", "package-lock.json", "index.html", "vite.config.ts",
  "svelte.config.js", "tsconfig.json", "src/main.ts", "src/App.svelte",
  "server/assistant.ts", "scripts/prepare-assets.mjs", "src/sim/policy-catalog.ts",
  "src/train/kernels/kernels.c", WASM, "third_party/POLLEN-APACHE-2.0.txt",
  "scripts/check-release.mjs", "scripts/check-release.test.mjs", "microduck_rl",
];
const ROOT_FILES = new Set([
  ...REQUIRED.filter((name) => !name.includes("/")), "LICENSE", "LICENSE.md",
  "NOTICE", "NOTICE.md", "SECURITY.md", "CHANGELOG.md", "AGENTS.md",
]);
const SOURCE_ROOTS = new Set(["src", "server", "scripts", "tests", "docs", "plans", "third_party"]);
const PRIVATE_COMPONENTS = new Set([
  ".git", ".aws", ".ssh", ".config", ".cache", "node_modules", "dist", "out",
  "local", "checkpoints", "checkpoint", "recordings", "reports", "test-results",
  "playwright-report", "coverage", "secrets", "credentials",
]);
const INERT_KEY = ["sk", "test-only-not-a-real-credential"].join("-");
const TOKEN_PATTERNS = [
  /\bsk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])/g,
  /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g,
  /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g,
  /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g,
  /-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----/g,
];

function git(cwd, args, maxBuffer = 32 * 1024 * 1024) {
  try {
    return execFileSync("git", ["--no-replace-objects", ...args], { cwd, maxBuffer, stdio: ["ignore", "pipe", "pipe"] });
  } catch {
    throw new Error("Git could not inspect the release index.");
  }
}

function decode(buffer) {
  return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
}

function pathProblem(name) {
  const parts = name.split("/");
  if (!name || name.startsWith("/") || /[\x00-\x1f\x7f\\]/.test(name) ||
      parts.some((part) => !part || part === "." || part === ".." || /[. ]$/.test(part))) {
    return "unsafe or nonportable path";
  }
  const lower = parts.map((part) => part.toLowerCase());
  const base = lower.at(-1);
  if (lower.some((part) => PRIVATE_COMPONENTS.has(part)) ||
      /^(?:\.env(?:\..*)?|.*\.env|\.npmrc|\.netrc|\.pypirc|\.ds_store|id_rsa|id_ed25519|credentials\.json|secrets\.json)$/.test(base) && name !== ".env.example" ||
      /\.(?:pem|key|p12|pfx|log|ckpt|pt|pth|safetensors|onnx|bin|zip|tgz|gz|mp4|webm)$/.test(base) ||
      /(?:^|[-_.])checkpoint(?:[-_.]|$)/.test(base) && !/\.(?:ts|js|mjs|md)$/.test(base)) {
    return "secret, generated, checkpoint, or private artifact path";
  }
  if (parts.length === 1 ? !ROOT_FILES.has(name) : !SOURCE_ROOTS.has(parts[0])) {
    return "path is outside the source release allowlist";
  }
  return null;
}

function hasCredential(text, name) {
  return TOKEN_PATTERNS.some((pattern) => {
    pattern.lastIndex = 0;
    for (const match of text.matchAll(pattern)) {
      if (name !== "server/assistant.test.ts" || match[0] !== INERT_KEY) return true;
    }
    return false;
  });
}

function redactedPath(name) {
  for (const pattern of TOKEN_PATTERNS) {
    pattern.lastIndex = 0;
    name = name.replace(pattern, "[REDACTED]");
  }
  return name;
}

function validEnvExample(text) {
  const seen = new Set();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const match = /^(OPENAI_API_KEY|OPENAI_MODEL)\s*=\s*(.*?)\s*$/.exec(line);
    if (!match || seen.has(match[1])) return false;
    seen.add(match[1]);
    if (match[1] === "OPENAI_API_KEY" && !["", "\"\"", "''"].includes(match[2])) return false;
    if (match[1] === "OPENAI_MODEL" && !["gpt-5.4-mini", "gpt-6-luna"].includes(match[2])) return false;
  }
  return seen.has("OPENAI_API_KEY");
}

// Inspect object IDs from the index, never the working copy or local environment files.
export function checkReleaseIndex(cwd = process.cwd()) {
  const problems = [];
  const entries = new Map();
  const foldedNames = new Set();
  const contents = new Map();
  const add = (name, reason) => problems.push({ path: redactedPath(name), reason });
  const listing = decode(git(cwd, ["ls-files", "--stage", "-z"]));
  for (const record of listing.split("\0").filter(Boolean)) {
    const match = /^([0-7]{6}) ([a-f0-9]{40}|[a-f0-9]{64}) ([0-3])\t([\s\S]+)$/.exec(record);
    if (!match) throw new Error("Malformed release index entry.");
    const [, mode, oid, stage, name] = match;
    if (stage !== "0") { add(name, "unresolved index conflict"); continue; }
    if (hasCredential(name, "")) { add(name, "possible credential in staged path"); continue; }
    const folded = name.normalize("NFC").toLowerCase();
    if (foldedNames.has(folded)) add(name, "case or Unicode-normalization path collision");
    foldedNames.add(folded);
    const invalidPath = pathProblem(name);
    if (invalidPath) { add(name, invalidPath); continue; }
    entries.set(name, { mode, oid });
    if (mode === "160000") {
      if (name !== "microduck_rl" || oid !== SUBMODULE_PIN) add(name, "unexpected or unpinned gitlink");
      continue;
    }
    if (name === "microduck_rl") { add(name, "expected the pinned submodule gitlink"); continue; }
    if (mode !== "100644" && mode !== "100755") { add(name, "symlink or unsupported index mode"); continue; }
    const size = Number(git(cwd, ["cat-file", "-s", oid], 1024).toString("ascii").trim());
    if (!Number.isSafeInteger(size) || size < 0 || size > MAX_BYTES) {
      add(name, "staged file exceeds the 2 MiB release limit"); continue;
    }
    const blob = git(cwd, ["show", "--no-ext-diff", "--no-textconv", oid], MAX_BYTES + 1024);
    if (blob.length !== size) { add(name, "staged blob size mismatch"); continue; }
    let text;
    if (name === WASM) {
      if (!blob.subarray(0, 8).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]))) add(name, "invalid kernel WASM header");
      text = blob.toString("latin1");
    } else {
      try { text = decode(blob); } catch { add(name, "unexpected binary source file"); continue; }
      if (text.includes("\0")) { add(name, "unexpected binary source file"); continue; }
    }
    if (hasCredential(text, name)) add(name, "possible credential or private key in staged content");
    contents.set(name, text);
  }
  for (const name of REQUIRED) if (!entries.has(name)) add(name, "required release file is not staged");
  if (contents.has(".env.example") && !validEnvExample(contents.get(".env.example"))) {
    add(".env.example", "template must contain a blank API key and only supported settings");
  }
  if (contents.has(".gitmodules")) {
    let valid = false;
    try {
      const config = decode(git(cwd, ["config", "--no-includes", "--null", `--blob=${entries.get(".gitmodules").oid}`, "--list"]));
      const fields = config.split("\0").filter(Boolean).sort();
      valid = JSON.stringify(fields) === JSON.stringify([
        "submodule.microduck_rl.path\nmicroduck_rl",
        `submodule.microduck_rl.url\n${SUBMODULE_URL}`,
      ].sort());
    } catch { /* Invalid Git configuration is a release error, never a fallback. */ }
    if (!valid) add(".gitmodules", "expected only the pinned upstream submodule path and URL");
  }
  if (contents.has("package.json") && contents.has("package-lock.json")) {
    try {
      const pkg = JSON.parse(contents.get("package.json"));
      const lock = JSON.parse(contents.get("package-lock.json"));
      if (pkg.private !== true || lock.lockfileVersion !== 3 || !pkg.name || !pkg.version ||
          lock.name !== pkg.name || lock.version !== pkg.version ||
          lock.packages?.[""]?.name !== pkg.name || lock.packages?.[""]?.version !== pkg.version) {
        throw new Error("manifest mismatch");
      }
    } catch { add("package-lock.json", "invalid or mismatched private package manifest and v3 lockfile"); }
  }
  return { filesChecked: entries.size, problems };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const result = checkReleaseIndex();
    if (result.problems.length) {
      console.error("Release index refused:");
      for (const item of result.problems) console.error(`  ${JSON.stringify(item.path)}: ${item.reason}`);
      process.exitCode = 1;
    } else console.log(`Release index: ${result.filesChecked} staged paths passed (index only).`);
  } catch {
    console.error("Release index inspection failed; verify the Git index and retry.");
    process.exitCode = 1;
  }
}
