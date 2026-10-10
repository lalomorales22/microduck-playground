import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { checkReleaseIndex } from "./check-release.mjs";

const PIN = "5bbe9637294d0c794edb185e284cfb1a77c6a0b4";
const CLI = fileURLToPath(new URL("./check-release.mjs", import.meta.url));
const key = () => ["sk", "proj", "a".repeat(48)].join("-");
const inertKey = () => ["sk", "test-only-not-a-real-credential"].join("-");

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "microduck-release-test-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const git = (...args) => execFileSync("git", args, { cwd: dir, stdio: ["ignore", "pipe", "pipe"] });
  git("init", "--quiet");
  const write = (name, content, stage = true) => {
    fs.mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
    fs.writeFileSync(path.join(dir, name), content);
    if (stage) git("add", "--", name);
  };
  for (const name of [
    ".gitignore", "README.md", "UPSTREAM.md", "NOTICE.md", "SECURITY.md", "index.html", "vite.config.ts", "svelte.config.js",
    "tsconfig.json", "src/main.ts", "src/App.svelte", "server/assistant.ts",
    "scripts/prepare-assets.mjs", "src/sim/policy-catalog.ts", "src/train/kernels/kernels.c",
    "third_party/POLLEN-APACHE-2.0.txt", "scripts/check-release.mjs", "scripts/check-release.test.mjs",
  ]) write(name, "source fixture\n");
  write(".env.example", "# No credential\nOPENAI_API_KEY=\nOPENAI_MODEL=gpt-5.4-mini\n");
  write(".gitmodules", '[submodule "microduck_rl"]\npath = microduck_rl\nurl = https://github.com/pollen-robotics/microduck_rl\n');
  write("package.json", JSON.stringify({ name: "fixture", version: "0.1.0", private: true }));
  write("package-lock.json", JSON.stringify({ name: "fixture", version: "0.1.0", lockfileVersion: 3, packages: { "": { name: "fixture", version: "0.1.0" } } }));
  write("src/train/kernels/kernels.wasm", Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  git("update-index", "--add", "--cacheinfo", `160000,${PIN},microduck_rl`);
  return { dir, git, write, check: () => checkReleaseIndex(dir) };
}

const refused = (result, name, reason) => assert.ok(result.problems.some((item) => item.path === name && item.reason.includes(reason)), JSON.stringify(result.problems));

test("complete source index, pinned gitlink, kernel binary and exact inert fixture pass", (t) => {
  const f = fixture(t);
  f.write("server/assistant.test.ts", `const KEY = ${JSON.stringify(inertKey())};`);
  f.write("src/train/checkpoint-store.ts", "// source, not checkpoint data\n");
  assert.deepEqual(f.check().problems, []);
});

test("index-only inspection ignores untracked secrets and unstaged source changes", (t) => {
  const f = fixture(t);
  f.write(".env", `OPENAI_API_KEY=${key()}`, false);
  f.write("server/assistant.ts", key(), false);
  assert.deepEqual(f.check().problems, []);
});

test("staged credential is refused even when its working copy is clean; CLI redacts value", (t) => {
  const f = fixture(t);
  f.write("src/main.ts", key());
  f.write("src/main.ts", "clean worktree", false);
  refused(f.check(), "src/main.ts", "credential");
  const run = spawnSync(process.execPath, [CLI], { cwd: f.dir, encoding: "utf8" });
  assert.equal(run.status, 1);
  assert.match(run.stderr, /src\/main\.ts/);
  assert.ok(!run.stderr.includes(key()));
});

test("Git replacement refs cannot hide the original staged credential", (t) => {
  const f = fixture(t);
  f.write("src/main.ts", key());
  const original = f.git("rev-parse", ":src/main.ts").toString().trim();
  const replacement = execFileSync("git", ["hash-object", "-w", "--stdin"], { cwd: f.dir, input: "clean replacement" }).toString().trim();
  f.git("replace", original, replacement);
  refused(f.check(), "src/main.ts", "credential");
});

test("inert-key exemption does not extend to other files or other keys", (t) => {
  const f = fixture(t);
  f.write("src/main.ts", inertKey());
  f.write("server/assistant.test.ts", `${inertKey()}\n${key()}`);
  refused(f.check(), "src/main.ts", "credential");
  refused(f.check(), "server/assistant.test.ts", "credential");
});

test("inert-key exemption requires the entire token, including trailing punctuation", (t) => {
  const f = fixture(t);
  f.write("server/assistant.test.ts", `${inertKey()}-`);
  refused(f.check(), "server/assistant.test.ts", "credential");
});

test("credential-shaped filenames are refused and redacted, including conflict diagnostics", (t) => {
  const f = fixture(t);
  f.write(`docs/${key()}.txt`, "ordinary text");
  const oid = f.git("rev-parse", ":src/main.ts").toString().trim();
  execFileSync("git", ["update-index", "--index-info"], { cwd: f.dir, input: `100644 ${oid} 1\tdocs/${key()}.conflict\n` });
  const result = f.check();
  refused(result, "docs/[REDACTED].txt", "credential");
  refused(result, "docs/[REDACTED].conflict", "conflict");
  assert.ok(!JSON.stringify(result).includes(key()));
});

test("common GitHub, AWS and private-key markers are refused without exposing matches", (t) => {
  const f = fixture(t);
  const values = ["ghp_" + "x".repeat(36), "github_pat_" + "x".repeat(60), "AKIA" + "A".repeat(16), ["-----BEGIN", "OPENSSH", "PRIVATE KEY-----"].join(" ")];
  for (const [i, value] of values.entries()) f.write(`docs/fixture${i}.txt`, value);
  const result = f.check();
  for (let i = 0; i < values.length; ++i) refused(result, `docs/fixture${i}.txt`, "credential");
  for (const value of values) assert.ok(!JSON.stringify(result).includes(value));
});

test("secret, generated, checkpoint, report and private paths are refused", (t) => {
  const f = fixture(t);
  const names = [".env", ".env.production", "local.env", "docs/.env.example", "docs/key.pem", "src/.ssh/config", "node_modules/pkg/index.js", "dist/index.html", "public/model/robot.xml", "checkpoints/run.json", "reports/result.json", "docs/export.checkpoint.json", "docs/model.onnx", "src/.npmrc", "docs/trace.log"];
  for (const name of names) f.write(name, "harmless staged fixture");
  const result = f.check();
  for (const name of names) assert.ok(result.problems.some((item) => item.path === name), name);
});

test("blank key template passes; populated or additional credential settings fail", (t) => {
  const f = fixture(t);
  for (const text of ["OPENAI_API_KEY=not-a-key\n", "OPENAI_API_KEY=\nTOKEN=example\n", "OPENAI_API_KEY=\nOPENAI_API_KEY=\n", "OPENAI_MODEL=unsupported\nOPENAI_API_KEY=\n"]) {
    f.write(".env.example", text);
    refused(f.check(), ".env.example", "template");
  }
  f.write(".env.example", "OPENAI_API_KEY=''\nOPENAI_MODEL=gpt-6-luna\n");
  assert.deepEqual(f.check().problems, []);
});

test("2 MiB boundary is enforced on staged bytes, not the working copy", (t) => {
  const f = fixture(t);
  f.write("docs/large.txt", "a".repeat(2 * 1024 * 1024));
  assert.deepEqual(f.check().problems, []);
  f.write("docs/large.txt", "a".repeat(2 * 1024 * 1024 + 1));
  f.write("docs/large.txt", "small", false);
  refused(f.check(), "docs/large.txt", "2 MiB");
});

test("symlinks, unexpected gitlinks and altered upstream pin fail", (t) => {
  const f = fixture(t);
  fs.mkdirSync(path.join(f.dir, "docs"));
  fs.symlinkSync("../src/main.ts", path.join(f.dir, "docs", "linked.ts"));
  f.git("add", "--", "docs/linked.ts");
  f.git("update-index", "--add", "--cacheinfo", `160000,${PIN},src/other`);
  f.git("update-index", "--cacheinfo", `160000,${"1".repeat(40)},microduck_rl`);
  const result = f.check();
  refused(result, "docs/linked.ts", "symlink");
  refused(result, "src/other", "gitlink");
  refused(result, "microduck_rl", "unpinned");
});

test("unresolved merge stages are rejected", (t) => {
  const f = fixture(t);
  const oid = f.git("rev-parse", ":src/main.ts").toString().trim();
  f.git("update-index", "--force-remove", "src/main.ts");
  execFileSync("git", ["update-index", "--index-info"], { cwd: f.dir, input: `100644 ${oid} 1\tsrc/main.ts\n100644 ${oid} 2\tsrc/main.ts\n` });
  refused(f.check(), "src/main.ts", "conflict");
});

test("nonportable index paths and case collisions fail", (t) => {
  const f = fixture(t);
  const oid = f.git("rev-parse", ":src/main.ts").toString().trim();
  for (const name of ["src/bad\nname.ts", "src/back\\slash.ts", "src/MAIN.ts"]) {
    f.git("update-index", "--add", "--cacheinfo", `100644,${oid},${name}`);
  }
  const result = f.check();
  refused(result, "src/bad\nname.ts", "path");
  refused(result, "src/back\\slash.ts", "path");
  assert.ok(result.problems.some((item) => item.reason.includes("collision")));
});

test("missing foundational source, mismatched manifest and changed submodule URL fail", (t) => {
  const f = fixture(t);
  f.git("update-index", "--force-remove", "src/train/kernels/kernels.c");
  f.write("package-lock.json", "{}");
  f.write(".gitmodules", '[submodule "microduck_rl"]\npath = microduck_rl\nurl = https://example.invalid/other\n');
  const result = f.check();
  refused(result, "src/train/kernels/kernels.c", "required");
  refused(result, "package-lock.json", "manifest");
  refused(result, ".gitmodules", "URL");
});

test("arbitrary binaries and fake kernel WASM fail", (t) => {
  const f = fixture(t);
  f.write("docs/image.dat", Buffer.from([255, 0, 254]));
  f.write("src/train/kernels/kernels.wasm", "not wasm");
  const result = f.check();
  refused(result, "docs/image.dat", "binary");
  refused(result, "src/train/kernels/kernels.wasm", "WASM");
});
