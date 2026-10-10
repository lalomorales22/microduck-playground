import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { type MotionFile } from "./format.ts";
import { deleteOwnedMotion, saveNewMotion } from "./motion-store.ts";

const motion = (): MotionFile => ({
  name: "Small nod", joints: ["head_pitch"], interp: "linear",
  keyframes: [{ t: 0, pose: [0.3] }, { t: 1, pose: [0.4] }],
});
const missing = () => new DOMException("Missing entry", "NotFoundError");
type Stage = "root" | "directory" | "lookup" | "create" | "writable" | "write" | "close" | "abort" | "remove";

function storage(t: TestContext, options: {
  files?: Record<string, string>;
  faults?: Partial<Record<Stage, Error>>;
  locks?: boolean;
} = {}) {
  const files = new Map(Object.entries(options.files ?? {}));
  const calls: string[] = [];
  const lockNames: string[] = [];
  const lockTails = new Map<string, Promise<unknown>>();
  function step(stage: Stage, name = ""): void {
    calls.push(`${stage}${name ? `:${name}` : ""}`);
    if (options.faults?.[stage]) throw options.faults[stage];
  }
  const directory = {
    async getFileHandle(name: string, flags?: { create?: boolean }) {
      step(flags?.create ? "create" : "lookup", name);
      if (!files.has(name)) {
        if (!flags?.create) throw missing();
        files.set(name, "");
      }
      return {
        async createWritable() {
          step("writable", name);
          let pending = files.get(name)!;
          return {
            async write(value: string) { step("write", name); pending = value; },
            async close() { step("close", name); files.set(name, pending); },
            async abort() { step("abort", name); },
          };
        },
      };
    },
    async removeEntry(name: string) {
      step("remove", name);
      if (!files.delete(name)) throw missing();
    },
  };
  const navigator = {
    storage: {
      async getDirectory() {
        step("root");
        return {
          async getDirectoryHandle(name: string, flags?: { create?: boolean }) {
            step("directory", `${name}:${flags?.create ? "create" : "existing"}`);
            return directory;
          },
        };
      },
    },
    locks: options.locks === false ? undefined : {
      request(name: string, callback: () => Promise<void>) {
        lockNames.push(name);
        const result = (lockTails.get(name) ?? Promise.resolve()).then(callback);
        lockTails.set(name, result.catch(() => {}));
        return result;
      },
    },
  };
  const original = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: navigator });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, "navigator", original);
    else Reflect.deleteProperty(globalThis, "navigator");
  });
  return { files, calls, lockNames };
}

test("exclusive save refuses a collision without changing the existing file", async (t) => {
  const state = storage(t, { files: { "existing.json": "original user bytes" } });
  await assert.rejects(saveNewMotion("existing", motion()), /already uses that filename/);
  assert.deepEqual([...state.files], [["existing.json", "original user bytes"]]);
  assert.deepEqual(state.lockNames, ["wicroduck.motion.existing"]);
  assert.deepEqual(state.calls, ["root", "directory:motions:create", "lookup:existing.json"]);
});

test("exclusive save writes the complete motion and closes the writer", async (t) => {
  const state = storage(t, { files: { "unrelated.json": "untouched" } });
  const file = motion();
  await saveNewMotion("ai-nod_123", file);
  assert.equal(state.files.get("ai-nod_123.json"), JSON.stringify(file));
  assert.equal(state.files.get("unrelated.json"), "untouched");
  assert.deepEqual(state.lockNames, ["wicroduck.motion.ai-nod_123"]);
  assert.deepEqual(state.calls, ["root", "directory:motions:create", "lookup:ai-nod_123.json",
    "create:ai-nod_123.json", "writable:ai-nod_123.json", "write:ai-nod_123.json", "close:ai-nod_123.json"]);
});

test("simultaneous saves under the same lock leave the first motion intact", async (t) => {
  const state = storage(t);
  const first = motion(), second = { ...motion(), name: "Different motion" };
  const results = await Promise.allSettled([saveNewMotion("same", first), saveNewMotion("same", second)]);
  assert.equal(results[0].status, "fulfilled");
  assert.equal(results[1].status, "rejected");
  assert.equal(state.files.get("same.json"), JSON.stringify(first));
  assert.equal(state.calls.filter((entry) => entry.startsWith("write:")).length, 1);
});

test("writer creation, write and close failures remove only the newly created file", async (t) => {
  for (const stage of ["writable", "write", "close"] as const) {
    await t.test(stage, async (sub) => {
      const failure = new DOMException(`${stage} failed`, "QuotaExceededError");
      const state = storage(sub, { files: { "user.json": "preserve" }, faults: { [stage]: failure } });
      await assert.rejects(saveNewMotion("new", motion()), (error) => error === failure);
      assert.deepEqual([...state.files], [["user.json", "preserve"]]);
      assert.equal(state.calls.at(-1), "remove:new.json");
      assert.equal(state.calls.includes("abort:new.json"), stage !== "writable");
    });
  }
});

test("an abort failure does not suppress the original write error or cleanup", async (t) => {
  const failure = new Error("write failed");
  const state = storage(t, { faults: { write: failure, abort: new Error("abort failed") } });
  await assert.rejects(saveNewMotion("new", motion()), (error) => error === failure);
  assert.equal(state.files.size, 0);
  assert.deepEqual(state.calls.slice(-2), ["abort:new.json", "remove:new.json"]);
});

test("non-missing lookup failures never create or delete a file", async (t) => {
  const failure = new DOMException("Access denied", "NotAllowedError");
  const state = storage(t, { files: { "new.json": "preserve" }, faults: { lookup: failure } });
  await assert.rejects(saveNewMotion("new", motion()), (error) => error === failure);
  assert.deepEqual([...state.files], [["new.json", "preserve"]]);
  assert.deepEqual(state.calls, ["root", "directory:motions:create", "lookup:new.json"]);
});

test("exclusive save refuses unavailable locks and invalid motion before storage access", async (t) => {
  const state = storage(t, { locks: false });
  await assert.rejects(saveNewMotion("new", motion()), /Exclusive motion saving is unavailable/);
  await assert.rejects(saveNewMotion("new", { ...motion(), keyframes: [] }));
  assert.deepEqual(state.calls, []);
});

test("strict rollback deletion removes only the named file without creating a directory", async (t) => {
  const state = storage(t, { files: { "owned.json": "new", "user.json": "preserve" } });
  await deleteOwnedMotion("owned");
  assert.deepEqual([...state.files], [["user.json", "preserve"]]);
  assert.deepEqual(state.calls, ["root", "directory:motions:existing", "remove:owned.json"]);
});

test("strict rollback deletion tolerates missing root, directory or file", async (t) => {
  for (const stage of ["root", "directory", "remove"] as const) {
    await t.test(stage, async (sub) => {
      const state = storage(sub, { faults: { [stage]: missing() } });
      await deleteOwnedMotion("missing");
      assert.equal(state.files.size, 0);
    });
  }
  await t.test("absent file", async (sub) => {
    const state = storage(sub);
    await deleteOwnedMotion("missing");
    assert.equal(state.calls.at(-1), "remove:missing.json");
  });
});

test("strict rollback deletion propagates permission and unknown failures at each storage stage", async (t) => {
  for (const stage of ["root", "directory", "remove"] as const) {
    for (const name of ["NotAllowedError", "UnknownError"] as const) {
      await t.test(`${stage}: ${name}`, async (sub) => {
        const failure = new DOMException("Storage unavailable", name);
        const state = storage(sub, { files: { "owned.json": "still present" }, faults: { [stage]: failure } });
        await assert.rejects(deleteOwnedMotion("owned"), (error) => error === failure);
        assert.equal(state.files.get("owned.json"), "still present");
      });
    }
  }
  await t.test("non-DOM missing-like error", async (sub) => {
    const failure = Object.assign(new Error("Not a browser missing-file result"), { name: "NotFoundError" });
    storage(sub, { faults: { remove: failure } });
    await assert.rejects(deleteOwnedMotion("owned"), (error) => error === failure);
  });
});

test("unsafe filenames are rejected by save and strict delete before touching storage", async (t) => {
  const state = storage(t);
  for (const name of ["", ".", "..", "../user", "sub/user", "sub\\user", "/absolute", "name.json",
    "with space", "builtin:nod", "name\0suffix", "a".repeat(65)]) {
    await assert.rejects(saveNewMotion(name, motion()), /Invalid new motion filename/);
    await assert.rejects(deleteOwnedMotion(name), /Invalid new motion filename/);
  }
  assert.deepEqual(state.calls, []);
  assert.deepEqual(state.lockNames, []);
});
