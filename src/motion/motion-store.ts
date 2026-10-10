// Motion files, stored alongside checkpoints in OPFS.
//
// The built-in library is merged into the listing rather than being copied
// into storage on first run: a built-in that is only ever read stays exactly
// what ships, and cannot rot into a stale copy of an older version of itself.
// Saving one under a new name is how you edit it.

import { listJson, readFile, readJson, removeJson, renameJson, writeJson } from "../opfs.ts";
import { BUILTIN_MOTIONS } from "./library.ts";
import { parseMotion, type Motion, type MotionFile } from "./format.ts";

const DIR = "motions";

export interface MotionEntry {
  /** Unique across both sources: built-ins are prefixed. */
  id: string;
  name: string;
  source: "builtin" | "saved";
  description: string;
  duration: number;
  loop: boolean;
  keyframes: number;
  /** Saved files only. */
  bytes?: number;
  modified?: number;
  /** Set when a stored file will not parse, so the UI can say so instead of
   *  silently dropping it. */
  error?: string;
}

export const BUILTIN_PREFIX = "builtin:";

function describe(id: string, source: MotionEntry["source"], motion: Motion): MotionEntry {
  return {
    id,
    name: motion.name,
    source,
    description: motion.description,
    duration: motion.duration,
    loop: motion.loop,
    keyframes: motion.keyframes.length,
  };
}

export function builtinEntries(): MotionEntry[] {
  return BUILTIN_MOTIONS.map((file) => {
    const motion = parseMotion(file);
    return describe(`${BUILTIN_PREFIX}${motion.name}`, "builtin", motion);
  });
}

/** Every motion the app can offer: the built-ins, then whatever is stored. */
export async function listMotions(): Promise<MotionEntry[]> {
  const stored = await listJson(DIR);
  const saved = await Promise.all(
    stored.map(async (f): Promise<MotionEntry> => {
      const raw = await readJson<MotionFile>(DIR, f.name);
      try {
        const motion = parseMotion(raw);
        return { ...describe(f.name, "saved", motion), bytes: f.bytes, modified: f.modified };
      } catch (err) {
        return {
          id: f.name, name: f.name, source: "saved", description: "", duration: 0,
          loop: false, keyframes: 0, bytes: f.bytes, modified: f.modified,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    }),
  );
  return [...builtinEntries(), ...saved];
}

/**
 * Load one by id, from either source.
 *
 * Returns the raw FILE rather than a parsed Motion: it has to cross a
 * postMessage boundary to reach the rollout workers, and a Motion holds
 * Float32Arrays and derived fields that would just be recomputed on the other
 * side anyway.
 */
export async function loadMotionFile(id: string): Promise<MotionFile | null> {
  if (id.startsWith(BUILTIN_PREFIX)) {
    const name = id.slice(BUILTIN_PREFIX.length);
    return BUILTIN_MOTIONS.find((m) => m.name === name) ?? null;
  }
  return readJson<MotionFile>(DIR, id);
}

/** Parsed, or a thrown MotionError naming what is wrong with the file. */
export async function loadMotion(id: string): Promise<Motion | null> {
  const file = await loadMotionFile(id);
  return file === null ? null : parseMotion(file);
}

export const saveMotion = (name: string, file: MotionFile): Promise<void> =>
  writeJson(DIR, name, file);

/** Assistant-created files must not replace a saved user motion. */
export async function saveNewMotion(name: string, file: MotionFile): Promise<void> {
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) throw new Error("Invalid new motion filename");
  parseMotion(file);
  if (!navigator.locks) throw new Error("Exclusive motion saving is unavailable in this browser");
  await navigator.locks.request(`wicroduck.motion.${name}`, async () => {
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle(DIR, { create: true });
    const filename = `${name}.json`;
    try {
      await directory.getFileHandle(filename);
      throw new Error("A saved motion already uses that filename");
    } catch (error) {
      if (!(error instanceof DOMException) || error.name !== "NotFoundError") throw error;
    }
    const handle = await directory.getFileHandle(filename, { create: true });
    try {
      const writable = await handle.createWritable();
      try { await writable.write(JSON.stringify(file)); await writable.close(); }
      catch (error) { await writable.abort().catch(() => {}); throw error; }
    } catch (error) {
      await directory.removeEntry(filename).catch(() => {});
      throw error;
    }
  });
}

export const deleteMotion = (name: string): Promise<void> => removeJson(DIR, name);

/** Rollback must report storage failures instead of silently leaving a new file. */
export async function deleteOwnedMotion(name: string): Promise<void> {
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) throw new Error("Invalid new motion filename");
  try {
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle(DIR);
    await directory.removeEntry(`${name}.json`);
  } catch (error) {
    if (!(error instanceof DOMException) || error.name !== "NotFoundError") throw error;
  }
}

export const renameMotion = (from: string, to: string): Promise<void> =>
  renameJson(DIR, from, to);

export const readMotionFile = (name: string): Promise<File> => readFile(DIR, name);

/**
 * Store a motion that came from outside — usually written by an agent against
 * docs/motion-format.md. Parsed before it is stored, so a bad file is rejected
 * at the point the author can still see why.
 */
export async function importMotion(name: string, text: string): Promise<Motion> {
  const motion = parseMotion(text);
  await saveMotion(name, JSON.parse(text) as MotionFile);
  return motion;
}
