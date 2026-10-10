// The Origin Private File System, as a small typed helper.
//
// OPFS rather than localStorage because the things stored here are megabytes,
// and rather than IndexedDB because the API is a plain file handle. Available
// on the main thread and in workers, which matters: the trainer runs in a
// worker and autosaves without a round trip.
//
// Two collections use it — trained runs and motion files — and they want the
// same six operations, so those live here once rather than being written twice
// with subtly different error behaviour.

export function opfsAvailable(): boolean {
  return typeof navigator !== "undefined" && typeof navigator.storage?.getDirectory === "function";
}

async function dir(collection: string): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(collection, { create: true });
}

/** Names become filenames, so keep them tame. */
export function sanitizeName(raw: string): string {
  return raw.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64);
}

export interface StoredFile {
  name: string;
  bytes: number;
  modified: number;
}

export async function writeJson(collection: string, name: string, data: unknown): Promise<void> {
  const handle = await (await dir(collection)).getFileHandle(`${name}.json`, { create: true });
  const writable = await handle.createWritable();
  try {
    await writable.write(JSON.stringify(data));
  } finally {
    await writable.close();
  }
}

/** Absent is a normal answer, not an error. */
export async function readJson<T>(collection: string, name: string): Promise<T | null> {
  try {
    const handle = await (await dir(collection)).getFileHandle(`${name}.json`);
    return JSON.parse(await (await handle.getFile()).text()) as T;
  } catch {
    return null;
  }
}

export async function exists(collection: string, name: string): Promise<boolean> {
  try {
    await (await dir(collection)).getFileHandle(`${name}.json`);
    return true;
  } catch {
    return false;
  }
}

/** The raw file, for download. */
export async function readFile(collection: string, name: string): Promise<File> {
  const handle = await (await dir(collection)).getFileHandle(`${name}.json`);
  return handle.getFile();
}

export async function listJson(collection: string): Promise<StoredFile[]> {
  const out: StoredFile[] = [];
  try {
    const d = await dir(collection) as FileSystemDirectoryHandle & {
      values(): AsyncIterableIterator<FileSystemHandle>;
    };
    for await (const entry of d.values()) {
      if (entry.kind !== "file" || !entry.name.endsWith(".json")) continue;
      const file = await (entry as FileSystemFileHandle).getFile();
      out.push({
        name: entry.name.replace(/\.json$/, ""),
        bytes: file.size,
        modified: file.lastModified,
      });
    }
  } catch {
    return [];
  }
  return out.sort((a, b) => b.modified - a.modified);
}

export async function removeJson(collection: string, name: string): Promise<void> {
  try {
    await (await dir(collection)).removeEntry(`${name}.json`);
  } catch {
    /* already gone */
  }
}

/**
 * Rename by copy-then-delete rather than FileSystemHandle.move(), which is
 * Chromium-only. These files are a few megabytes at most; the copy is cheap
 * and it works everywhere OPFS does.
 */
export async function renameJson(collection: string, from: string, to: string): Promise<void> {
  if (from === to) return;
  const data = await readJson<unknown>(collection, from);
  if (data === null) throw new Error(`no file named "${from}"`);
  await writeJson(collection, to, data);
  await removeJson(collection, from);
}
