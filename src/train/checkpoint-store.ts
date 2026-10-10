// Checkpoint persistence in the Origin Private File System.
//
// A run that cannot survive a reload is a run you cannot leave overnight, so
// this is M2 work rather than a later nicety. The filesystem mechanics live in
// src/opfs.ts, which motion files share.

import {
  exists, listJson, opfsAvailable, readFile, readJson, removeJson, renameJson,
  sanitizeName, writeJson, type StoredFile,
} from "../opfs.ts";
import { checkpointMetadata, checkpointNetwork, validateCheckpointResume } from "./checkpoint-task.ts";

const DIR = "checkpoints";

export { opfsAvailable, sanitizeName };
export type CheckpointInfo = StoredFile;

export const saveCheckpoint = (name: string, data: unknown): Promise<void> =>
  writeJson(DIR, name, data);

export const loadCheckpoint = <T>(name: string): Promise<T | null> => readJson<T>(DIR, name);

export const checkpointExists = (name: string): Promise<boolean> => exists(DIR, name);

export const renameCheckpoint = (from: string, to: string): Promise<void> =>
  renameJson(DIR, from, to);

/** The raw file, for download. */
export const readCheckpointFile = (name: string): Promise<File> => readFile(DIR, name);

export const listCheckpoints = (): Promise<CheckpointInfo[]> => listJson(DIR);

export const deleteCheckpoint = (name: string): Promise<void> => removeJson(DIR, name);

/** Store a checkpoint that came from outside — a download from another
 *  machine. Validated enough to fail loudly rather than at training time. */
export async function importCheckpoint(name: string, text: string): Promise<void> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("not valid JSON");
  }
  checkpointMetadata(parsed);
  checkpointNetwork(parsed);
  validateCheckpointResume(parsed);
  await saveCheckpoint(name, parsed);
}
