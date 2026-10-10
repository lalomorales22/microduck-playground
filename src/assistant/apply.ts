import type { MotionFile } from "../motion/format.ts";
import type { WorldConfig } from "../sim/world.ts";
import {
  ASSISTANT_LIMITS,
  validateProposal,
  type AssistantContext,
  type TrainingDraft,
} from "./contract.ts";

export interface AssistantApplyPorts {
  getContext(): Promise<AssistantContext>;
  applyWorld(world: WorldConfig): Promise<boolean>;
  /** Must reject an existing ID without changing that saved motion. */
  saveMotion(id: string, file: MotionFile): Promise<void>;
  deleteMotion(id: string): Promise<void>;
  setMotion(file: MotionFile): void;
  /** Restores the caller's captured draft when it has no valid motion yet. */
  restoreEmptyMotion?(): void;
  setTraining(draft: TrainingDraft): void;
}

const NO_CHANGES = { message: "Validate context", world: null, motion: null, training: null };
const BUSY = new Set(["loading", "running", "stopping"]);

function draftOf(context: AssistantContext): TrainingDraft {
  const { task, iterations, seed, envs, rolloutWorkers, useCurrentWorld, hiddenPreset } = context.training;
  return { task, iterations, seed, envs, rolloutWorkers, useCurrentWorld, hiddenPreset };
}

function snapshot(context: AssistantContext): AssistantContext {
  validateProposal(NO_CHANGES, context);
  return structuredClone(context);
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function signature(context: AssistantContext, ownMotionId: string | null = null): string {
  return canonical({
    world: context.world,
    currentMotion: context.currentMotion,
    training: draftOf(context),
    motions: context.motions.filter((motion) => motion.id !== ownMotionId)
      .sort((a, b) => a.id.localeCompare(b.id)),
  });
}

function checkCurrent(
  current: AssistantContext,
  expected: string,
  ownMotion: { id: string; name: string } | null = null,
): void {
  if (BUSY.has(current.training.status)) {
    throw new Error(`Training is ${current.training.status}; stop it before applying a proposal`);
  }
  const saved = ownMotion && current.motions.find((motion) => motion.id === ownMotion.id);
  if (ownMotion && (!saved || saved.name !== ownMotion.name)) {
    throw new Error("The newly saved motion is missing or changed while applying the proposal");
  }
  if (signature(current, ownMotion?.id ?? null) !== expected) {
    throw new Error("The proposal is stale: the world, current motion, saved motions, or training draft changed");
  }
}

function motionId(name: string, current: AssistantContext): string {
  const safeName = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 52) || "motion";
  const id = `ai-${safeName}-${crypto.randomUUID().slice(0, 8)}`;
  if (current.motions.some((motion) => motion.id === id)) {
    throw new Error("The generated motion ID already exists; apply the proposal again to choose a new ID");
  }
  return id;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Applies validated data only. The caller serializes applies and locks edits. */
export async function applyAssistantProposal(
  raw: unknown,
  capturedContext: AssistantContext,
  ports: AssistantApplyPorts,
): Promise<{ message: string; motionId: string | null }> {
  const captured = snapshot(capturedContext);
  const proposal = validateProposal(raw, captured);
  const expected = signature(captured);
  const current = snapshot(await ports.getContext());
  checkCurrent(current, expected);
  validateProposal(proposal, current);

  const newMotion = proposal.motion;
  if (newMotion && current.motions.length >= ASSISTANT_LIMITS.motions) {
    throw new Error("The motion library is full; remove a motion before applying a generated motion");
  }
  const newId = newMotion ? motionId(newMotion.name!, current) : null;
  const worldChanged = proposal.world !== null && canonical(proposal.world) !== canonical(current.world);
  const nextTraining = proposal.training ? {
    ...proposal.training,
    task: proposal.training.task === "$generated_motion" ? `motion:${newId}` : proposal.training.task,
  } : null;
  const trainingChanged = nextTraining !== null && canonical(nextTraining) !== canonical(draftOf(current));
  let saved = false;
  let worldAttempted = false;
  let motionAttempted = false;
  let trainingAttempted = false;
  let step = "saving the generated motion";

  try {
    if (newMotion && newId) {
      // Ownership starts only after an exclusive save succeeds. A rejected
      // save can be a collision with somebody else's file, which we never delete.
      await ports.saveMotion(newId, structuredClone(newMotion));
      saved = true;
    }
    step = "checking the current state";
    const latest = snapshot(await ports.getContext());
    checkCurrent(latest, expected, newMotion && newId ? { id: newId, name: newMotion.name! } : null);
    validateProposal(proposal, latest);

    if (worldChanged) {
      step = "applying the world";
      worldAttempted = true;
      if (!await ports.applyWorld(structuredClone(proposal.world!))) {
        throw new Error("World application was refused");
      }
    }
    // All asynchronous prerequisites finish before changing either draft.
    if (newMotion) {
      step = "setting the current motion";
      motionAttempted = true;
      ports.setMotion(structuredClone(newMotion));
    }
    if (trainingChanged) {
      step = "configuring training";
      trainingAttempted = true;
      ports.setTraining(structuredClone(nextTraining!));
    }
  } catch (error) {
    const failures: string[] = [];
    if (trainingAttempted) {
      try { ports.setTraining(draftOf(current)); }
      catch (rollbackError) { failures.push(`training draft: ${errorText(rollbackError)}`); }
    }
    if (motionAttempted) {
      if (current.currentMotion) {
        try { ports.setMotion(structuredClone(current.currentMotion)); }
        catch (rollbackError) { failures.push(`current motion: ${errorText(rollbackError)}`); }
      } else if (ports.restoreEmptyMotion) {
        try { ports.restoreEmptyMotion(); }
        catch (rollbackError) { failures.push(`empty motion draft: ${errorText(rollbackError)}`); }
      } else {
        failures.push("current motion: the adapter cannot restore an empty motion selection");
      }
    }
    if (worldAttempted) {
      try {
        if (!await ports.applyWorld(structuredClone(current.world))) failures.push("world: restoration was refused");
      } catch (rollbackError) { failures.push(`world: ${errorText(rollbackError)}`); }
    }
    if (saved && newId) {
      try { await ports.deleteMotion(newId); }
      catch (rollbackError) { failures.push(`new motion cleanup: ${errorText(rollbackError)}`); }
    }
    const detail = failures.length ? ` Rollback incomplete: ${failures.join("; ")}.` : "";
    throw new Error(`Could not apply the proposal while ${step}: ${errorText(error)}.${detail}`);
  }

  const changes: string[] = [];
  if (worldChanged) changes.push("World updated.");
  if (newMotion) changes.push(`Motion saved in a new slot (${newId}) and selected.`);
  if (nextTraining) changes.push(trainingChanged ? "Training configured, not started." : "Training configuration unchanged; training was not started.");
  return { message: changes.join(" ") || "No changes requested.", motionId: newId };
}
