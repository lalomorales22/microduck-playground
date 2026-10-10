import { validateProposal, type AssistantCommand, type AssistantContext } from "./contract.ts";

export interface AssistantCommandPorts {
  getContext(): Promise<AssistantContext>;
  startTraining(): Promise<string>;
  resumeTraining(name: string): Promise<string>;
  stopTraining(runId: string): string;
  evaluateCheckpoint(name: string): Promise<string>;
  replayCheckpoint(name: string): Promise<string>;
}

const draft = (context: AssistantContext) => {
  const { task, iterations, seed, envs, rolloutWorkers, useCurrentWorld, hiddenPreset } = context.training;
  return { task, iterations, seed, envs, rolloutWorkers, useCurrentWorld, hiddenPreset };
};
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const busy = (context: AssistantContext) => [context.training.status, context.run?.status]
  .some(status => status !== undefined && ["loading", "running", "stopping"].includes(status));

export function assertRunRequested(expectedRunId: string | null,
  run: { runId: string | null; status: string; error: string | null }): void {
  if (run.status === "error") throw new Error(run.error ?? "Training failed to start");
  if (!expectedRunId || run.runId !== expectedRunId || !["loading", "running"].includes(run.status))
    throw new Error("The requested run was canceled or replaced before startup completed.");
}

/** Commands have no rollback: validate both snapshots before dispatching one
 * bounded local operation, and let its lifecycle supply the receipt. */
export async function executeAssistantCommand(
  command: AssistantCommand, captured: AssistantContext, ports: AssistantCommandPorts,
): Promise<string> {
  const proposal = { message: "Pending command", world: null, motion: null, training: null, command };
  const checked = validateProposal(proposal, captured).command;
  if (!checked) throw new Error("No command requested");
  const current = await ports.getContext();
  validateProposal(proposal, current);
  if (checked.kind === "stop_training") {
    if (!checked.target || checked.target !== captured.run?.runId || checked.target !== current.run?.runId)
      throw new Error("The active run changed. Ask for a new stop command.");
    if (!busy(current) || current.run?.status === "stopping") throw new Error("This run is no longer accepting a stop request.");
    return ports.stopTraining(checked.target);
  }
  if (busy(current)) throw new Error("Wait for the current training run to stop before this command.");
  if (checked.kind === "start_training") {
    if (captured.run?.runId !== current.run?.runId || !same(draft(captured), draft(current)) ||
        (current.training.useCurrentWorld && !same(captured.world, current.world)))
      throw new Error("The training setup changed. Ask for a fresh start command with the current budget.");
    if (current.training.task.startsWith("motion:")) {
      const id = current.training.task.slice(7);
      const before = captured.motions.find(item => item.id === id), after = current.motions.find(item => item.id === id);
      if (!before || !after || before.name !== after.name || before.modified !== after.modified || before.bytes !== after.bytes ||
          (!id.startsWith("builtin:") && (before.modified === undefined || before.bytes === undefined)))
        throw new Error("The saved motion changed or has no revision. Ask for a fresh start command.");
    }
    return ports.startTraining();
  }
  const before = captured.checkpoints?.find(item => item.name === checked.target);
  const after = current.checkpoints?.find(item => item.name === checked.target);
  if (!checked.target || !before || !after || before.name !== after.name || before.iteration !== after.iteration ||
      before.modified !== after.modified || before.bytes !== after.bytes)
    throw new Error("The saved checkpoint changed or is missing. Ask for a fresh command.");
  if (checked.kind === "resume_training") {
    if (captured.run?.runId !== current.run?.runId || captured.training.iterations !== current.training.iterations ||
        captured.training.rolloutWorkers !== current.training.rolloutWorkers)
      throw new Error("The continuation budget changed. Ask for a fresh resume command.");
    return ports.resumeTraining(checked.target);
  }
  return checked.kind === "evaluate_checkpoint"
    ? ports.evaluateCheckpoint(checked.target)
    : ports.replayCheckpoint(checked.target);
}
