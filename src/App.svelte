<script lang="ts">
  import { onDestroy } from "svelte";
  import Rail from "./lib/Rail.svelte";
  import Stage from "./lib/Stage.svelte";
  import Inspector from "./lib/Inspector.svelte";
  import MotionMakerView from "./lib/MotionMakerView.svelte";
  import MotionInspector from "./lib/MotionInspector.svelte";
  import StatusBar from "./lib/StatusBar.svelte";
  import TrainView from "./lib/TrainView.svelte";
  import GuideView from "./lib/GuideView.svelte";
  import FilesView from "./lib/FilesView.svelte";
  import DebugView from "./lib/DebugView.svelte";
  import AssistantView from "./lib/AssistantView.svelte";
  import { Session, WORLD_STORAGE_KEY } from "./lib/session.svelte";
  import { MakerSession } from "./lib/maker.svelte";
  import { TrainingSession } from "./lib/training.svelte";
  import { applyAssistantProposal } from "./assistant/apply.ts";
  import { assistantTrainerConfig } from "./assistant/training.ts";
  import { assertRunRequested, executeAssistantCommand } from "./assistant/commands.ts";
  import type { AssistantCommand, AssistantContext, AssistantProposal, AssistantRunSnapshot, TrainingDraft } from "./assistant/contract.ts";
  import { listCheckpoints } from "./train/checkpoint-store.ts";
  import { taskLabel } from "./train/env/tasks.ts";
  import { listMotions, saveNewMotion, deleteOwnedMotion } from "./motion/motion-store.ts";
  import { newDraft } from "./motion/edit.ts";
  import type { MotionFile } from "./motion/format.ts";
  import type { View } from "./lib/views.ts";

  const TITLES: Record<View, { title: string; crumb: string }> = {
    guide: { title: "Guide", crumb: "What this is and how to use it" },
    sim: { title: "Microduck playground", crumb: "MuJoCo · CPU physics" },
    motion: { title: "Motion maker", crumb: "Keyframe a pose over time" },
    train: { title: "Train", crumb: "Microduck · PPO" },
    assistant: { title: "Assistant", crumb: "OpenAI · Run control" },
    files: { title: "Files", crumb: "Checkpoints saved in this browser" },
    debug: { title: "Debug", crumb: "Throughput and environment instruments" },
  };

  const session = new Session();
  const maker = new MakerSession();
  const training = new TrainingSession();
  let assistantApplying = $state(false);
  let commandBusy = false;
  let view = $state<View>("sim");
  onDestroy(() => training.dispose());

  const go = (next: View) => { if (!assistantApplying) view = next; };
  const monitor: AssistantRunSnapshot = $derived.by(() => {
    const result = training.evaluation;
    return {
      runId: training.runId, task: training.runTask ? taskLabel(training.runTask) : null,
      status: training.stopping ? "stopping" : training.status,
      completionReason: training.completionReason, iteration: training.iteration,
      targetIteration: training.targetIteration, totalSteps: training.totalSteps,
      reward: training.last?.rewardPerStep ?? null, standing: training.last?.standingFraction ?? null,
      checkpoint: training.savedName ?? result?.checkpointName ?? training.resumeFrom,
      evaluation: result ? {
        checkpointName: result.checkpointName, iteration: result.iteration, verdict: result.verdict,
        reasons: result.reasons, heightGain: result.jump?.heightGain ?? null,
        flightSeconds: result.jump?.flightSeconds ?? null,
        stableLandingSeconds: result.jump?.stableLandingSeconds ?? null,
      } : null,
    };
  });
  async function assistantContext(): Promise<AssistantContext> {
    const motions = (await listMotions()).filter(m => !m.error).map(({ id, name, modified, bytes }) => ({
      id, name, ...(modified === undefined ? {} : { modified }), ...(bytes === undefined ? {} : { bytes }),
    }));
    const checkpoints = (await listCheckpoints()).slice(0, 100).map(({ name, modified, bytes }) => ({
      name, modified, bytes, iteration: name === training.savedName ? training.savedAt : null,
    }));
    return JSON.parse(JSON.stringify({
      world: session.world,
      motions,
      currentMotion: maker.problem ? null : JSON.parse(maker.json()),
      run: monitor, checkpoints,
      training: {
        task: training.task, iterations: training.iterations, seed: training.seed,
        envs: training.envs, rolloutWorkers: training.rolloutWorkers,
        useCurrentWorld: training.useCurrentWorld, hiddenPreset: training.hiddenPreset,
        status: training.stopping ? "stopping" : training.status,
        iteration: training.iteration, reward: training.last?.rewardPerStep ?? null,
      },
    })) as AssistantContext;
  }
  function configureTraining(draft: TrainingDraft): void {
    const config = assistantTrainerConfig(draft);
    if (draft.useCurrentWorld) {
      // Training reads this snapshot when New run is pressed. Do not claim a
      // scene was configured if persistence would leave an older world active.
      localStorage.setItem(WORLD_STORAGE_KEY, JSON.stringify(session.world));
    }
    training.task = draft.task;
    training.iterations = draft.iterations;
    training.seed = draft.seed;
    training.envs = draft.envs;
    training.rolloutWorkers = draft.rolloutWorkers;
    training.useCurrentWorld = draft.useCurrentWorld;
    training.hiddenPreset = draft.hiddenPreset;
    training.stepsPerIter = config.stepsPerIter;
    training.resumeFrom = null;
    training.resumeSummary = null;
    training.error = null;
    training.refreshWorld();
  }
  async function applyProposal(proposal: AssistantProposal, context: AssistantContext): Promise<string> {
    if (assistantApplying || commandBusy) throw new Error("Another assistant operation is in progress");
    if (proposal.command) throw new Error("Run commands require their own approval button");
    if (training.evaluationStatus === "running") throw new Error("Wait for evaluation or cancel it before applying changes");
    if (proposal.world && (!session.ready || session.worldBusy || session.policyBusy)) {
      throw new Error("Wait for the simulator to finish loading before applying a scene");
    }
    const priorDraft = JSON.parse(JSON.stringify(maker.draft));
    if (proposal.motion && maker.problem && JSON.stringify(priorDraft) !== JSON.stringify(newDraft())) {
      throw new Error("Resolve the current Motion maker draft error before replacing it");
    }
    assistantApplying = true;
    const savedAs = maker.savedAs;
    const priorTraining = {
      stepsPerIter: training.stepsPerIter, resumeFrom: training.resumeFrom,
      resumeSummary: training.resumeSummary, error: training.error,
    };
    try {
      const result = await applyAssistantProposal(proposal, context, {
        getContext: assistantContext,
        applyWorld: value => session.applyWorld(value, "Assistant scene"),
        saveMotion: saveNewMotion,
        deleteMotion: deleteOwnedMotion,
        restoreEmptyMotion: () => { maker.draft = priorDraft; },
        setMotion: (file: MotionFile) => {
          if (!maker.loadJson(JSON.stringify(file))) throw new Error(maker.saveError ?? "Motion could not be opened");
        },
        setTraining: configureTraining,
      });
      if (result.motionId) maker.savedAs = result.motionId;
      // These lists are views of the files already committed above; a refresh
      // failure must not incorrectly report that the proposal was rolled back.
      await maker.refreshLibrary().catch(() => {});
      await training.refreshMotions().catch(() => {});
      if (proposal.world) session.frameWorld();
      return result.message;
    } catch (error) {
      maker.savedAs = savedAs;
      Object.assign(training, priorTraining);
      throw error;
    } finally { assistantApplying = false; }
  }

  function requireIdle(): void {
    if (training.status === "loading" || training.status === "running" || training.stopping)
      throw new Error("Stop the current training run first");
    if (training.evaluationStatus === "running") throw new Error("Wait for evaluation or cancel it first");
  }
  async function evaluateCheckpoint(name: string): Promise<void> {
    requireIdle();
    await training.evaluateCheckpoint(name);
    if (training.evaluationStatus === "error") throw new Error(training.evaluationError ?? "Evaluation failed");
  }
  async function replayCheckpoint(name: string): Promise<void> {
    requireIdle();
    if (!session.ready || session.worldBusy || session.policyBusy) throw new Error("Wait for the simulator to finish loading");
    if (!(await session.replayCheckpoint(name))) throw new Error(session.policyError ?? "Checkpoint could not be opened");
    view = "sim";
    session.frameWorld();
  }
  async function executeCommand(command: AssistantCommand, captured: AssistantContext): Promise<string> {
    if (commandBusy || assistantApplying) throw new Error("Another assistant operation is in progress");
    commandBusy = true;
    try {
      return await executeAssistantCommand(command, captured, {
        getContext: assistantContext,
        startTraining: async () => {
          requireIdle();
          // Reuse the reviewed bounded draft. No command can resume/overwrite a
          // saved policy or silently change the number of rollout steps.
          configureTraining(captured.training);
          const motion = captured.training.task.startsWith("motion:")
            ? captured.motions.find(item => item.id === captured.training.task.slice(7)) : undefined;
          const revision = motion?.modified !== undefined && motion.bytes !== undefined
            ? { id: motion.id, modified: motion.modified, bytes: motion.bytes } : undefined;
          const pending = training.start(false, revision), requestedRunId = training.runId;
          await pending;
          assertRunRequested(requestedRunId, training);
          return `Run requested: ${training.runName ?? training.task}, up to ${captured.training.iterations} iterations. ${training.status === "running" ? "Training is running." : "The worker is loading."} Skill unverified.`;
        },
        resumeTraining: async name => {
          requireIdle();
          training.resumeFrom = name;
          training.iterations = captured.training.iterations;
          training.rolloutWorkers = captured.training.rolloutWorkers;
          const pending = training.start(true), requestedRunId = training.runId;
          await pending;
          assertRunRequested(requestedRunId, training);
          return `Continuation requested from ${name}, up to ${captured.training.iterations} additional iterations. New checkpoint: ${training.runName}. Original preserved; skill unverified.`;
        },
        stopTraining: runId => {
          if (training.runId !== runId) throw new Error("The active run changed");
          training.stop();
          return training.stopping ? "Stop requested. The worker will finish this iteration and save before stopping." : "Loading canceled. No new checkpoint was created.";
        },
        evaluateCheckpoint: async name => {
          await evaluateCheckpoint(name);
          const report = training.evaluation;
          if (!report || report.checkpointName !== name || training.evaluationStatus !== "done")
            throw new Error("Evaluation was canceled. No qualification was recorded.");
          return `Evaluation of ${name}, iteration ${report.iteration}: ${report.verdict}. ${report.reasons.join(" ")} One deterministic 3-second trial; not a robustness certification.`;
        },
        replayCheckpoint: async name => {
          await replayCheckpoint(name);
          return `${name} loaded paused, upright, in its saved training scene. Playback is still an unassisted policy test.`;
        },
      });
    } finally { commandBusy = false; }
  }

  // The stage mounts on the first visit to Simulate and stays mounted after —
  // booting MuJoCo and the policy takes seconds and costs a 21 MB download, so
  // it should not happen behind someone reading the Guide, and it should not
  // happen twice. It is fully paused while hidden: a duck stepping in the
  // background would skew the throughput harness next door.
  let simMounted = $state(false);
  // The Motion maker boots a SECOND model. The two workspaces want the robot
  // in different states at the same moment — one running a policy, one holding
  // a pose being edited — and one MjData cannot be both. Only one ever steps:
  // whichever is not up front is fully paused.
  let makerMounted = $state(false);
  let trainMounted = $state(false);
  let assistantMounted = $state(false);
  $effect(() => {
    if (view === "train") trainMounted = true;
    if (view === "assistant") assistantMounted = true;
    if (view === "sim") {
      simMounted = true;
      // A run saved in the Train workspace after this list was first built
      // should still show up in the policy picker.
      if (session.ready) void session.refreshPolicies();
    }
    if (view === "motion") {
      makerMounted = true;
      // A motion uploaded in Files, or saved from here earlier, should be in
      // the Open list without a reload.
      if (maker.ready) void maker.refreshLibrary();
    }
    session.setActive(view === "sim");
    maker.setActive(view === "motion");
  });
</script>

<!-- The inspector column only exists for workspaces that have one; without
     this the Train view would sit off-centre next to an empty gutter. -->
<div class="shell" class:no-aside={view !== "sim" && view !== "motion"}>
  <Rail bind:view disabled={assistantApplying} />

  <header class="topbar">
    <h1>{TITLES[view].title}</h1>
    <span class="crumb">{TITLES[view].crumb}</span>
  </header>

  <!-- Kept in the DOM once mounted, hidden when another workspace is up front. -->
  {#if simMounted}
    <div class="main" class:hidden={view !== "sim"}>
      <Stage {session} />
    </div>
  {/if}
  {#if makerMounted}
    <div class="main maker-main" class:hidden={view !== "motion"}>
      <MotionMakerView session={maker} />
    </div>
  {/if}
  {#if view === "sim"}
    <Inspector {session} />
  {:else if view === "motion"}
    <MotionInspector session={maker} />
  {:else if view === "files"}
    <FilesView />
  {:else if view === "debug"}
    <DebugView />
  {:else if view === "guide"}
    <GuideView go={(v) => (view = v)} />
  {/if}

  {#if trainMounted}
    <div class="training-workspace" class:hidden={view !== "train"}>
      <TrainView active={view === "train"} session={training} {replayCheckpoint} />
    </div>
  {/if}

  {#if assistantMounted}
    <div class="assistant-workspace" class:hidden={view !== "assistant"}>
      <AssistantView active={view === "assistant"} getContext={assistantContext} {applyProposal} {go}
        {monitor} {executeCommand} stopRun={() => training.stop()} {evaluateCheckpoint} {replayCheckpoint}
        evaluationState={{ status: training.evaluationStatus, progress: training.evaluationProgress,
          error: training.evaluationError, name: training.evaluatingName }} cancelEvaluation={() => training.cancelEvaluation()} />
    </div>
  {/if}

  <StatusBar {session} {maker} {view} />
</div>

<style>
  .shell {
    height: 100%;
    display: grid;
    grid-template-columns: 180px minmax(0, 1fr) 300px;
    grid-template-rows: 44px minmax(0, 1fr) auto;
    grid-template-areas:
      "rail topbar topbar"
      "rail main   aside"
      "rail status status";
  }

  .shell.no-aside {
    grid-template-columns: 208px minmax(0, 1fr);
    grid-template-rows: 44px minmax(0, 1fr) auto;
    grid-template-areas:
      "rail topbar"
      "rail main"
      "rail status";
  }

  .topbar {
    grid-area: topbar;
    display: flex; align-items: center; gap: 10px;
    padding: 0 14px;
    background: var(--panel); border-bottom: 1px solid var(--line);
  }
  h1 { margin: 0; font-size: 13px; font-weight: 700; letter-spacing: 0; }
  .crumb { font-size: 11px; color: var(--muted); }

  .main { grid-area: main; display: grid; min-height: 0; min-width: 0; }
  /* The maker lays out its own rows, so it must not be stretched into one. */
  .main.maker-main { display: contents; }
  .main.hidden { display: none; }
  .training-workspace { grid-area: main; display: flex; flex-direction: column; min-width: 0; min-height: 0; overflow: hidden; }
  .training-workspace.hidden { display: none; }
  .assistant-workspace { grid-area: main; min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }
  .assistant-workspace.hidden { display: none; }

  @media (max-width: 1040px) {
    .shell {
      grid-template-columns: 208px minmax(0, 1fr);
      grid-template-rows: 44px minmax(200px, 1fr) minmax(0, 38%) auto;
      grid-template-areas:
        "rail topbar"
        "rail main"
        "rail aside"
        "rail status";
    }
  }
  @media (max-width: 860px) {
    .shell, .shell.no-aside { grid-template-columns: 56px minmax(0, 1fr); }
    .crumb { display: none; }
  }
</style>
