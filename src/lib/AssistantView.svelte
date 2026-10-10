<script lang="ts">
  import { onMount, tick, untrack } from "svelte";
  import type { AssistantCommand, AssistantContext, AssistantProposal, AssistantRunSnapshot } from "../assistant/contract.ts";
  import type { View } from "./views.ts";
  import { AssistantSession } from "./assistant.svelte";

  let { active = true, getContext, applyProposal, go, monitor, executeCommand, stopRun, evaluateCheckpoint, replayCheckpoint,
    evaluationState = { status: "idle", progress: 0, error: null, name: null }, cancelEvaluation = () => {} }: {
    active?: boolean;
    getContext: () => Promise<AssistantContext>;
    applyProposal: (proposal: AssistantProposal, context: AssistantContext) => Promise<string>;
    go: (view: View) => void;
    monitor: AssistantRunSnapshot;
    executeCommand: (command: AssistantCommand, context: AssistantContext) => Promise<string>;
    stopRun: () => void;
    evaluateCheckpoint: (name: string) => Promise<void>;
    replayCheckpoint: (name: string) => Promise<void>;
    evaluationState?: { status: "idle" | "running" | "done" | "error"; progress: number; error: string | null; name: string | null };
    cancelEvaluation?: () => void;
  } = $props();
  const s = new AssistantSession();
  let mounted = $state(false);
  let thread: HTMLDivElement;
  let composer: HTMLTextAreaElement;
  let localAction = $state<"evaluate" | "replay" | null>(null);
  let actionError = $state<string | null>(null);
  let stopRequestedRunId = $state<string | null>(null);
  let cancelRequested = $state(false);
  const examples = ["Build a short obstacle course", "Start a short jump training run", "Review my latest training result"];
  onMount(() => { mounted = true; return () => s.dispose(); });
  $effect(() => {
    if (mounted && active) untrack(() => { if (!s.busy) void s.refreshStatus(); });
  });
  $effect(() => {
    s.messages.length; s.busy; s.error;
    void tick().then(() => { if (thread) thread.scrollTop = thread.scrollHeight; });
  });
  const fillPrompt = (prompt: string) => { s.draft = prompt; composer?.focus(); };
  const taskName = (task: string) => task === "$generated_motion" ? "Generated motion" : task.replaceAll("_", " ").replace(/^motion:/, "Motion: ");
  const submit = () => { if (s.canSend) void s.send(getContext); };
  const commandLabel = (kind: AssistantCommand["kind"]) => ({ start_training: "Start training", resume_training: "Resume training", stop_training: "Stop training", evaluate_checkpoint: "Evaluate checkpoint", replay_checkpoint: "Open trained policy" })[kind];
  const runBusy = $derived(["loading", "running", "stopping"].includes(monitor.status));
  const evaluationBusy = $derived(evaluationState.status === "running");
  const operationBusy = $derived(runBusy || evaluationBusy || !!localAction || s.applying);
  const runLabel = $derived(monitor.status === "loading" ? "Loading environment" : monitor.status === "running" ? "Training" : monitor.status === "stopping" ? "Saving and stopping" : monitor.status === "evaluating" ? "Evaluating checkpoint" : monitor.status === "replaying" ? "Loading trained policy" : monitor.completionReason === "completed" ? "Budget completed" : monitor.completionReason === "error" || monitor.status === "error" ? "Run error" : monitor.completionReason === "stopped" ? "Stopped by request" : monitor.runId ? "Run stopped" : "No active run");
  const metric = (value: number | null, digits = 3) => value === null ? "Not measured" : value.toFixed(digits);
  const measurement = (value: number | null, unit: string) => value === null ? "Not measured" : `${value.toFixed(3)} ${unit}`;
  $effect(() => { if (!evaluationBusy) cancelRequested = false; });
  const reviewRun = () => {
    if (!s.consent || !s.configured || s.busy || s.applying || s.draft.trim()) return;
    s.draft = "Review the current run and its latest checkpoint evaluation. Separate training completion from demonstrated skill, explain any failed criteria, and suggest one bounded next step.";
    submit();
  };
  const requestStop = () => {
    if (!monitor.runId || !runBusy || monitor.status === "stopping" || stopRequestedRunId === monitor.runId) return;
    stopRequestedRunId = monitor.runId;
    actionError = null;
    try { stopRun(); } catch (error) { stopRequestedRunId = null; actionError = error instanceof Error ? error.message.slice(0, 500) : "Could not request stop."; }
  };
  const checkpointAction = async (kind: "evaluate" | "replay") => {
    const name = monitor.checkpoint;
    if (!name || operationBusy) return;
    localAction = kind;
    actionError = null;
    try { await (kind === "evaluate" ? evaluateCheckpoint(name) : replayCheckpoint(name)); }
    catch (error) { actionError = error instanceof Error ? error.message.slice(0, 500) : "Checkpoint action failed."; }
    finally { localAction = null; }
  };
</script>

<section class="assistant-view" aria-label="Assistant workspace">
  <header class="assistant-header">
    <div><h2>Assistant</h2><span class="runtime">Microduck / MuJoCo CPU</span></div>
    <div class="connection" role="status">
      <span class="connection-dot" class:connected={s.configured}></span>
      <span>{s.statusLoading ? "Checking configuration" : s.configured ? "Key configured" : "Not configured"}</span>
      {#if s.model}<span class="model">{s.model}</span>{/if}
      <button class="icon" aria-label="Refresh assistant connection" title="Refresh assistant connection" disabled={s.statusLoading || s.busy} onclick={() => void s.refreshStatus()}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5M6.1 7A7 7 0 0 1 18 5l2 3M4 16l2 3a7 7 0 0 0 11.9-2" /></svg>
      </button>
    </div>
  </header>

  {#if s.statusChecked && !s.configured}
    <div class="connection-message" role="status">
      <strong>{s.connectionError ? "Assistant connection unavailable" : "OpenAI is not configured on this server."}</strong>
      <span>{s.connectionError ?? "The server needs OPENAI_API_KEY. No key is stored in this browser."}</span>
    </div>
  {/if}

  <section class="run-monitor" aria-label="Training run monitor">
    <div class="run-heading"><div><h3>{monitor.task ? taskName(monitor.task) : "Training"}</h3><span class="run-status" role="status">{runLabel}</span></div><span class="runtime">Local CPU</span></div>
    {#if monitor.runId}
      <div class="run-id">{monitor.runId}</div>
      <progress aria-label="Training iteration progress" max={Math.max(1, monitor.targetIteration)} value={Math.min(monitor.iteration, Math.max(1, monitor.targetIteration))}></progress>
      <dl class="run-metrics"><div><dt>Iterations</dt><dd>{monitor.iteration} / {monitor.targetIteration}</dd></div><div><dt>Reward</dt><dd>{metric(monitor.reward)}</dd></div><div><dt>Upright</dt><dd>{monitor.standing === null ? "Not measured" : `${(monitor.standing * 100).toFixed(1)}%`}</dd></div><div><dt>Environment steps</dt><dd>{monitor.totalSteps.toLocaleString()}</dd></div></dl>
    {/if}
    {#if monitor.evaluation}
      {@const evaluation = monitor.evaluation}
      {#if evaluation.checkpointName !== monitor.checkpoint}<span class="checkpoint-name">Evaluation checkpoint: {evaluation.checkpointName}</span>{/if}
      <details class="evaluation"><summary><span class:qualified={evaluation.verdict === "passed"}>{evaluation.verdict === "passed" ? "Evaluation passed" : evaluation.verdict === "failed" ? "Evaluation failed" : "Skill not qualified"}</span><span> / iteration {evaluation.iteration}</span></summary><div class="evaluation-body"><span class="detail">{evaluation.checkpointName}</span><dl class="evaluation-metrics"><div><dt>Height gain</dt><dd>{measurement(evaluation.heightGain, "m")}</dd></div><div><dt>Flight</dt><dd>{measurement(evaluation.flightSeconds, "s")}</dd></div><div><dt>Stable landing</dt><dd>{measurement(evaluation.stableLandingSeconds, "s")}</dd></div></dl>{#if evaluation.reasons.length}<ul>{#each evaluation.reasons as reason}<li>{reason}</li>{/each}</ul>{/if}</div></details>
    {:else if monitor.runId}<span class="qualification">Skill not evaluated</span>{/if}
    {#if monitor.checkpoint}<div class="checkpoint-name">Checkpoint: {monitor.checkpoint}</div>{/if}
    {#if evaluationBusy}
      <div class="evaluation-progress"><span role="status">Evaluating checkpoint / {(Math.max(0, Math.min(1, evaluationState.progress)) * 100).toFixed(0)}%</span><button disabled={cancelRequested} onclick={() => { cancelRequested = true; cancelEvaluation(); }}>{cancelRequested ? "Cancel requested" : "Cancel evaluation"}</button></div>
      {#if evaluationState.name}<span class="checkpoint-name">{evaluationState.name}</span>{/if}
      <progress aria-label="Checkpoint evaluation progress" max="1" value={Math.max(0, Math.min(1, evaluationState.progress))}></progress>
    {/if}
    {#if evaluationState.status === "error" && evaluationState.error}<p class="error" role="alert">{evaluationState.error.slice(0, 500)}</p>{/if}
    <div class="monitor-actions">
      {#if runBusy}<button class="stop" disabled={monitor.status === "stopping" || stopRequestedRunId === monitor.runId} onclick={requestStop}>{monitor.status === "stopping" || stopRequestedRunId === monitor.runId ? "Stop requested" : "Stop run"}</button>{/if}
      {#if monitor.checkpoint}<button disabled={operationBusy} onclick={() => void checkpointAction("evaluate")}>{localAction === "evaluate" ? "Evaluating" : "Evaluate checkpoint"}</button><button disabled={operationBusy} onclick={() => void checkpointAction("replay")}>{localAction === "replay" ? "Loading policy" : "Open trained policy"}</button>{/if}
      <button disabled={!s.consent || !s.configured || s.busy || s.applying || !!s.draft.trim() || (!monitor.runId && !monitor.evaluation && !monitor.checkpoint)} title={!s.consent ? "Enable context sharing to review this run" : s.draft.trim() ? "Send or clear the current draft first" : "Send this run snapshot to the assistant"} onclick={reviewRun}>Review run</button>
    </div>
    {#if actionError}<p class="error" role="alert">{actionError}</p>{/if}
  </section>

  <div class="thread" bind:this={thread} role="log" aria-label="Assistant conversation" aria-live="polite" aria-relevant="additions">
    {#if !s.messages.length}
      <div class="empty">
        <h3>What shall we make?</h3>
        <div class="examples">{#each examples as prompt (prompt)}<button onclick={() => fillPrompt(prompt)}>{prompt}<span aria-hidden="true">&#8599;</span></button>{/each}</div>
      </div>
    {/if}
    {#each s.messages as entry (entry.id)}
      <article class:user={entry.role === "user"} class="message">
        <div class="message-label">{entry.role === "user" ? "You" : "Assistant"}</div>
        <div class="message-body">
          <p class="message-content">{entry.content}</p>
          {#if entry.proposal}
            {@const proposal = entry.proposal}
            <section class="proposal" aria-label={proposal.command ? "Proposed action" : "Proposed changes"}>
              <div class="proposal-heading"><h3>{proposal.command ? "Proposed action" : "Proposed changes"}</h3><span class="proposal-status" class:applied={entry.proposalStatus === "applied"}>{entry.proposalStatus === "pending" ? "Awaiting review" : entry.proposalStatus === "applying" ? "Requesting" : entry.proposalStatus === "applied" ? proposal.command ? "Requested" : "Applied" : entry.proposalStatus === "dismissed" ? "Dismissed" : entry.proposalStatus === "failed" ? "Failed" : "Superseded"}</span></div>
              {#if proposal.command}
                <div class="proposal-row"><span class="kind">Action</span><div><strong>{commandLabel(proposal.command.kind)}</strong>{#if proposal.command.target}<span class="detail">{proposal.command.target}</span>{/if}{#if proposal.command.kind === "start_training" && entry.context}<span class="detail">{taskName(entry.context.training.task)} / {entry.context.training.iterations} iterations / seed {entry.context.training.seed}</span><span class="detail">{entry.context.training.envs * entry.context.training.rolloutWorkers} environments / {entry.context.training.hiddenPreset === "reference" ? "512 / 256 / 128" : "128 / 64"} network / {entry.context.training.useCurrentWorld ? "current scene snapshot" : "default floor"}</span><span class="warning">New bounded run / local CPU / skill unverified</span>{:else if proposal.command.kind === "resume_training" && entry.context}<span class="detail">{entry.context.training.iterations} additional iterations / {entry.context.training.rolloutWorkers} rollout workers</span><span class="detail">Saved task, network, seed, and scene</span><span class="warning">New checkpoint fork / original preserved / local CPU</span>{:else if proposal.command.kind === "evaluate_checkpoint"}<span class="detail">Bounded local checkpoint evaluation</span>{:else if proposal.command.kind === "replay_checkpoint"}<span class="detail">Load this saved policy in Simulate</span>{/if}</div></div>
              {/if}
              {#if proposal.world}
                <div class="proposal-row"><span class="kind">Scene</span><div><strong>{proposal.world.objects.length} objects</strong><span class="detail">{proposal.world.objects.length ? proposal.world.objects.map((object) => `${object.id} (${object.kind})`).join(", ") : "Default floor"}</span></div></div>
              {/if}
              {#if proposal.motion}
                <div class="proposal-row"><span class="kind">Motion</span><div><strong>{proposal.motion.name ?? "Generated motion"}</strong><span class="detail">{proposal.motion.keyframes.length} frames / {proposal.motion.keyframes.at(-1)?.t.toFixed(2)} s / {proposal.motion.loop ? "loop" : "single pass"}</span><span class="detail">New saved motion; replaces Motion maker draft</span><span class="warning">Unverified balance and physical feasibility</span></div></div>
              {/if}
              {#if proposal.training}
                <div class="proposal-row"><span class="kind">Training</span><div><strong>{taskName(proposal.training.task)}</strong><span class="detail">{proposal.training.iterations} iterations / seed {proposal.training.seed} / {proposal.training.envs * proposal.training.rolloutWorkers} environments</span><span class="detail">{proposal.training.hiddenPreset === "reference" ? "512 / 256 / 128" : "128 / 64"} network / {proposal.training.useCurrentWorld ? "current world snapshot" : "default floor"}</span><span class="warning">Configuration only / training will not start</span></div></div>
              {/if}
              <details><summary>Review JSON</summary><pre>{JSON.stringify(proposal, null, 2)}</pre></details>
              {#if entry.applicationError}<p class="error" role="alert">{entry.applicationError}</p>{/if}
              {#if entry.proposalStatus === "pending" || entry.proposalStatus === "applying"}
                <div class="proposal-actions"><button class="primary" disabled={s.busy || s.applying || !!localAction || (evaluationBusy && proposal.command?.kind !== "stop_training")} onclick={() => proposal.command ? void s.execute(entry.id, executeCommand) : void s.apply(entry.id, applyProposal)}>{entry.proposalStatus === "applying" ? "Requesting action" : proposal.command ? commandLabel(proposal.command.kind) : "Apply proposal"}</button><button disabled={s.busy || s.applying} onclick={() => s.dismiss(entry.id)}>Dismiss</button></div>
              {:else if entry.proposalStatus === "applied"}
                {#if entry.applicationResult}<p class="application-result" role="status">{entry.applicationResult}</p>{/if}
                <div class="proposal-actions workspace-links">{#if proposal.world}<button onclick={() => go("sim")}>Open scene</button>{/if}{#if proposal.motion}<button onclick={() => go("motion")}>Open motion</button>{/if}{#if proposal.training}<button onclick={() => go("train")}>Open training</button>{/if}</div>
              {/if}
            </section>
          {/if}
        </div>
      </article>
    {/each}
    {#if s.busy}<div class="pending" role="status"><span class="pending-dot"></span>Waiting for assistant<button onclick={() => s.cancel()}>Cancel request</button></div>{/if}
    {#if s.error}<div class="request-message"><p class="error" role="alert">{s.error}</p>{#if s.retryAvailable}<button disabled={!s.consent || !s.configured || s.busy || s.applying} onclick={() => void s.send(getContext, true)}>Retry request</button>{/if}</div>{/if}
    {#if s.notice}<div class="request-message"><p class="muted" role="status">{s.notice}</p>{#if s.retryAvailable}<button disabled={!s.consent || !s.configured || s.busy || s.applying} onclick={() => void s.send(getContext, true)}>Retry request</button>{/if}</div>{/if}
  </div>

  <form class="composer" onsubmit={(event) => { event.preventDefault(); submit(); }}>
    <label class="consent"><input type="checkbox" bind:checked={s.consent} disabled={s.busy}>Send chat, scene, motion, and training context to OpenAI</label>
    <div class="input-row"><textarea bind:this={composer} bind:value={s.draft} aria-label="Message to assistant" placeholder="Ask for a scene, motion, or training setup" maxlength="8000" rows="3" disabled={s.applying} onkeydown={(event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); submit(); } }}></textarea><button class="send icon primary" type="submit" aria-label="Send message" title="Send message" disabled={!s.canSend}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7" /></svg></button></div>
    <div class="composer-foot"><span>Actions require confirmation / chat stays in this session</span><span>{s.draft.length.toLocaleString()} / 8,000</span></div>
  </form>
</section>

<style>
  .assistant-view{display:flex;flex-direction:column;min-width:0;min-height:0;height:100%;background:var(--viewport);letter-spacing:0}
  .assistant-header{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:15px 22px;border-bottom:1px solid var(--line);flex:none;flex-wrap:wrap}.assistant-header>div:first-child{display:flex;align-items:center;gap:12px;flex-wrap:wrap}h2{margin:0;font-size:14px;letter-spacing:0}.runtime{font-size:10px;color:var(--muted)}.connection{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--muted);flex-wrap:wrap}.connection-dot{width:6px;height:6px;border-radius:50%;background:var(--muted);flex:none}.connection-dot.connected{background:var(--ok)}.model{font-family:ui-monospace,monospace;font-size:10px;overflow-wrap:anywhere}.connection .icon{width:28px;min-height:28px}
  button{font:inherit;font-size:12px;color:var(--ink);background:var(--panel-hi);border:1px solid var(--line);border-radius:5px;padding:7px 11px;min-height:34px;cursor:pointer}button:hover:not(:disabled){border-color:var(--line-hi)}button:disabled{opacity:.4;cursor:not-allowed}.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600}.primary:hover:not(:disabled){background:var(--accent-hi)}.icon{display:inline-flex;align-items:center;justify-content:center;width:34px;padding:0;flex:none}.icon svg{height:17px;width:17px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}button:focus-visible,textarea:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:3px}
  .connection-message{display:flex;flex-direction:column;gap:5px;padding:12px 22px;background:var(--panel);border-bottom:1px solid var(--line);font-size:11px;color:var(--muted);line-height:1.5;flex:none;overflow-wrap:anywhere}.connection-message strong{color:var(--ink);font-size:12px;font-weight:550}.thread{min-height:0;flex:1;overflow:auto;overscroll-behavior:contain;padding:0 22px 20px}.empty{padding:28px 0 18px;max-width:520px}.empty h3{margin:0 0 14px;font-size:15px;font-weight:600}.examples{display:flex;flex-direction:column;align-items:stretch}.examples button{display:flex;justify-content:space-between;align-items:center;background:transparent;border:0;border-bottom:1px solid var(--line);border-radius:0;text-align:left;padding:11px 0;gap:12px}.examples button span{color:var(--muted)}
  .run-monitor{flex:none;padding:12px 22px;display:flex;flex-direction:column;gap:8px;border-bottom:1px solid var(--line);background:var(--panel);max-height:38%;overflow:auto;font-size:11px;min-width:0}.run-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}.run-heading>div{display:flex;align-items:center;gap:12px;flex-wrap:wrap;min-width:0}.run-heading h3{margin:0;font-size:12px;font-weight:600;overflow-wrap:anywhere}.run-status{color:var(--muted);font-size:10px}.run-id,.checkpoint-name{color:var(--muted);font-family:ui-monospace,monospace;font-size:9px;overflow-wrap:anywhere}.run-monitor progress{display:block;width:100%;height:4px;border:0;accent-color:var(--control-accent);border-radius:0;background:var(--line)}.run-monitor progress::-webkit-progress-bar{background:var(--line)}.run-monitor progress::-webkit-progress-value{background:var(--accent-text)}.run-metrics,.evaluation-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:0}.run-metrics div,.evaluation-metrics div{min-width:0}.run-monitor dt{color:var(--muted);font-size:9px;margin-bottom:3px}.run-monitor dd{margin:0;font-size:11px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.monitor-actions{display:flex;gap:6px;flex-wrap:wrap}.monitor-actions button{font-size:10px;padding:5px 8px;min-height:29px}.stop{color:var(--danger)}.qualification{font-size:10px;color:var(--warn)}.evaluation summary{font-size:10px}.evaluation summary>span:first-child{color:var(--warn)}.evaluation summary>span.qualified{color:var(--ok)}.evaluation-body{display:flex;flex-direction:column;gap:8px;padding-top:6px;overflow-wrap:anywhere}.evaluation-metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.evaluation ul{margin:0;padding-left:16px;line-height:1.5;font-size:10px}
  .evaluation-progress{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:10px}.evaluation-progress button{font-size:10px;min-height:28px;padding:4px 8px}
  .message{display:grid;grid-template-columns:72px minmax(0,1fr);gap:12px;border-bottom:1px solid var(--line);padding:21px 0;min-width:0}.message-label{font-size:11px;font-weight:600;color:var(--muted);padding-top:3px}.message.user .message-label{color:var(--ink)}.message-body{min-width:0;max-width:900px}.message-content{margin:0;font-size:13px;line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere}.proposal{margin-top:17px;border-top:1px solid var(--line);padding-top:13px;display:flex;flex-direction:column;gap:12px;min-width:0}.proposal-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.proposal h3{margin:0;font-size:12px;letter-spacing:0}.proposal-status{font-size:10px;color:var(--muted)}.proposal-status.applied{color:var(--ok)}.proposal-row{display:grid;grid-template-columns:62px minmax(0,1fr);gap:10px;font-size:11px;line-height:1.5}.kind{color:var(--muted)}.proposal-row>div{display:flex;flex-direction:column;gap:3px;min-width:0;overflow-wrap:anywhere}.proposal-row strong{font-size:12px;font-weight:550}.detail{color:var(--muted)}.warning{font-size:10px;color:var(--warn)}details{font-size:11px;color:var(--muted)}summary{cursor:pointer;padding:4px 0;width:fit-content}pre{margin:8px 0 0;padding:10px 0;border-top:1px solid var(--line);max-height:260px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:10px;line-height:1.6;color:var(--muted)}.proposal-actions{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.application-result{margin:0;color:var(--ok);font-size:11px;line-height:1.5;overflow-wrap:anywhere}.workspace-links button{font-size:11px;min-height:30px;padding:5px 9px}
  .pending{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-top:18px;font-size:12px;color:var(--muted)}.pending button{margin-left:auto;min-height:30px;font-size:11px}.pending-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse 1.2s infinite}@keyframes pulse{50%{opacity:.25}}.request-message{padding-top:16px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.error{margin:0;color:var(--danger);font-size:12px;line-height:1.6;overflow-wrap:anywhere}.muted{margin:0;color:var(--muted);font-size:12px}.request-message button{font-size:11px;min-height:30px}
  .composer{flex:none;border-top:1px solid var(--line);padding:12px 22px 13px;background:var(--panel)}.consent{display:flex;align-items:flex-start;gap:8px;font-size:11px;color:var(--muted);margin-bottom:11px;line-height:1.4}.consent input{margin:1px 0 0;accent-color:var(--control-accent);flex:none}.input-row{display:flex;align-items:flex-end;gap:9px}textarea{flex:1;width:100%;min-width:0;min-height:74px;max-height:200px;resize:vertical;font:inherit;font-size:12px;line-height:1.55;letter-spacing:0;color:var(--ink);background:var(--viewport);border:1px solid var(--line);border-radius:5px;padding:10px 12px}.send{height:36px;width:36px;margin-bottom:1px}.composer-foot{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:9px;line-height:1.5;margin-top:8px}
  @media(max-width:700px){.assistant-header{padding:12px;gap:9px}.assistant-header>div:first-child{gap:8px}.connection{width:100%;font-size:10px}.connection .icon{margin-left:auto}.connection-message{padding:11px 12px;font-size:10px}.run-monitor{padding:10px 12px;gap:7px}.run-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.run-heading>div{gap:6px}.run-heading{align-items:flex-start}.thread{padding:0 12px 14px}.message{grid-template-columns:minmax(0,1fr);gap:7px;padding:17px 0}.message-label{font-size:10px;padding:0}.message-content{font-size:12px}.proposal-row{grid-template-columns:53px minmax(0,1fr);gap:8px}.composer{padding:11px 12px}.consent{font-size:10px}.composer-foot{font-size:8px;gap:5px}.composer-foot span:last-child{margin-left:auto}.empty{padding-top:22px}.empty h3{font-size:14px}.examples button{font-size:11px}.input-row{gap:7px}textarea{font-size:12px;padding:9px;min-height:70px}.send{width:32px;height:34px}.runtime{font-size:9px}}
</style>
