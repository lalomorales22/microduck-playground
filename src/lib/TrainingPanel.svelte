<script lang="ts">
  import { onMount } from "svelte";
  import { BUILTIN_TASKS } from "../train/env/tasks.ts";
  import type { TrainingSession } from "./training.svelte";

  let { active = true, session, replayCheckpoint }: {
    active?: boolean; session: TrainingSession; replayCheckpoint?: (name: string) => Promise<void>;
  } = $props();
  const s = $derived(session);
  let mounted = $state(false);
  onMount(() => {
    mounted = true;
    const refresh = () => s.refreshWorld();
    window.addEventListener("storage", refresh);
    window.addEventListener("focus", refresh);
    return () => {
      window.removeEventListener("storage", refresh);
      window.removeEventListener("focus", refresh);
    };
  });

  const taskGroups = [
    { label: "Balance", kinds: ["hold_pose", "standup"] },
    { label: "Locomotion", kinds: ["walk_forward", "walk_backward", "turn_left", "turn_right"] },
    { label: "Acrobatics", kinds: ["jump", "somersault", "backflip"] },
  ];
  const budgets = [{ label: "Quick", iterations: 50 }, { label: "Standard", iterations: 400 }, { label: "Extended", iterations: 1000 }];
  const motion = $derived(s.motions.find((m) => m.id === s.motionId) ?? null);
  const busy = $derived(s.status === "running" || s.status === "loading");
  const blocked = $derived(busy || s.evaluationStatus === "running");
  $effect(() => {
    if (!mounted || !active || busy) return;
    void s.refreshCheckpoints();
    void s.refreshMotions().catch((error) => { s.error = String(error); });
    s.refreshWorld();
  });
  const uprightLabel = $derived(s.runTask?.kind === "hold_pose" || s.runTask?.kind === "standup" ? "Standing" : "Upright");
  const taskMetrics = $derived(Object.entries(s.last?.taskMetrics ?? {}));
  const num = (n: number) => Math.round(n).toLocaleString();
  const metricName = (name: string) => name.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");
  const statusLabel = $derived(s.stopping ? "Saving and stopping" : s.status === "loading" ? "Loading environment" : s.status === "running" ? "Training" : s.status === "stopped" ? s.completionReason === "completed" ? "Completed budget" : "Stopped" : s.status === "error" ? "Error" : "Ready");
  const evaluationLabel = $derived(s.evaluationStatus === "running" ? "Evaluating checkpoint" : s.evaluationStatus === "error" ? "Evaluation failed" : s.evaluation?.verdict === "passed" ? "Jump check passed" : s.evaluation?.verdict === "failed" ? "Needs more training" : s.evaluation ? "Not qualified" : "Not evaluated");
  const evaluate = (name: string) => { void s.evaluateCheckpoint(name).catch((error) => { s.evaluationError = error instanceof Error ? error.message : String(error); }); };
  const replay = (name: string) => { void replayCheckpoint?.(name).catch((error) => { s.error = error instanceof Error ? error.message : String(error); }); };
  const W = 720, H = 120, PAD = 4;
  const path = $derived.by(() => {
    const h = s.history;
    if (h.length < 2) return { reward: "", standing: "", min: 0, max: 0 };
    const min = Math.min(...h.map((p) => p.reward));
    const max = Math.max(...h.map((p) => p.reward));
    const x = (i: number) => PAD + (i / (h.length - 1)) * (W - 2 * PAD);
    const line = (pick: (p: (typeof h)[number]) => number, lo: number, hi: number) =>
      h.map((p, i) => {
        const t = (pick(p) - lo) / (hi - lo || 1);
        return `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${(H - PAD - t * (H - 2 * PAD)).toFixed(1)}`;
      }).join("");
    return { reward: line((p) => p.reward, min, max), standing: line((p) => p.standing, 0, 1), min, max };
  });
</script>

<section class="train-panel" aria-label="Policy training">
  <header class="sub">
    <div class="heading"><h3>PPO training</h3><span class="runtime">MuJoCo / browser CPU</span></div>
    <span class="status" class:active={busy} role="status">{statusLabel}</span>
    <div class="buttons">
      {#if busy}
        <button onclick={() => s.saveAs()} disabled={!s.iteration || s.stopping}>Save as</button>
        <button onclick={() => s.stop()} disabled={s.stopping}>Stop</button>
      {:else}
        {#if s.iteration && s.status !== "error"}<button disabled={blocked} onclick={() => s.saveAs()}>Save as</button>{/if}
        <button class="primary" disabled={blocked} onclick={() => void s.start(false)}>New run</button>
      {/if}
    </div>
  </header>

  <div class="setup">
    <label class="task-picker">Task
      <select aria-label="Training task" bind:value={s.task} disabled={blocked}>
        {#each taskGroups as group (group.label)}
          <optgroup label={group.label}>
            {#each BUILTIN_TASKS.filter((task) => group.kinds.includes(task.kind)) as task (task.kind)}
              <option value={task.kind}>{task.label}{task.experimental ? " (experimental)" : ""}</option>
            {/each}
          </optgroup>
        {/each}
        {#if s.motions.length}
          <optgroup label="Motion targets">
            {#each s.motions as m (m.id)}<option value={"motion:" + m.id}>{m.name}{m.source === "builtin" ? "" : " (yours)"}</option>{/each}
          </optgroup>
        {/if}
      </select>
    </label>
    <div class="task-detail">
      {#if s.taskInfo?.experimental}<span class="badge experimental">Experimental objective</span>{/if}
      <span>{s.taskInfo?.metricLabel ?? "Joint-angle tracking"}</span>
      {#if motion}<span>{motion.duration.toFixed(1)} s / {motion.keyframes} keyframes / {motion.loop ? "loop" : "single pass"}</span>{/if}
    </div>
  </div>

  <div class="config">
    <fieldset class="budget" disabled={blocked}>
      <legend>Training budget</legend>
      <div class="segments">
        {#each budgets as budget (budget.iterations)}
          <label class:selected={s.iterations === budget.iterations}>
            <input type="radio" name="training-budget" value={budget.iterations} checked={s.iterations === budget.iterations} onchange={() => s.applyBudget(budget.iterations)}>
            {budget.label}<small>{budget.iterations}</small>
          </label>
        {/each}
      </div>
    </fieldset>
    <label>Iterations<input aria-label="Training iterations" type="number" min="1" max="5000" step="1" bind:value={s.iterations} disabled={blocked}></label>
    <label>Seed<input aria-label="Training seed" type="number" min="0" max="4294967295" step="1" bind:value={s.seed} disabled={blocked}></label>
    <label>Network<select aria-label="Training network" bind:value={s.hiddenPreset} disabled={blocked}><option value="small">128 / 64</option><option value="reference">512 / 256 / 128</option></select></label>
    <label>Rollout workers<select aria-label="Rollout workers" bind:value={s.rolloutWorkers} disabled={blocked}>{#each [...new Set([1, 2, 4, 6, 8, s.rolloutWorkers])].sort((a, b) => a - b) as n (n)}<option value={n} disabled={n * s.envs > 128}>{n}</option>{/each}</select></label>
    <label>Envs / worker<select aria-label="Environments per worker" bind:value={s.envs} disabled={blocked}>{#each [...new Set([16, 32, 64, 128, s.envs])].sort((a, b) => a - b) as n (n)}<option value={n} disabled={n * s.rolloutWorkers > 128}>{n}</option>{/each}</select></label>
  </div>

  <div class="world-row">
    <label class="toggle"><input type="checkbox" bind:checked={s.useCurrentWorld} disabled={blocked} onchange={() => s.refreshWorld()}>Current world snapshot</label>
    <span class="muted">{s.worldObjectCount} objects</span>
    <button class="compact" onclick={() => s.refreshWorld()} disabled={busy}>Refresh</button>
    {#if s.useCurrentWorld && s.worldError}<span class="error">{s.worldError}</span>{/if}
  </div>

  {#if s.error}<p class="error" role="alert">{s.error}</p>{/if}

  {#if s.runName}
    <div class="run-status"><strong>{s.runTask?.kind === "motion" ? s.runTask.motion.name ?? "Motion target" : s.runTaskInfo?.label ?? s.taskLabel}</strong><span>{s.runWorldLabel}</span><span class="run-name">{s.runName}</span></div>
  {/if}

  {#if s.history.length > 1}
    <svg class="chart" viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img" aria-label="Training reward and upright fraction">
      <path d={path.standing} class="standing" /><path d={path.reward} class="reward" />
    </svg>
    <div class="legend"><span class="key reward">Reward / step</span><span>{path.min.toFixed(2)} to {path.max.toFixed(2)}</span><span class="key standing">{uprightLabel}</span><span>0 to 100%</span><span>Exploring policy</span></div>
  {/if}

  {#if s.last}
    {@const l = s.last}
    <dl class="stats">
      <div><dt>Iteration</dt><dd>{num(l.iteration)} / {num(s.targetIteration)}</dd></div>
      <div><dt>Envs</dt><dd>{s.totalEnvs}</dd></div>
      <div><dt>Steps</dt><dd>{num(l.totalSteps)}</dd></div>
      <div><dt>Reward / step</dt><dd>{l.rewardPerStep.toFixed(3)}</dd></div>
      <div><dt>{uprightLabel}</dt><dd>{(l.standingFraction * 100).toFixed(1)}%</dd></div>
      <div><dt>Episode return</dt><dd>{l.episodeReturn.toFixed(1)}</dd></div>
      <div><dt>Non-finite steps</dt><dd class:error={l.nonFiniteSteps > 0}>{l.nonFiniteSteps}</dd></div>
      <div><dt>KL</dt><dd>{l.approxKl.toFixed(4)}</dd></div>
      <div><dt>Learning rate</dt><dd>{l.lr.toExponential(1)}</dd></div>
      <div><dt>Iteration time</dt><dd>{(l.rolloutMs + l.updateMs).toFixed(0)} ms</dd></div>
      <div><dt>Rollout / update</dt><dd>{l.rolloutMs.toFixed(0)} / {l.updateMs.toFixed(0)} ms</dd></div>
      <div><dt>Forward / backward / opt</dt><dd>{l.fwdMs.toFixed(0)} / {l.bwdMs.toFixed(0)} / {l.optMs.toFixed(0)} ms</dd></div>
    </dl>
    {#if taskMetrics.length}
      <div class="metric-heading"><h4>Measured task metrics</h4>{#if s.runTaskInfo?.experimental}<span class="badge experimental">Experimental / not skill certification</span>{/if}</div>
      <dl class="stats task-metrics">{#each taskMetrics as [name, value] (name)}<div><dt>{metricName(name)}</dt><dd>{Number.isFinite(value) ? value.toFixed(4) : "Invalid"}</dd></div>{/each}</dl>
    {/if}
  {/if}

  <section class="evaluation" aria-label="Checkpoint evaluation">
    <div class="runs-heading"><h4>Skill check</h4><strong class:passed={s.evaluation?.verdict === "passed"} class:failed={s.evaluation?.verdict === "failed"} role="status">{evaluationLabel}</strong></div>
    {#if s.evaluationStatus === "running"}
      <div class="evaluation-progress"><progress aria-label="Evaluation progress" max="1" value={s.evaluationProgress}></progress><span>{Math.round(s.evaluationProgress * 100)}%</span><button class="compact" onclick={() => s.cancelEvaluation()}>Cancel evaluation</button></div>
      <p class="muted run-name">{s.evaluatingName}</p>
    {:else if s.evaluation}
      {@const report = s.evaluation}
      <p class="muted run-name">{report.checkpointName} / iteration {report.iteration}</p>
      <p class="muted">One upright-start replay / {report.seconds.toFixed(1)} seconds / {report.worldObjects} scene objects</p>
      <dl class="stats evaluation-stats">
        <div><dt>Height gain</dt><dd>{((report.jump?.heightGain ?? report.metrics.heightGain) * 100).toFixed(1)} cm</dd></div>
        <div><dt>Upright frames</dt><dd>{(report.metrics.uprightFraction * 100).toFixed(1)}%</dd></div>
        {#if report.jump}
          <div><dt>Contact-free flight</dt><dd>{report.jump.flightSeconds.toFixed(2)} s</dd></div>
          <div><dt>Stable landing</dt><dd>{report.jump.stableLandingSeconds.toFixed(2)} s</dd></div>
        {/if}
      </dl>
      {#if report.reasons.length}<ul class="evaluation-reasons">{#each report.reasons as reason}<li>{reason}</li>{/each}</ul>{/if}
      <div class="evaluation-actions"><button class="compact" disabled={blocked} onclick={() => evaluate(report.checkpointName)}>Evaluate again</button>{#if replayCheckpoint}<button class="compact" disabled={blocked} onclick={() => replay(report.checkpointName)}>Replay checkpoint</button>{/if}</div>
    {:else if s.savedName && !busy}
      <button class="compact" disabled={blocked} onclick={() => evaluate(s.savedName!)}>Evaluate saved checkpoint</button>
    {/if}
    {#if s.evaluationError}<p class="error" role="alert">{s.evaluationError}</p>{/if}
  </section>

  <section class="runs" aria-label="Saved training runs">
    <div class="runs-heading"><h4>Saved runs <span>{s.checkpoints.length}</span></h4><button class="compact" disabled={busy} onclick={() => void s.refreshCheckpoints()}>Refresh</button></div>
    {#if s.checkpoints.length}
      <ul>
        {#each s.checkpoints as c (c.name)}
          <li class:picked={s.resumeFrom === c.name}>
            <label class="checkpoint-choice"><input type="radio" name="resume-checkpoint" checked={s.resumeFrom === c.name} disabled={blocked} onchange={() => void s.selectCheckpoint(c.name)}><span>{c.name}</span></label>
            <span class="meta">{(c.bytes / 1024 ** 2).toFixed(1)} MB / {new Date(c.modified).toLocaleString()}</span>
            <div class="checkpoint-actions"><button class="compact" disabled={blocked} onclick={() => evaluate(c.name)}>Evaluate</button>{#if replayCheckpoint}<button class="compact" disabled={blocked} onclick={() => replay(c.name)}>Replay</button>{/if}<button class="del" disabled={blocked} onclick={() => { if (confirm(`Delete saved run "${c.name}"?`)) void s.remove(c.name); }}>Delete</button></div>
          </li>
        {/each}
      </ul>
      {#if s.resumeFrom}
        <div class="resume-row"><span>{s.resumeSummary ?? s.resumeFrom}</span><button disabled={blocked} onclick={() => void s.start(true)}>Resume as new run</button></div>
      {/if}
    {:else}<p class="muted">No saved runs</p>{/if}
  </section>

  <p class="foot">
    {#if !s.opfs}<span class="error">Checkpoint storage unavailable</span>
    {:else if s.savedName}Saved {s.savedName} / iteration {s.savedAt}
    {:else if s.status === "stopped" && !s.iteration}No iterations completed / no checkpoint saved
    {:else}Per-run autosave / every 25 iterations + stop{/if}
    {#if s.resumedAt} / resumed at {s.resumedAt}{/if}
    {#if s.paramCount} / {s.simd ? "SIMD kernels" : "JavaScript fallback"}{#if s.rolloutLabel} / {s.rolloutLabel}{/if}{/if}
  </p>
</section>

<style>
  .train-panel{display:flex;flex-direction:column;gap:15px;min-width:0;letter-spacing:0}
  .sub,.heading,.buttons,.world-row,.run-status,.runs-heading,.resume-row,.metric-heading{display:flex;align-items:center;gap:10px;flex-wrap:wrap;min-width:0}
  .heading{flex:1}.buttons{gap:6px;margin-left:auto}.runtime,.muted,.meta{color:var(--muted);font-size:11px}.status{color:var(--muted);font-size:11px}.status.active{color:var(--ok)}
  h3{margin:0;font-size:14px;letter-spacing:0}h4{margin:0;font-size:11px;font-weight:650;letter-spacing:0;color:var(--muted)}h4 span{font-weight:400;margin-left:5px}p{margin:0;font-size:12px;line-height:1.5}
  button{font:inherit;font-size:12px;font-weight:600;color:var(--ink);background:var(--panel-hi);border:1px solid var(--line);border-radius:5px;min-height:34px;padding:6px 12px;cursor:pointer;white-space:nowrap}
  button:hover:not(:disabled){border-color:var(--line-hi)}button:disabled{opacity:.4;cursor:not-allowed}button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}button.primary:hover:not(:disabled){background:var(--accent-hi)}button.compact{min-height:28px;padding:3px 8px;font-size:11px}
  button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--focus);outline-offset:3px}
  .setup{display:flex;align-items:flex-end;gap:12px 20px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:14px}.task-picker{flex:1 1 260px;max-width:420px}.task-detail{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:11px;color:var(--muted);min-height:34px}
  label{display:flex;flex-direction:column;gap:6px;font-size:11px;color:var(--muted);min-width:0}select,input[type=number]{font:inherit;font-size:12px;color:var(--ink);background:var(--panel-hi);border:1px solid var(--line);border-radius:5px;padding:7px 9px;min-height:34px;min-width:0;width:100%;letter-spacing:0}select:disabled,input:disabled{opacity:.5}.task-picker select{width:100%}
  .config{display:grid;grid-template-columns:minmax(250px,1.8fr) repeat(2,minmax(115px,1fr));gap:12px 16px}.config>label{min-width:0}.budget{border:0;padding:0;margin:0;min-width:0}.budget legend{font-size:11px;color:var(--muted);margin-bottom:6px;padding:0}.segments{display:flex;border:1px solid var(--line);border-radius:5px;overflow:hidden;min-height:34px}.segments label{position:relative;flex:1;display:flex;flex-direction:row;align-items:center;justify-content:center;gap:4px;padding:7px 6px;color:var(--muted);cursor:pointer;font-size:11px}.segments label+label{border-left:1px solid var(--line)}.segments label.selected{background:var(--panel-hi);color:var(--ink)}.segments input{position:absolute;opacity:0;width:1px;height:1px}.segments label:has(input:focus-visible){outline:2px solid var(--focus);outline-offset:-2px}.segments small{font-size:10px;opacity:.7}.budget:disabled{opacity:.5}
  .world-row{font-size:11px;border-bottom:1px solid var(--line);padding-bottom:14px}.toggle{flex-direction:row;align-items:center;gap:7px;font-size:12px;color:var(--ink)}input[type=checkbox],input[type=radio]{accent-color:var(--control-accent);margin:0;flex:none}.badge{font-size:10px;line-height:1.3;padding:3px 6px;border:1px solid var(--line);border-radius:3px}.experimental{color:var(--warn)}
  .run-status{font-size:11px;color:var(--muted)}.run-status strong{color:var(--ink)}.run-name{overflow-wrap:anywhere;font-family:ui-monospace,monospace;font-size:10px}.chart{width:100%;height:120px;border-bottom:1px solid var(--line)}.chart path{fill:none;stroke-width:1.5;vector-effect:non-scaling-stroke}.chart path.reward{stroke:var(--accent-text)}.chart path.standing{stroke:var(--ok)}.legend{display:flex;gap:7px 13px;align-items:center;flex-wrap:wrap;font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}.key{display:flex;align-items:center;gap:5px}.key::before{content:"";width:10px;height:2px}.key.reward::before{background:var(--accent-text)}.key.standing::before{background:var(--ok)}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:12px 18px;margin:0;padding:12px 0;font-size:12px}.stats div{display:flex;flex-direction:column;gap:4px;min-width:0}.stats dt{font-size:10px;color:var(--muted);letter-spacing:0;text-transform:none}.stats dd{margin:0;font-weight:600;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.task-metrics{padding-top:0}.metric-heading{border-top:1px solid var(--line);padding-top:12px}
  .runs{display:flex;flex-direction:column;gap:8px;border-top:1px solid var(--line);padding-top:14px}.runs-heading{justify-content:space-between}.runs ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;max-height:270px;overflow:auto}.runs li{display:flex;align-items:center;gap:10px;padding:8px 5px;border-bottom:1px solid var(--line);font-size:11px;min-width:0}.runs li.picked{background:var(--panel)}.checkpoint-choice{flex:1;display:flex;flex-direction:row;align-items:center;gap:8px;color:var(--ink);min-width:0}.checkpoint-choice span{overflow-wrap:anywhere}.meta{font-size:10px;font-variant-numeric:tabular-nums}.del{padding:2px 7px;min-height:27px;font-size:10px;color:var(--muted);background:none}.del:hover:not(:disabled){color:var(--danger);border-color:var(--danger)}.resume-row{justify-content:space-between;font-size:11px;color:var(--muted)}.resume-row span{overflow-wrap:anywhere}.error{color:var(--danger);font-size:12px;overflow-wrap:anywhere}.foot{font-size:10px;line-height:1.6;color:var(--muted);overflow-wrap:anywhere}
  .evaluation{display:flex;flex-direction:column;gap:10px;border-top:1px solid var(--line);padding-top:14px;min-width:0}.evaluation strong{font-size:12px}.passed{color:var(--ok)}.failed{color:var(--warn)}.evaluation-progress,.evaluation-actions,.checkpoint-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-width:0}.evaluation-progress{font-size:11px;color:var(--muted)}progress{accent-color:var(--control-accent);min-width:100px;max-width:100%;flex:1}.evaluation-reasons{margin:0;padding-left:18px;color:var(--muted);font-size:11px;line-height:1.6}.evaluation-stats{padding:0}.checkpoint-actions{margin-left:auto}
  @media(min-width:1300px){.config{grid-template-columns:minmax(270px,1.8fr) repeat(5,minmax(95px,1fr))}}
  @media(max-width:700px){.config{grid-template-columns:repeat(2,minmax(0,1fr))}.budget{grid-column:1/-1}.heading{flex-basis:100%}.buttons{margin-left:auto}.sub{gap:8px}.task-picker{max-width:none;flex-basis:100%}.task-detail{min-height:0}.setup{gap:9px}.runs li{flex-wrap:wrap}.checkpoint-choice{flex-basis:calc(100% - 68px)}.meta{order:3;margin-left:21px}.stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.runtime{font-size:10px}}
</style>
