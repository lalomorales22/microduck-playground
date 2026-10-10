<script lang="ts">
  // The instruments: raw throughput and a bare environment rollout. Neither
  // trains anything — they exist to answer "is this machine fast enough" and
  // "does the environment still step correctly", which is why they are here
  // rather than cluttering the Train workspace.
  import { Bench, PLAN_ESTIMATE } from "./bench.svelte";

  const bench = new Bench();

  const mb = (bytes: number) => `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  const shortModel = (xml: string) => xml.replace("robot_", "").replace(".xml", "");
  const rate = (n: number) => Math.round(n).toLocaleString();
</script>

<div class="debug">
  <section>
    <header>
      <div>
        <h2>Throughput harness</h2>
        <p>
          How fast can this machine step the duck? Every cell spins up a fresh
          worker pool, allocates its environments, and steps them all at once —
          what matters is what the machine sustains with every core busy, not
          one worker in isolation.
        </p>
      </div>
      <div class="actions">
        {#if bench.running}
          <button onclick={() => bench.stop()}>Stop</button>
        {:else}
          <button class="primary" onclick={() => void bench.run()}>Run sweep</button>
        {/if}
      </div>
    </header>

    {#if bench.crashed.length}
      <div class="warn">
        <strong>{bench.crashed.length} configuration{bench.crashed.length === 1 ? "" : "s"}
        crashed the tab</strong> and will be skipped:
        {bench.crashed.map((c) => c.label).join(", ")}.
        A big pool can exhaust memory badly enough to kill the renderer, which
        the page cannot catch — so each cell is written down before it runs and
        one with no completion is treated as fatal.
        <button class="link" onclick={() => bench.forgetCrashes()}>Try them again</button>
      </div>
    {/if}

    <div class="config">
      <fieldset>
        <legend>Model</legend>
        {#each bench.allModels as m (m)}
          <label>
            <input
              type="checkbox"
              checked={bench.models.includes(m)}
              disabled={bench.running}
              onchange={() => (bench.models = bench.toggle(bench.models, m))}
            />
            {shortModel(m)}
          </label>
        {/each}
      </fieldset>

      <fieldset>
        <legend>Envs / worker</legend>
        {#each [1, 16, 64, 128] as n (n)}
          <label>
            <input
              type="checkbox"
              checked={bench.envsPerWorker.includes(n)}
              disabled={bench.running}
              onchange={() => (bench.envsPerWorker = bench.toggle(bench.envsPerWorker, n))}
            />
            {n}
          </label>
        {/each}
      </fieldset>

      <fieldset>
        <legend>Workers <small>({bench.cores} cores)</small></legend>
        {#each bench.allWorkerCounts as n (n)}
          <label>
            <input
              type="checkbox"
              checked={bench.workerCounts.includes(n)}
              disabled={bench.running}
              onchange={() => (bench.workerCounts = bench.toggle(bench.workerCounts, n))}
            />
            {n}
          </label>
        {/each}
      </fieldset>

      <fieldset>
        <legend>Options</legend>
        <label>
          <input type="checkbox" bind:checked={bench.withPolicy} disabled={bench.running} />
          Policy forward
        </label>
        <label>
          <input type="checkbox" bind:checked={bench.capMemory} disabled={bench.running} />
          Cap arena (1M)
        </label>
        <label>
          Window
          <select bind:value={bench.durationMs} disabled={bench.running}>
            <option value={1000}>1 s</option>
            <option value={2000}>2 s</option>
            <option value={5000}>5 s</option>
          </select>
        </label>
      </fieldset>
    </div>

    {#if bench.error}<p class="error">{bench.error}</p>{/if}

    {#if bench.running || bench.results.length}
      <div class="progress">
        <div style:width="{bench.progress.total ? (bench.progress.done / bench.progress.total) * 100 : 0}%"></div>
      </div>
    {/if}

    {#if bench.results.length}
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Workers</th><th>Envs/w</th><th>Envs</th>
            <th>Control steps/s</th><th>× realtime</th><th>MB/env</th><th>Heap</th>
          </tr>
        </thead>
        <tbody>
          {#each bench.results as r, i (i)}
            <tr class:best={r === bench.best} class:bad={!!r.error || r.oom}>
              <td>{shortModel(r.robotXml)}</td>
              <td class="n">{r.workers}</td>
              <td class="n">{r.envsPerWorker}</td>
              <td class="n">{r.envs}</td>
              <td class="n strong">
                {#if r.error}<span class="err">{r.error}</span>{:else}{rate(r.controlStepsPerSec)}{/if}
              </td>
              <td class="n">{r.error ? "" : `${r.realtimeFactor.toFixed(0)}×`}</td>
              <td class="n">{r.error ? "" : (r.bytesPerEnv / 1024 ** 2).toFixed(1)}</td>
              <td class="n">{r.error ? "" : mb(r.heapTotalBytes)}{r.oom ? " ⚠" : ""}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}

    {#if bench.best && !bench.running}
      {@const best = bench.best}
      {@const hours = bench.referenceHours ?? 0}
      <div class="verdict" class:pass={best.controlStepsPerSec >= PLAN_ESTIMATE}>
        <strong>{rate(best.controlStepsPerSec)} control steps/s</strong>
        — {shortModel(best.robotXml)}, {best.workers} workers × {best.envsPerWorker} envs.
        The plan assumed {rate(PLAN_ESTIMATE)}; the full reference recipe
        (1.47 B steps) would take <strong>{hours.toFixed(1)} h</strong>,
        a 2,000-iteration run <strong>{((hours * 60 * 2000) / 15000).toFixed(0)} min</strong>.
      </div>
    {/if}
  </section>

  <section class="env">
    <header>
      <div>
        <h2>Environment rollout</h2>
        <p>
          The real training env — reset, reward, termination — in a worker pool,
          driven by a randomly initialised policy. The reward here is a floor,
          not a result; whether the reward stack ranks good behaviour above bad
          is settled by <code>npm run check:env</code>.
        </p>
      </div>
      <div class="actions">
        <button disabled={bench.envRunning} onclick={() => void bench.runEnv()}>
          {bench.envRunning ? "Rolling out…" : "Run rollout"}
        </button>
      </div>
    </header>

    {#if bench.envError}
      <p class="error">{bench.envError}</p>
    {:else if bench.envStats}
      {@const s = bench.envStats}
      <dl class="stats">
        <div><dt>Envs</dt><dd>{s.envs}</dd></div>
        <div><dt>Control steps</dt><dd>{s.controlSteps.toLocaleString()}</dd></div>
        <div><dt>Throughput</dt><dd>{rate(s.stepsPerSec)}/s</dd></div>
        <div><dt>× realtime</dt><dd>{s.realtimeFactor.toFixed(0)}×</dd></div>
        <div><dt>Episodes</dt><dd>{s.episodes}</dd></div>
        <div><dt>Reward / step</dt><dd>{s.rewardPerStep.toFixed(3)}</dd></div>
        <div><dt>Standing</dt><dd>{(s.standingFraction * 100).toFixed(1)}%</dd></div>
      </dl>
      <table class="terms">
        <tbody>
          {#each Object.entries(s.breakdown).sort((a, b) => b[1] - a[1]) as [name, value] (name)}
            <tr><td>{name}</td><td class="n" class:neg={value < 0}>{value.toFixed(4)}</td></tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
</div>

<style>
  .debug {
    grid-area: main;
    display: flex; flex-direction: column; gap: 24px;
    padding: 18px 22px; overflow-y: auto; background: var(--viewport);
  }
  section { display: flex; flex-direction: column; gap: 12px; }
  .env { padding-top: 20px; border-top: 1px solid var(--line); }

  header { display: flex; align-items: flex-start; gap: 20px; }
  header > div:first-child { flex: 1; min-width: 0; }
  h2 { margin: 0; font-size: 14px; letter-spacing: 0; }
  p { margin: 6px 0 0; font-size: 12px; line-height: 1.55; color: var(--muted); max-width: 78ch; }
  code { background: var(--panel-hi); padding: 1px 4px; border-radius: 4px; font-size: 0.92em; }

  button {
    font: inherit; font-size: 12px; font-weight: 600;
    color: var(--ink); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 7px;
    padding: 8px 14px; cursor: pointer; white-space: nowrap;
  }
  button:hover { border-color: var(--line-hi); }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button.primary:hover { background: var(--accent-hi); }
  button.link {
    padding: 0; border: 0; background: none; font-size: inherit; font-weight: 600;
    color: var(--accent-text); text-decoration: underline;
  }

  .warn {
    padding: 10px 12px; border-radius: 8px; font-size: 11px; line-height: 1.6;
    color: var(--muted); background: var(--panel);
    border: 1px solid color-mix(in srgb, var(--warn) 45%, var(--line));
  }
  .warn strong { color: var(--warn); }

  .config { display: flex; flex-wrap: wrap; gap: 10px; }
  fieldset {
    display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center;
    margin: 0; padding: 8px 12px 9px;
    border: 1px solid var(--line); border-radius: 8px; background: var(--panel);
  }
  legend {
    padding: 0 4px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0; color: var(--muted);
  }
  legend small { font-weight: 500; letter-spacing: 0; text-transform: none; }
  fieldset label { display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer; }
  fieldset input { accent-color: var(--control-accent); cursor: pointer; }
  select {
    font: inherit; font-size: 11px; color: var(--ink);
    background: var(--panel-hi); border: 1px solid var(--line);
    border-radius: 5px; padding: 1px 4px;
  }

  .progress { height: 2px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--accent); transition: width 0.2s ease; }

  table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 5px 12px 5px 0; text-align: left; white-space: nowrap; }
  th {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0; color: var(--muted); border-bottom: 1px solid var(--line);
  }
  td { border-bottom: 1px solid var(--line); }
  .n { text-align: right; }
  .strong { font-weight: 700; }
  tr.best td { color: var(--accent-text); }
  tr.bad td { color: var(--muted); }
  .err { color: var(--danger); font-weight: 400; }

  .verdict {
    padding: 11px 13px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel);
    font-size: 12px; line-height: 1.6; color: var(--muted); max-width: 78ch;
  }
  .verdict strong { color: var(--ink); }
  .verdict.pass { border-color: color-mix(in srgb, var(--ok) 45%, var(--line)); }
  .error { color: var(--danger); font-size: 12px; }

  .stats { display: flex; flex-wrap: wrap; gap: 6px 22px; margin: 0; font-size: 12px; }
  .stats div { display: flex; flex-direction: column; gap: 2px; }
  .stats dt { font-size: 10px; text-transform: uppercase; letter-spacing: 0; color: var(--muted); }
  .stats dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }

  .terms { font-size: 11px; }
  .terms td { padding: 2px 16px 2px 0; border: 0; color: var(--muted); }
  .terms td.n { text-align: right; color: var(--ink); }
  .terms td.neg { color: var(--warn); }
</style>
