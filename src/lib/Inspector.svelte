<script lang="ts">
  import type { Session } from "./session.svelte";
  import { objectCatalogue, worldPresets, MAX_WORLD_OBJECTS, type ObjectKind, type WorldObject } from "../sim/world.ts";
  const { session }: { session: Session } = $props();
  let tab = $state<"robot" | "scene">("robot");
  let kind = $state<ObjectKind>("ball");
  let preset = $state("playground");
  let fileInput = $state<HTMLInputElement>();
  let draft = $state<WorldObject | null>(null);
  const blocked = $derived(!session.ready || session.worldBusy || session.policyBusy);
  $effect(() => {
    const selected = session.object;
    draft = selected ? JSON.parse(JSON.stringify(selected)) : null;
  });
  function resize(axis: number, value: number): void {
    if (!draft) return;
    draft.size[axis] = value;
    if (draft.kind === "ball") draft.size = [value, value, value];
    else if (draft.kind === "cylinder" && axis < 2) { draft.size[0] = value; draft.size[1] = value; }
  }
  const swatches = [[0.1, 0.83, 0.9, 1], [1, 0.36, 0.31, 1], [0.23, 0.83, 0.5, 1], [1, 0.78, 0.23, 1], [0.56, 0.6, 0.64, 1]] as const;
</script>

<aside class="inspector">
  <div class="tabs" role="tablist" aria-label="Inspector">
    <button role="tab" aria-selected={tab === "robot"} onclick={() => tab = "robot"}>Robot</button>
    <button role="tab" aria-selected={tab === "scene"} onclick={() => tab = "scene"}>Scene <span>{session.world.objects.length}</span></button>
  </div>
  <div class="content">
    {#if tab === "robot"}
      <section>
        <h2>Driving policy</h2>
        <select aria-label="Driving policy" value={session.activePolicy} disabled={blocked} onchange={e => void session.selectPolicy(e.currentTarget.value)}>
          {#each session.policies as p (p.id)}
            <option value={p.id}>{p.label}{p.kind === "checkpoint" ? " (saved training scene)" : ""}</option>
          {/each}
        </select>
        <p class="muted">{session.policyDescription}</p>
        {#if session.selectedCheckpointName}
          <p class="muted">Scene: {session.checkpointSceneMatches ? "saved training scene" : "modified scene"} · {session.paused ? "paused" : "running"}</p>
          <button disabled={blocked} onclick={() => session.selectedCheckpointName && void session.replayCheckpoint(session.selectedCheckpointName)}>Restart in training scene</button>
        {/if}
        {#if session.policyError}<p class="error" role="alert">{session.policyError}</p>{/if}
      </section>

      {#if session.commandEnabled}
        <section>
          <h2>Velocity command</h2>
          <label class="slider">Forward <output>{session.forward.toFixed(2)} m/s</output>
            <input aria-label="Forward velocity" type="range" min="-0.25" max="0.25" step="0.01" value={session.forward} oninput={e => { session.forward = e.currentTarget.valueAsNumber; session.setCommands(); }} disabled={blocked}/>
          </label>
          <label class="slider">Sideways <output>{session.lateral.toFixed(2)} m/s</output>
            <input aria-label="Lateral velocity" type="range" min="-0.15" max="0.15" step="0.01" value={session.lateral} oninput={e => { session.lateral = e.currentTarget.valueAsNumber; session.setCommands(); }} disabled={blocked}/>
          </label>
          <label class="slider">Turn left <output>{session.turn.toFixed(2)} rad/s</output>
            <input aria-label="Turn velocity" type="range" min="-1" max="1" step="0.05" value={session.turn} oninput={e => { session.turn = e.currentTarget.valueAsNumber; session.setCommands(); }} disabled={blocked}/>
          </label>
          <div class="row">
            <button disabled={blocked} onclick={() => { session.forward = 0.25; session.lateral = 0; session.turn = 0; session.setCommands(); }}>Walk forward</button>
            <button disabled={blocked} onclick={() => session.stopMoving()}>Stop moving</button>
          </div>
        </section>
      {/if}

      <section>
        <h2>Recovery</h2>
        <button disabled={blocked} onclick={() => session.knockDown()}>Knock down &amp; stand up</button>
        <div class="row">
          <button disabled={blocked} onclick={() => session.push()}>Push</button>
          <button disabled={blocked} onclick={() => session.standUp()}>Stand up</button>
          <button class="icon" aria-label="Reset" title="Reset robot and objects" disabled={blocked} onclick={() => session.reset()}>↺</button>
        </div>
        <label class="toggle"><input type="checkbox" checked={session.autoRepeat} onchange={e => session.setAutoRepeat(e.currentTarget.checked)} disabled={blocked}/> Repeat knockdowns</label>
      </section>
      <section>
        <h2>Simulation</h2>
        <label class="toggle"><input type="checkbox" checked={session.follow} onchange={e => session.setFollow(e.currentTarget.checked)}/> Follow robot</label>
        <label class="toggle"><input type="checkbox" bind:checked={session.showCollision} disabled={blocked}/> Collision geometry</label>
        <label class="field">Playback speed
          <select aria-label="Playback speed" bind:value={session.speed}><option value={0.25}>0.25x</option><option value={0.5}>0.5x</option><option value={1}>1x</option></select>
        </label>
        <dl><div><dt>Physics / control</dt><dd>200 / 50 Hz</dd></div><div><dt>Backend</dt><dd>MuJoCo WASM · CPU</dd></div></dl>
      </section>
    {:else}
      <section>
        <h2>Scene kit</h2>
        <div class="row"><select aria-label="Scene preset" bind:value={preset} disabled={blocked}>
          {#each worldPresets as p (p.id)}<option value={p.id}>{p.label}</option>{/each}
        </select><button disabled={blocked} onclick={() => void session.loadPreset(preset)}>Load</button></div>
        <div class="row"><button onclick={() => session.exportWorld()} disabled={!session.ready}>Export JSON</button><button onclick={() => fileInput?.click()} disabled={blocked}>Import JSON</button></div>
        <input class="file" type="file" accept="application/json,.json" bind:this={fileInput} onchange={e => { const file = e.currentTarget.files?.[0]; if (file) void session.importWorld(file); e.currentTarget.value = ""; }}/>
      </section>
      <section>
        <h2>Object library <span>{session.world.objects.length}/{MAX_WORLD_OBJECTS}</span></h2>
        <div class="row"><select aria-label="Object type" bind:value={kind} disabled={blocked}>
          {#each objectCatalogue as item (item.kind)}<option value={item.kind}>{item.label}</option>{/each}
        </select><button class="primary" disabled={blocked || session.world.objects.length >= MAX_WORLD_OBJECTS} onclick={() => void session.addObject(kind)}>Add object</button></div>
        {#if session.world.objects.length}
          <select aria-label="Selected object" bind:value={session.selectedObject} disabled={blocked}>
            <option value="">Select an object</option>
            {#each session.world.objects as o (o.id)}<option value={o.id}>{o.id} · {o.dynamic ? "movable" : "fixed"}</option>{/each}
          </select>
        {/if}
      </section>
      {#if draft}
        <section>
          <h2>{draft.kind} <span>{session.selectedSpeed.toFixed(2)} m/s</span></h2>
          <fieldset disabled={blocked}>
            <legend>Position · meters</legend>
            <div class="triple">{#each ["X", "Y", "Z"] as axis, i}
              <label>{axis}<input aria-label={`Object position ${axis}`} type="number" step="0.05" min={i === 2 ? 0 : -5} max={i === 2 ? 3 : 5} bind:value={draft.position[i]}/></label>
            {/each}</div>
          </fieldset>
          <fieldset disabled={blocked}>
            <legend>Size · meters</legend>
            <div class="triple">{#each ["X", "Y", "Z"] as axis, i}
              <label>{axis}<input aria-label={`Object size ${axis}`} type="number" step="0.01" min="0.02" max="2" value={draft.size[i]} oninput={e => resize(i, e.currentTarget.valueAsNumber)}/></label>
            {/each}</div>
          </fieldset>
          <label class="field">Yaw · degrees<input aria-label="Object yaw" type="number" step="15" min="-360" max="360" value={Math.round(draft.yaw * 180 / Math.PI)} onchange={e => { if (draft) draft.yaw = e.currentTarget.valueAsNumber * Math.PI / 180; }}/></label>
          {#if objectCatalogue.find(c => c.kind === draft?.kind)?.dynamic}
            <label class="toggle"><input type="checkbox" bind:checked={draft.dynamic} disabled={blocked}/> Movable body</label>
            <label class="field">Mass · kg<input aria-label="Object mass" type="number" step="0.01" min="0.01" max="20" bind:value={draft.mass} disabled={!draft.dynamic || blocked}/></label>
          {/if}
          {#if draft.kind === "steps"}
            <label class="field">Step count<input aria-label="Step count" type="number" min="2" max="8" step="1" bind:value={draft.steps}/></label>
          {/if}
          <div class="swatches" role="group" aria-label="Object color">{#each swatches as color, i}
            <button class="swatch" aria-label={`Object color ${i + 1}`} title={["Cyan", "Coral", "Green", "Yellow", "Gray"][i]} aria-pressed={draft.color.every((c, j) => c === color[j])} style:background={`rgb(${color.slice(0, 3).map(c => c * 255).join(",")})`} onclick={() => { if (draft) draft.color = [...color]; }}></button>
          {/each}</div>
          <button class="primary" disabled={blocked} onclick={() => draft && void session.updateObject(draft)}>Apply &amp; reset scene</button>
          <div class="row"><button disabled={blocked || !draft.dynamic} onclick={() => session.nudgeObject()}>Nudge object</button><button disabled={blocked || session.world.objects.length >= MAX_WORLD_OBJECTS} onclick={() => void session.duplicateObject()}>Duplicate</button><button class="icon danger" aria-label="Delete object" title="Delete selected object" disabled={blocked} onclick={() => void session.removeObject()}>×</button></div>
        </section>
      {/if}
    {/if}
    {#if session.worldError}<p class="error" role="alert">{session.worldError}</p>{/if}
  </div>
</aside>

<style>
  .inspector { grid-area: aside; display: flex; flex-direction: column; min-height: 0; min-width: 0; background: var(--panel); border-left: 1px solid var(--line); font-size: 12px; }
  .tabs { display: grid; grid-template-columns: 1fr 1fr; flex: none; border-bottom: 1px solid var(--line); padding: 6px; gap: 4px; }
  .tabs button { border-color: transparent; background: transparent; padding: 8px; }
  .tabs button[aria-selected="true"] { background: var(--panel-hi); border-color: var(--line-hi); color: var(--ink); }
  .tabs span, h2 span { color: var(--muted); font-weight: 500; }
  .content { padding: 14px; overflow: auto; min-height: 0; display: flex; flex-direction: column; gap: 22px; }
  section { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
  h2 { display: flex; justify-content: space-between; margin: 0; font-size: 11px; font-weight: 650; letter-spacing: 0; color: var(--muted-hi); }
  button, select, input[type="number"] { font: inherit; color: var(--ink); background: var(--panel-hi); border: 1px solid var(--line); border-radius: 5px; min-width: 0; }
  button { min-height: 32px; padding: 6px 8px; cursor: pointer; font-size: 11px; font-weight: 600; }
  button:hover:not(:disabled) { border-color: var(--line-hi); }
  button:disabled, select:disabled { opacity: 0.4; cursor: not-allowed; }
  .primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
  select { width: 100%; padding: 7px; font-size: 11px; }
  input[type="number"] { width: 78px; padding: 6px; font-size: 11px; }
  input[type="range"] { width: 100%; margin: 4px 0 0; accent-color: var(--control-accent); }
  input[type="checkbox"] { accent-color: var(--control-accent); margin: 0; }
  .row { display: flex; gap: 6px; align-items: stretch; }
  .row > :not(.icon) { flex: 1; min-width: 0; }
  .icon { width: 32px; height: 32px; padding: 0; font-size: 20px; flex: none; }
  .danger { color: var(--danger); }
  .toggle { display: flex; gap: 8px; align-items: center; font-size: 11px; }
  .field { display: flex; justify-content: space-between; gap: 8px; align-items: center; color: var(--muted); }
  .field select { width: 80px; }
  .slider { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px; font-size: 11px; color: var(--muted); }
  output { font-variant-numeric: tabular-nums; color: var(--ink); }
  .muted, .error { margin: 0; color: var(--muted); font-size: 10px; line-height: 1.5; overflow-wrap: anywhere; }
  .error { color: var(--danger); }
  dl { margin: 0; display: grid; gap: 7px; font-size: 10px; }
  dl div { display: flex; justify-content: space-between; gap: 8px; } dt { color: var(--muted); } dd { margin: 0; }
  fieldset { margin: 0; padding: 0; border: 0; min-width: 0; }
  legend { margin-bottom: 7px; color: var(--muted); font-size: 10px; }
  .triple { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; }
  .triple label { display: flex; flex-direction: column; gap: 4px; font-size: 10px; color: var(--muted); }
  .triple input { width: 100%; }
  .swatches { display: flex; gap: 8px; }
  .swatch { width: 25px; height: 25px; min-height: 25px; padding: 0; border: 2px solid transparent; border-radius: 50%; }
  .swatch[aria-pressed="true"] { outline: 2px solid var(--ink); outline-offset: 2px; }
  .file { display: none; }
  @media (max-width: 1040px) { .inspector { border-left: 0; border-top: 1px solid var(--line); } .content { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; } }
  @media (max-width: 480px) { .content { display: flex; padding: 12px; } }
</style>
