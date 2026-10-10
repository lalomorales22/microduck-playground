<script lang="ts">
  import { onMount } from "svelte";
  import { addKey, closeLoop, moveKey, removeKey, setDuration } from "../motion/edit.ts";
  import type { MakerSession } from "./maker.svelte";

  const { session }: { session: MakerSession } = $props();

  let canvas: HTMLCanvasElement;
  let track = $state<HTMLDivElement | null>(null);
  let jsonText = $state("");
  /** What a pointer drag on the timeline is doing: scrubbing, or moving a key. */
  let dragging = $state<{ kind: "scrub" } | { kind: "key"; index: number } | null>(null);

  onMount(() => {
    void session.start(canvas);
    return () => session.destroy();
  });

  const dur = $derived(Math.max(session.duration, 0.001));
  const pct = (t: number) => `${(t / dur) * 100}%`;

  /** Pointer x -> time, measured on the LANE, which is the element the
   *  keyframes are positioned in — measuring the padded box instead puts every
   *  click a few pixels out and the end key outside its own timeline. */
  function timeAt(e: PointerEvent): number {
    if (!track) return 0;
    const rect = track.getBoundingClientRect();
    const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width);
    return rect.width > 0 ? (x / rect.width) * dur : 0;
  }

  function down(e: PointerEvent, kind: "scrub" | number): void {
    track?.setPointerCapture(e.pointerId);
    if (kind === "scrub") {
      dragging = { kind: "scrub" };
      session.seek(timeAt(e));
    } else {
      // Clicking a key selects it AND parks the playhead on it, so the pose
      // panel is immediately editing what was clicked.
      dragging = { kind: "key", index: kind };
      session.seek(session.draft.keys[kind].t);
    }
  }

  function move(e: PointerEvent): void {
    if (!dragging) return;
    if (dragging.kind === "scrub") {
      session.seek(timeAt(e));
      return;
    }
    // Key 0 is pinned at t = 0: it is the motion's origin, and letting it
    // slide would mean the file no longer starts where it says it does.
    if (dragging.index === 0) return;
    const at = moveKey(session.draft, dragging.index, timeAt(e));
    session.seek(session.draft.keys[at].t);
    dragging = { kind: "key", index: at };
  }

  function up(e: PointerEvent): void {
    track?.releasePointerCapture(e.pointerId);
    dragging = null;
  }

  function addAtPlayhead(): void {
    session.selected = addKey(session.draft, session.time, session.poseAtPlayhead());
    session.refresh();
  }

  function deleteSelected(): void {
    if (removeKey(session.draft, session.selected)) session.seek(session.time);
  }

  function saveAs(): void {
    const raw = globalThis.prompt?.("Save this motion as:", session.savedAs ?? session.draft.name);
    if (!raw) return;
    void session.save(raw);
  }

  const clock = $derived(`${session.time.toFixed(2)} / ${session.duration.toFixed(2)} s`);
</script>

<div class="maker">
  <div class="toolbar">
    <input class="title" bind:value={session.draft.name} placeholder="untitled" aria-label="Motion name" />

    <select
      value=""
      onchange={(e) => { void session.open(e.currentTarget.value); e.currentTarget.value = ""; }}
      aria-label="Open a motion"
    >
      <option value="" disabled>Open…</option>
      {#each session.library as m (m.id)}
        <option value={m.id}>{m.name}{m.source === "builtin" ? " (built-in)" : ""}</option>
      {/each}
    </select>

    <button onclick={() => session.reset()}>New</button>
    <button class="primary" onclick={() => void session.save()}>Save</button>
    <button onclick={saveAs}>Save as…</button>

    <span class="gap"></span>

    <label class="check">
      <input type="checkbox" bind:checked={session.draft.loop} onchange={() => session.refresh()} />
      Loop
    </label>
    <label class="field">
      Ease
      <select bind:value={session.draft.interp} onchange={() => session.refresh()}>
        <option value="cubic">Smooth</option>
        <option value="linear">Linear</option>
      </select>
    </label>
    <label class="field">
      Length
      <input
        type="number" min="0.2" max="30" step="0.1" value={session.duration.toFixed(2)}
        onchange={(e) => { setDuration(session.draft, Number(e.currentTarget.value)); session.seek(0); }}
      />s
    </label>
  </div>

  <div class="stage">
    <canvas bind:this={canvas}></canvas>
    {#if session.error}
      <div class="overlay error"><strong>Failed to start</strong><span>{session.error}</span></div>
    {:else if !session.ready}
      <div class="overlay">
        <div class="spinner"></div>
        <span>{session.loadStage}</span>
        <div class="bar"><div style:width="{session.loadProgress}%"></div></div>
      </div>
    {/if}
  </div>

  <div class="transport">
    <button
      class="primary"
      onclick={() => (session.playing ? session.pause() : session.play())}
      disabled={!session.ready}
    >{session.playing ? "Pause" : "Play"}</button>
    <button onclick={() => session.seek(0)} disabled={!session.ready}>Rewind</button>
    <span class="clock">{clock}</span>

    <select value={session.mode} onchange={(e) => session.setMode(e.currentTarget.value as "preview" | "physics")}>
      <option value="preview">Preview — no physics</option>
      <option value="physics">Physics — open loop</option>
    </select>

    <span class="gap"></span>

    <button onclick={addAtPlayhead} disabled={!session.ready || session.selected >= 0}>
      Add keyframe
    </button>
    <button
      onclick={deleteSelected}
      disabled={session.selected <= 0 || session.selected >= session.draft.keys.length - 1}
    >Delete keyframe</button>
  </div>

  <!-- The timeline. Click or drag the track to scrub; drag a diamond to move
       that keyframe in time. -->
  <div
    class="timeline"
    bind:this={track}
    onpointerdown={(e) => down(e, "scrub")}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
    role="slider"
    tabindex="0"
    aria-label="Motion timeline"
    aria-valuemin="0"
    aria-valuemax={session.duration}
    aria-valuenow={session.time}
  >
    <div class="track"></div>
    <div class="head" style:left={pct(session.time)}></div>
    {#each session.draft.keys as k, i (i)}
      <button
        class="key"
        class:sel={session.selected === i}
        class:pinned={i === 0}
        style:left={pct(k.t)}
        onpointerdown={(e) => { e.stopPropagation(); down(e, i); }}
        title="{k.t.toFixed(2)} s{i === 0 ? ' — pinned to the start' : ''}"
        aria-label="Keyframe at {k.t.toFixed(2)} seconds"
      ></button>
    {/each}
  </div>

  <div class="foot">
    {#if session.saveError}
      <p class="error">{session.saveError}</p>
    {:else if session.problem}
      <p class="warn">
        Not saveable yet: {session.problem}
        {#if session.draft.loop && session.problem.includes("repeat the first")}
          <button class="inline" onclick={() => { closeLoop(session.draft); session.refresh(); }}>
            Close the loop
          </button>
        {/if}
      </p>
    {:else if session.notice}
      <p class="ok">{session.notice}</p>
    {:else}
      <p>
        {session.mode === "preview"
          ? "Preview places the joints exactly as written — no physics, no falling."
          : "Open loop against gravity. It will fall: standing is an active behaviour on this robot, not a pose. Training a policy on the motion is what makes it survivable."}
      </p>
    {/if}

    <details class="io">
      <summary>JSON</summary>
      <p class="io-hint">
        Hand <code>docs/motion-format.md</code> to an AI agent, paste what it
        writes back here, and press Load.
      </p>
      <textarea
        bind:value={jsonText}
        placeholder={"{\n  \"name\": \"bow\",\n  \"joints\": [\"neck_pitch\"],\n  \"keyframes\": [ ... ]\n}"}
        spellcheck="false"
      ></textarea>
      <div class="io-buttons">
        <button onclick={() => session.loadJson(jsonText)} disabled={!jsonText.trim()}>Load</button>
        <button onclick={() => (jsonText = session.json())}>Copy this motion in</button>
      </div>
    </details>
  </div>
</div>

<style>
  .maker {
    grid-area: main;
    display: grid; grid-template-rows: auto minmax(220px, 1fr) auto auto auto;
    min-height: 0; min-width: 0; overflow: auto; background: var(--viewport);
  }

  .toolbar, .transport {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; background: var(--panel);
    border-bottom: 1px solid var(--line); flex-wrap: wrap;
  }
  .transport { border-top: 1px solid var(--line); border-bottom: 0; }
  .gap { flex: 1; }

  .title {
    font: inherit; font-size: 13px; font-weight: 700; color: var(--ink);
    background: none; border: 1px solid transparent; border-radius: 6px;
    padding: 4px 6px; min-width: 120px; max-width: 220px;
  }
  .title:hover, .title:focus { border-color: var(--line); background: var(--panel-hi); outline: none; }

  button {
    font: inherit; font-size: 11px; font-weight: 600;
    color: var(--ink); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 6px;
    padding: 5px 10px; cursor: pointer; white-space: nowrap;
  }
  button:hover:not(:disabled) { border-color: var(--line-hi); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button.primary:hover:not(:disabled) { background: var(--accent-hi); }

  select, input[type="number"] {
    font: inherit; font-size: 11px; color: var(--ink);
    background: var(--panel-hi); border: 1px solid var(--line);
    border-radius: 6px; padding: 4px 6px;
  }
  input[type="number"] { width: 60px; }
  .field, .check {
    display: flex; align-items: center; gap: 5px;
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0; color: var(--muted); white-space: nowrap;
  }
  .check input { accent-color: var(--control-accent); cursor: pointer; }
  .clock { font-size: 11px; font-variant-numeric: tabular-nums; color: var(--muted); }

  .stage { position: relative; min-height: 0; }
  canvas { display: block; width: 100%; height: 100%; }
  .overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px;
    background: color-mix(in srgb, var(--viewport) 92%, transparent);
    font-size: 12px; color: var(--muted); text-align: center;
  }
  .spinner {
    width: 24px; height: 24px; border-radius: 50%;
    border: 2px solid var(--line); border-top-color: var(--accent);
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .bar { width: 200px; height: 2px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .bar > div { height: 100%; background: var(--accent); transition: width 0.15s ease; }
  .overlay.error { color: var(--danger); }

  /* The lane is inset from the box so a keyframe at t = 0 or t = duration
     sits fully inside its own timeline rather than straddling the border. */
  .timeline {
    position: relative; height: 42px; margin: 0 22px;
    cursor: pointer; touch-action: none;
  }
  .timeline::before {
    content: ""; position: absolute; inset: 0 -10px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  }
  .track {
    position: absolute; left: 0; right: 0; top: 50%;
    height: 2px; margin-top: -1px; background: var(--line-hi); border-radius: 1px;
  }
  .head {
    position: absolute; top: 4px; bottom: 4px; width: 2px;
    margin-left: -1px; background: var(--accent-text); border-radius: 1px;
  }
  .key {
    position: absolute; top: 50%; width: 11px; height: 11px; padding: 0;
    margin: -5.5px 0 0 -5.5px; border-radius: 2px;
    background: var(--panel-hi); border: 1px solid var(--line-hi);
    transform: rotate(45deg); cursor: grab;
  }
  .key:hover { border-color: var(--accent-text); }
  .key.sel { background: var(--accent); border-color: var(--accent-text); }
  .key.pinned { cursor: pointer; }

  .foot { padding: 8px 12px 12px; display: flex; flex-direction: column; gap: 8px; }
  .foot p { margin: 0; font-size: 11px; line-height: 1.55; color: var(--muted); max-width: 92ch; }
  .foot .warn { color: var(--warn); }
  .foot .ok { color: var(--ok); }
  .foot .error { color: var(--danger); }
  .inline {
    margin-left: 6px; padding: 2px 7px; font-size: 10px;
    background: none; color: var(--accent-text); border-color: var(--accent-text);
  }

  .io summary {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0; color: var(--muted); cursor: pointer;
  }
  .io-hint { margin: 8px 0 6px !important; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10px; }
  textarea {
    width: 100%; height: 110px; resize: vertical;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
    color: var(--ink); background: var(--panel); border: 1px solid var(--line);
    border-radius: 6px; padding: 8px;
  }
  .io-buttons { display: flex; gap: 6px; margin-top: 6px; }
</style>
