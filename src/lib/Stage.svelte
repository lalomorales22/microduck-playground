<script lang="ts">
  import { onMount } from "svelte";
  import type { Session } from "./session.svelte";

  const { session }: { session: Session } = $props();

  let canvas: HTMLCanvasElement;

  onMount(() => {
    void session.start(canvas);
    return () => session.destroy();
  });
</script>

<div class="stage">
  <canvas bind:this={canvas}></canvas>

  {#if session.ready}
    <div class="transport" role="toolbar" aria-label="Simulation playback">
      <button aria-label={session.paused ? "Play simulation" : "Pause simulation"} title={session.paused ? "Play" : "Pause"} onclick={() => session.setPaused(!session.paused)}>{session.paused ? "▶" : "Ⅱ"}</button>
      <button aria-label="Step simulation" title="Advance one control step" disabled={!session.paused || session.worldBusy} onclick={() => void session.singleStep()}>▹|</button>
      <button aria-label="Reset scene" title="Reset robot and objects" disabled={session.worldBusy} onclick={() => session.reset()}>↺</button>
      <span class="time">{session.simTime.toFixed(2)} s</span>
      <button aria-label="Frame scene" title="Frame scene" onclick={() => session.frameWorld()}>⊞</button>
    </div>
    <div class="scene-tag">{session.sceneName} <span>{session.world.objects.length} objects</span></div>
  {/if}

  {#if session.error}
    <div class="overlay error">
      <strong>Failed to start</strong>
      <span>{session.error}</span>
    </div>
  {:else if !session.ready}
    <div class="overlay">
      <div class="spinner"></div>
      <span>{session.loadStage}</span>
      <div class="bar"><div style:width="{session.loadProgress}%"></div></div>
    </div>
  {/if}
</div>

<style>
  .stage { position: relative; min-height: 0; background: var(--viewport); }
  canvas { display: block; width: 100%; height: 100%; }
  .transport { position: absolute; top: 12px; left: 12px; display: flex; align-items: center; gap: 4px; padding: 4px; background: color-mix(in srgb, var(--panel) 96%, transparent); border: 1px solid var(--line); border-radius: 6px; }
  .transport button { display: grid; place-items: center; width: 30px; height: 30px; border: 0; border-radius: 4px; color: var(--ink); background: transparent; font-size: 17px; cursor: pointer; }
  .transport button:hover { background: var(--panel-hi); }
  .transport button:disabled { opacity: 0.35; cursor: not-allowed; }
  .time { min-width: 66px; text-align: center; font: 11px ui-monospace, monospace; color: var(--muted-hi); }
  .scene-tag { position: absolute; bottom: 12px; left: 14px; font-size: 11px; pointer-events: none; }
  .scene-tag span { margin-left: 8px; color: var(--muted); }

  .overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px;
    padding: 24px; text-align: center;
    background: color-mix(in srgb, var(--viewport) 92%, transparent);
    font-size: 12px; color: var(--muted);
  }
  .spinner {
    width: 24px; height: 24px; border-radius: 50%;
    border: 2px solid var(--line); border-top-color: var(--accent);
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .bar { width: 200px; height: 2px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .bar > div { height: 100%; background: var(--accent); transition: width 0.15s ease; }

  .error { color: var(--danger); }
  .error strong { font-size: 13px; }
  .error span { max-width: 46ch; line-height: 1.5; }
</style>
