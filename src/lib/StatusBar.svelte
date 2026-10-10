<script lang="ts">
  import { PHASE_LABEL, type Session } from "./session.svelte";
  import type { MakerSession } from "./maker.svelte";
  import type { View } from "./views.ts";

  const { session, maker, view }: { session: Session; maker: MakerSession; view: View } =
    $props();

  // One bar, two workspaces that own the viewport. Showing the Simulate
  // session's telemetry while the Motion maker is up front would be reporting
  // on a duck nobody is looking at.
  const inMaker = $derived(view === "motion");
</script>

<footer class="status">
{#if view === "assistant"}
  <span>Assistant</span>
  <span class="spacer"></span>
  <span class="dim">OpenAI API · Simulation and training remain local</span>
{:else if inMaker}
  <span class="phase" data-phase={maker.ready ? (maker.playing ? "standing" : "settling") : "loading"}>
    <i></i>{maker.started ? (maker.ready ? (maker.playing ? "Playing" : "Paused") : "Loading") : "Idle"}
  </span>
  {#if maker.started}
    <span><b>Motion</b>{maker.draft.name || "untitled"}</span>
    <span><b>Time</b>{maker.time.toFixed(2)} / {maker.duration.toFixed(2)} s</span>
    <span><b>Keys</b>{maker.draft.keys.length}</span>
    <span><b>Trunk</b>{maker.trunkCm.toFixed(1)} cm</span>
  {/if}
  <span class="spacer"></span>
  <span class="dim">{maker.mode === "preview" ? "Kinematic preview" : "Open-loop physics"}</span>
{:else}
  <!-- data-phase drives the dot colour; "loading" leaves it neutral rather
       than showing the initial "standing" green before the sim exists. -->
  <span class="phase" data-phase={session.ready ? session.phase : "loading"}>
    <i></i>{session.started ? (session.ready ? PHASE_LABEL[session.phase] : "Loading") : "Idle"}
  </span>
  {#if session.started}
    <span><b>Upright</b>{session.uprightPct}%</span>
    <span><b>Trunk</b>{session.heightCm.toFixed(1)} cm</span>
    <span><b>Policy</b>{session.policyHz} Hz</span>
  {:else}
    <span class="dim">Simulator loads on first visit</span>
  {/if}
  <span class="spacer"></span>
  <!-- The ACTIVE policy, not the shipped one: the viewport can be driven by a
       run trained here, and saying otherwise would be a lie in the one place
       that is always on screen. -->
  <span class="dim">MuJoCo WASM · {session.activePolicyLabel}</span>
{/if}
</footer>

<style>
  .status {
    grid-area: status;
    display: flex; align-items: center; gap: 16px;
    height: 26px; padding: 0 12px;
    background: var(--panel); border-top: 1px solid var(--line);
    font-size: 11px; font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .status span { display: flex; align-items: center; gap: 6px; }
  b { font-weight: 500; color: var(--muted); }

  .phase { font-weight: 600; }
  i { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
  .phase[data-phase="standing"] i { background: var(--ok); }
  .phase[data-phase="recovering"] i { background: var(--accent); }
  .phase[data-phase="limp"] i,
  .phase[data-phase="settling"] i { background: var(--warn); }

  .spacer { flex: 1; }
  .dim { color: var(--muted); overflow: hidden; text-overflow: ellipsis; }
  @media (max-width: 860px) { .status .dim { display: none; } }
  @media (max-width: 480px) {
    .status { height: auto; min-height: 26px; flex-wrap: wrap; gap: 4px 8px; padding: 4px 8px; }
    .status span { gap: 4px; max-width: 100%; flex-wrap: wrap; white-space: normal; overflow-wrap: anywhere; }
    .status .spacer { display: none; }
  }
</style>
