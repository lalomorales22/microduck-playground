<script lang="ts">
  import { JOINT_LIMITS, JOINT_NAMES } from "../sim/microduck.ts";
  import { addKey, mirrorLegs, setJoint, undriveJoint } from "../motion/edit.ts";
  import type { MakerSession } from "./maker.svelte";

  const { session }: { session: MakerSession } = $props();

  /** Grouped the way the robot is, not the way the array is. */
  const GROUPS: { label: string; joints: number[] }[] = [
    { label: "Left leg", joints: [0, 1, 2, 3, 4] },
    { label: "Head & neck", joints: [5, 6, 7, 8] },
    { label: "Right leg", joints: [9, 10, 11, 12, 13] },
  ];

  const deg = (rad: number) => `${Math.round((rad * 180) / Math.PI)}°`;
  const short = (name: string) => name.replace(/^(left|right)_/, "");

  // The angles the sliders show: the selected key's, or the interpolated pose
  // when the playhead sits between keys.
  const pose = $derived.by(() => {
    const key = session.draft.keys[session.selected];
    return key ? key.pose : Array.from(session.poseAtPlayhead());
  });
  const onKey = $derived(session.selected >= 0);

  /**
   * Editing a joint is also how it joins the motion, and how a keyframe gets
   * created — the auto-key behaviour every timeline tool has. Making either
   * one a separate step would be friction for no benefit.
   */
  function edit(joint: number, value: number): void {
    let index = session.selected;
    if (index < 0) index = addKey(session.draft, session.time, session.poseAtPlayhead());
    setJoint(session.draft, index, joint, value);
    session.selected = index;
    session.refresh();
  }

  function toggle(joint: number): void {
    if (session.draft.driven[joint]) undriveJoint(session.draft, joint);
    else session.draft.driven[joint] = true;
    session.refresh();
  }

  function mirror(from: "left" | "right"): void {
    let index = session.selected;
    if (index < 0) index = addKey(session.draft, session.time, session.poseAtPlayhead());
    mirrorLegs(session.draft, index, from);
    session.selected = index;
    session.refresh();
  }
</script>

<aside class="joints">
  <section>
    <h2>Pose</h2>
    {#if onKey}
      <p class="hint">
        Editing keyframe {session.selected + 1} of {session.draft.keys.length},
        at {session.draft.keys[session.selected].t.toFixed(2)} s.
      </p>
    {:else}
      <p class="hint warn">
        Between keyframes. Moving a slider adds one at {session.time.toFixed(2)} s.
      </p>
    {/if}
  </section>

  {#each GROUPS as group (group.label)}
    <section>
      <h2>{group.label}</h2>
      {#each group.joints as j (j)}
        <div class="joint" class:off={!session.draft.driven[j]}>
          <button
            class="dot"
            class:on={session.draft.driven[j]}
            title={session.draft.driven[j]
              ? "Part of this motion — click to drop it and reset to the reference angle"
              : "Not part of this motion — click to include it"}
            onclick={() => toggle(j)}
            aria-label="Toggle {JOINT_NAMES[j]}"
          ></button>
          <span class="name">{short(JOINT_NAMES[j])}</span>
          <input
            type="range"
            min={JOINT_LIMITS[j][0]}
            max={JOINT_LIMITS[j][1]}
            step="0.005"
            value={pose[j]}
            oninput={(e) => edit(j, Number(e.currentTarget.value))}
            disabled={!session.ready}
          />
          <span class="val">{deg(pose[j])}</span>
        </div>
      {/each}
      {#if group.label === "Left leg"}
        <button class="mirror" onclick={() => mirror("left")}>Mirror left → right</button>
      {:else if group.label === "Right leg"}
        <button class="mirror" onclick={() => mirror("right")}>Mirror right → left</button>
      {/if}
    </section>
  {/each}

  <p class="note">
    The legs' joint axes are mirrored, so a symmetric pose needs opposite
    signs. Use the mirror buttons rather than typing them twice.
  </p>
</aside>

<style>
  .joints {
    grid-area: aside;
    display: flex; flex-direction: column; gap: 14px;
    padding: 14px; overflow-y: auto;
    background: var(--panel); border-left: 1px solid var(--line);
  }
  section { display: flex; flex-direction: column; gap: 5px; }
  h2 {
    margin: 0 0 2px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0; color: var(--muted);
  }
  .hint { margin: 0; font-size: 10px; line-height: 1.5; color: var(--muted); }
  .hint.warn { color: var(--warn); }

  .joint { display: grid; grid-template-columns: 8px 62px 1fr 34px; align-items: center; gap: 6px; }
  .joint.off { opacity: 0.55; }
  .name { font-size: 10px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; }
  .val { font-size: 10px; text-align: right; font-variant-numeric: tabular-nums; }

  .dot {
    width: 8px; height: 8px; padding: 0; border-radius: 50%;
    border: 1px solid var(--line-hi); background: none; cursor: pointer;
  }
  .dot.on { background: var(--accent); border-color: var(--accent-text); }

  input[type="range"] { width: 100%; accent-color: var(--control-accent); cursor: pointer; }
  input[type="range"]:disabled { opacity: 0.4; cursor: not-allowed; }

  .mirror {
    margin-top: 3px; font: inherit; font-size: 10px; font-weight: 600;
    color: var(--muted); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 6px;
    padding: 4px 8px; cursor: pointer;
  }
  .mirror:hover { color: var(--ink); border-color: var(--line-hi); }

  .note {
    margin: auto 0 0; padding-top: 10px;
    border-top: 1px solid var(--line);
    font-size: 10px; line-height: 1.5; color: var(--muted);
  }

  @media (max-width: 1040px) {
    .joints {
      border-left: 0; border-top: 1px solid var(--line);
      flex-direction: row; flex-wrap: wrap; gap: 20px;
    }
    section { flex: 1 1 220px; }
    .note { display: none; }
  }
</style>
