<script lang="ts">
  import { VIEWS, type View } from "./views.ts";

  let { view = $bindable(), disabled = false }: { view: View; disabled?: boolean } = $props();
</script>

<nav class="rail">
  <div class="brand">
    <span class="mark">🐤</span>
    <span class="name">
      Wicroduck
      <small>in-browser RL</small>
    </span>
  </div>

  <ul>
    {#each VIEWS as item (item.id)}
      <li>
        <button
          class:active={view === item.id}
          {disabled}
          aria-label={item.label}
          title={item.label}
          aria-current={view === item.id ? "page" : undefined}
          onclick={() => (view = item.id)}
        >
          {#if item.id === "guide"}
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 5a2 2 0 0 1 2-2h5v18H6a2 2 0 0 1-2-2z" />
              <path d="M20 5a2 2 0 0 0-2-2h-5v18h5a2 2 0 0 0 2-2z" />
            </svg>
          {:else if item.id === "sim"}
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 2.6 21 7.4v9.2L12 21.4 3 16.6V7.4z" />
              <path d="M3 7.4 12 12m0 0 9-4.6M12 12v9.4" />
            </svg>
          {:else if item.id === "motion"}
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M3 12h4m4 0h2m4 0h4" />
              <path d="M9 9.5 11.5 12 9 14.5 6.5 12z" />
              <path d="M17 9.5 19.5 12 17 14.5 14.5 12z" />
              <path d="M4 5v14" />
            </svg>
          {:else if item.id === "train"}
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M3 20h18" />
              <path d="M5 20V9m4.7 11V4m4.6 16v-7M19 20v-4" />
            </svg>
          {:else if item.id === "assistant"}
            <span class="assistant-icon" aria-hidden="true">✦</span>
          {:else if item.id === "files"}
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2.5h9A1.5 1.5 0 0 1 21 10v7.5A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5z" />
            </svg>
          {:else}
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="13" r="5" />
              <path d="M12 8V5m0 0-2-2m2 2 2-2M7.5 9.5 5 7m14 2.5L21.5 7M7 13H3m18 0h-4M7.5 16.5 5 19m14-2.5L21.5 19" />
            </svg>
          {/if}
          <span class="label">
            {item.label}
            <small>{item.hint}</small>
          </span>
        </button>
      </li>
    {/each}
  </ul>

  <a class="repo" href="https://github.com/pollen-robotics/microduck_rl" target="_blank" rel="noreferrer">
    microduck_rl ↗
  </a>
</nav>

<style>
  .rail {
    grid-area: rail;
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px 10px;
    background: var(--panel);
    border-right: 1px solid var(--line);
    min-width: 0;
  }

  .brand { display: flex; align-items: center; gap: 9px; padding: 4px 6px 14px; }
  .mark {
    display: grid; place-items: center;
    width: 28px; height: 28px; flex: none;
    font-size: 16px; line-height: 1;
    background: var(--accent); border-radius: 7px;
  }
  .name {
    display: flex; flex-direction: column; line-height: 1.2;
    font-weight: 700; font-size: 13px; letter-spacing: 0;
  }
  .name small { font-weight: 500; font-size: 10px; color: var(--muted); letter-spacing: 0; }

  ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }

  button {
    position: relative;
    display: flex; align-items: center; gap: 10px;
    width: 100%; padding: 8px 10px;
    font: inherit; font-size: 13px; text-align: left;
    color: var(--muted); background: none;
    border: 0; border-radius: 8px; cursor: pointer;
  }
  button:hover { background: var(--panel-hi); color: var(--ink); }
  button:disabled { cursor: wait; opacity: 0.65; }
  .assistant-icon { width: 17px; height: 17px; display: grid; place-items: center; flex: none; }
  button.active { background: var(--panel-hi); color: var(--ink); }
  /* Accent edge marks the active workspace without shouting. */
  button.active::before {
    content: ""; position: absolute; left: 0; top: 9px; bottom: 9px;
    width: 2px; border-radius: 2px; background: var(--accent);
  }

  svg {
    width: 17px; height: 17px; flex: none;
    fill: none; stroke: currentColor; stroke-width: 1.6;
    stroke-linecap: round; stroke-linejoin: round;
  }
  .label { display: flex; flex-direction: column; line-height: 1.25; min-width: 0; }
  .label small { font-size: 10px; color: var(--muted); }
  button.active .label small { color: var(--muted-hi); }

  .repo {
    margin-top: auto; padding: 8px 10px;
    font-size: 11px; color: var(--muted); text-decoration: none;
  }
  .repo:hover { color: var(--ink); }

  /* Narrow viewports: icons only. */
  @media (max-width: 860px) {
    .rail { padding: 12px 8px; }
    .name, .label, .repo { display: none; }
    button { justify-content: center; padding: 10px; }
    .brand { justify-content: center; padding-bottom: 12px; }
  }
</style>
