<script lang="ts">
  import { onMount } from "svelte";
  import {
    deleteCheckpoint, importCheckpoint, listCheckpoints, loadCheckpoint,
    opfsAvailable, readCheckpointFile, renameCheckpoint, sanitizeName,
    type CheckpointInfo,
  } from "../train/checkpoint-store.ts";
  import {
    deleteMotion, importMotion, listMotions, loadMotionFile, readMotionFile,
    renameMotion, saveMotion, type MotionEntry,
  } from "../motion/motion-store.ts";

  interface Row extends CheckpointInfo {
    iteration?: number;
    task?: string;
    hidden?: string;
  }

  let rows = $state<Row[]>([]);
  let motions = $state<MotionEntry[]>([]);
  let error = $state<string | null>(null);
  let busy = $state(false);
  const available = opfsAvailable();

  async function refresh(): Promise<void> {
    const list = await listCheckpoints();
    // Read each file's header for the details worth showing. A handful of
    // multi-megabyte files, only when this tab is open.
    rows = await Promise.all(
      list.map(async (c) => {
        const data = await loadCheckpoint<{
          iteration?: number; task?: string; config?: { hidden?: number[] };
        }>(c.name);
        return {
          ...c,
          iteration: data?.iteration,
          task: data?.task,
          hidden: data?.config?.hidden?.join("/"),
        };
      }),
    );
  }

  async function refreshMotions(): Promise<void> {
    motions = await listMotions();
  }

  onMount(() => {
    void refresh();
    void refreshMotions();
  });

  async function guard(fn: () => Promise<void>): Promise<void> {
    busy = true;
    error = null;
    try {
      await fn();
      await refresh();
      await refreshMotions();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      busy = false;
    }
  }

  function saveBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    // Revoked on the next tick — revoking immediately can cancel the download.
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }

  async function download(name: string): Promise<void> {
    saveBlob(await readCheckpointFile(name), `${name}.json`);
  }

  /**
   * Built-ins are not in storage, so they are re-serialised on the way out.
   * Pretty-printed rather than minified: the point of downloading one is
   * usually to hand it to an agent or edit it by hand.
   */
  async function downloadMotion(m: MotionEntry): Promise<void> {
    if (m.source === "saved") {
      saveBlob(await readMotionFile(m.id), `${m.name}.json`);
      return;
    }
    const file = await loadMotionFile(m.id);
    saveBlob(new Blob([JSON.stringify(file, null, 2)], { type: "application/json" }), `${m.name}.json`);
  }

  /** Copy a built-in into storage so it can be edited and renamed. */
  function fork(m: MotionEntry): void {
    const raw = globalThis.prompt?.("Save a copy of this motion as:", `${m.name}-copy`);
    if (!raw) return;
    const name = sanitizeName(raw);
    if (!name) return;
    void guard(async () => {
      const file = await loadMotionFile(m.id);
      if (!file) throw new Error(`motion "${m.id}" is gone`);
      await saveMotion(name, { ...file, name });
    });
  }

  function renameMotionRow(m: MotionEntry): void {
    const raw = globalThis.prompt?.("Rename this motion:", m.id);
    if (!raw) return;
    const to = sanitizeName(raw);
    if (!to || to === m.id) return;
    void guard(() => renameMotion(m.id, to));
  }

  function removeMotion(m: MotionEntry): void {
    if (!globalThis.confirm?.(`Delete motion “${m.name}”? This cannot be undone.`)) return;
    void guard(() => deleteMotion(m.id));
  }

  async function uploadMotion(e: Event): Promise<void> {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    const name = sanitizeName(file.name.replace(/\.json$/i, ""));
    // importMotion parses before it stores, so a bad file is rejected here
    // with the parser's message rather than surfacing later as a training
    // error nobody can connect back to the file.
    await guard(async () => void await importMotion(name || "motion", await file.text()));
  }

  function rename(row: Row): void {
    const raw = globalThis.prompt?.("Rename this run:", row.name);
    if (!raw) return;
    const to = sanitizeName(raw);
    if (!to || to === row.name) return;
    void guard(() => renameCheckpoint(row.name, to));
  }

  function remove(row: Row): void {
    if (!globalThis.confirm?.(`Delete “${row.name}”? This cannot be undone.`)) return;
    void guard(() => deleteCheckpoint(row.name));
  }

  async function upload(e: Event): Promise<void> {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = ""; // so the same file can be picked twice
    if (!file) return;
    const name = sanitizeName(file.name.replace(/\.json$/i, ""));
    await guard(async () => importCheckpoint(name || "imported", await file.text()));
  }

  const mb = (b: number) => `${(b / 1024 ** 2).toFixed(1)} MB`;
  const when = (t: number) => new Date(t).toLocaleString();
  const total = $derived(rows.reduce((n, r) => n + r.bytes, 0));
</script>

<div class="files">
  <header>
    <div>
      <h2>Saved runs</h2>
      <p>
        Checkpoints live in this browser's private filesystem — not in a folder
        you can browse, which is why they get a manager. Download one to keep
        it or move it to another machine; upload it back to carry on.
      </p>
    </div>
    <label class="upload">
      <input type="file" accept="application/json,.json" onchange={upload} disabled={busy} />
      <span>Upload…</span>
    </label>
  </header>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  {#if !available}
    <p class="empty">This browser has no private filesystem, so runs cannot be saved.</p>
  {:else if rows.length === 0}
    <p class="empty">
      Nothing saved yet. Train a policy, then use <strong>Save as…</strong> in
      the Train workspace.
    </p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Name</th><th>Task</th><th class="n">Iteration</th>
          <th>Net</th><th class="n">Size</th><th>Modified</th><th></th>
        </tr>
      </thead>
      <tbody>
        {#each rows as r (r.name)}
          <tr>
            <td class="name">{r.name}</td>
            <td>{r.task ?? "—"}</td>
            <td class="n">{r.iteration ?? "—"}</td>
            <td class="mono">{r.hidden ?? "—"}</td>
            <td class="n">{mb(r.bytes)}</td>
            <td class="dim">{when(r.modified)}</td>
            <td class="actions">
              <button disabled={busy} onclick={() => void download(r.name)}>Download</button>
              <button disabled={busy} onclick={() => rename(r)}>Rename</button>
              <button class="danger" disabled={busy} onclick={() => remove(r)}>Delete</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
    <p class="foot">{rows.length} run{rows.length === 1 ? "" : "s"} · {mb(total)} total</p>
  {/if}

  <header class="second">
    <div>
      <h2>Motions</h2>
      <p>
        A motion is a small JSON file of joint angles over time — the input to
        a tracking task. Author one in the <strong>Motion maker</strong>, or
        have an AI agent write it from <code>docs/motion-format.md</code>. The
        built-ins are always here; download one as a starting point.
      </p>
    </div>
    <label class="upload">
      <input type="file" accept="application/json,.json" onchange={uploadMotion} disabled={busy} />
      <span>Upload motion…</span>
    </label>
  </header>

  <table class="motions">
    <thead>
      <tr>
        <th>Name</th><th>Source</th><th class="n">Length</th>
        <th class="n">Keys</th><th>Plays</th><th>Description</th><th></th>
      </tr>
    </thead>
    <tbody>
      {#each motions as m (m.id)}
        <tr>
          <td class="name">{m.name}</td>
          <td class="dim">{m.source === "builtin" ? "built-in" : "yours"}</td>
          <td class="n">{m.error ? "—" : `${m.duration.toFixed(1)} s`}</td>
          <td class="n">{m.error ? "—" : m.keyframes}</td>
          <td class="dim">{m.error ? "—" : m.loop ? "looping" : "once"}</td>
          <td class="desc" class:bad={!!m.error}>{m.error ?? m.description}</td>
          <td class="actions">
            <button disabled={busy} onclick={() => void downloadMotion(m)}>Download</button>
            {#if m.source === "builtin"}
              <button disabled={busy} onclick={() => fork(m)}>Save a copy</button>
            {:else}
              <button disabled={busy} onclick={() => renameMotionRow(m)}>Rename</button>
              <button class="danger" disabled={busy} onclick={() => removeMotion(m)}>Delete</button>
            {/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .files {
    grid-area: main;
    display: flex; flex-direction: column; gap: 14px;
    padding: 20px 24px; overflow-y: auto; background: var(--viewport);
  }
  header { display: flex; align-items: flex-start; gap: 20px; }
  header.second { margin-top: 10px; padding-top: 18px; border-top: 1px solid var(--line); }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
  header > div { flex: 1; min-width: 0; }
  h2 { margin: 0; font-size: 15px; letter-spacing: 0; }
  p { margin: 6px 0 0; font-size: 12px; line-height: 1.6; color: var(--muted); max-width: 74ch; }

  .upload input { display: none; }
  .upload span {
    display: inline-block; font-size: 12px; font-weight: 600;
    color: var(--ink); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 7px;
    padding: 8px 14px; cursor: pointer; white-space: nowrap;
  }
  .upload span:hover { border-color: var(--line-hi); }

  table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 16px 6px 0; text-align: left; white-space: nowrap; }
  th {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0; color: var(--muted); border-bottom: 1px solid var(--line);
  }
  td { border-bottom: 1px solid var(--line); }
  .n { text-align: right; }
  .name { font-weight: 600; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
  .dim { color: var(--muted); }
  .desc { white-space: normal; max-width: 42ch; color: var(--muted); }
  .desc.bad { color: var(--danger); }

  .actions { display: flex; gap: 6px; padding-right: 0; }
  .actions button {
    font: inherit; font-size: 11px; font-weight: 500;
    color: var(--muted); background: none;
    border: 1px solid var(--line); border-radius: 6px;
    padding: 3px 9px; cursor: pointer;
  }
  .actions button:hover:not(:disabled) { color: var(--ink); border-color: var(--line-hi); }
  .actions button.danger:hover:not(:disabled) { color: var(--danger); border-color: var(--danger); }
  .actions button:disabled { opacity: 0.4; cursor: not-allowed; }

  .empty, .foot { font-size: 12px; color: var(--muted); }
  .error { color: var(--danger); font-size: 12px; }
</style>
