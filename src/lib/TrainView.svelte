<script lang="ts">
  // Training, and nothing else. The throughput harness and the bare
  // environment rollout live in Debug — they are instruments, not the task.
  import TrainingPanel from "./TrainingPanel.svelte";
  import type { TrainingSession } from "./training.svelte";
  let { active = true, session, replayCheckpoint }: {
    active?: boolean; session: TrainingSession; replayCheckpoint?: (name: string) => Promise<void>;
  } = $props();
</script>

<div class="train">
  <TrainingPanel {active} {session} {replayCheckpoint} />
</div>

<style>
  .train {
    min-width: 0; min-height: 0; flex: 1;
    display: flex; flex-direction: column;
    padding: 18px 22px; overflow-y: auto; background: var(--viewport);
  }
  @media (max-width: 700px) {
    .train { padding: 14px 12px; }
  }
</style>
