import { validateTrainingConfig } from "../train/checkpoint-task.ts";
import { DEFAULT_PPO } from "../train/ppo.ts";
import { DEFAULT_TRAINER, type TrainerConfig } from "../train/trainer.ts";
import type { TrainingDraft } from "./contract.ts";

/** New assistant drafts must not inherit a resumed checkpoint's batch size. */
export function assistantTrainerConfig(draft: TrainingDraft): TrainerConfig {
  if (draft.hiddenPreset !== "small" && draft.hiddenPreset !== "reference") {
    throw new Error("Assistant training hiddenPreset must be small or reference");
  }
  return validateTrainingConfig({
    ...DEFAULT_TRAINER,
    envs: draft.envs,
    seed: draft.seed,
    stepsPerIter: 24,
    hidden: draft.hiddenPreset === "small" ? [128, 64] : [512, 256, 128],
    ppo: { ...DEFAULT_PPO, epochs: 4 },
  });
}
