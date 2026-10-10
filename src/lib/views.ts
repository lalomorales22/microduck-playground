/** The workspaces the left rail switches between. */
export type View = "guide" | "sim" | "motion" | "train" | "assistant" | "files" | "debug";

export const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "guide", label: "Guide", hint: "Start here" },
  { id: "sim", label: "Simulate", hint: "Run a trained policy" },
  { id: "motion", label: "Motion maker", hint: "Author a motion" },
  { id: "train", label: "Train", hint: "PPO in your browser" },
  { id: "assistant", label: "Assistant", hint: "Scenes, motions and training" },
  { id: "files", label: "Files", hint: "Saved runs" },
  { id: "debug", label: "Debug", hint: "Instruments" },
];
