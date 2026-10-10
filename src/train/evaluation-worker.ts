/// <reference lib="webworker" />
import { setAssetBase } from "../asset-url.ts";
import { loadModelAssets } from "../sim/scene.ts";
import { readCheckpointFile } from "./checkpoint-store.ts";
import { EVALUATION_ROBOT, MAX_EVALUATION_CHECKPOINT_BYTES, runCheckpointEvaluation, validateEvaluationName } from "./evaluation.ts";
import type { FromEvaluationWorker, ToEvaluationWorker } from "./evaluation-protocol.ts";

const post = (message: FromEvaluationWorker) => self.postMessage(message);
let started = false;

async function evaluate(message: ToEvaluationWorker): Promise<void> {
  if (!message || message.type !== "evaluate") throw new Error("Unsupported evaluation request");
  if (started) throw new Error("Evaluation worker already started");
  started = true;
  const name = validateEvaluationName(message.name);
  const base = new URL(message.baseUrl);
  if (base.origin !== self.location.origin || !["http:", "https:"].includes(base.protocol) || base.username || base.password) {
    throw new Error("Evaluation assets must use the current origin");
  }
  setAssetBase(base.href);
  const file = await readCheckpointFile(name);
  if (file.size < 2 || file.size > MAX_EVALUATION_CHECKPOINT_BYTES) throw new Error("Checkpoint size exceeds the evaluation budget");
  const bytes = await file.arrayBuffer();
  const checkpointSha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), b => b.toString(16).padStart(2, "0")).join("");
  const checkpoint: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  post({ type: "progress", progress: 0 });
  const assets = await loadModelAssets(() => {}, [EVALUATION_ROBOT]);
  const report = await runCheckpointEvaluation(assets, checkpoint, {
    checkpointName: name, checkpointSha256,
    onProgress: progress => post({ type: "progress", progress }),
  });
  post({ type: "result", report });
}

self.onmessage = (event: MessageEvent<ToEvaluationWorker>) => {
  void evaluate(event.data).catch((error: unknown) => post({ type: "error", error: error instanceof Error ? error.message : String(error) }));
};
