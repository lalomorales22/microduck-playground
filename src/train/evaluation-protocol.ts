import type { EvaluationReport } from "./evaluation.ts";

export type ToEvaluationWorker = { type: "evaluate"; name: string; baseUrl: string };
export type FromEvaluationWorker =
  | { type: "progress"; progress: number }
  | { type: "result"; report: EvaluationReport }
  | { type: "error"; error: string };
