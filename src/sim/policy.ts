// Policies the viewport can run.
//
// Two sources, one interface: the shipped ONNX checkpoints, and networks
// trained in the Train workspace. The controller does not care which — it
// hands over a 61-float observation and gets 14 joint targets back.

import * as ort from "onnxruntime-web/wasm";
import { assetUrl } from "../asset-url.ts";
import { CTRL_DT, NUM_JOINTS, OBS_SIZE } from "./microduck.ts";
import { ActorCritic } from "../train/ac-policy.ts";
import { BUILTIN_TASKS, taskCommandWriter, taskLabel } from "../train/env/tasks.ts";
import { checkpointMetadata, checkpointNetwork } from "../train/checkpoint-task.ts";
import type { WorldConfig } from "./world.ts";

export interface PolicyRunner {
  /** Human-readable source, for the UI. */
  readonly label: string;
  readonly taskLabel?: string;
  readonly experimental?: boolean;
  readonly automaticRecovery?: boolean;
  readonly world?: WorldConfig;
  reset?(): void;
  dispose?(): Promise<void>;
  /** One observation in, one action out. Batch of 1: this is deployment. */
  run(obs: Float32Array): Promise<Float32Array>;
}

// Configured on first load, never at module scope — see loadMujoco() for why.
// Static hosting sends no COOP/COEP headers, so SharedArrayBuffer — and with
// it multi-threaded ORT — is unavailable. One thread is plenty: the net is
// ~800 KB and runs at 50 Hz.
let configured = false;
function configureOrt(): void {
  if (configured) return;
  ort.env.wasm.wasmPaths = assetUrl("vendor/ort/");
  ort.env.wasm.numThreads = 1;
  configured = true;
}

/**
 * A shipped ONNX checkpoint.
 *
 * These were traced for single-robot deployment, so the batch dimension is
 * fixed at 1 — fine here, and the reason a vectorized replay has to call them
 * once per environment.
 */
export class Policy implements PolicyRunner {
  private readonly session: ort.InferenceSession;
  private readonly inputName: string;
  private readonly outputName: string;
  readonly label: string;

  private constructor(
    session: ort.InferenceSession,
    inputName: string,
    outputName: string,
    label: string,
  ) {
    this.session = session;
    this.inputName = inputName;
    this.outputName = outputName;
    this.label = label;
  }

  static async load(url: string, label = url.split("/").pop() ?? url): Promise<Policy> {
    configureOrt();
    const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    return new Policy(session, session.inputNames[0], session.outputNames[0], label);
  }

  async run(obs: Float32Array): Promise<Float32Array> {
    const feeds: Record<string, ort.Tensor> = {
      [this.inputName]: new ort.Tensor("float32", obs, [1, obs.length]),
    };
    const out = await this.session.run(feeds);
    return out[this.outputName].data as Float32Array;
  }

  async dispose(): Promise<void> { await this.session.release(); }
}

/**
 * A network trained in this app, restored from a checkpoint.
 *
 * Runs the distribution MEAN, not a sample: exploration noise belongs in
 * training, and a deployed policy is deterministic. The observation normalizer
 * comes back with the weights — a policy fed unnormalized observations does
 * not error, it just behaves badly, which is the whole reason the reference
 * project insists the normalizer be baked into an export.
 */
export class CheckpointPolicy implements PolicyRunner {
  readonly label: string;
  readonly iteration: number;
  readonly taskLabel: string;
  readonly experimental: boolean;
  readonly automaticRecovery: boolean;
  readonly world: WorldConfig;
  #ac: ActorCritic;
  #obs = new Float32Array(OBS_SIZE);
  #time = 0;
  #commands: ReturnType<typeof taskCommandWriter>;

  constructor(checkpoint: unknown, label: string) {
    const metadata = checkpointMetadata(checkpoint);
    const network = checkpointNetwork(checkpoint);
    this.#commands = taskCommandWriter(metadata.task);
    this.taskLabel = taskLabel(metadata.task);
    this.experimental = BUILTIN_TASKS.find(item => item.kind === metadata.task.kind)?.experimental ?? false;
    this.automaticRecovery = metadata.task.kind === "standup";
    this.world = metadata.world;
    this.#ac = new ActorCritic(
      OBS_SIZE,
      NUM_JOINTS,
      Math.random,
      1,
      network.hidden,
      null, // the viewport runs one observation at a time; SIMD buys nothing
    );
    this.#ac.load(network.policy);
    this.iteration = network.iteration;
    this.label = label;
  }

  run(obs: Float32Array): Promise<Float32Array> {
    if (obs.length !== OBS_SIZE || obs.some(x => !Number.isFinite(x))) throw new Error("invalid policy observation");
    this.#obs.set(obs);
    this.#commands(this.#time, this.#obs);
    const mean = this.#ac.actMean(this.#obs, 1);
    if (mean.some(x => !Number.isFinite(x))) throw new Error("checkpoint produced non-finite actions");
    this.#time += CTRL_DT;
    // Copy: the net's output buffer is reused on the next call.
    return Promise.resolve(mean.slice(0, this.#ac.actDim));
  }

  reset(): void { this.#time = 0; }
}
