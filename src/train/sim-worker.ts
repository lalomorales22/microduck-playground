/// <reference lib="webworker" />
// One worker owns one compiled MjModel and a batch of MjData, and reports how
// fast it can step them. Envs are stepped round-robin inside the worker rather
// than one env per worker: the benchmark that motivated this measured ~1.9x
// better per-step throughput that way.

import { setAssetBase } from "../asset-url.ts";
import { compileScene, loadModelAssets, type ModelAssets } from "../sim/scene.ts";
import { DECIMATION, NUM_JOINTS, OBS_SIZE } from "../sim/microduck.ts";
import { Mlp } from "./mlp.ts";
import type { FromWorker, ToWorker, WorkerStats } from "./protocol.ts";

let assets: ModelAssets | null = null;
let model: import("../sim/mujoco").MjModel | null = null;
let datas: import("../sim/mujoco").MjData[] = [];
let mlp: Mlp | null = null;
let obsBatch: Float32Array | null = null;

const post = (msg: FromWorker) => self.postMessage(msg);

/**
 * Total wasm heap size. Emscripten's HEAPU8 is not exported by the MuJoCo
 * bindings, but every MuJoCo array is a view onto that heap, so any view's
 * backing buffer is the whole thing.
 */
function heapBytes(): number {
  if (model) return (model.geom_size as Float64Array).buffer.byteLength;
  return 0;
}

async function init(baseUrl: string, config: ToWorker & { type: "init" }): Promise<void> {
  setAssetBase(baseUrl);
  const { robotXml, envs, memory, withPolicy } = config.config;

  // Only this cell's model, and only the meshes it references.
  assets = await loadModelAssets(() => {}, [robotXml]);
  ({ model } = compileScene(assets, { robotXml, memory }));
  const heapAfterCompileBytes = heapBytes();

  // Allocating past the wasm32 2 GB ceiling throws from inside MuJoCo. That is
  // a real answer for the sweep ("this config does not fit"), not a crash, so
  // keep whatever was allocated and report it.
  let oom = false;
  for (let i = 0; i < envs; i++) {
    try {
      const d = new assets.mujoco.MjData(model);
      assets.mujoco.mj_resetDataKeyframe(model, d, 0);
      assets.mujoco.mj_forward(model, d);
      datas.push(d);
    } catch {
      oom = true;
      break;
    }
  }

  const heapTotalBytes = heapBytes();
  if (withPolicy && datas.length > 0) {
    mlp = new Mlp(OBS_SIZE, NUM_JOINTS, datas.length);
    obsBatch = new Float32Array(datas.length * OBS_SIZE);
  }

  const stats: WorkerStats = {
    heapAfterCompileBytes,
    heapTotalBytes,
    bytesPerEnv: datas.length ? (heapTotalBytes - heapAfterCompileBytes) / datas.length : 0,
    ngeom: model.ngeom,
    nq: model.nq,
    envs: datas.length,
    oom,
  };
  post({ type: "ready", stats });
}

/**
 * Step every environment until the deadline, counting physics steps. One
 * "control step" is DECIMATION physics steps plus (optionally) one policy
 * forward pass over the whole batch — the shape of a real rollout.
 */
function run(durationMs: number): void {
  if (!assets || !model || datas.length === 0) {
    post({ type: "error", message: "worker not initialised" });
    return;
  }
  const { mujoco } = assets;
  const batch = datas.length;
  const deadline = performance.now() + durationMs;
  let physicsSteps = 0;
  const start = performance.now();

  while (performance.now() < deadline) {
    // A chunk of control steps between clock reads: performance.now() is not
    // free, and checking it every physics step would show up in the number.
    for (let c = 0; c < 8; c++) {
      if (mlp && obsBatch) {
        // Gather a policy-shaped observation. The values are not meaningful in
        // M0 — the cost of touching qpos per env is.
        for (let e = 0; e < batch; e++) {
          const qpos = datas[e].qpos;
          const off = e * OBS_SIZE;
          for (let j = 0; j < OBS_SIZE; j++) obsBatch[off + j] = qpos[j % model.nq];
        }
        mlp.forward(obsBatch, batch);
      }
      for (let s = 0; s < DECIMATION; s++) {
        for (let e = 0; e < batch; e++) mujoco.mj_step(model, datas[e]);
      }
      physicsSteps += DECIMATION * batch;
    }
  }
  post({ type: "result", physicsSteps, elapsedMs: performance.now() - start });
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      void init(msg.baseUrl, msg).catch((err: unknown) =>
        post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
      );
    } else if (msg.type === "run") {
      run(msg.durationMs);
    } else if (msg.type === "dispose") {
      datas = [];
      model = null;
      assets = null;
      mlp = null;
      self.close();
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
