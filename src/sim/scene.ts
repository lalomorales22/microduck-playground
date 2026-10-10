// Assembles the MuJoCo scene the app simulates: the robot MJCF from the
// microduck_rl submodule, wrapped in a world that adds a floor, pins the
// timestep the policies were trained at, and declares the STAND keyframe.
//
// Split so a worker can reuse the pieces: fetching the assets into a VFS is
// separate from compiling a model out of it, because the throughput harness
// compiles several model variants from one VFS.

import { loadMujoco, type MjData, type MjModel, type Mujoco } from "./mujoco.ts";
import { DEFAULT_POSE, TIMESTEP } from "./microduck.ts";
import { assetUrl } from "../asset-url.ts";
import { defaultWorld, worldXml, type WorldConfig } from "./world.ts";

/** Deferred: `assetUrl` needs a base, which a worker only has after init. */
const modelDir = () => assetUrl("model/microduck");

export interface ModelEntry {
  file: string;
  /** Only the meshes this variant's geoms actually reference. */
  meshes: string[];
}

export interface Manifest {
  models: ModelEntry[];
  /** Union across every variant. */
  meshes: string[];
}

export interface Progress {
  (stage: string, done?: number, total?: number): void;
}

export interface ModelAssets {
  mujoco: Mujoco;
  vfs: import("./mujoco").MjVFS;
  manifest: Manifest;
}

export interface SceneOptions {
  /** Optional real collision objects; omitted preserves the original floor-only scene. */
  world?: WorldConfig;
  /**
   * MJCF to include, e.g. "robot_walk-nv.xml". Defaults to the first entry in
   * the manifest. The "-nv" twins carry no visual geoms: identical dynamics,
   * a fraction of the meshes, and far less compile memory — always the right
   * pick for training, never for the viewport.
   */
  robotXml?: string;
  /**
   * MuJoCo arena size per MjData, e.g. "1M". Left unset, MuJoCo picks a
   * generous default that costs ~13 MB per MjData — which is the binding
   * constraint on how many environments fit in a worker's 2 GB wasm heap.
   */
  memory?: string;
}

export interface Simulation {
  mujoco: Mujoco;
  model: MjModel;
  data: MjData;
  /** Index of the STAND keyframe, for mj_resetDataKeyframe. */
  standKey: number;
}

/**
 * The robot MJCF has no world around it: no floor, no keyframe, and a 2 ms
 * timestep that does not match the 5 ms the policies were trained with. This
 * wraps it into a scene that fixes all three.
 *
 * The `<include>` is resolved by the MuJoCo compiler against the VFS, so the
 * robot XML and its meshes just have to be present under the same names the
 * MJCF uses (`meshdir="assets"`).
 */
export function sceneXml(robotXml: string, options: SceneOptions = {}): string {
  const pose = Array.from(DEFAULT_POSE).join(" ");
  const size = options.memory ? `<size memory="${options.memory}"/>` : "";
  const world = worldXml(options.world ?? defaultWorld());
  const worldPose = world.qpos.length ? ` ${world.qpos.join(" ")}` : "";
  return `<mujoco model="microduck-scene">
  <include file="${robotXml}"/>
  <option timestep="${TIMESTEP}"/>
  ${size}
  ${world.assets}
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" rgba="0.28 0.3 0.34 1"/>
    ${world.bodies}
  </worldbody>
  <keyframe>
    <!-- STAND2, the pose the policies treat as zero: free joint (7) then the
         14 hinges in document order, which is the actuator order. -->
    <key name="STAND" qpos="0 0 0.12 1 0 0 0 ${pose}${worldPose}" ctrl="${pose}"/>
  </keyframe>
</mujoco>`;
}

async function fetchBytes(url: string, init?: RequestInit): Promise<Uint8Array> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/**
 * Boot the MuJoCo runtime and load the requested MJCF variants into one VFS,
 * along with just the meshes those variants reference.
 *
 * Subsetting matters: the throughput harness starts one worker per core and
 * each pays for its own copy, so loading `robot_walk-nv.xml`'s 4 meshes instead
 * of all 38 is the difference between a fast pool start and a slow one.
 */
export async function loadModelAssets(
  onProgress: Progress = () => {},
  wanted?: string[],
): Promise<ModelAssets> {
  onProgress("Loading MuJoCo runtime");
  const dir = modelDir();
  const [mujoco, manifest] = await Promise.all([
    loadMujoco(),
    fetch(`${dir}/manifest.json`).then((r) => {
      if (!r.ok) throw new Error(`manifest.json -> HTTP ${r.status}; run \`npm run prepare-assets\``);
      return r.json() as Promise<Manifest>;
    }),
  ]);

  const models = wanted
    ? manifest.models.filter((m) => wanted.includes(m.file))
    : manifest.models.slice(0, 1);
  if (models.length === 0) {
    throw new Error(`no such model in manifest: ${wanted?.join(", ")}`);
  }
  const meshes = [...new Set(models.flatMap((m) => m.meshes))];

  const vfs = new mujoco.MjVFS();

  onProgress("Loading robot model", 0, meshes.length);
  await Promise.all(
    models.map(async (m) => vfs.addBuffer(m.file, await fetchBytes(`${dir}/${m.file}`))),
  );

  let loaded = 0;
  await Promise.all(
    meshes.map(async (f) => {
      // The throughput harness spins up a worker pool per sweep cell and each
      // worker needs its own copy in its own wasm heap. force-cache keeps that
      // to one network round trip for the whole sweep.
      const buf = await fetchBytes(`${dir}/assets/${f}`, { cache: "force-cache" });
      vfs.addBuffer(`assets/${f}`, buf);
      onProgress("Loading robot model", ++loaded, meshes.length);
    }),
  );
  return { mujoco, vfs, manifest };
}

/** Compile one scene out of already-loaded assets. */
export function compileScene(
  { mujoco, vfs, manifest }: ModelAssets,
  options: SceneOptions = {},
): { model: MjModel; standKey: number } {
  const robotXml = options.robotXml ?? manifest.models[0].file;
  const model = mujoco.MjModel.from_xml_string(sceneXml(robotXml, options), vfs);
  const standKey = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND");
  return { model, standKey };
}

export async function loadSimulation(
  onProgress: Progress = () => {},
  options: SceneOptions = {},
): Promise<Simulation> {
  const assets = await loadModelAssets(onProgress, options.robotXml ? [options.robotXml] : undefined);
  onProgress("Compiling physics");
  const { model, standKey } = compileScene(assets, options);
  const data = new assets.mujoco.MjData(model);
  assets.mujoco.mj_resetDataKeyframe(model, data, standKey);
  assets.mujoco.mj_forward(model, data);
  return { mujoco: assets.mujoco, model, data, standKey };
}
