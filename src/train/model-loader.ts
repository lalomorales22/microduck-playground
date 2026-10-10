// Compiling a scene from local files, for Node-side checks.
//
// The browser path goes through src/sim/scene.ts and fetch(); this is the same
// compile against the filesystem, so a headless check exercises the real
// VecEnv and Trainer rather than a parallel implementation.

import { sceneXml } from "../sim/scene.ts";
import type { MjModel, Mujoco } from "../sim/mujoco.ts";

export interface LoadedModel {
  mujoco: Mujoco;
  model: MjModel;
  standKey: number;
}

export async function loadModelFromDisk(
  modelDir: string,
  robotXml: string,
  fs: typeof import("node:fs"),
  path: typeof import("node:path"),
): Promise<LoadedModel> {
  const { default: loadMujoco } = await import("@mujoco/mujoco");
  const mujoco = (await loadMujoco()) as unknown as Mujoco;
  const manifest = JSON.parse(
    fs.readFileSync(path.join(modelDir, "manifest.json"), "utf8"),
  ) as { models: { file: string; meshes: string[] }[] };
  const entry = manifest.models.find((m) => m.file === robotXml);
  if (!entry) throw new Error(`${robotXml} missing from manifest; run npm run prepare-assets`);

  const vfs = new mujoco.MjVFS();
  vfs.addBuffer(robotXml, new Uint8Array(fs.readFileSync(path.join(modelDir, robotXml))));
  for (const mesh of entry.meshes) {
    vfs.addBuffer(
      `assets/${mesh}`,
      new Uint8Array(fs.readFileSync(path.join(modelDir, "assets", mesh))),
    );
  }
  const model = mujoco.MjModel.from_xml_string(sceneXml(robotXml), vfs);
  const standKey = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND");
  return { mujoco, model, standKey };
}
