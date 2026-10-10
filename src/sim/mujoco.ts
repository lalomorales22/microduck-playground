// Thin, hand-written typing over the parts of the official @mujoco/mujoco
// WebAssembly bindings this app touches, plus the loader.
//
// The runtime is fetched as a plain static file from /vendor/mujoco/ (copied
// there by scripts/prepare-assets.mjs) rather than imported through the
// bundler: mujoco.js resolves its .wasm sidecar relative to its own URL, and
// keeping both side by side under public/ is the one arrangement that works
// identically in dev and in a static production build.

import { assetUrl } from "../asset-url.ts";

/** A MuJoCo array view. The WASM heap can grow and detach earlier views, so
 *  never cache these across a step — always re-read from model/data. */
export type MjArray = Float64Array | Float32Array | Int32Array | Uint8Array;

export interface MjModel {
  nq: number; nv: number; nu: number; nbody: number; ngeom: number;
  nmesh: number; nmat: number; nkey: number; nsensordata: number;
  nmeshvert: number; nmeshface: number;
  opt: { timestep: number; gravity: MjArray };

  readonly geom_type: MjArray;
  readonly geom_bodyid: MjArray;
  readonly geom_dataid: MjArray;
  readonly geom_group: MjArray;
  readonly geom_matid: MjArray;
  readonly geom_size: MjArray;
  readonly geom_rgba: MjArray;
  readonly mat_rgba: MjArray;

  /** [njnt, 2] hinge limits. Only meaningful where jnt_limited is set. */
  readonly jnt_range: MjArray;

  readonly mesh_vertadr: MjArray;
  readonly mesh_vertnum: MjArray;
  readonly mesh_vert: MjArray;
  readonly mesh_normal: MjArray;
  readonly mesh_faceadr: MjArray;
  readonly mesh_facenum: MjArray;
  readonly mesh_face: MjArray;

  jnt(nameOrId: string | number): { qposadr: number; dofadr: number };
  sensor(nameOrId: string | number): { adr: number; dim: number };
  delete(): void;
}

export interface MjData {
  readonly time: number;
  readonly qpos: MjArray;
  readonly qvel: MjArray;
  readonly ctrl: MjArray;
  readonly sensordata: MjArray;
  readonly geom_xpos: MjArray;
  readonly geom_xmat: MjArray;
  readonly xfrc_applied: MjArray;
  body(nameOrId: string | number): { xpos: MjArray; xquat: MjArray };
  delete(): void;
}

export interface MjVFS {
  addBuffer(name: string, data: Uint8Array): void;
  deleteFile(name: string): void;
}

export interface Mujoco {
  MjVFS: new () => MjVFS;
  MjModel: { from_xml_string(xml: string, vfs?: MjVFS): MjModel };
  MjData: new (model: MjModel) => MjData;
  mjtObj: Record<string, { value: number }>;
  mj_step(model: MjModel, data: MjData): void;
  mj_forward(model: MjModel, data: MjData): void;
  mj_resetData(model: MjModel, data: MjData): void;
  mj_resetDataKeyframe(model: MjModel, data: MjData, key: number): void;
  mj_name2id(model: MjModel, type: number, name: string): number;
}

/** mjtGeom values we can turn into three.js geometry. */
export const GEOM = {
  PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3,
  ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7,
} as const;

let cached: Promise<Mujoco> | null = null;

export function loadMujoco(): Promise<Mujoco> {
  // Resolved on first call, never at module scope: a worker only learns the
  // page's base URL from its init message, and this module is imported long
  // before that lands.
  cached ??= import(/* @vite-ignore */ assetUrl("vendor/mujoco/mujoco.js")).then(
    (mod: { default: () => Promise<Mujoco> }) => mod.default(),
  );
  return cached;
}
