/** Serializable, bounded playground objects in MuJoCo's meter/radian, Z-up frame. */
export type Vec3 = [number, number, number];
export type Color4 = [number, number, number, number];
export const objectKinds = ["ball", "box", "cylinder", "ramp", "steps", "hurdle", "platform", "domino"] as const;
export type ObjectKind = typeof objectKinds[number];

export interface WorldObject {
  id: string;
  kind: ObjectKind;
  /** Body center; size is full dimensions, not MuJoCo half-extents. */
  position: Vec3;
  size: Vec3;
  yaw: number;
  dynamic: boolean;
  mass: number;
  color: Color4;
  steps?: number;
}

export interface WorldConfig { version: 1; objects: WorldObject[] }
export const MAX_WORLD_OBJECTS = 32;
export const worldStorageKey = "wicroduck.world.v1";
export const objectCatalogue: { kind: ObjectKind; label: string; dynamic: boolean }[] = [
  { kind: "ball", label: "Ball", dynamic: true },
  { kind: "box", label: "Box", dynamic: true },
  { kind: "cylinder", label: "Cylinder", dynamic: true },
  { kind: "ramp", label: "Ramp", dynamic: false },
  { kind: "steps", label: "Steps", dynamic: false },
  { kind: "hurdle", label: "Hurdle", dynamic: false },
  { kind: "platform", label: "Platform", dynamic: false },
  { kind: "domino", label: "Domino", dynamic: true },
];

const colors: Record<ObjectKind, Color4> = {
  ball: [0.1, 0.83, 0.9, 1], box: [1, 0.36, 0.31, 1], cylinder: [0.23, 0.83, 0.5, 1],
  ramp: [0.2, 0.68, 0.77, 1], steps: [0.33, 0.79, 0.54, 1], hurdle: [1, 0.61, 0.27, 1],
  platform: [0.56, 0.6, 0.64, 1], domino: [0.95, 0.38, 0.48, 1],
};
const sizes: Record<ObjectKind, Vec3> = {
  ball: [0.1, 0.1, 0.1], box: [0.1, 0.1, 0.1], cylinder: [0.09, 0.09, 0.14],
  ramp: [0.6, 0.4, 0.12], steps: [0.6, 0.4, 0.18], hurdle: [0.05, 0.38, 0.16],
  platform: [0.4, 0.4, 0.1], domino: [0.025, 0.1, 0.18],
};

export function defaultWorld(): WorldConfig { return { version: 1, objects: [] }; }
export function worldBodyName(id: string): string { return `world_${id}`; }
export function worldJointName(id: string): string { return `world_joint_${id}`; }

export function createWorldObject(kind: ObjectKind, id: string): WorldObject {
  if (!objectKinds.includes(kind)) throw new Error("Unknown world object kind");
  const size = [...sizes[kind]] as Vec3;
  return { id, kind, size, position: [0.65, 0, size[2] / 2 + 0.002], yaw: 0,
    dynamic: objectCatalogue.find((entry) => entry.kind === kind)!.dynamic,
    mass: 0.12, color: [...colors[kind]], ...(kind === "steps" ? { steps: 3 } : {}) };
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
}
function fields(value: Record<string, unknown>, allowed: string[], label: string): void {
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new Error(`${label}: unknown field ${key}`);
}
function scalar(value: unknown, low: number, high: number, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high)
    throw new Error(`${label} must be finite and between ${low} and ${high}`);
  return value;
}
function vector(value: unknown, count: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== count) throw new Error(`${label} must have ${count} numbers`);
  return Array.from(value, (item) => scalar(item, -10, 10, label));
}

/** Reject unknown fields and invalid geometry instead of silently repairing a saved scene. */
export function validateWorld(value: unknown): WorldConfig {
  const world = record(value, "World");
  fields(world, ["version", "objects"], "World");
  if (world.version !== 1) throw new Error("World version must be 1");
  if (!Array.isArray(world.objects) || world.objects.length > MAX_WORLD_OBJECTS)
    throw new Error(`World objects must be an array of at most ${MAX_WORLD_OBJECTS} entries`);
  const ids = new Set<string>();
  const objects = Array.from(world.objects, (raw, index): WorldObject => {
    const item = record(raw, `Object ${index}`);
    fields(item, ["id", "kind", "position", "size", "yaw", "dynamic", "mass", "color", "steps"], `Object ${index}`);
    if (typeof item.id !== "string" || !/^[a-zA-Z][a-zA-Z0-9_-]{0,39}$/.test(item.id))
      throw new Error("Object id must start with a letter and use at most 40 letters, digits, _ or -");
    if (ids.has(item.id)) throw new Error(`Duplicate object id: ${item.id}`);
    ids.add(item.id);
    if (!objectKinds.includes(item.kind as ObjectKind)) throw new Error(`Unknown object kind: ${String(item.kind)}`);
    const kind = item.kind as ObjectKind;
    const position = vector(item.position, 3, `${item.id}.position`) as Vec3;
    scalar(position[0], -5, 5, "Position x"); scalar(position[1], -5, 5, "Position y");
    scalar(position[2], 0, 3, "Position z");
    const size = vector(item.size, 3, `${item.id}.size`) as Vec3;
    size.forEach((part) => scalar(part, 0.02, 2, "Object size"));
    if ((kind === "ball" && (size[0] !== size[1] || size[0] !== size[2])) ||
        (kind === "cylinder" && size[0] !== size[1]))
      throw new Error(`${kind} requires matching circular dimensions`);
    if (typeof item.dynamic !== "boolean") throw new Error("Object dynamic must be boolean");
    if (item.dynamic && !objectCatalogue.find((entry) => entry.kind === kind)!.dynamic)
      throw new Error(`${kind} must remain static`);
    const color = vector(item.color, 4, `${item.id}.color`) as Color4;
    color.forEach((part) => scalar(part, 0, 1, "Object color"));
    const result: WorldObject = { id: item.id, kind, position, size, dynamic: item.dynamic,
      yaw: scalar(item.yaw, -2 * Math.PI, 2 * Math.PI, "Object yaw"),
      mass: scalar(item.mass, 0.01, 20, "Object mass"), color };
    if (kind === "steps") {
      result.steps = scalar(item.steps ?? 3, 2, 8, "Step count");
      if (!Number.isInteger(result.steps)) throw new Error("Step count must be an integer");
    } else if (item.steps !== undefined) throw new Error("Only steps objects have a step count");
    return result;
  });
  return { version: 1, objects };
}

/** Place a new object near the origin without overlapping props or the robot spawn. */
export function placeNewObject(world: WorldConfig, object: WorldObject): WorldObject {
  const existing = validateWorld(world);
  const combined = validateWorld({ version: 1, objects: [...existing.objects, object] });
  const candidate = combined.objects[combined.objects.length - 1];
  const margin = 0.06;
  const footprint = (item: WorldObject) => {
    const c = Math.abs(Math.cos(item.yaw)), s = Math.abs(Math.sin(item.yaw));
    return { x: item.position[0], y: item.position[1],
      hx: (c * item.size[0] + s * item.size[1]) / 2,
      hy: (s * item.size[0] + c * item.size[1]) / 2 };
  };
  // Boxes conservatively cover ramps, stairs, and rotated circular props too.
  const occupied = [{ x: 0, y: 0, hx: 0.22, hy: 0.22 }, ...existing.objects.map(footprint)];
  const { hx, hy } = footprint(candidate);
  const positions: [number, number][] = [];
  for (let ix = -20; ix <= 20; ix++) {
    for (let iy = -20; iy <= 20; iy++) positions.push([ix * 0.25, iy * 0.25]);
  }
  positions.sort(([ax, ay], [bx, by]) =>
    (ax * ax + ay * ay) - (bx * bx + by * by) || bx - ax || ay - by);
  for (const [x, y] of positions) {
    if (occupied.every(other => Math.abs(x - other.x) >= hx + other.hx + margin ||
        Math.abs(y - other.y) >= hy + other.hy + margin)) {
      candidate.position = [x, y, candidate.position[2]];
      return candidate;
    }
  }
  throw new Error("No clear placement is available within the playground bounds");
}

function placed(kind: ObjectKind, id: string, x: number, y: number, overrides: Partial<WorldObject> = {}): WorldObject {
  const object = createWorldObject(kind, id);
  return { ...object, position: [x, y, object.position[2]], ...overrides };
}

export const worldPresets: { id: string; label: string; description: string; world: WorldConfig }[] = [
  { id: "plain", label: "Clear floor", description: "Open floor", world: defaultWorld() },
  { id: "playground", label: "Playground", description: "Movable props and low obstacles", world: { version: 1, objects: [
    placed("ball", "ball-1", 0.55, -0.15), placed("box", "box-1", 0.8, 0.2),
    placed("cylinder", "cylinder-1", 0.9, -0.4), placed("ramp", "ramp-1", 1.3, 0.45),
    placed("steps", "steps-1", 1.9, -0.3), placed("hurdle", "hurdle-1", -0.75, 0.35),
    placed("platform", "platform-1", -0.85, -0.45),
  ] } },
  { id: "obstacle-course", label: "Obstacle course", description: "Ramp, stairs and hurdles", world: { version: 1, objects: [
    placed("ramp", "ramp-1", 0.8, 0), placed("steps", "steps-1", 1.5, 0),
    placed("hurdle", "hurdle-1", 2.15, 0), placed("platform", "platform-1", 2.65, 0),
  ] } },
  { id: "ball-play", label: "Ball play", description: "Three free-moving balls", world: { version: 1, objects: [
    placed("ball", "ball-1", 0.55, 0),
    placed("ball", "ball-2", 0.8, 0.3, { color: [1, 0.36, 0.31, 1] }),
    placed("ball", "ball-3", 0.9, -0.3, { color: [0.23, 0.83, 0.5, 1] }),
  ] } },
  { id: "domino-run", label: "Domino run", description: "Nine movable dominoes", world: { version: 1,
    objects: Array.from({ length: 9 }, (_, i) => placed("domino", `domino-${i + 1}`, 0.55 + i * 0.08, 0)) } },
];

/** Internal MJCF fragments, including freejoint coordinates in document order. */
export function worldXml(value: WorldConfig): { assets: string; bodies: string; qpos: number[] } {
  const world = validateWorld(value);
  const assets: string[] = [], bodies: string[] = [], qpos: number[] = [];
  for (const object of world.objects) {
    const { id, kind, size: [x, y, z], position, yaw, color, mass, dynamic } = object;
    const quat = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
    const geoms: string[] = [];
    const geom = (type: string, size: number[], offset: number[] = [0, 0, 0], extra = "") => {
      geoms.push(`<geom name="world_geom_${id}_${geoms.length}" type="${type}" size="${size.join(" ")}" pos="${offset.join(" ")}" rgba="${color.join(" ")}" group="0" contype="1" conaffinity="1" friction="0.8 0.02 0.002" ${dynamic ? `mass="${mass}"` : ""} ${extra}/>`);
    };
    if (kind === "ball") geom("sphere", [x / 2]);
    else if (kind === "cylinder") geom("cylinder", [x / 2, z / 2]);
    else if (kind === "steps") {
      const count = object.steps!;
      for (let i = 0; i < count; ++i) {
        const height = z * (i + 1) / count;
        geom("box", [x / count / 2, y / 2, height / 2], [-x / 2 + x * (i + 0.5) / count, 0, (height - z) / 2]);
      }
    } else if (kind === "hurdle") {
      const post = Math.min(x, y * 0.12), bar = Math.min(z * 0.18, 0.025);
      geom("box", [x / 2, post / 2, (z - bar) / 2], [0, (y - post) / 2, -bar / 2]);
      geom("box", [x / 2, post / 2, (z - bar) / 2], [0, -(y - post) / 2, -bar / 2]);
      geom("box", [x / 2, y / 2, bar / 2], [0, 0, (z - bar) / 2]);
    } else if (kind === "ramp") {
      const vertices = [-x / 2, -y / 2, -z / 2, x / 2, -y / 2, -z / 2,
        x / 2, y / 2, -z / 2, -x / 2, y / 2, -z / 2, x / 2, -y / 2, z / 2, x / 2, y / 2, z / 2];
      assets.push(`<mesh name="world_mesh_${id}" vertex="${vertices.join(" ")}" face="0 2 1 0 3 2 1 2 5 1 5 4 0 1 4 3 5 2 0 4 5 0 5 3" smoothnormal="true"/>`);
      geom("mesh", [1, 1, 1], [0, 0, 0], `mesh="world_mesh_${id}"`);
    } else geom("box", [x / 2, y / 2, z / 2]);
    if (dynamic) qpos.push(...position, ...quat);
    bodies.push(`<body name="${worldBodyName(id)}" pos="${position.join(" ")}" quat="${quat.join(" ")}">${dynamic ? `<freejoint name="${worldJointName(id)}"/>` : ""}${geoms.join("")}</body>`);
  }
  return { assets: assets.length ? `<asset>${assets.join("")}</asset>` : "", bodies: bodies.join("\n"), qpos };
}
