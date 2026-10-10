// Real WASM physics gate: no rendering mocks or XML-only success claims.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco, { type MjModel, type MjData } from "@mujoco/mujoco";
import { compileScene, type ModelAssets } from "../src/sim/scene.ts";
import { DEFAULT_POSE } from "../src/sim/microduck.ts";
import { createWorldObject, defaultWorld, MAX_WORLD_OBJECTS, objectKinds, placeNewObject, validateWorld,
  worldBodyName, worldJointName, worldPresets, type WorldConfig, type WorldObject } from "../src/sim/world.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = path.join(ROOT, "public/model/microduck");
const ROBOT_XML = "robot_walk-nv.xml";
const good = (): WorldConfig => ({ version: 1, objects: [createWorldObject("ball", "ball-1")] });
const invalid: unknown[] = [null, {}, { version: 2, objects: [] }, { version: 1, objects: {} },
  { ...defaultWorld(), extra: true }, { version: 1, objects: new Array(1) },
  { version: 1, objects: Array.from({ length: MAX_WORLD_OBJECTS + 1 }, (_, i) => createWorldObject("box", `b${i}`)) }];
for (const change of [
  { id: 'bad"/>' }, { id: "" }, { kind: "cone" }, { position: [0, 0, NaN] },
  { position: [6, 0, 0] }, { position: [0, 0, -1] }, { position: [0, 0, 4] },
  { size: [0, 0, 0] }, { size: [3, 3, 3] }, { size: [0.1, 0.2, 0.1] },
  { size: new Array(3) }, { color: new Array(4) }, { mass: 0 }, { mass: Infinity },
  { yaw: NaN }, { yaw: 10 }, { color: [1, 2, 0, 1] }, { dynamic: "yes" },
  { steps: 3 }, { unknown: true },
]) invalid.push({ version: 1, objects: [{ ...good().objects[0], ...change }] });
invalid.push({ version: 1, objects: [good().objects[0], good().objects[0]] });
invalid.push({ version: 1, objects: [{ ...createWorldObject("ramp", "ramp"), dynamic: true }] });
invalid.push({ version: 1, objects: [{ ...createWorldObject("steps", "stairs"), steps: 2.5 }] });
invalid.forEach((value) => assert.throws(() => validateWorld(value)));
const source = good(), normalized = validateWorld(source);
normalized.objects[0].position[0] = 2;
assert.equal(source.objects[0].position[0], 0.65, "validation must return detached arrays");
assert.deepEqual(validateWorld(JSON.parse(JSON.stringify(source))), source);
assert.equal(defaultWorld().objects.length, 0);
for (const preset of worldPresets) assert.deepEqual(validateWorld(preset.world), preset.world);
for (const kind of objectKinds) validateWorld({ version: 1, objects: [createWorldObject(kind, "object")] });
console.log(`PASS ${invalid.length} invalid worlds rejected; presets, detached copies and JSON round trip`);

function xyBounds(object: WorldObject) {
  const points = [-1, 1].flatMap(sx => [-1, 1].map(sy => {
    const x = sx * object.size[0] / 2, y = sy * object.size[1] / 2;
    return [object.position[0] + x * Math.cos(object.yaw) - y * Math.sin(object.yaw),
      object.position[1] + x * Math.sin(object.yaw) + y * Math.cos(object.yaw)];
  }));
  return { left: Math.min(...points.map(p => p[0])), right: Math.max(...points.map(p => p[0])),
    bottom: Math.min(...points.map(p => p[1])), top: Math.max(...points.map(p => p[1])) };
}
function clearBounds(a: ReturnType<typeof xyBounds>, b: ReturnType<typeof xyBounds>): boolean {
  const gap = 0.06 - 1e-10;
  return a.left >= b.right + gap || b.left >= a.right + gap ||
    a.bottom >= b.top + gap || b.bottom >= a.top + gap;
}
const populated = defaultWorld();
for (let round = 0; round < 4; round++) {
  for (const [index, kind] of objectKinds.entries()) {
    const object = createWorldObject(kind, `${kind}-${round}`);
    object.yaw = round * Math.PI / 4 + index * Math.PI / 12;
    object.position[2] = 0.21;
    const beforeWorld = JSON.stringify(populated), beforeObject = JSON.stringify(object);
    const placed = placeNewObject(populated, object);
    assert.deepEqual(placeNewObject(populated, object), placed, "placement is deterministic");
    assert.equal(JSON.stringify(populated), beforeWorld, "placement does not mutate existing world");
    assert.equal(JSON.stringify(object), beforeObject, "placement does not mutate input object");
    assert.notEqual(placed, object);
    assert.notEqual(placed.size, object.size);
    assert.notEqual(placed.color, object.color);
    assert.equal(placed.position[2], object.position[2], "placement preserves Z");
    assert.equal(placed.yaw, object.yaw, "placement preserves yaw");
    assert(Math.abs(placed.position[0]) <= 5 && Math.abs(placed.position[1]) <= 5);
    const bounds = xyBounds(placed);
    assert(clearBounds(bounds, { left: -0.22, right: 0.22, bottom: -0.22, top: 0.22 }), "robot spawn stays clear");
    for (const prior of populated.objects) assert(clearBounds(bounds, xyBounds(prior)),
      `${placed.id} clears yaw-expanded footprint of ${prior.id}`);
    populated.objects.push(placed);
  }
}
assert.equal(populated.objects.length, MAX_WORLD_OBJECTS);
assert.throws(() => placeNewObject(populated, createWorldObject("box", "overflow")), /at most/);
assert.throws(() => placeNewObject(good(), good().objects[0]), /Duplicate/);
assert.throws(() => placeNewObject(defaultWorld(), { ...good().objects[0], yaw: NaN }), /finite/);
const packed: WorldConfig = { version: 1, objects: [-4, 0, 4].flatMap((x, i) => [-4, 0, 4].map((y, j) => ({
  ...createWorldObject("platform", `packed-${i}-${j}`), position: [x, y, 0.1] as [number, number, number],
  size: [2, 2, 0.2] as [number, number, number],
}))) };
assert.throws(() => placeNewObject(packed, { ...createWorldObject("box", "large"), size: [2, 2, 0.2] }),
  /No clear placement/, "a bounded search fails instead of overlapping an occupied floor");
console.log("PASS 32 deterministic placements across all 8 kinds clear rotated footprints and robot spawn; full/duplicate/packed rejection");

const mujoco = await loadMujoco();
const manifest = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "manifest.json"), "utf8"));
const entry = manifest.models.find((item: { file: string }) => item.file === ROBOT_XML);
assert(entry, "run prepare-assets before checking world physics");
const visualEntry = manifest.models[0] as { file: string; meshes: string[] };
const vfs = new mujoco.MjVFS();
for (const filename of new Set([ROBOT_XML, visualEntry.file])) {
  vfs.addBuffer(filename, fs.readFileSync(path.join(MODEL_DIR, filename)));
}
for (const mesh of new Set<string>([...entry.meshes, ...visualEntry.meshes])) {
  vfs.addBuffer(`assets/${mesh}`, fs.readFileSync(path.join(MODEL_DIR, "assets", mesh)));
}
const assets = { mujoco, vfs, manifest } as unknown as ModelAssets;

function simulation(world?: WorldConfig, robotXml = ROBOT_XML) {
  const compiled = compileScene(assets, { robotXml, world });
  const model = compiled.model as unknown as MjModel;
  const data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, compiled.standKey);
  mujoco.mj_forward(model, data);
  return { model, data, standKey: compiled.standKey, free() { data.delete(); model.delete(); } };
}
function bodyId(model: MjModel, id: string): number {
  const result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, worldBodyName(id));
  assert(result > 0, `body ${id} exists`);
  return result;
}
function contactPairs(model: MjModel, data: MjData): Set<string> {
  const pairs = new Set<string>();
  const contacts = data.contact;
  for (let i = 0; i < data.ncon; i++) {
    const contact = contacts.get(i)!;
    const pair = [Number(model.geom_bodyid[contact.geom1]), Number(model.geom_bodyid[contact.geom2])].sort((a, b) => a - b);
    pairs.add(pair.join(":"));
    contact.delete();
  }
  contacts.delete();
  return pairs;
}
function pairKey(a: number, b: number): string { return [a, b].sort((x, y) => x - y).join(":"); }
function stepped(model: MjModel, data: MjData, count: number): Set<string> {
  const contacts = new Set<string>();
  for (let i = 0; i < count; ++i) {
    mujoco.mj_step(model, data);
    for (const pair of contactPairs(model, data)) contacts.add(pair);
    assert(Array.from(data.qpos as Float64Array).every(Number.isFinite), "qpos stays finite");
    assert(Array.from(data.qvel as Float64Array).every(Number.isFinite), "qvel stays finite");
  }
  return contacts;
}
function checkRampNormals(model: MjModel, world: WorldConfig): void {
  for (const object of world.objects.filter((item) => item.kind === "ramp")) {
    const mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH.value, `world_mesh_${object.id}`);
    assert(mesh >= 0, "ramp mesh exists");
    assert.equal(Number(model.mesh_vertnum[mesh]), Number(model.mesh_normalnum[mesh]), "ramp has vertex normals");
    assert.equal(Number(model.mesh_vertadr[mesh]), Number(model.mesh_normaladr[mesh]), "viewer normal offset matches");
    const adr = Number(model.mesh_faceadr[mesh]) * 3, count = Number(model.mesh_facenum[mesh]) * 3;
    assert.deepEqual(Array.from(model.mesh_face.slice(adr, adr + count)),
      Array.from(model.mesh_facenormal.slice(adr, adr + count)), "viewer face/normal indices match");
  }
}

const baseline = simulation();
const original = Array.from(baseline.data.qpos as Float64Array);
assert.equal(baseline.model.nq, 21);
assert.equal(baseline.model.nu, 14);
assert.deepEqual(original, [0, 0, 0.12, 1, 0, 0, 0, ...DEFAULT_POSE]);
const empty = simulation(defaultWorld());
assert.deepEqual(Array.from(empty.data.qpos as Float64Array), original);
assert.equal(empty.model.ngeom, baseline.model.ngeom);
empty.free();
baseline.free();
for (const preset of worldPresets) {
  const sim = simulation(preset.world);
  assert.deepEqual(Array.from(sim.data.qpos.slice(0, 21)), original, "robot STAND layout is unchanged");
  assert.equal(sim.model.nq, 21 + 7 * preset.world.objects.filter((object) => object.dynamic).length);
  let offset = 21;
  for (const object of preset.world.objects) {
    bodyId(sim.model, object.id);
    if (!object.dynamic) continue;
    assert.equal(Number(sim.model.jnt(worldJointName(object.id)).qposadr), offset);
    const expected = [...object.position, Math.cos(object.yaw / 2), 0, 0, Math.sin(object.yaw / 2)];
    expected.forEach((value, i) => assert(Math.abs(sim.data.qpos[offset + i] - value) < 1e-12));
    offset += 7;
  }
  checkRampNormals(sim.model, preset.world);
  stepped(sim.model, sim.data, 10);
  sim.free();
}
console.log("PASS default STAND and all preset freejoint/body layouts; ramp mesh renderer contract");
const playground = worldPresets.find((preset) => preset.id === "playground")!.world;
const visualScene = simulation(playground, visualEntry.file);
assert.deepEqual(Array.from(visualScene.data.qpos.slice(0, 21)), original, "visual robot STAND layout is unchanged");
checkRampNormals(visualScene.model, playground);
stepped(visualScene.model, visualScene.data, 10);
visualScene.free();
console.log(`PASS full visual ${visualEntry.file} with playground ramp normals and finite stepping`);

const rotated = createWorldObject("box", "rotated");
rotated.yaw = 0.73;
rotated.position = [1, 1, 0.2];
const rotatedScene = simulation({ version: 1, objects: [rotated] });
assert(Math.abs(rotatedScene.data.qpos[24] - Math.cos(rotated.yaw / 2)) < 1e-12);
assert(Math.abs(rotatedScene.data.qpos[27] - Math.sin(rotated.yaw / 2)) < 1e-12);
rotatedScene.free();

const drops = ["ball", "box", "cylinder", "domino"].map((kind, i) => ({
  ...createWorldObject(kind as WorldObject["kind"], `drop-${i}`), position: [0.8 + i * 0.4, -1.5, 0.55] as [number, number, number],
}));
const drop = simulation({ version: 1, objects: drops });
const dropContacts = stepped(drop.model, drop.data, 240);
for (const object of drops) {
  const id = bodyId(drop.model, object.id), z = Number(drop.data.body(id).xpos[2]);
  assert(dropContacts.has(pairKey(0, id)), `${object.kind} makes actual floor contact`);
  assert(z < 0.2 && z > 0.009, `${object.kind} falls and settles above the floor: ${z}`);
}
const ballJoint = drop.model.jnt(worldJointName("drop-0"));
const ballQpos = Number(ballJoint.qposadr), ballDof = Number(ballJoint.dofadr);
const oldX = Number(drop.data.qpos[ballQpos]);
drop.data.qvel[ballDof] = 0.7;
stepped(drop.model, drop.data, 120);
assert(drop.data.qpos[ballQpos] > oldX + 0.1, "free ball moves under a real initial velocity");
mujoco.mj_resetDataKeyframe(drop.model, drop.data, drop.standKey);
assert.equal(drop.data.qpos[ballQpos], drops[0].position[0], "reset restores world freejoint position");
drop.free();
console.log("PASS ball, box, cylinder and domino dynamic drop/contact; ball impulse and reset");

for (const kind of ["ramp", "steps", "hurdle", "platform"] as const) {
  const obstacle = createWorldObject(kind, "obstacle");
  obstacle.position = [1.5, -1.5, obstacle.size[2] / 2];
  const ball = createWorldObject("ball", "probe");
  ball.position = [1.5 + (kind === "steps" || kind === "ramp" ? obstacle.size[0] * 0.3 : 0), -1.5, 0.6];
  const sim = simulation({ version: 1, objects: [obstacle, ball] });
  const obstacleId = bodyId(sim.model, obstacle.id), ballId = bodyId(sim.model, ball.id);
  const contacts = stepped(sim.model, sim.data, 300);
  assert(contacts.has(pairKey(obstacleId, ballId)), `${kind} must collide with a real falling ball`);
  const position = Array.from(sim.data.body(ballId).xpos as Float64Array);
  if (kind === "ramp") assert(position[0] < ball.position[0] - 0.1, "ball rolls down the real ramp");
  else assert(position[2] > obstacle.size[2] + 0.025, `${kind} supports the ball above its top`);
  sim.free();
}
console.log("PASS ramp, steps, hurdle and platform actual ball contacts; ramp rolling motion");

const dominos = worldPresets.find((preset) => preset.id === "domino-run")!.world;
const dominoScene = simulation(dominos);
const firstDof = Number(dominoScene.model.jnt(worldJointName("domino-1")).dofadr);
dominoScene.data.qvel[firstDof] = 1.1;
const dominoContacts = stepped(dominoScene.model, dominoScene.data, 300);
assert(dominoContacts.has(pairKey(bodyId(dominoScene.model, "domino-1"), bodyId(dominoScene.model, "domino-2"))),
  "a moving domino collides with its neighbor");
let moved = 0;
for (const domino of dominos.objects) {
  const qadr = Number(dominoScene.model.jnt(worldJointName(domino.id)).qposadr);
  if (Math.abs(dominoScene.data.qpos[qadr] - domino.position[0]) > 0.02) moved++;
}
assert(moved >= 3, "contact transfers motion through the domino row");
dominoScene.free();
console.log(`PASS domino chain transfers motion to ${moved} bodies`);
vfs.delete();
console.log("WORLD PHYSICS CHECKS PASSED");
