// A generic MuJoCo -> three.js view: it builds one three.js mesh per MuJoCo
// geom straight from the compiled model's arrays, then each frame copies
// data.geom_xpos / data.geom_xmat onto them. Nothing here knows about the
// duck, so it renders any MJCF the sim module compiles.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GEOM, type MjData, type MjModel } from "../sim/mujoco.ts";

/** MuJoCo is z-up, three.js is y-up. One rotation on the root holds the
 *  whole scene instead of converting every pose. */
const Z_UP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const BACKGROUND_COLOR = 0xf5f6f7;
const FLOOR_COLOR = 0xcbd0d2;

/** MuJoCo geom groups: 2 is the visual shell, 3 the collision proxies. The
 *  all-collisions model carries both, at the same poses. */
export const GROUP_VISUAL = 2;
export const GROUP_COLLISION = 3;

function meshGeometry(model: MjModel, meshId: number): THREE.BufferGeometry {
  const vertAdr = model.mesh_vertadr[meshId];
  const vertNum = model.mesh_vertnum[meshId];
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];

  const position = new Float32Array(vertNum * 3);
  const normal = new Float32Array(vertNum * 3);
  for (let i = 0; i < vertNum * 3; i++) {
    position[i] = model.mesh_vert[vertAdr * 3 + i];
    normal[i] = model.mesh_normal[vertAdr * 3 + i];
  }
  const index = new Uint32Array(faceNum * 3);
  for (let i = 0; i < faceNum * 3; i++) index[i] = model.mesh_face[faceAdr * 3 + i];

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(position, 3));
  // MuJoCo emits one normal per vertex for these meshes (mesh_normalnum ==
  // mesh_vertnum and mesh_facenormal == mesh_face), so the normals can be
  // indexed exactly like the positions instead of unwelding every face.
  geo.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  return geo;
}

/** Primitive geoms, sized so a unit build can be scaled by geom_size. */
function primitiveGeometry(type: number, size: Float64Array | number[]): THREE.BufferGeometry | null {
  switch (type) {
    case GEOM.PLANE:
      // MuJoCo planes face local +Z, which is what an unrotated three.js
      // PlaneGeometry already does — the z-up flip lives on the scene root.
      // A size of 0 means "infinite"; a big quad reads as a floor either way.
      return new THREE.PlaneGeometry((size[0] || 10) * 2, (size[1] || 10) * 2, 1, 1);
    case GEOM.SPHERE:
      return new THREE.SphereGeometry(size[0], 24, 16);
    case GEOM.CAPSULE:
      return new THREE.CapsuleGeometry(size[0], size[1] * 2, 6, 16).rotateX(Math.PI / 2);
    case GEOM.ELLIPSOID:
      return new THREE.SphereGeometry(1, 24, 16).scale(size[0], size[1], size[2]);
    case GEOM.CYLINDER:
      return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 24).rotateX(Math.PI / 2);
    case GEOM.BOX:
      return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    default:
      return null; // heightfields and anything else: not rendered
  }
}

export interface ViewerOptions {
  /** Follow the trunk with the camera target instead of orbiting a fixed point. */
  followBody?: number;
}

export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly root = new THREE.Object3D();
  /** three.js object per MuJoCo geom id; null where the geom is not drawn. */
  private readonly geomObjects: (THREE.Mesh | null)[] = [];
  private readonly collisionObjects: THREE.Mesh[] = [];
  /** One geometry per mesh asset, shared by every geom referencing it; owned
   *  here so it is disposed once rather than once per geom. */
  private meshCache = new Map<number, THREE.BufferGeometry>();
  private readonly followTarget = new THREE.Vector3();
  private readonly followDelta = new THREE.Vector3();
  private followBody: number | null = null;
  private followEnabled = true;
  private readonly resizeObserver: ResizeObserver;
  private readonly mat = new THREE.Matrix4();

  private readonly canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene.background = new THREE.Color(BACKGROUND_COLOR);
    this.scene.fog = new THREE.Fog(BACKGROUND_COLOR, 1.5, 6);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.position.set(0.55, 0.28, 0.55);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0.08, 0);
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 4;

    this.root.quaternion.copy(Z_UP);
    this.scene.add(this.root);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x858b88, 1.35);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(0.6, 1.2, 0.5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    // Tight ortho frustum: the whole robot is ~25 cm tall, so the default
    // camera box would spend its entire depth range on empty space.
    const s = 0.6;
    key.shadow.camera.left = -s; key.shadow.camera.right = s;
    key.shadow.camera.top = s; key.shadow.camera.bottom = -s;
    key.shadow.camera.near = 0.1; key.shadow.camera.far = 4;
    key.shadow.bias = -0.0015;
    this.scene.add(key);
    this.scene.add(new THREE.DirectionalLight(0xffffff, 0.5).translateX(-1).translateY(0.5));

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement ?? canvas);
    this.resize();
  }

  /** (Re)build the render objects for a compiled model. */
  build(model: MjModel, options: ViewerOptions = {}): void {
    this.disposeObjects();
    this.root.clear();
    this.followBody = options.followBody ?? null;
    const meshCache = this.meshCache;

    for (let g = 0; g < model.ngeom; g++) {
      const type = model.geom_type[g];
      const group = model.geom_group[g];
      const size = [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];

      let geometry: THREE.BufferGeometry | null;
      let meshId = -1;
      if (type === GEOM.MESH) {
        meshId = model.geom_dataid[g];
        geometry = meshCache.get(meshId) ?? null;
        if (!geometry) {
          geometry = meshGeometry(model, meshId);
          meshCache.set(meshId, geometry);
        }
      } else {
        geometry = primitiveGeometry(type, size);
      }
      if (!geometry) {
        this.geomObjects.push(null);
        continue;
      }

      const matid = model.geom_matid[g];
      const rgba = matid >= 0 ? model.mat_rgba : model.geom_rgba;
      const base = matid >= 0 ? matid * 4 : g * 4;
      const isCollision = group === GROUP_COLLISION;
      const material = new THREE.MeshStandardMaterial({
        color: type === GEOM.PLANE ? new THREE.Color(FLOOR_COLOR)
          : new THREE.Color(rgba[base], rgba[base + 1], rgba[base + 2]),
        opacity: isCollision ? 0.35 : rgba[base + 3],
        transparent: isCollision || rgba[base + 3] < 1,
        roughness: type === GEOM.PLANE ? 0.95 : 0.55,
        metalness: type === GEOM.PLANE ? 0 : 0.15,
        wireframe: isCollision,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData.meshId = meshId;
      mesh.castShadow = type !== GEOM.PLANE;
      mesh.receiveShadow = true;
      mesh.visible = !isCollision;
      mesh.matrixAutoUpdate = false;
      this.root.add(mesh);
      this.geomObjects.push(mesh);
      if (isCollision) this.collisionObjects.push(mesh);
    }
  }

  setCollisionVisible(visible: boolean): void {
    for (const m of this.collisionObjects) m.visible = visible;
  }

  setFollowEnabled(enabled: boolean): void {
    this.followEnabled = enabled;
    if (enabled) {
      this.controls.target.copy(this.followTarget);
      this.camera.position.copy(this.followTarget).add(new THREE.Vector3(0.55, 0.28, 0.55));
    }
  }

  frameWorld(center = [0, 0, 0.1], radius = 1): void {
    this.followEnabled = false;
    this.controls.target.set(center[0], center[2], -center[1]);
    const fov = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * Math.min(1, this.camera.aspect));
    const distance = Math.max(0.8, radius * 1.15 / Math.sin(fov / 2));
    this.controls.maxDistance = Math.max(4, distance * 2);
    this.camera.position.copy(this.controls.target).add(new THREE.Vector3(0.8, 1.1, 1.2).normalize().multiplyScalar(distance));
    this.scene.fog = new THREE.Fog(BACKGROUND_COLOR, distance + 3, distance + 10);
    this.controls.update();
  }

  /** Copy the current MuJoCo geom poses onto the render objects. */
  sync(model: MjModel, data: MjData): void {
    const xpos = data.geom_xpos;
    const xmat = data.geom_xmat;
    for (let g = 0; g < model.ngeom; g++) {
      const obj = this.geomObjects[g];
      if (!obj) continue;
      const p = g * 3, m = g * 9;
      // MuJoCo stores row-major 3x3; three.js Matrix4.set takes row-major too.
      this.mat.set(
        xmat[m + 0], xmat[m + 1], xmat[m + 2], xpos[p + 0],
        xmat[m + 3], xmat[m + 4], xmat[m + 5], xpos[p + 1],
        xmat[m + 6], xmat[m + 7], xmat[m + 8], xpos[p + 2],
        0, 0, 0, 1,
      );
      obj.matrix.copy(this.mat);
      obj.matrixWorldNeedsUpdate = true;
    }

    if (this.followBody !== null) {
      const bp = data.body(this.followBody).xpos;
      // Body pose is in MuJoCo's z-up frame; the root carries the flip.
      this.followTarget.set(bp[0], bp[1], bp[2]).applyQuaternion(Z_UP);
      if (this.followEnabled) {
        this.followDelta.copy(this.followTarget).sub(this.controls.target).multiplyScalar(0.15);
        this.controls.target.add(this.followDelta);
        this.camera.position.add(this.followDelta);
      }
    }
  }

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const box = this.canvas.parentElement ?? this.canvas;
    const w = box.clientWidth;
    const h = box.clientHeight;
    // A hidden stage (another workspace up front) measures 0x0. Keep the last
    // good size rather than collapsing the buffer and the camera aspect —
    // the observer fires again with real numbers when it comes back.
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.disposeObjects();
    this.renderer.dispose();
  }

  private disposeObjects(): void {
    for (const obj of this.geomObjects) {
      if (!obj) continue;
      // Mesh-asset geometries are shared and disposed via meshCache below;
      // primitives are one-per-geom and disposed here.
      if (!this.meshCache.has(obj.userData.meshId ?? -1)) obj.geometry.dispose();
      const mat = obj.material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat.dispose();
    }
    for (const geo of this.meshCache.values()) geo.dispose();
    this.meshCache = new Map();
    this.geomObjects.length = 0;
    this.collisionObjects.length = 0;
  }
}
