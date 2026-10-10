// Scene replacements wait for in-flight inference before freeing MuJoCo data.
import { compileScene, loadModelAssets, type ModelAssets, type Simulation } from "../sim/scene.ts";
import { CheckpointPolicy, Policy, type PolicyRunner } from "../sim/policy.ts";
import { SHIPPED_POLICIES } from "../sim/policy-catalog.ts";
import { listCheckpoints, loadCheckpoint } from "../train/checkpoint-store.ts";
import { MicroduckController, type Phase } from "../sim/controller.ts";
import { CTRL_DT, TRUNK_BODY } from "../sim/microduck.ts";
import {
  createWorldObject, defaultWorld, placeNewObject, validateWorld, worldPresets, worldJointName,
  type WorldConfig, type WorldObject, type ObjectKind,
} from "../sim/world.ts";
import { Viewer } from "../render/viewer.ts";
import { assetUrl } from "../asset-url.ts";

export const POLICY_NAME = "alpha_stand.onnx";
export const WORLD_STORAGE_KEY = "wicroduck.world.v1";
export interface PolicyOption { id: string; label: string; kind: "shipped" | "checkpoint" }
export const PHASE_LABEL: Record<Phase, string> = {
  standing: "Policy active", limp: "Falling", settling: "Settling", recovering: "Getting up",
};

function safeCheckpointName(name: unknown): name is string {
  return typeof name === "string" && /^[A-Za-z0-9._-]{1,64}$/.test(name) &&
    !name.includes("..") && !/^\.+$/.test(name);
}

export class Session {
  loadStage = $state("Starting...");
  loadProgress = $state(0);
  ready = $state(false);
  started = $state(false);
  error = $state<string | null>(null);
  phase = $state<Phase>("standing");
  uprightPct = $state(0);
  heightCm = $state(0);
  policyHz = $state(0);
  simTime = $state(0);
  paused = $state(false);
  speed = $state(1);
  follow = $state(true);
  forward = $state(0);
  lateral = $state(0);
  turn = $state(0);
  policies = $state<PolicyOption[]>([]);
  activePolicy = $state(POLICY_NAME);
  policyError = $state<string | null>(null);
  policyBusy = $state(false);
  checkpointWorld = $state<WorldConfig | null>(null);
  checkpointLabel = $state("");
  world = $state<WorldConfig>(defaultWorld());
  worldBusy = $state(false);
  worldError = $state<string | null>(null);
  selectedObject = $state("");
  selectedSpeed = $state(0);
  sceneName = $state("Custom");

  #autoRepeat = $state(false);
  #showCollision = $state(false);
  #active = true;
  #viewer: Viewer | null = null;
  #assets: ModelAssets | null = null;
  #sim: Simulation | null = null;
  #controller: MicroduckController | null = null;
  #policy: PolicyRunner | null = null;
  #cache = new Map<string, PolicyRunner>();
  #step: Promise<void> | null = null;
  #frame = 0;
  #clockReset = true;
  #repeatTimer: ReturnType<typeof setTimeout> | undefined;
  #disposed = false;
  #objectSerial = 0;

  get autoRepeat(): boolean { return this.#autoRepeat; }
  get activePolicyLabel(): string { return this.policies.find(p => p.id === this.activePolicy)?.label ?? this.activePolicy; }
  get commandEnabled(): boolean { return !!SHIPPED_POLICIES.find(p => p.id === this.activePolicy)?.commands; }
  get policyDescription(): string { return SHIPPED_POLICIES.find(p => p.id === this.activePolicy)?.description ?? this.checkpointLabel; }
  get selectedCheckpointName(): string | null { return this.activePolicy.startsWith("ckpt:") ? this.activePolicy.slice(5) : null; }
  get checkpointSceneMatches(): boolean {
    return this.checkpointWorld !== null && JSON.stringify(this.world) === JSON.stringify(this.checkpointWorld);
  }
  get object(): WorldObject | undefined { return this.world.objects.find(o => o.id === this.selectedObject); }
  get showCollision(): boolean { return this.#showCollision; }
  set showCollision(value: boolean) { this.#showCollision = value; this.#viewer?.setCollisionVisible(value); }

  async start(canvas: HTMLCanvasElement): Promise<void> {
    this.started = true;
    try {
      const viewer = new Viewer(canvas);
      this.#viewer = viewer;
      viewer.render();
      try {
        const saved = localStorage.getItem(WORLD_STORAGE_KEY);
        if (saved) this.world = validateWorld(JSON.parse(saved));
        else this.world = validateWorld(worldPresets.find(p => p.id === "playground")?.world ?? defaultWorld());
      } catch (err) { this.worldError = `Saved scene not loaded: ${String(err)}`; }
      this.#assets = await loadModelAssets((stage, done, total) => {
        this.loadStage = total ? `${stage} ${done}/${total}` : stage;
        this.loadProgress = total ? ((done ?? 0) / total) * 100 : 0;
      });
      if (this.#disposed) return;
      this.loadStage = "Loading policy";
      this.#policy = await this.#loadPolicy(POLICY_NAME);
      if (this.#disposed) return;
      this.#replaceScene(this.world);
      this.sceneName = worldPresets.find(p => JSON.stringify(validateWorld(p.world)) === JSON.stringify(this.world))?.label ?? "Custom";
      await this.refreshPolicies();
      this.#persistWorld();
      this.ready = true;
      this.#run();
    } catch (err) { this.error = err instanceof Error ? err.message : String(err); }
  }

  #replaceScene(world: WorldConfig, policy: PolicyRunner | null = this.#policy): void {
    if (!this.#assets || !policy || !this.#viewer) throw new Error("Simulation is not ready");
    const { model, standKey } = compileScene(this.#assets, { world });
    const { mujoco } = this.#assets;
    const previous = this.#sim;
    let data: Simulation["data"] | null = null;
    let viewerTouched = false;
    try {
      data = new mujoco.MjData(model);
      const next = { mujoco, model, data, standKey };
      mujoco.mj_resetDataKeyframe(model, data, standKey);
      mujoco.mj_forward(model, data);
      const controller = new MicroduckController(next, policy);
      controller.onRecoveryEnd = () => {
        if (this.#autoRepeat) this.#scheduleRepeat();
      };
      const trunkId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY);
      viewerTouched = true;
      this.#viewer.build(model, { followBody: trunkId });
      this.#viewer.setCollisionVisible(this.#showCollision);
      this.#viewer.setFollowEnabled(this.follow);
      this.#viewer.sync(model, data);
      this.#viewer.render();
      policy.reset?.();
      // Keep the previous simulation alive until the replacement can render.
      this.#sim = next;
      this.#controller = controller;
      this.#policy = policy;
      this.stopMoving();
      this.#clockReset = true;
    } catch (err) {
      let restoration = "";
      if (viewerTouched && previous) {
        try {
          const trunkId = mujoco.mj_name2id(previous.model, mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY);
          this.#viewer.build(previous.model, { followBody: trunkId });
          this.#viewer.setCollisionVisible(this.#showCollision);
          this.#viewer.setFollowEnabled(this.follow);
          this.#viewer.sync(previous.model, previous.data);
          this.#viewer.render();
        } catch (restoreError) { restoration = `; previous scene rendering could not be restored: ${String(restoreError)}`; }
      }
      data?.delete(); model.delete();
      throw new Error(`${String(err)}${restoration}`);
    }
    previous?.data.delete();
    previous?.model.delete();
  }

  #run(): void {
    let last = performance.now() / 1000;
    let accumulator = 0, steps = 0, window = 0;
    const tick = async (): Promise<void> => {
      if (this.#disposed) return;
      const now = performance.now() / 1000;
      const dt = Math.min(Math.max(0, now - last), 0.1);
      last = now;
      if (this.#clockReset) { accumulator = 0; this.#clockReset = false; }
      if (this.#active && !this.worldBusy && !this.policyBusy) {
        if (!this.paused) {
          accumulator += dt * this.speed;
          try {
            while (accumulator >= CTRL_DT && !this.worldBusy && !this.policyBusy && !this.paused) {
              accumulator -= CTRL_DT;
              this.#step = this.#controller!.step();
              await this.#step;
              this.#step = null;
              if (this.#disposed) return;
              steps++;
            }
          } catch (err) {
            this.#step = null;
            this.paused = true;
            this.policyError = `Simulation paused: ${String(err)}`;
          }
        } else accumulator = 0;
        window += dt;
        if (window >= 0.5) { this.policyHz = Math.round(steps / window); steps = 0; window = 0; }
        this.#render();
      } else accumulator = 0;
      this.#frame = requestAnimationFrame(() => void tick());
    };
    void tick();
  }

  #render(): void {
    if (!this.#sim || !this.#controller || !this.#viewer || this.#disposed) return;
    const t = this.#controller.telemetry();
    this.phase = t.phase;
    this.uprightPct = Math.round(Math.max(0, -t.gravityZ) * 100);
    this.heightCm = Math.round(t.height * 1000) / 10;
    this.simTime = t.simTime;
    const selected = this.object;
    if (selected?.dynamic) {
      const adr = this.#sim.model.jnt(worldJointName(selected.id)).dofadr;
      const v = this.#sim.data.qvel;
      this.selectedSpeed = Math.hypot(v[adr], v[adr + 1], v[adr + 2]);
    } else this.selectedSpeed = 0;
    this.#viewer.sync(this.#sim.model, this.#sim.data);
    this.#viewer.render();
  }

  async #loadPolicy(id: string): Promise<PolicyRunner> {
    const cached = this.#cache.get(id);
    if (cached) return cached;
    const entry = SHIPPED_POLICIES.find(p => p.id === id);
    if (!entry) throw new Error(`Unknown shipped policy: ${id}`);
    const policy = await Policy.load(assetUrl(`policies/${entry.id}`), entry.label);
    this.#cache.set(id, policy);
    return policy;
  }

  async refreshPolicies(): Promise<void> {
    const saved = await listCheckpoints();
    this.policies = [
      ...SHIPPED_POLICIES.map(p => ({ id: p.id, label: p.label, kind: "shipped" as const })),
      ...saved.map(c => ({ id: `ckpt:${c.name}`, label: c.name, kind: "checkpoint" as const })),
    ];
  }

  async replayCheckpoint(name: string): Promise<boolean> {
    if (!safeCheckpointName(name)) {
      this.policyError = "Checkpoint name must be a safe filename of at most 64 characters";
      return false;
    }
    return this.selectPolicy(`ckpt:${name}`);
  }

  async selectPolicy(id: string): Promise<boolean> {
    if (!this.ready || this.policyBusy || this.worldBusy || this.#disposed) return false;
    this.policyBusy = true;
    this.policyError = null;
    try {
      await this.#step;
      if (this.#disposed) return false;
      let policy: PolicyRunner;
      if (id.startsWith("ckpt:")) {
        const name = id.slice(5);
        if (!safeCheckpointName(name)) throw new Error("Checkpoint name must be a safe filename of at most 64 characters");
        const ckpt = await loadCheckpoint<unknown>(name);
        if (!ckpt) throw new Error(`Checkpoint "${name}" is gone`);
        policy = new CheckpointPolicy(ckpt, name);
      } else policy = await this.#loadPolicy(id);
      if (this.#disposed) return false;
      const savedWorld = policy instanceof CheckpointPolicy ? validateWorld(policy.world) : null;
      if (savedWorld && JSON.stringify(savedWorld) !== JSON.stringify(this.world)) {
        this.#replaceScene(savedWorld, policy);
        this.world = savedWorld;
        if (!savedWorld.objects.some(o => o.id === this.selectedObject)) this.selectedObject = savedWorld.objects[0]?.id ?? "";
        this.#persistWorld();
      } else {
        this.#controller!.setPolicy(policy);
        this.#policy = policy;
      }
      this.setAutoRepeat(false);
      this.activePolicy = id;
      this.checkpointWorld = savedWorld;
      this.checkpointLabel = policy instanceof CheckpointPolicy ? `${policy.taskLabel} · iteration ${policy.iteration}${policy.experimental ? " · experimental" : ""}` : "";
      if (savedWorld) {
        this.sceneName = "Training scene";
        this.setPaused(true);
      }
      this.stopMoving();
      try { this.#render(); }
      catch (err) {
        this.setPaused(true);
        this.policyError = `Policy loaded, but rendering failed: ${String(err)}`;
      }
      return true;
    } catch (err) { this.policyError = String(err); return false; }
    finally { this.policyBusy = false; this.#clockReset = true; }
  }

  setActive(value: boolean): void { this.#active = value; this.#clockReset = true; }
  setPaused(value: boolean): void { this.paused = value; this.#clockReset = true; }
  setFollow(value: boolean): void { this.follow = value; this.#viewer?.setFollowEnabled(value); }
  frameWorld(): void {
    const low = [-0.2, -0.2, 0], high = [0.2, 0.2, 0.3];
    for (const o of this.world.objects) for (let axis = 0; axis < 3; axis++) {
      const half = axis === 2 ? o.size[2] / 2 : Math.hypot(o.size[0], o.size[1]) / 2;
      low[axis] = Math.min(low[axis], o.position[axis] - half);
      high[axis] = Math.max(high[axis], o.position[axis] + half);
    }
    const center = low.map((v, i) => (v + high[i]) / 2);
    const radius = Math.hypot(...low.map((v, i) => (high[i] - v) / 2));
    this.follow = false; this.#viewer?.frameWorld(center, radius);
  }
  setCommands(): void {
    if (!this.commandEnabled) { this.stopMoving(); return; }
    this.#controller?.setCommand(this.forward, this.lateral, this.turn);
  }
  stopMoving(): void { this.forward = 0; this.lateral = 0; this.turn = 0; this.#controller?.setCommand(0, 0, 0); }

  async singleStep(): Promise<void> {
    if (!this.ready || !this.paused || this.worldBusy || this.policyBusy || this.#step) return;
    try { this.#step = this.#controller!.step(); await this.#step; this.#render(); }
    catch (err) { this.policyError = String(err); }
    finally { this.#step = null; }
  }

  async #act(action: (controller: MicroduckController) => void): Promise<void> {
    if (!this.ready || this.worldBusy || this.policyBusy) return;
    this.worldBusy = true;
    try { await this.#step; if (this.#controller && !this.#disposed) action(this.#controller); this.#render(); }
    catch (err) { this.worldError = String(err); }
    finally { this.worldBusy = false; this.#clockReset = true; }
  }
  knockDown(): void { void this.#act(c => c.knockDown()); }
  push(): void { void this.#act(c => c.push()); }
  standUp(): void { void this.#act(c => c.standUp()); }
  reset(): void { this.setAutoRepeat(false); this.stopMoving(); void this.#act(c => c.reset()); }
  #scheduleRepeat(): void {
    clearTimeout(this.#repeatTimer);
    this.#repeatTimer = setTimeout(() => {
      this.#repeatTimer = undefined;
      if (!this.#autoRepeat || this.#disposed) return;
      if (this.policyBusy || this.worldBusy) { this.#scheduleRepeat(); return; }
      this.knockDown();
    }, 1200);
  }
  setAutoRepeat(value: boolean): void {
    this.#autoRepeat = value;
    clearTimeout(this.#repeatTimer);
    this.#repeatTimer = undefined;
    if (value) this.knockDown();
  }

  #persistWorld(): void {
    try { localStorage.setItem(WORLD_STORAGE_KEY, JSON.stringify(this.world)); }
    catch { this.worldError = "Scene is active, but browser storage is unavailable. Export JSON to keep it."; }
  }

  async applyWorld(value: unknown, name = "Custom"): Promise<boolean> {
    if (!this.ready || this.worldBusy || this.policyBusy) return false;
    this.worldBusy = true;
    this.worldError = null;
    this.setAutoRepeat(false);
    try {
      const next = validateWorld(value);
      await this.#step;
      if (this.#disposed) return false;
      this.#replaceScene(next);
      this.world = next;
      if (!next.objects.some(o => o.id === this.selectedObject)) this.selectedObject = next.objects[0]?.id ?? "";
      this.sceneName = name;
      this.#persistWorld();
      this.#render();
      return true;
    } catch (err) { this.worldError = String(err); return false; }
    finally { this.worldBusy = false; this.#clockReset = true; }
  }

  async loadPreset(id: string): Promise<void> {
    const preset = worldPresets.find(p => p.id === id);
    if (preset && await this.applyWorld(preset.world, preset.label)) this.frameWorld();
  }
  async addObject(kind: ObjectKind): Promise<void> {
    try {
      let id: string;
      do { id = `${kind}-${++this.#objectSerial}`; } while (this.world.objects.some(o => o.id === id));
      const object = placeNewObject(this.world, createWorldObject(kind, id));
      if (await this.applyWorld({ ...this.world, objects: [...this.world.objects, object] })) this.selectedObject = id;
    } catch (err) { this.worldError = String(err); }
  }
  async updateObject(object: WorldObject): Promise<void> {
    await this.applyWorld({ ...this.world, objects: this.world.objects.map(o => o.id === object.id ? object : o) });
  }
  async removeObject(): Promise<void> {
    await this.applyWorld({ ...this.world, objects: this.world.objects.filter(o => o.id !== this.selectedObject) });
  }
  async duplicateObject(): Promise<void> {
    if (!this.object) return;
    try {
      const clone = JSON.parse(JSON.stringify(this.object)) as WorldObject;
      do { clone.id = `${clone.kind}-${++this.#objectSerial}`; } while (this.world.objects.some(o => o.id === clone.id));
      const placed = placeNewObject(this.world, clone);
      if (await this.applyWorld({ ...this.world, objects: [...this.world.objects, placed] })) this.selectedObject = placed.id;
    } catch (err) { this.worldError = String(err); }
  }
  nudgeObject(): void {
    const object = this.object;
    if (!object?.dynamic) return;
    void this.#act(() => {
      if (!this.#sim) return;
      const adr = this.#sim.model.jnt(worldJointName(object.id)).dofadr;
      this.#sim.data.qvel[adr] += 0.4;
      this.#sim.data.qvel[adr + 2] += 0.1;
    });
  }
  exportWorld(): void {
    const url = URL.createObjectURL(new Blob([JSON.stringify(this.world, null, 2)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = "microduck-scene.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async importWorld(file: File): Promise<void> {
    try {
      if (file.size > 100_000) throw new Error("Scene JSON must be smaller than 100 KB");
      await this.applyWorld(JSON.parse(await file.text()), file.name.replace(/\.json$/i, ""));
    } catch (err) { this.worldError = String(err); }
  }
  destroy(): void {
    this.#disposed = true;
    cancelAnimationFrame(this.#frame);
    clearTimeout(this.#repeatTimer);
    this.#viewer?.dispose(); this.#viewer = null;
    void (async () => {
      try { await this.#step; } catch { /* Inference failures are already surfaced. */ }
      this.#sim?.data.delete(); this.#sim?.model.delete(); this.#sim = null;
    })();
  }
}
