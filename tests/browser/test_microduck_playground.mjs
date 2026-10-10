#!/usr/bin/env node
// Uses installed Playwright + sharp. No installs, visible-browser access, or training.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const sharp = require("sharp");
const options = { url: "http://127.0.0.1:8767", output: null };
const supplied = new Set();
for (let i = 2; i < process.argv.length; i += 2) {
  const key = { "--url": "url", "--output": "output" }[process.argv[i]];
  assert(key && process.argv[i + 1] && !process.argv[i + 1].startsWith("--") && !supplied.has(key),
    "Usage: node local/test_microduck_playground.mjs --output NEW_DIRECTORY [--url LOCAL_URL]");
  supplied.add(key);
  options[key] = process.argv[i + 1];
}
assert(options.output, "--output must name a new evidence directory");
const url = new URL(options.url);
assert(["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) && ["http:", "https:"].includes(url.protocol), "Only a local HTTP(S) application is permitted");
const output = path.resolve(options.output);
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.mkdir(output); // Atomic new-only leaf: rejects existing dirs, files and symlinks.

const KINDS = ["ball", "box", "cylinder", "ramp", "steps", "hurdle", "platform", "domino"];
const MOTIONS = ["nod", "look_around", "squat", "bow", "march_in_place", "walk_cycle_reference", "sidestep_reference", "crouch_hold", "jump_preparation", "tuck_preparation", "roll_preparation", "dance_sway", "neck_stretch"];
const WORLD_KEY = "wicroduck.world.v1";
const report = {
  schema_version: 1, started_at: new Date().toISOString(), url: url.href, output,
  node: process.version, browser_executable: chromium.executablePath(),
  scope: "Fresh isolated browser contexts only; scene edits affect test localStorage, never the user's tab or OPFS. No training run or checkpoint operation.",
  limitations: [
    "Canvas changes establish rendered activity, not a validated gait, jump, recovery, or CUDA acceleration.",
    "Walk controls are checked for policy selection and displayed commands; distance or gait quality is not qualified.",
    "Existing object configurations, not their instantaneous simulated positions, are exported and persisted.",
  ],
  results: [], passed: false,
};
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const button = (page, name) => page.getByRole("button", { name, exact: true });
const world = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), WORLD_KEY);
const clock = async page => Number.parseFloat(await page.locator(".transport .time").textContent());

function xyBounds(object) {
  assert.equal(typeof object.id, "string", "Object needs an id for placement evidence");
  assert(Array.isArray(object.position) && object.position.length === 3 && object.position.every(Number.isFinite), `${object.id}: invalid position`);
  assert(Array.isArray(object.size) && object.size.length === 3 && object.size.every(size => Number.isFinite(size) && size > 0), `${object.id}: invalid dimensions`);
  assert(Number.isFinite(object.yaw), `${object.id}: invalid yaw`);
  // Transform all four footprint corners independently of the app's placement
  // helper. This conservatively encloses rotated boxes, ramps and circular props.
  const corners = [-1, 1].flatMap(sx => [-1, 1].map(sy => {
    const x = sx * object.size[0] / 2, y = sy * object.size[1] / 2;
    return [object.position[0] + x * Math.cos(object.yaw) - y * Math.sin(object.yaw),
      object.position[1] + x * Math.sin(object.yaw) + y * Math.cos(object.yaw)];
  }));
  return { left: Math.min(...corners.map(corner => corner[0])), right: Math.max(...corners.map(corner => corner[0])),
    bottom: Math.min(...corners.map(corner => corner[1])), top: Math.max(...corners.map(corner => corner[1])) };
}
function placementAudit(scene, onlyId = null) {
  assert.equal(scene.version, 1);
  assert(Array.isArray(scene.objects) && scene.objects.length > 0, "Placement check needs scene objects");
  assert.equal(new Set(scene.objects.map(object => object.id)).size, scene.objects.length, "Duplicate placement ids");
  const boxes = scene.objects.map(object => ({ id: object.id, ...xyBounds(object) }));
  if (onlyId !== null) assert(boxes.some(box => box.id === onlyId), "Duplicated object missing from scene JSON");
  const spawn = { id: "robot spawn", left: -0.22, right: 0.22, bottom: -0.22, top: 0.22 };
  const margin = 0.06, tolerance = 1e-9;
  const comparisons = [];
  const check = (a, b) => {
    const gap = Math.max(a.left - b.right, b.left - a.right, a.bottom - b.top, b.bottom - a.top);
    assert(gap >= margin - tolerance, `${a.id} overlaps or lacks ${margin}m clearance from ${b.id}: separating-axis gap ${gap}m`);
    comparisons.push({ a: a.id, b: b.id, separating_axis_gap_m: gap });
  };
  boxes.forEach((box, index) => {
    if (onlyId !== null && box.id !== onlyId) return;
    check(box, spawn);
    boxes.forEach((other, otherIndex) => {
      if (onlyId === null ? otherIndex > index : other.id !== onlyId) check(box, other);
    });
  });
  return { scope: onlyId === null ? "All initial automatic placements" : `Only new duplicate ${onlyId}; manually edited existing pairs are not requalified`,
    robot_spawn_bounds: spawn, required_clearance_m: margin, numerical_tolerance_m: tolerance,
    rotated_xy_bounds: boxes, comparisons, minimum_checked_gap_m: Math.min(...comparisons.map(pair => pair.separating_axis_gap_m)) };
}

async function waitReady(page) {
  await page.waitForFunction(() => {
    const error = document.querySelector(".stage .overlay.error");
    if (error) throw new Error(error.textContent);
    return !!document.querySelector(".main:not(.hidden) .transport") && !document.querySelector(".main:not(.hidden) .stage .overlay");
  }, undefined, { timeout: 120_000 });
}
async function waitScene(page, count) {
  await page.waitForFunction(({ key, count }) => {
    const raw = localStorage.getItem(key);
    const add = [...document.querySelectorAll("button")].find(item => item.textContent.trim() === "Add object");
    return raw && JSON.parse(raw).objects.length === count && add && !add.disabled;
  }, { key: WORLD_KEY, count });
}
async function pause(page) {
  if (await button(page, "Pause simulation").count()) await button(page, "Pause simulation").click();
  await button(page, "Play simulation").waitFor();
  await page.waitForTimeout(150); // Allow a control step already in flight to finish.
}
async function screenshot(page, result, name) {
  const file = `${result.viewport.name}-${name}.png`;
  const bytes = await page.screenshot({ path: path.join(output, file), fullPage: true, timeout: 30_000 });
  result.screenshots.push({ file, sha256: hash(bytes) });
}
async function canvasFrame(page, result, name) {
  const file = `${result.viewport.name}-${name}.png`;
  const bytes = await page.locator(".main:not(.hidden) .stage canvas").screenshot({ path: path.join(output, file), timeout: 30_000 });
  const metadata = await sharp(bytes).metadata();
  // Locator screenshots include overlaid HTML. Exclude the clock/toolbar and
  // scene label so changing text alone cannot satisfy the motion pixel gate.
  const region = { left: 8, top: 60, width: metadata.width - 16, height: metadata.height - 92 };
  assert(region.width > 0 && region.height > 0, "Canvas too small for a scene-only pixel region");
  const { data, info } = await sharp(bytes).extract(region).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  assert.equal(info.channels, 3);
  const sums = [0, 0, 0], squares = [0, 0, 0], colors = new Set();
  for (let p = 0; p < data.length; p += 3) {
    colors.add((data[p] << 16) | (data[p + 1] << 8) | data[p + 2]);
    for (let c = 0; c < 3; c++) { sums[c] += data[p + c]; squares[c] += data[p + c] ** 2; }
  }
  const pixels = info.width * info.height;
  const stddev = sums.map((sum, c) => Math.sqrt(Math.max(0, squares[c] / pixels - (sum / pixels) ** 2)));
  const evidence = { file, sha256: hash(bytes), width: metadata.width, height: metadata.height, pixel_analysis_region: region, unique_rgb_colors: colors.size, channel_stddev: stddev };
  result.frames.push(evidence);
  assert(colors.size >= 128 && Math.max(...stddev) >= 5, "Canvas is blank or lacks rendered detail");
  return { data, width: info.width, height: info.height };
}
function frameDifference(a, b) {
  assert.equal(a.width, b.width); assert.equal(a.height, b.height);
  let total = 0, changed = 0;
  for (let p = 0; p < a.data.length; p += 3) {
    let maximum = 0;
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a.data[p + c] - b.data[p + c]);
      total += d; maximum = Math.max(maximum, d);
    }
    if (maximum > 8) changed++;
  }
  return { mean_absolute_rgb_difference: total / a.data.length, changed_pixel_fraction_gt8: changed / (a.width * a.height) };
}
async function layout(page, result, name, training = false) {
  const measured = await page.evaluate(training => {
    const selectors = [".shell", ".rail", ".topbar", "footer.status", ...(training ? [".training-workspace:not(.hidden)", ".train"] : [".main:not(.hidden)", ".stage", ".inspector", ".inspector .content"])];
    const issues = [], boxes = {};
    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (!element) { issues.push(`Missing ${selector}`); continue; }
      const r = element.getBoundingClientRect();
      boxes[selector] = { x: r.x, y: r.y, width: r.width, height: r.height, scroll_width: element.scrollWidth, client_width: element.clientWidth };
      if (r.left < -1 || r.top < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1) issues.push(`${selector} outside viewport`);
      if (element.scrollWidth > element.clientWidth + 1) issues.push(`${selector} horizontal content overflow`);
    }
    if (document.documentElement.scrollWidth > innerWidth + 1 || document.documentElement.scrollHeight > innerHeight + 1) issues.push("Document overflow");
    const footer = boxes["footer.status"];
    if (footer && footer.height > 60) issues.push("Footer consumes an oversized grid row");
    if (!training) {
      const r = document.querySelector(".main:not(.hidden) canvas")?.getBoundingClientRect();
      if (!r || r.width < 180 || r.height < 160) issues.push("Canvas below inspectable 180x160 CSS pixels");
    }
    return { boxes, issues };
  }, training);
  result.layouts.push({ name, ...measured });
  assert.deepEqual(measured.issues, [], `${name}: layout issues`);
}
async function checkpointNames(page) {
  return page.evaluate(async () => {
    if (!navigator.storage?.getDirectory) return { available: false, files: [] };
    const root = await navigator.storage.getDirectory();
    const files = [];
    async function visit(directory, prefix) {
      for await (const [name, handle] of directory.entries()) {
        if (handle.kind === "file") files.push(prefix + name);
        else await visit(handle, `${prefix}${name}/`);
      }
    }
    await visit(root, "");
    return { available: true, files: files.sort() };
  });
}

let browser;
try {
  browser = await chromium.launch({ headless: true, timeout: 30_000 });
  report.browser_version = browser.version();
  for (const viewport of [{ name: "desktop", width: 1360, height: 900 }, { name: "mobile", width: 390, height: 844 }]) {
    const result = { viewport, passed: false, checks: [], screenshots: [], frames: [], layouts: [], page_errors: [], console_errors: [], failed_requests: [], failed_http: [], assets: [] };
    report.results.push(result);
    const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, acceptDownloads: true });
    const page = await context.newPage();
    page.setDefaultTimeout(20_000);
    page.setDefaultNavigationTimeout(60_000);
    page.on("pageerror", e => result.page_errors.push(String(e)));
    page.on("console", message => { if (message.type() === "error") result.console_errors.push(message.text()); });
    page.on("requestfailed", request => result.failed_requests.push({ url: request.url(), error: request.failure()?.errorText }));
    page.on("response", response => {
      if (response.status() >= 400) result.failed_http.push({ url: response.url(), status: response.status() });
      if (/\.(wasm|onnx|xml|stl)(?:\?|$)/i.test(response.url())) result.assets.push({ url: response.url(), status: response.status() });
    });
    const watchdog = setTimeout(() => { result.timeout = "Viewport exceeded 300-second budget"; void context.close().catch(() => {}); }, 300_000);
    try {
      await page.goto(url.href, { waitUntil: "domcontentloaded" });
      await waitReady(page);
      result.opfs_before = await checkpointNames(page);
      assert.deepEqual(result.opfs_before.files, [], "Fresh test context unexpectedly contains files");
      await pause(page);
      const t0 = await clock(page);
      await page.waitForTimeout(250);
      assert.equal(await clock(page), t0, "Paused clock advanced");
      await button(page, "Step simulation").click();
      await page.waitForFunction(t => Math.abs(Number.parseFloat(document.querySelector(".transport .time").textContent) - t - 0.02) < 1e-6, t0);
      result.pause_step = { before: t0, after: await clock(page), expected_control_step_s: 0.02 };
      result.checks.push("pause_and_single_control_step");

      await page.getByRole("tab", { name: /^Scene/ }).click();
      await page.getByLabel("Scene preset", { exact: true }).selectOption("plain");
      await button(page, "Load").click();
      await waitScene(page, 0);
      for (const [i, kind] of KINDS.entries()) {
        await page.getByLabel("Object type", { exact: true }).selectOption(kind);
        await button(page, "Add object").click();
        await waitScene(page, i + 1);
        assert.equal((await world(page)).objects.at(-1).kind, kind);
      }
      result.checks.push("clear_preset_and_add_all_eight_object_kinds");
      const original = await world(page);
      result.initial_placement = placementAudit(original);
      result.checks.push("all_initial_objects_clear_each_other_and_robot_spawn");
      const ball = original.objects.find(item => item.kind === "ball");
      await page.getByLabel("Selected object", { exact: true }).selectOption(ball.id);
      await page.getByLabel("Object position X", { exact: true }).fill("-0.65");
      await page.getByLabel("Object position Z", { exact: true }).fill("0.12");
      await page.getByLabel("Object size X", { exact: true }).fill("0.14");
      await page.getByLabel("Object mass", { exact: true }).fill("0.2");
      await button(page, "Object color 2").click();
      await button(page, "Apply & reset scene").click();
      await page.waitForFunction(({ key, id }) => {
        const object = JSON.parse(localStorage.getItem(key)).objects.find(item => item.id === id);
        return object.position[0] === -0.65 && object.position[2] === 0.12 && object.size.every(x => x === 0.14) && object.mass === 0.2 && object.color[0] === 1;
      }, { key: WORLD_KEY, id: ball.id });
      await page.getByLabel("Object size X", { exact: true }).scrollIntoViewIfNeeded();
      await screenshot(page, result, "object-edit");
      await layout(page, result, "object-edit");
      result.checks.push("edit_object_position_dimensions_mass_and_color");

      await button(page, "Duplicate").click();
      await waitScene(page, 9);
      const duplicateId = await page.getByLabel("Selected object", { exact: true }).inputValue();
      assert.notEqual(duplicateId, ball.id);
      const duplicateWorld = await world(page);
      result.duplicate_placement = placementAudit(duplicateWorld, duplicateId);
      result.checks.push("duplicate_clears_all_existing_objects_and_robot_spawn");
      const duplicated = duplicateWorld.objects.find(item => item.id === duplicateId);
      assert.deepEqual(duplicated.size, [0.14, 0.14, 0.14]);
      await button(page, "Nudge object").click();
      await page.waitForFunction(() => [...document.querySelectorAll(".inspector h2 span")].some(span => /m\/s$/.test(span.textContent.trim()) && Number.parseFloat(span.textContent) > 0.3));
      result.nudge_speed = await page.locator(".inspector h2 span").filter({ hasText: /m\/s/ }).textContent();
      await button(page, "Delete object").click();
      await waitScene(page, 8);
      assert(!(await world(page)).objects.some(item => item.id === duplicateId));
      result.checks.push("duplicate_nudge_and_delete_test_object");

      const preserved = await world(page);
      await page.locator("input[type=file]").setInputFiles({ name: "invalid.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ version: 1, objects: [], unexpected: true })) });
      await page.getByRole("alert").filter({ hasText: /unknown field/ }).waitFor();
      assert.deepEqual(await world(page), preserved, "Rejected import altered the existing scene");
      result.checks.push("invalid_import_preserves_scene");

      const downloaded = page.waitForEvent("download");
      await button(page, "Export JSON").click();
      const download = await downloaded;
      const exportedFile = `${viewport.name}-exported-world.json`;
      await download.saveAs(path.join(output, exportedFile));
      assert.equal(await download.failure(), null);
      const bytes = await fs.readFile(path.join(output, exportedFile));
      const exported = JSON.parse(bytes.toString("utf8"));
      assert.deepEqual(exported, preserved);
      result.export = { file: exportedFile, sha256: hash(bytes), object_count: exported.objects.length };
      await page.getByLabel("Scene preset", { exact: true }).selectOption("plain");
      await button(page, "Load").click(); await waitScene(page, 0);
      await page.locator("input[type=file]").setInputFiles({ name: "probe-scene.json", mimeType: "application/json", buffer: bytes });
      await waitScene(page, 8);
      assert.deepEqual(await world(page), exported);
      await page.reload({ waitUntil: "domcontentloaded" });
      await waitReady(page); await pause(page);
      assert.deepEqual(await world(page), exported, "Scene configuration did not survive reload");
      result.checks.push("export_valid_import_and_reload_persistence");

      await page.getByLabel("Driving policy", { exact: true }).selectOption("velstand.onnx");
      await button(page, "Walk forward").waitFor();
      await button(page, "Walk forward").click();
      assert.equal(await page.getByLabel("Forward velocity", { exact: true }).inputValue(), "0.25");
      assert.equal((await page.locator("label.slider output").allTextContents())[0], "0.25 m/s");
      await button(page, "Stop moving").click();
      assert.equal(await page.getByLabel("Forward velocity", { exact: true }).inputValue(), "0");
      await page.getByLabel("Forward velocity", { exact: true }).focus();
      await page.keyboard.press("ArrowRight");
      assert.equal(await page.getByLabel("Forward velocity", { exact: true }).inputValue(), "0.01");
      await button(page, "Stop moving").click();
      result.checks.push("walking_policy_and_command_controls");

      await button(page, "Frame scene").click();
      await page.waitForTimeout(300);
      await screenshot(page, result, "full-scene");
      await layout(page, result, "full-scene");
      // Inspect motion at robot scale, then lock the camera so tracking cannot
      // satisfy the pixel gate. Keep the wide scene screenshot separately.
      await page.getByLabel("Follow robot", { exact: true }).check();
      await page.getByLabel("Follow robot", { exact: true }).uncheck();
      const before = await canvasFrame(page, result, "canvas-before");
      const motionStart = await clock(page);
      await button(page, "Push").click();
      await button(page, "Play simulation").click();
      await page.waitForFunction(start => Number.parseFloat(document.querySelector(".transport .time").textContent) >= start + 1,
        motionStart, { timeout: 30_000 });
      await pause(page);
      const after = await canvasFrame(page, result, "canvas-after");
      result.motion = { ...frameDifference(before, after), simulated_seconds: await clock(page) - motionStart,
        framing: "Robot scale, follow disabled before both captures" };
      assert(result.motion.mean_absolute_rgb_difference >= 0.05 && result.motion.changed_pixel_fraction_gt8 >= 0.001, "No visible motion after playback and Push");
      result.checks.push("nonblank_canvas_and_visible_physics_motion");

      await button(page, "Train").click();
      await page.waitForFunction(() => document.querySelectorAll('select[aria-label="Training task"] optgroup[label="Motion targets"] option').length >= 13);
      result.motion_options = await page.locator('select[aria-label="Training task"] optgroup[label="Motion targets"] option').allTextContents();
      for (const name of MOTIONS) assert(result.motion_options.includes(name), `Training task missing ${name}`);
      await screenshot(page, result, "training");
      await layout(page, result, "training", true);
      assert.equal(await page.locator(".train-panel .sub .status[role=status]").textContent(), "Ready", "Test unexpectedly started training");
      result.opfs_after = await checkpointNames(page);
      assert.deepEqual(result.opfs_after, result.opfs_before, "Test created or changed OPFS file names");
      result.checks.push("training_layout_and_all_thirteen_motion_options_without_training");

      assert(result.assets.some(asset => /velstand\.onnx(?:\?|$)/i.test(asset.url)), "Walking policy asset was not loaded");
      assert.deepEqual(result.page_errors, [], "Page errors");
      assert.deepEqual(result.console_errors, [], "Console errors");
      assert.deepEqual(result.failed_requests, [], "Failed requests");
      assert.deepEqual(result.failed_http, [], "Failed HTTP responses");
      result.checks.push("no_page_console_request_or_asset_errors");
      result.passed = true;
    } catch (error) {
      result.error = error.stack ?? String(error);
      await screenshot(page, result, "failure").catch(() => {});
    } finally {
      clearTimeout(watchdog);
      await context.close();
    }
    console.log(JSON.stringify({ viewport: viewport.name, passed: result.passed, checks: result.checks, error: result.error }));
  }
  report.passed = report.results.length === 2 && report.results.every(result => result.passed);
} catch (error) {
  report.error = error.stack ?? String(error);
} finally {
  await browser?.close();
  report.finished_at = new Date().toISOString();
  await fs.writeFile(path.join(output, "results.json"), `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });
}
console.log(`Playground evidence: ${path.join(output, "results.json")}`);
if (!report.passed) process.exitCode = 1;
