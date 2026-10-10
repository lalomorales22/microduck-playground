#!/usr/bin/env node
// Isolated UI integration only: assistant endpoints are mocked, never live OpenAI.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const sharp = require("sharp");
const options = { url: "http://127.0.0.1:8767", output: null, viewport: "both" };
const supplied = new Set();
for (let i = 2; i < process.argv.length; i += 2) {
  const key = { "--url": "url", "--output": "output", "--viewport": "viewport" }[process.argv[i]];
  assert(key && process.argv[i + 1] && !process.argv[i + 1].startsWith("--") && !supplied.has(key),
    "Usage: node local/test_microduck_assistant.mjs --output NEW_DIRECTORY [--url LOCAL_URL] [--viewport desktop|mobile|both]");
  supplied.add(key);
  options[key] = process.argv[i + 1];
}
assert(options.output, "--output must name a new evidence directory");
assert(["desktop", "mobile", "both"].includes(options.viewport), "Invalid --viewport");
const url = new URL(options.url);
const loopback = hostname => ["localhost", "127.0.0.1", "[::1]"].includes(hostname);
assert(loopback(url.hostname) && ["http:", "https:"].includes(url.protocol), "Only a loopback HTTP(S) application is permitted");
const output = path.resolve(options.output);
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.mkdir(output); // Reserve a new leaf, rejecting existing files, dirs and symlinks.

const WORLD_KEY = "wicroduck.world.v1";
const CONTEXT_LABEL = "Send chat, scene, motion, and training context to OpenAI";
const SENTINEL_PATH = "checkpoints/assistant-e2e-existing-autosave.json";
const SENTINEL_TEXT = JSON.stringify({ fixture: "Opaque autosave preservation sentinel, not a trained policy", nonce: "assistant-e2e-v1" });
const QUOTA_ERROR = "Mock quota exhausted. Retry after the test fixture recovers.";
const JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"];
const POSE = [0, -0.087266, -0.457924, -0.00494, 0.452984, 0.349066, 0.349066, 0, 0, 0, 0.087266, 0.457924, 0.00494, -0.452984];
const proposal = {
  message: "Review this physical scene, joint reference and unstarted training draft.",
  world: { version: 1, objects: [
    { id: "assistant_ball", kind: "ball", position: [0.65, 0, 0.07], size: [0.12, 0.12, 0.12], yaw: 0, dynamic: true, mass: 0.12, color: [0.1, 0.83, 0.9, 1] },
    { id: "assistant_platform", kind: "platform", position: [-0.65, 0, 0.05], size: [0.3, 0.3, 0.1], yaw: 0.2, dynamic: false, mass: 1, color: [0.56, 0.6, 0.64, 1] },
  ] },
  motion: { format: 1, name: "assistant_e2e_reference", description: "Untrained small head nod joint reference", loop: false, units: "rad", interp: "linear", joints: JOINTS,
    keyframes: [{ t: 0, pose: POSE }, { t: 1, pose: POSE.map((value, index) => value + (index === 6 ? 0.1 : 0)) }] },
  training: { task: "$generated_motion", iterations: 50, seed: 42, envs: 16, rolloutWorkers: 1, useCurrentWorld: true, hiddenPreset: "small" },
};
const viewports = [{ name: "desktop", width: 1360, height: 900 }, { name: "mobile", width: 390, height: 844 }]
  .filter(viewport => options.viewport === "both" || options.viewport === viewport.name);
const report = {
  schema_version: 1, started_at: new Date().toISOString(), url: url.href, output,
  node: process.version, browser_executable: chromium.executablePath(), endpoint_verification: "MOCKED",
  scope: "Fresh browser contexts; only /api/assistant/status and /api/assistant/chat responses are mocked. No user's profile, live API key, OpenAI model, training run or checkpoint replay is used.",
  limitations: [
    "Mocked endpoints qualify browser validation, consent and apply wiring, not the server proxy, credentials, remote model availability or model-generated quality.",
    "The motion is an explicit untrained joint reference. Rendering, a physics control step and prop velocity are not learned behavior or task success.",
    "The autosave fixture is an opaque byte-preservation sentinel, not a valid checkpoint or evidence of resume compatibility.",
    "Scene persistence checks object configurations, not instantaneous simulated positions.",
    "Cancellation checks the actual aborted fetch and a bounded late-response observation, not a synthetic response that bypasses AbortController.",
  ], results: [], passed: false,
};
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const button = (page, name) => page.getByRole("button", { name, exact: true });
const world = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)) ?? { version: 1, objects: [] }, WORLD_KEY);
const clock = async page => Number.parseFloat(await page.locator(".transport .time").textContent());
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
async function eventually(predicate, message, timeout = 20_000) {
  const deadline = Date.now() + timeout;
  do { if (await predicate()) return; await sleep(50); } while (Date.now() < deadline);
  throw new Error(message);
}
async function waitReady(page) {
  await page.waitForFunction(() => {
    const error = document.querySelector(".stage .overlay.error");
    if (error) throw new Error(error.textContent);
    return !!document.querySelector(".main:not(.hidden) .transport") && !document.querySelector(".main:not(.hidden) .stage .overlay");
  }, undefined, { timeout: 120_000 });
}
async function pause(page) {
  if (await button(page, "Pause simulation").count()) await button(page, "Pause simulation").click();
  await button(page, "Play simulation").waitFor();
  await page.waitForTimeout(150);
}
async function files(page) {
  return page.evaluate(async () => {
    const result = {};
    const visit = async (directory, prefix) => {
      for await (const [name, handle] of directory.entries()) {
        if (handle.kind === "directory") await visit(handle, `${prefix}${name}/`);
        else result[prefix + name] = await (await handle.getFile()).text();
      }
    };
    await visit(await navigator.storage.getDirectory(), "");
    return result;
  });
}
const fileEvidence = stored => Object.fromEntries(Object.entries(stored).map(([name, text]) => [name, { bytes: Buffer.byteLength(text), sha256: hash(text) }]));
async function seedAutosave(page) {
  assert.deepEqual(await files(page), {}, "Fresh context unexpectedly has OPFS files");
  await page.evaluate(async ({ name, text }) => {
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle("checkpoints", { create: true });
    const file = await directory.getFileHandle(name, { create: true });
    const writer = await file.createWritable();
    try { await writer.write(text); } finally { await writer.close(); }
  }, { name: path.basename(SENTINEL_PATH), text: SENTINEL_TEXT });
}
async function trainingDraft(page) {
  return {
    task: await page.getByLabel("Training task", { exact: true }).inputValue(),
    iterations: Number(await page.getByLabel("Training iterations", { exact: true }).inputValue()),
    seed: Number(await page.getByLabel("Training seed", { exact: true }).inputValue()),
    hiddenPreset: await page.getByLabel("Training network", { exact: true }).inputValue(),
    rolloutWorkers: Number(await page.getByLabel("Rollout workers", { exact: true }).inputValue()),
    envs: Number(await page.getByLabel("Environments per worker", { exact: true }).inputValue()),
    useCurrentWorld: await page.getByLabel("Current world snapshot", { exact: true }).isChecked(),
    status: await page.locator(".train-panel .sub .status[role=status]").textContent(),
  };
}
async function screenshot(page, result, name, canvas = false) {
  const file = `${result.viewport.name}-${name}.png`;
  const bytes = canvas
    ? await page.locator(".main:not(.hidden) canvas").screenshot({ path: path.join(output, file), timeout: 30_000 })
    : await page.screenshot({ path: path.join(output, file), fullPage: true, timeout: 30_000 });
  const evidence = { file, sha256: hash(bytes) };
  if (canvas) {
    const metadata = await sharp(bytes).metadata();
    const region = name === "motion-canvas"
      ? { left: 8, top: 8, width: metadata.width - 16, height: metadata.height - 16 }
      : { left: 8, top: 60, width: metadata.width - 16, height: metadata.height - 92 };
    assert(region.width > 0 && region.height > 0, "Canvas too small for scene-only region");
    const { data, info } = await sharp(bytes).extract(region).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    assert.equal(info.channels, 3);
    const colors = new Set(), sums = [0, 0, 0], squares = [0, 0, 0];
    for (let i = 0; i < data.length; i += 3) {
      colors.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
      for (let c = 0; c < 3; c++) { sums[c] += data[i + c]; squares[c] += data[i + c] ** 2; }
    }
    const pixels = info.width * info.height;
    const stddev = sums.map((sum, c) => Math.sqrt(Math.max(0, squares[c] / pixels - (sum / pixels) ** 2)));
    Object.assign(evidence, { width: metadata.width, height: metadata.height, region, unique_rgb_colors: colors.size, channel_stddev: stddev });
    assert(colors.size >= 128 && Math.max(...stddev) >= 5, "Canvas is blank or lacks rendered detail");
  }
  result.screenshots.push(evidence);
}
async function layout(page, result, name) {
  const measurement = await page.evaluate(() => {
    const issues = [], boxes = {};
    for (const selector of [".shell", ".rail", ".topbar", "footer.status"]) {
      const element = document.querySelector(selector);
      if (!element) { issues.push(`Missing ${selector}`); continue; }
      const r = element.getBoundingClientRect();
      boxes[selector] = { x: r.x, y: r.y, width: r.width, height: r.height };
      if (r.left < -1 || r.top < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1) issues.push(`${selector} outside viewport`);
      if (element.scrollWidth > element.clientWidth + 1) issues.push(`${selector} horizontal overflow`);
    }
    for (const element of document.querySelectorAll("textarea, input, select, button")) {
      const r = element.getBoundingClientRect();
      if (r.width && r.height && (r.left < -1 || r.right > innerWidth + 1)) issues.push(`Control overflows horizontally: ${element.getAttribute("aria-label") ?? element.textContent.trim()}`);
    }
    if (document.documentElement.scrollWidth > innerWidth + 1 || document.documentElement.scrollHeight > innerHeight + 1) issues.push("Document overflow");
    if (boxes["footer.status"]?.height > 60) issues.push("Oversized footer row");
    return { boxes, issues };
  });
  result.layouts.push({ name, ...measurement });
  assert.deepEqual(measurement.issues, [], `${name}: layout issues`);
}

let browser;
try {
  browser = await chromium.launch({ headless: true, timeout: 30_000 });
  report.browser_version = browser.version();
  for (const viewport of viewports) {
    const result = { viewport, passed: false, endpoint_verification: "MOCKED", checks: [], screenshots: [], layouts: [], requests: [], page_errors: [], console_errors: [], failed_requests: [], failed_http: [], blocked_network: [], route_errors: [] };
    report.results.push(result);
    const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, serviceWorkers: "block" });
    const page = await context.newPage();
    page.setDefaultTimeout(20_000);
    page.setDefaultNavigationTimeout(60_000);
    let configured = false, responseMode = "valid", releasePending = null;
    const expectedAborts = new Set(), expectedHttpErrors = new Set();
    page.on("pageerror", error => result.page_errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") result.console_errors.push({ text: message.text(), url: message.location().url }); });
    page.on("requestfailed", request => result.failed_requests.push({ url: request.url(), error: request.failure()?.errorText, expected_cancellation: expectedAborts.has(request) }));
    page.on("response", response => { if (response.status() >= 400) result.failed_http.push({ url: response.url(), status: response.status(), expected_mock_error: expectedHttpErrors.has(response.request()) }); });
    await context.route("**/*", async route => {
      const request = route.request(), requestUrl = new URL(request.url());
      try {
        if (!loopback(requestUrl.hostname) && ["http:", "https:"].includes(requestUrl.protocol)) {
          result.blocked_network.push(request.url());
          await route.abort("blockedbyclient");
          return;
        }
        if (requestUrl.pathname === "/api/assistant/status") {
          assert.equal(request.method(), "GET");
          result.requests.push({ endpoint: "status", verification: "MOCKED", configured });
          await route.fulfill({ json: { configured, model: "mock-model-not-live" } });
          return;
        }
        if (requestUrl.pathname === "/api/assistant/chat") {
          assert.equal(request.method(), "POST");
          assert.equal(request.headers()["x-microduck-assistant"], "1");
          assert.match(request.headers()["content-type"] ?? "", /^application\/json(?:;|$)/);
          const body = request.postDataJSON(), mode = responseMode;
          result.requests.push({ endpoint: "chat", verification: "MOCKED", mode, body });
          assert(configured, "Chat request escaped the missing-key UI gate");
          if (mode === "server-error") {
            expectedHttpErrors.add(request);
            await route.fulfill({ status: 429, json: { error: { code: "mock_quota", message: QUOTA_ERROR } } });
          } else if (mode === "cancel") {
            expectedAborts.add(request);
            await new Promise(resolve => { releasePending = resolve; });
            await route.fulfill({ json: proposal }).catch(error => {
              result.cancelled_late_response = { rejected_by_browser: true, detail: String(error) };
            });
          } else await route.fulfill({ json: mode === "invalid" ? { ...proposal, unexpected: "Must reject this field" } : proposal });
          return;
        }
        assert(!requestUrl.pathname.startsWith("/api/assistant/"), "Unexpected assistant API path must not reach a live server");
        await route.continue();
      } catch (error) {
        result.route_errors.push(String(error));
        await route.abort("failed").catch(() => {});
      }
    });
    const watchdog = setTimeout(() => { result.timeout = "Viewport exceeded 300-second budget"; releasePending?.(); void context.close().catch(() => {}); }, 300_000);
    const chatRequests = () => result.requests.filter(request => request.endpoint === "chat");
    const send = async (text, mode = "valid") => {
      responseMode = mode;
      const previous = chatRequests().length;
      await page.getByLabel("Message to assistant", { exact: true }).fill(text);
      await button(page, "Send message").click();
      await eventually(() => chatRequests().length === previous + 1, "Mock chat request was not observed");
      return chatRequests().at(-1);
    };
    try {
      await page.goto(url.href, { waitUntil: "domcontentloaded" });
      await waitReady(page); await pause(page); await seedAutosave(page);
      const baselineWorld = await world(page), baselineFiles = await files(page);
      result.opfs_before = fileEvidence(baselineFiles);
      await button(page, "Train").click();
      await page.getByLabel("Training task", { exact: true }).waitFor();
      const baselineDraft = await trainingDraft(page);
      assert.equal(baselineDraft.status, "Ready");
      await button(page, "Assistant").click();
      await page.getByText("OpenAI is not configured on this server.", { exact: true }).waitFor();
      assert(!(await page.getByLabel(CONTEXT_LABEL, { exact: true }).isChecked()), "Context sharing must be opt-in");
      await page.getByLabel("Message to assistant", { exact: true }).fill("Propose a small scene only after the connection is configured.");
      await page.getByLabel(CONTEXT_LABEL, { exact: true }).check();
      assert(await button(page, "Send message").isDisabled(), "Missing-key status must block even a nonempty, consented request");
      assert.equal(chatRequests().length, 0);
      await page.getByLabel(CONTEXT_LABEL, { exact: true }).uncheck();
      await screenshot(page, result, "missing-key"); await layout(page, result, "missing-key");
      result.checks.push("mocked_missing_key_blocks_chat_and_context_defaults_off");

      configured = true;
      await button(page, "Refresh assistant connection").click();
      await eventually(() => page.getByText("OpenAI is not configured on this server.", { exact: true }).count().then(count => count === 0), "Configured status did not replace missing-key notice");
      await page.getByLabel("Message to assistant", { exact: true }).fill("Propose a small scene, head nod reference and training draft. Do not apply yet.");
      assert(await button(page, "Send message").isDisabled(), "Context consent must gate requests even when configured");
      await page.waitForTimeout(100);
      assert.equal(chatRequests().length, 0, "Request was sent without context consent");
      await page.getByLabel(CONTEXT_LABEL, { exact: true }).check();
      await send("Propose a small scene, head nod reference and training draft. Do not apply yet.");
      await button(page, "Apply proposal").waitFor();
      assert.deepEqual(await world(page), baselineWorld, "Proposal automatically changed the scene");
      assert.deepEqual(await files(page), baselineFiles, "Proposal automatically wrote OPFS");
      assert.deepEqual(await trainingDraft(page), baselineDraft, "Proposal automatically changed the training draft");
      await button(page, "Dismiss").click();
      assert.equal(await button(page, "Apply proposal").count(), 0);
      result.checks.push("no_request_without_context_consent_and_unapplied_proposal_is_inert");

      await page.getByLabel(CONTEXT_LABEL, { exact: true }).check();
      const withContext = await send("Use the shared scene and motion context to propose the same small reference.");
      assert(withContext.body.context && typeof withContext.body.context === "object", "Opted-in context missing");
      assert.deepEqual(withContext.body.context.world, baselineWorld);
      assert(Array.isArray(withContext.body.context.motions) && withContext.body.context.motions.length >= 13);
      assert(withContext.body.context.training && Object.hasOwn(withContext.body.context, "currentMotion"));
      assert(withContext.body.context.currentMotion === null || typeof withContext.body.context.currentMotion === "object", "Current motion must follow its nullable contract");
      assert.equal(withContext.body.context.currentMotion, null, "Fresh never-visited Maker must exercise the pristine empty-draft first-apply path");
      await button(page, "Apply proposal").waitFor();
      await screenshot(page, result, "proposal-review"); await layout(page, result, "proposal-review");
      assert.deepEqual(await files(page), baselineFiles);
      await button(page, "Apply proposal").click();
      await button(page, "Open training").waitFor();
      assert.deepEqual(await world(page), proposal.world);
      const appliedFiles = await files(page);
      assert.equal(appliedFiles[SENTINEL_PATH], SENTINEL_TEXT, "Existing autosave was overwritten");
      const newFiles = Object.keys(appliedFiles).filter(name => !(name in baselineFiles));
      assert.equal(newFiles.length, 1, "Apply should create one reference file, not checkpoints");
      assert.match(newFiles[0], /^motions\/ai-[a-zA-Z0-9._-]+\.json$/);
      assert.deepEqual(JSON.parse(appliedFiles[newFiles[0]]), proposal.motion);
      const savedId = path.basename(newFiles[0], ".json");
      result.applied_motion_id = savedId;
      result.opfs_after_apply = fileEvidence(appliedFiles);
      await screenshot(page, result, "applied"); await layout(page, result, "applied");
      result.checks.push("explicit_apply_creates_scene_and_one_saved_reference_preserving_autosave");
      result.checks.push("pristine_untouched_empty_maker_accepts_first_generated_reference");

      await button(page, "Open training").click();
      await eventually(async () => (await page.getByLabel("Training task", { exact: true }).inputValue()) === `motion:${savedId}`, "Generated motion was not linked to the staged training task");
      const appliedDraft = await trainingDraft(page);
      assert.deepEqual(appliedDraft, { ...proposal.training, task: `motion:${savedId}`, status: "Ready" });
      assert(await button(page, "New run").isEnabled());
      assert.equal(await page.locator(".train-panel .run-status").count(), 0, "Assistant auto-started a training run");
      await screenshot(page, result, "training-draft"); await layout(page, result, "training-draft");
      result.training_draft = appliedDraft;
      result.checks.push("training_link_resolves_saved_motion_and_stages_parameters_without_starting");

      await button(page, "Assistant").click(); await button(page, "Open motion").click();
      await eventually(async () => (await page.getByLabel("Motion name", { exact: true }).inputValue()) === proposal.motion.name, "Proposed reference was not loaded in Motion maker");
      const mode = page.locator("select").filter({ has: page.locator('option[value="preview"]') });
      assert.equal(await mode.inputValue(), "preview", "Reference must open in explicit non-physics preview mode");
      await page.getByRole("button", { name: "Keyframe at 1.00 seconds", exact: true }).waitFor();
      await page.waitForFunction(() => {
        const active = document.querySelector(".main.maker-main:not(.hidden)");
        const error = active?.querySelector(".overlay.error");
        if (error) throw new Error(error.textContent);
        const play = [...(active?.querySelectorAll("button") ?? [])].find(item => item.textContent.trim() === "Play");
        return play && !play.disabled && !active.querySelector(".overlay");
      }, undefined, { timeout: 120_000 });
      await screenshot(page, result, "motion-reference"); await layout(page, result, "motion-reference");
      await screenshot(page, result, "motion-canvas", true);
      result.checks.push("motion_link_opens_untrained_keyframe_reference_not_fake_learned_animation");

      await button(page, "Assistant").click(); await button(page, "Open scene").click();
      await waitReady(page); await pause(page);
      await page.getByRole("tab", { name: /^Scene/ }).click();
      await page.getByLabel("Selected object", { exact: true }).selectOption("assistant_ball");
      await button(page, "Nudge object").click();
      await page.waitForFunction(() => [...document.querySelectorAll(".inspector h2 span")].some(span => /m\/s$/.test(span.textContent.trim()) && Number.parseFloat(span.textContent) > 0.3));
      result.prop_velocity = await page.locator(".inspector h2 span").filter({ hasText: /m\/s/ }).textContent();
      const beforeStep = await clock(page);
      await button(page, "Step simulation").click();
      await page.waitForFunction(start => Math.abs(Number.parseFloat(document.querySelector(".transport .time").textContent) - start - 0.02) < 1e-6, beforeStep);
      result.physics_step = { before: beforeStep, after: await clock(page), expected_seconds: 0.02 };
      await button(page, "Frame scene").click();
      await screenshot(page, result, "physical-scene"); await screenshot(page, result, "physical-canvas", true);
      await layout(page, result, "physical-scene");
      result.checks.push("applied_scene_has_rendered_geometry_and_live_mujoco_prop_velocity_and_step");

      await button(page, "Assistant").click();
      const invalidWorld = await world(page), invalidFiles = await files(page);
      await send("Return an invalid test proposal that must be rejected.", "invalid");
      await page.getByRole("alert").filter({ hasText: /unknown|unexpected|invalid|field/i }).waitFor();
      assert.equal(await button(page, "Apply proposal").count(), 0);
      assert.deepEqual(await world(page), invalidWorld);
      assert.deepEqual(await files(page), invalidFiles);
      assert.deepEqual(await trainingDraft(page), appliedDraft);
      result.checks.push("invalid_mock_response_is_rejected_without_mutation");

      await send("Return the mocked nested quota error.", "server-error");
      await page.getByRole("alert").filter({ hasText: QUOTA_ERROR }).waitFor();
      assert(await button(page, "Retry request").isEnabled(), "Nested server error did not expose Retry");
      assert.equal(await button(page, "Apply proposal").count(), 0);
      assert.deepEqual(await world(page), invalidWorld);
      assert.deepEqual(await files(page), invalidFiles);
      assert.deepEqual(await trainingDraft(page), appliedDraft);
      await screenshot(page, result, "mocked-quota-error"); await layout(page, result, "mocked-quota-error");
      result.checks.push("mocked_nested_429_message_and_retry_are_visible_without_mutation");

      await send("Propose another scene, but wait for explicit review.");
      await button(page, "Apply proposal").waitFor();
      await button(page, "Simulate").click(); await waitReady(page); await pause(page);
      await page.getByRole("tab", { name: /^Scene/ }).click();
      await page.getByLabel("Object type", { exact: true }).selectOption("box");
      await button(page, "Add object").click();
      await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).objects.length === 3, WORLD_KEY);
      const changedWorld = await world(page);
      await button(page, "Assistant").click(); await button(page, "Apply proposal").click();
      await page.getByRole("alert").filter({ hasText: /changed|stale|again/i }).waitFor();
      assert.deepEqual(await world(page), changedWorld, "Stale proposal replaced the manually changed scene");
      assert.deepEqual(await files(page), invalidFiles, "Stale proposal wrote a reference or checkpoint");
      assert.deepEqual(await trainingDraft(page), appliedDraft);
      await screenshot(page, result, "stale-rejected"); await layout(page, result, "stale-rejected");
      if (await button(page, "Dismiss").count()) await button(page, "Dismiss").click();
      result.checks.push("scene_changed_after_proposal_causes_stale_apply_rejection");

      await button(page, "Motion maker").click();
      await page.getByLabel("Loop", { exact: true }).check();
      await page.getByText(/Not saveable yet:.*last keyframe/).waitFor();
      await button(page, "Assistant").click();
      const invalidDraftRequest = await send("Propose another reference while an edited invalid motion must be preserved.");
      assert.equal(invalidDraftRequest.body.context.currentMotion, null);
      await button(page, "Apply proposal").click();
      await page.getByRole("alert").filter({ hasText: /current Motion maker draft error/i }).waitFor();
      assert.deepEqual(await world(page), changedWorld);
      assert.deepEqual(await files(page), invalidFiles);
      assert.deepEqual(await trainingDraft(page), appliedDraft);
      await button(page, "Motion maker").click();
      assert(await page.getByLabel("Loop", { exact: true }).isChecked(), "Rejected apply altered the edited invalid draft");
      assert.equal(await page.getByLabel("Motion name", { exact: true }).inputValue(), proposal.motion.name);
      await page.getByLabel("Loop", { exact: true }).uncheck();
      await button(page, "Assistant").click(); await button(page, "Dismiss").click();
      result.checks.push("edited_invalid_maker_draft_is_refused_and_preserved_distinct_from_pristine_empty");

      await send("This request will be cancelled before its mocked response.", "cancel");
      await button(page, "Cancel request").click();
      await eventually(async () => !(await button(page, "Cancel request").count()), "Cancellation did not return UI to idle");
      await page.getByText("Request canceled.", { exact: true }).waitFor();
      releasePending?.(); releasePending = null;
      await eventually(() => result.failed_requests.some(request => request.expected_cancellation && /ABORTED/i.test(request.error ?? "")), "Cancelled fetch did not emit an aborted request");
      await page.waitForTimeout(250);
      result.cancellation = { fetch_abort_observed: true, late_response_observation_ms: 250 };
      assert.equal(await button(page, "Apply proposal").count(), 0, "Cancelled response produced an applicable proposal");
      assert.deepEqual(await world(page), changedWorld);
      assert.deepEqual(await files(page), invalidFiles);
      assert.deepEqual(await trainingDraft(page), appliedDraft);
      await screenshot(page, result, "cancelled"); await layout(page, result, "cancelled");
      result.checks.push("cancelled_mock_fetch_and_bounded_late_response_observation_preserve_state");
      result.opfs_final = fileEvidence(await files(page));
      assert.equal((await files(page))[SENTINEL_PATH], SENTINEL_TEXT);
      assert.deepEqual(result.page_errors, [], "Page errors");
      result.expected_console_errors = result.console_errors.filter(error => error.url === new URL("/api/assistant/chat", url).href && /status of 429/.test(error.text));
      assert(result.expected_console_errors.length <= 1, "More console errors than the one deliberate mocked 429");
      assert.deepEqual(result.console_errors.filter(error => !result.expected_console_errors.includes(error)), [], "Unexpected console errors");
      assert.deepEqual(result.failed_http.filter(response => !response.expected_mock_error), [], "Unexpected HTTP or asset failures");
      assert.equal(result.failed_http.filter(response => response.expected_mock_error && response.status === 429).length, 1, "Deliberate mocked 429 was not observed exactly once");
      assert.deepEqual(result.route_errors, [], "Mock routing failures");
      assert.deepEqual(result.blocked_network, [], "Application attempted non-loopback requests");
      assert.deepEqual(result.failed_requests.filter(request => !request.expected_cancellation), [], "Unexpected failed requests");
      result.checks.push("no_live_openai_requests_no_asset_errors_and_autosave_bytes_preserved");
      result.passed = true;
    } catch (error) {
      result.error = error.stack ?? String(error);
      await screenshot(page, result, "failure").catch(() => {});
    } finally {
      clearTimeout(watchdog); releasePending?.();
      await context.close();
      await fs.writeFile(path.join(output, `${viewport.name}-results.json`), `${JSON.stringify(result, null, 2)}\n`, { flag: "wx" });
    }
    console.log(JSON.stringify({ viewport: viewport.name, passed: result.passed, checks: result.checks, error: result.error }));
  }
  report.passed = report.results.length === viewports.length && report.results.every(result => result.passed);
} catch (error) {
  report.error = error.stack ?? String(error);
} finally {
  await browser?.close();
  report.finished_at = new Date().toISOString();
  await fs.writeFile(path.join(output, "results.json"), `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });
}
console.log(`MOCKED assistant integration evidence: ${path.join(output, "results.json")}`);
if (!report.passed) process.exitCode = 1;
