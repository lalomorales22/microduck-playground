#!/usr/bin/env node
// Mocked assistant transport, real browser CPU training/evaluation, isolated OPFS.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const sharp = require("sharp");
const options = { url: "http://127.0.0.1:8767/", output: null, viewport: "both", checkpoint: null };
const seen = new Set();
for (let i = 2; i < process.argv.length; i += 2) {
  const key = { "--url": "url", "--output": "output", "--viewport": "viewport", "--checkpoint": "checkpoint" }[process.argv[i]];
  assert(key && process.argv[i + 1] && !process.argv[i + 1].startsWith("--") && !seen.has(key),
    "Usage: node local/test_microduck_run_assistant.mjs --output NEW_DIRECTORY [--url LOCAL_URL] [--viewport desktop|mobile|both] [--checkpoint JSON_FILE]");
  seen.add(key); options[key] = process.argv[i + 1];
}
assert(options.output && ["both", "desktop", "mobile"].includes(options.viewport));
const url = new URL(options.url), output = path.resolve(options.output);
const local = hostname => ["localhost", "127.0.0.1", "[::1]"].includes(hostname);
assert(local(url.hostname) && ["http:", "https:"].includes(url.protocol), "Only loopback HTTP(S) is permitted");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
let importedBytes = null, importedCheckpoint = null;
if (options.checkpoint) {
  options.checkpoint = path.resolve(options.checkpoint);
  importedBytes = await fs.readFile(options.checkpoint);
  assert(importedBytes.length < 64 * 1024 * 1024, "Checkpoint fixture exceeds 64 MiB");
  importedCheckpoint = JSON.parse(importedBytes.toString());
  assert.equal(importedCheckpoint.version, 1);
}
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.mkdir(output);
const WORLD_KEY = "wicroduck.world.v1", IMPORT_NAME = "e2e-user-jump-50";
const TEST_WORLD = { version: 1, objects: [{ id: "e2e_platform", kind: "platform", position: [0.8, 0.6, 0.05], size: [0.3, 0.3, 0.1], yaw: 0.2, dynamic: false, mass: 1, color: [0.3, 0.7, 0.6, 1] }] };
const viewports = [{ name: "desktop", width: 1360, height: 900 }, { name: "mobile", width: 390, height: 844 }].filter(v => options.viewport === "both" || options.viewport === v.name);
const report = {
  schema_version: 1, started_at: new Date().toISOString(), url: url.href, output, endpoint_verification: "MOCKED",
  scope: "Fresh isolated Chromium contexts. Mocked assistant status/chat only; actual MuJoCo CPU worker training, frozen-checkpoint evaluation and replay. No live provider call or user's browser profile.",
  limitations: ["Two-iteration training and a two-additional-iteration continuation are lifecycle smoke tests, not evidence of learning a jump.", "Physical qualification is one deterministic saved-checkpoint trial, not robustness or hardware certification.", "The optional stop test starts at most one additional two-iteration run; a completion race is recorded explicitly, never described as a successful stop.", "All HTTP assistant replies are synthetic. API key, model and server proxy behavior are not verified."],
  imported_source: importedBytes ? { path: options.checkpoint, bytes: importedBytes.length, sha256: hash(importedBytes), iteration: importedCheckpoint.iteration } : null,
  results: [], passed: false,
};
const button = (page, name) => page.getByRole("button", { name, exact: true });
const actions = page => page.getByRole("region", { name: "Proposed action", exact: true });
const monitor = page => page.getByRole("region", { name: "Training run monitor", exact: true });
const world = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)) ?? { version: 1, objects: [] }, WORLD_KEY);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate, message, timeout = 20_000) {
  const deadline = Date.now() + timeout;
  do { if (await predicate()) return; await sleep(50); } while (Date.now() < deadline);
  throw new Error(message);
}
async function waitSim(page) {
  await page.waitForFunction(() => {
    const active = document.querySelector(".main:not(.hidden)");
    const error = active?.querySelector(".overlay.error");
    if (error) throw new Error(error.textContent);
    return active?.querySelector(".transport") && !active.querySelector(".overlay");
  }, undefined, { timeout: 120_000 });
}
async function pause(page) {
  if (await button(page, "Pause simulation").count()) await button(page, "Pause simulation").click();
  await button(page, "Play simulation").waitFor(); await page.waitForTimeout(100);
}
async function storedFiles(page) {
  return page.evaluate(async () => {
    const files = {};
    async function walk(dir, prefix) {
      for await (const [name, handle] of dir.entries()) {
        if (handle.kind === "directory") await walk(handle, `${prefix}${name}/`);
        else files[prefix + name] = await (await handle.getFile()).text();
      }
    }
    await walk(await navigator.storage.getDirectory(), ""); return files;
  });
}
const receipt = files => Object.fromEntries(Object.entries(files).map(([name, text]) => [name, { bytes: Buffer.byteLength(text), sha256: hash(text) }]));
function preserved(before, after) {
  for (const [name, bytes] of Object.entries(before)) assert.equal(after[name], bytes, `Existing checkpoint changed: ${name}`);
}
async function screenshot(page, result, name, canvas = false) {
  const file = `${result.viewport.name}-${name}.png`;
  const bytes = canvas
    ? await page.locator(".main:not(.hidden) canvas").screenshot({ path: path.join(output, file), timeout: 20_000 })
    : await page.screenshot({ path: path.join(output, file), fullPage: true, timeout: 20_000 });
  const entry = { file, sha256: hash(bytes) };
  if (canvas) {
    const metadata = await sharp(bytes).metadata();
    const region = { left: 8, top: 60, width: metadata.width - 16, height: metadata.height - 92 };
    assert(region.width > 0 && region.height > 0);
    const { data, info } = await sharp(bytes).extract(region).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    assert.equal(info.channels, 3);
    const colors = new Set(), sums = [0, 0, 0], squares = [0, 0, 0];
    for (let i = 0; i < data.length; i += 3) {
      colors.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
      for (let c = 0; c < 3; c++) { sums[c] += data[i + c]; squares[c] += data[i + c] ** 2; }
    }
    const pixels = info.width * info.height;
    const stddev = sums.map((sum, c) => Math.sqrt(Math.max(0, squares[c] / pixels - (sum / pixels) ** 2)));
    Object.assign(entry, { width: metadata.width, height: metadata.height, region, colors: colors.size, stddev });
    assert(colors.size >= 128 && Math.max(...stddev) >= 5, "Replay canvas is blank");
  }
  result.screenshots.push(entry);
}
async function layout(page, result, name) {
  const issues = await page.evaluate(() => {
    const problems = [];
    if (document.documentElement.scrollWidth > innerWidth + 1 || document.documentElement.scrollHeight > innerHeight + 1) problems.push("Document overflow");
    for (const selector of [".shell", ".topbar", ".rail", "footer.status", ".assistant-workspace:not(.hidden)"]) {
      const element = document.querySelector(selector); if (!element) continue;
      const r = element.getBoundingClientRect();
      if (r.left < -1 || r.right > innerWidth + 1 || r.top < -1 || r.bottom > innerHeight + 1 || element.scrollWidth > element.clientWidth + 1) problems.push(selector);
    }
    for (const element of document.querySelectorAll("button,input,select,textarea")) {
      const r = element.getBoundingClientRect();
      if (r.width && r.height && (r.left < -1 || r.right > innerWidth + 1)) problems.push(element.getAttribute("aria-label") ?? element.textContent.trim());
    }
    return problems;
  });
  result.layouts.push({ name, issues }); assert.deepEqual(issues, [], `${name}: overflow`);
}
const commandNames = { start_training: "Start training", resume_training: "Resume training", stop_training: "Stop training", evaluate_checkpoint: "Evaluate checkpoint", replay_checkpoint: "Open trained policy" };
let browser;
for (const signal of ["SIGINT", "SIGTERM"]) process.once(signal, () => {
  report.interrupted = signal;
  void browser?.close().catch(() => {});
});
try {
  browser = await chromium.launch({ headless: true, timeout: 30_000 }); report.browser = browser.version();
  for (const viewport of viewports) {
    if (report.interrupted) break;
    const viewportStarted = Date.now();
    const result = { viewport, started_at: new Date(viewportStarted).toISOString(), passed: false, endpoint_verification: "MOCKED", timings: [], checks: [], screenshots: [], layouts: [], chats: [], workers: [], page_errors: [], console_errors: [], failed_requests: [], failed_http: [], blocked_network: [], route_errors: [] };
    report.results.push(result);
    const mark = phase => {
      const elapsed_seconds = (Date.now() - viewportStarted) / 1000;
      result.timings.push({ phase, elapsed_seconds });
      console.log(`${viewport.name}: ${phase} (${elapsed_seconds.toFixed(1)}s)`);
    };
    const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, serviceWorkers: "block" });
    const page = await context.newPage(); page.setDefaultTimeout(15_000); page.setDefaultNavigationTimeout(60_000);
    let responseCommand = null, responseTarget = null, cancellationPhase = null;
    page.on("worker", worker => result.workers.push(worker.url()));
    page.on("pageerror", error => result.page_errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") result.console_errors.push(message.text()); });
    page.on("requestfailed", request => result.failed_requests.push({ url: request.url(), error: request.failure()?.errorText,
      expected_worker_cancellation: cancellationPhase && /ABORTED/i.test(request.failure()?.errorText ?? "") ? cancellationPhase : null }));
    page.on("response", response => { if (response.status() >= 400) result.failed_http.push({ url: response.url(), status: response.status() }); });
    await context.route("**/*", async route => {
      const request = route.request(), target = new URL(request.url());
      try {
        if (["http:", "https:"].includes(target.protocol) && !local(target.hostname)) { result.blocked_network.push(request.url()); await route.abort("blockedbyclient"); return; }
        if (target.pathname === "/api/assistant/status") { assert.equal(request.method(), "GET"); await route.fulfill({ json: { configured: true, model: "mock-run-controller-not-live" } }); return; }
        if (target.pathname === "/api/assistant/chat") {
          assert.equal(request.method(), "POST"); assert.equal(request.headers()["x-microduck-assistant"], "1");
          const body = request.postDataJSON();
          const resolvedTarget = responseCommand === "stop_training" ? body.context.run?.runId ?? null : responseTarget;
          const command = responseCommand ? { kind: responseCommand, target: resolvedTarget } : null;
          result.chats.push({ verification: "MOCKED", command, body });
          await route.fulfill({ json: { message: command ? "Review this bounded local action before requesting it. Completion does not prove a learned skill." : "Snapshot recorded. No action requested; completed budget and measured qualification are separate.", world: null, motion: null, training: null, command } });
          return;
        }
        assert(!target.pathname.startsWith("/api/assistant/"), "Unknown assistant path must not reach a live server");
        await route.continue();
      } catch (error) { result.route_errors.push(String(error)); await route.abort("failed").catch(() => {}); }
    });
    const watchdog = setTimeout(() => { result.timeout = "Viewport exceeded 300-second budget"; void context.close().catch(() => {}); }, 300_000);
    const send = async (kind = null, target = null, text = null) => {
      responseCommand = kind; responseTarget = target; const before = result.chats.length;
      await page.getByLabel("Message to assistant", { exact: true }).fill(text ?? (kind ? `Please propose ${kind} for this current context.` : "Summarize the current run and evaluation without taking action."));
      await button(page, "Send message").click();
      await until(() => result.chats.length === before + 1, "Mocked chat request missing");
      return result.chats.at(-1).body.context;
    };
    const approve = async kind => {
      const action = actions(page).last();
      await action.getByRole("button", { name: commandNames[kind], exact: true }).click();
      return action;
    };
    const snapshot = async () => {
      await button(page, "Assistant").click();
      const data = await send();
      await page.getByText("Snapshot recorded. No action requested; completed budget and measured qualification are separate.", { exact: true }).last().waitFor();
      assert(data.run && Array.isArray(data.checkpoints));
      for (const checkpoint of data.checkpoints) assert(Object.keys(checkpoint).every(key => ["name", "iteration", "bytes", "modified"].includes(key)), "Checkpoint tensors leaked into assistant context");
      return data;
    };
    const evaluationDone = async () => {
      await until(async () => (await monitor(page).locator(".evaluation summary").count()) > 0 && !(await button(page, "Cancel evaluation").count()), "Physical evaluation did not finish", 90_000);
    };
    const replay = async (name, checkpoint, label) => {
      const preservedFiles = await storedFiles(page);
      const proposed = await send("replay_checkpoint", name);
      assert(proposed.checkpoints.some(entry => entry.name === name));
      await actions(page).last().getByRole("button", { name: "Open trained policy", exact: true }).waitFor();
      assert.deepEqual(await storedFiles(page), preservedFiles);
      await approve("replay_checkpoint"); await waitSim(page);
      await until(async () => (await page.getByLabel("Driving policy", { exact: true }).inputValue()) === `ckpt:${name}`, "Replayed checkpoint did not become the selected driving policy");
      assert.equal(await page.getByLabel("Driving policy", { exact: true }).inputValue(), `ckpt:${name}`);
      assert.equal(await button(page, "Pause simulation").count(), 0, "Replay started running automatically");
      await button(page, "Play simulation").waitFor();
      assert.deepEqual(await world(page), checkpoint.world ?? { version: 1, objects: [] }, "Replay did not restore the saved world");
      const time = Number.parseFloat(await page.locator(".transport .time").textContent());
      assert.equal(time, 0, "Replay did not reset its clock");
      await page.waitForTimeout(150);
      assert.equal(Number.parseFloat(await page.locator(".transport .time").textContent()), time, "Paused replay advanced");
      const status = await page.locator("footer.status").textContent(), upright = Number(status.match(/Upright\s*(\d+)%/)?.[1]);
      assert(upright >= 95, `Replay did not start upright: ${status}`);
      await screenshot(page, result, `${label}-replay`); await screenshot(page, result, `${label}-canvas`, true); await layout(page, result, `${label}-replay`);
      assert.deepEqual(await storedFiles(page), preservedFiles, "Replay rewrote checkpoint files");
      result[`${label}_replay`] = { checkpoint: name, clock: time, upright_percent: upright, world: await world(page) };
      await button(page, "Assistant").click();
    };
    try {
      await page.goto(url.href, { waitUntil: "domcontentloaded" }); await waitSim(page); await pause(page);
      mark("simulator-ready");
      assert.deepEqual(await storedFiles(page), {}, "Fresh context has existing OPFS files");
      if (importedBytes) {
        await button(page, "Files").click();
        await page.locator(".files > header input[type=file]").first().setInputFiles({ name: `${IMPORT_NAME}.json`, mimeType: "application/json", buffer: importedBytes });
        await page.locator(".files tbody .name").filter({ hasText: IMPORT_NAME }).waitFor();
        await until(async () => !(await page.locator(".files > header input[type=file]").first().isDisabled()), "Checkpoint import did not finish");
        const imported = JSON.parse((await storedFiles(page))[`checkpoints/${IMPORT_NAME}.json`]);
        assert.deepEqual(imported, importedCheckpoint, "Imported checkpoint content changed");
      }
      const baselineFiles = await storedFiles(page); result.baseline_files = receipt(baselineFiles);
      await button(page, "Simulate").click(); await waitSim(page); await pause(page);
      await page.getByRole("tab", { name: /^Scene/ }).click();
      await page.locator(".inspector input[type=file]").setInputFiles({ name: "run-e2e-world.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(TEST_WORLD)) });
      await until(async () => { try { assert.deepEqual(await world(page), TEST_WORLD); return true; } catch { return false; } }, "Test world was not imported");
      await button(page, "Train").click();
      await page.getByLabel("Training task", { exact: true }).selectOption("jump");
      await page.getByLabel("Training iterations", { exact: true }).fill("2");
      await page.getByLabel("Training seed", { exact: true }).fill("123");
      await page.getByLabel("Training network", { exact: true }).selectOption("small");
      await page.getByLabel("Environments per worker", { exact: true }).selectOption("16");
      await page.getByLabel("Rollout workers", { exact: true }).selectOption("1");
      await page.getByLabel("Current world snapshot", { exact: true }).check();
      await button(page, "Assistant").click();
      await page.getByLabel("Send chat, scene, motion, and training context to OpenAI", { exact: true }).check();
      await send("start_training");
      await actions(page).last().getByRole("button", { name: "Start training", exact: true }).waitFor();
      assert.deepEqual(await storedFiles(page), baselineFiles, "Unapproved start wrote a checkpoint");
      await monitor(page).getByText("No active run", { exact: true }).waitFor();
      assert.equal(await monitor(page).locator(".run-id").count(), 0, "Unapproved start created a run");
      await screenshot(page, result, "start-awaiting-approval"); await layout(page, result, "start-awaiting-approval");
      await button(page, "Train").click(); await page.getByLabel("Training seed", { exact: true }).fill("124");
      await button(page, "Assistant").click(); await approve("start_training");
      await actions(page).last().getByRole("alert").filter({ hasText: /changed|stale|review|again/i }).waitFor();
      assert.deepEqual(await storedFiles(page), baselineFiles);
      result.checks.push("unapproved_start_is_inert_and_changed_training_draft_rejects_stale_approval");
      if (await actions(page).last().getByRole("button", { name: "Dismiss", exact: true }).count()) await actions(page).last().getByRole("button", { name: "Dismiss", exact: true }).click();
      const startContext = await send("start_training");
      assert.equal(startContext.training.iterations, 2); assert.equal(startContext.training.seed, 124);
      await approve("start_training");
      await until(async () => !(await page.getByRole("button", { name: "Requesting action", exact: true }).count()), "Start request remained pending");
      mark("real-two-iteration-training-requested");
      const busyContext = await send("start_training", null, "Propose a second start while this worker is busy; it must be rejected.");
      assert(["loading", "running", "stopping"].includes(busyContext.run.status), "Training finished before the bounded busy-command probe; no busy claim can be made");
      await page.getByRole("alert").filter({ hasText: /busy|running|active|stop|idle/i }).waitFor();
      result.busy_run_id = busyContext.run.runId;
      result.checks.push("second_start_is_rejected_against_actual_busy_worker_context");
      await monitor(page).getByText("Budget completed", { exact: true }).waitFor({ timeout: 120_000 });
      await evaluationDone();
      const finished = await snapshot();
      assert.equal(finished.run.runId, busyContext.run.runId, "Rejected command spawned or replaced the run");
      assert.equal(finished.run.completionReason, "completed"); assert.equal(finished.run.iteration, 2); assert.equal(finished.run.targetIteration, 2);
      assert.equal(finished.run.totalSteps, 768);
      const savedName = finished.run.checkpoint, trainedFiles = await storedFiles(page);
      assert.equal(typeof savedName, "string"); preserved(baselineFiles, trainedFiles);
      const newPaths = Object.keys(trainedFiles).filter(name => !(name in baselineFiles));
      assert.deepEqual(newPaths, [`checkpoints/${savedName}.json`], "Unexpected new training files");
      const checkpoint = JSON.parse(trainedFiles[newPaths[0]]);
      assert.equal(checkpoint.iteration, 2); assert.equal(checkpoint.totalSteps, 768); assert.equal(checkpoint.config.stepsPerIter, 24);
      assert.equal(checkpoint.config.seed, 124); assert.deepEqual(checkpoint.world, TEST_WORLD);
      assert(finished.run.evaluation && ["passed", "failed", "not-qualified"].includes(finished.run.evaluation.verdict));
      assert.equal(finished.run.evaluation.checkpointName, savedName); assert.equal(finished.run.evaluation.iteration, 2);
      result.completed_run = finished.run; result.trained_files = receipt(trainedFiles);
      mark("real-two-iteration-training-and-auto-evaluation-complete");
      await screenshot(page, result, "completed-and-evaluated"); await layout(page, result, "completed-and-evaluated");
      result.checks.push("real_two_iteration_budget_completion_is_separate_from_saved_checkpoint_physical_verdict");

      await send("evaluate_checkpoint", savedName); await approve("evaluate_checkpoint");
      cancellationPhase = "explicit-evaluation-cancel";
      await button(page, "Cancel evaluation").click();
      await until(async () => !(await page.getByRole("progressbar", { name: "Checkpoint evaluation progress", exact: true }).count()), "Evaluation cancellation did not finish", 30_000);
      await until(async () => !(await page.getByRole("button", { name: "Requesting action", exact: true }).count()), "Canceled command did not settle", 30_000);
      await actions(page).last().getByRole("alert").filter({ hasText: /cancelled|canceled/i }).waitFor();
      const canceled = await snapshot();
      assert.equal(canceled.run.evaluation, null, "Cancelled evaluation retained a qualification verdict");
      await page.waitForTimeout(200); cancellationPhase = null;
      assert.deepEqual(await storedFiles(page), trainedFiles, "Canceled evaluation altered checkpoint bytes");
      result.checks.push("explicit_evaluation_can_be_canceled_without_checkpoint_mutation");
      await send("evaluate_checkpoint", savedName); await approve("evaluate_checkpoint"); await evaluationDone();
      const evaluated = await snapshot();
      assert.equal(evaluated.run.evaluation.checkpointName, savedName); assert.equal(evaluated.run.evaluation.iteration, 2);
      assert.deepEqual(await storedFiles(page), trainedFiles);
      result.repeated_evaluation = evaluated.run.evaluation;
      await replay(savedName, checkpoint, "trained");
      mark("explicit-evaluation-cancel-reevaluation-and-replay-complete");
      result.checks.push("approved_saved_checkpoint_evaluation_and_paused_upright_saved_world_replay_preserve_bytes");
      if (importedCheckpoint) {
        await send("evaluate_checkpoint", IMPORT_NAME); await approve("evaluate_checkpoint"); await evaluationDone();
        const importedEvaluation = await snapshot();
        assert.equal(importedEvaluation.run.evaluation.checkpointName, IMPORT_NAME);
        assert.equal(importedEvaluation.run.evaluation.iteration, importedCheckpoint.iteration);
        result.imported_evaluation = importedEvaluation.run.evaluation;
        await replay(IMPORT_NAME, importedCheckpoint, "imported");
        assert.deepEqual(await storedFiles(page), trainedFiles);
        result.checks.push("original_user_checkpoint_copy_evaluates_and_replays_without_altering_any_checkpoint");
        mark("original-checkpoint-copy-evaluation-and-replay-complete");
      }
      await button(page, "Train").click();
      await page.getByLabel("Training task", { exact: true }).selectOption("hold_pose");
      await page.getByLabel("Training seed", { exact: true }).fill("999");
      await page.getByLabel("Current world snapshot", { exact: true }).uncheck();
      await button(page, "Assistant").click();
      const resumeContext = await send("resume_training", savedName);
      assert.equal(resumeContext.training.iterations, 2);
      await actions(page).last().getByRole("button", { name: "Resume training", exact: true }).waitFor();
      assert.deepEqual(await storedFiles(page), trainedFiles, "Unapproved continuation wrote checkpoint files");
      assert.equal(await monitor(page).locator(".run-id").textContent(), finished.run.runId, "Unapproved continuation replaced the active run");
      await approve("resume_training");
      await until(async () => !(await page.getByRole("button", { name: "Requesting action", exact: true }).count()), "Continuation request remained pending");
      await monitor(page).getByText("Budget completed", { exact: true }).waitFor({ timeout: 120_000 });
      await evaluationDone();
      const resumed = await snapshot(), resumedFiles = await storedFiles(page);
      assert.notEqual(resumed.run.runId, finished.run.runId);
      assert.notEqual(resumed.run.checkpoint, savedName);
      assert.equal(resumed.run.task, "jump"); assert.equal(resumed.run.completionReason, "completed");
      assert.equal(resumed.run.iteration, 4); assert.equal(resumed.run.targetIteration, 4); assert.equal(resumed.run.totalSteps, 1536);
      preserved(trainedFiles, resumedFiles);
      const resumedPaths = Object.keys(resumedFiles).filter(name => !(name in trainedFiles));
      assert.deepEqual(resumedPaths, [`checkpoints/${resumed.run.checkpoint}.json`], "Continuation did not create exactly one new checkpoint");
      const resumedCheckpoint = JSON.parse(resumedFiles[resumedPaths[0]]);
      assert.equal(resumedCheckpoint.iteration, 4); assert.equal(resumedCheckpoint.totalSteps, 1536);
      assert.deepEqual(resumedCheckpoint.config, checkpoint.config); assert.deepEqual(resumedCheckpoint.world, checkpoint.world);
      assert.deepEqual(resumedCheckpoint.taskSpec, checkpoint.taskSpec);
      assert(resumedCheckpoint.optimizer.t > checkpoint.optimizer.t, "Continuation did not advance the saved optimizer");
      result.resumed_run = resumed.run; result.resumed_files = receipt(resumedFiles);
      mark("two-additional-iteration-resume-fork-complete");
      await screenshot(page, result, "resumed-fork-completed"); await layout(page, result, "resumed-fork-completed");
      result.checks.push("approved_resume_adds_two_iterations_in_new_fork_with_original_config_world_and_checkpoint_bytes_preserved");
      // One additional bounded run exercises loading/running stop. Never loop
      // until a timing-sensitive state appears, and never overwrite the first run.
      await send("start_training"); await approve("start_training");
      await until(async () => !(await page.getByRole("button", { name: "Requesting action", exact: true }).count()), "Second bounded start remained pending");
      const stopContext = await send("stop_training");
      assert(stopContext.run.runId && stopContext.run.runId !== resumed.run.runId, "Second start failed to create a distinct bounded run");
      assert.equal(stopContext.run.targetIteration, 2);
      if (["loading", "running", "stopping"].includes(stopContext.run.status)) {
        cancellationPhase = "explicit-loading-stop";
        await approve("stop_training");
        await until(async () => /Stopped by request|Budget completed|Run error/.test(await monitor(page).textContent()), "Bounded stop did not settle", 90_000);
        await until(async () => !(await button(page, "Cancel evaluation").count()), "Post-stop evaluation did not settle", 90_000);
        const stopped = await snapshot();
        assert.equal(stopped.run.runId, stopContext.run.runId);
        assert(["stopped", "completed"].includes(stopped.run.completionReason));
        assert(stopped.run.iteration <= 2);
        result.stop_probe = { requested: true, completion_race: stopped.run.completionReason === "completed", run: stopped.run };
        await page.waitForTimeout(200); cancellationPhase = null;
      } else {
        assert.equal(stopContext.run.completionReason, "completed"); assert.equal(stopContext.run.iteration, 2);
        result.stop_probe = { requested: false, completion_race: true, run: stopContext.run, reason: "Second two-iteration budget completed before the stop proposal captured a busy state" };
      }
      const finalFiles = await storedFiles(page); preserved(resumedFiles, finalFiles);
      result.final_files = receipt(finalFiles); result.checks.push("bounded_stop_probe_preserves_all_preexisting_checkpoints");
      const snapshotDir = path.join(output, `${viewport.name}-checkpoints`);
      await fs.mkdir(snapshotDir);
      result.checkpoint_snapshots = [];
      for (const [name, content] of Object.entries(finalFiles)) {
        assert(/^checkpoints\/[A-Za-z0-9_-]+\.json$/.test(name), "Unexpected OPFS checkpoint path");
        const file = path.join(`${viewport.name}-checkpoints`, path.basename(name));
        await fs.writeFile(path.join(output, file), content, { flag: "wx" });
        result.checkpoint_snapshots.push({ file, source_opfs: name, bytes: Buffer.byteLength(content), sha256: hash(content) });
      }
      mark("bounded-stop-probe-complete");
      await screenshot(page, result, "final-monitor"); await layout(page, result, "final-monitor");
      assert.deepEqual(result.page_errors, []); assert.deepEqual(result.console_errors, []);
      assert.deepEqual(result.failed_http, []); assert.deepEqual(result.blocked_network, []); assert.deepEqual(result.route_errors, []);
      assert.deepEqual(result.failed_requests.filter(request => !request.expected_worker_cancellation), [], "Unexpected failed requests");
      assert(result.workers.some(worker => /worker/i.test(worker)), "No real worker was observed");
      result.checks.push("no_unexpected_browser_asset_or_network_errors_and_real_worker_observed");
      result.passed = true;
    } catch (error) {
      result.error = error.stack ?? String(error); await screenshot(page, result, "failure").catch(() => {});
    } finally {
      clearTimeout(watchdog); await context.close();
      result.elapsed_seconds = (Date.now() - viewportStarted) / 1000;
      await fs.writeFile(path.join(output, `${viewport.name}-results.json`), `${JSON.stringify(result, null, 2)}\n`, { flag: "wx" });
    }
    console.log(JSON.stringify({ viewport: viewport.name, passed: result.passed, checks: result.checks, error: result.error }));
  }
  report.passed = !report.interrupted && report.results.length === viewports.length && report.results.every(result => result.passed);
} catch (error) { report.error = error.stack ?? String(error); }
finally {
  await browser?.close();
  if (importedBytes) {
    report.imported_source.unchanged = hash(await fs.readFile(options.checkpoint)) === report.imported_source.sha256;
    if (!report.imported_source.unchanged) report.passed = false;
  }
  report.finished_at = new Date().toISOString();
  await fs.writeFile(path.join(output, "results.json"), `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });
}
console.log(`MOCKED assistant / REAL CPU run evidence: ${path.join(output, "results.json")}`);
if (!report.passed) process.exitCode = report.interrupted ? 130 : 1;
