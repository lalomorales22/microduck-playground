#!/usr/bin/env node
// Isolated, local-only theme probes using the installed Playwright and sharp runtimes.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const sharp = require("sharp");
const options = { url: "http://127.0.0.1:8768/", output: null };
const selfTest = process.argv.length === 3 && process.argv[2] === "--self-test";
const seen = new Set();
if (!selfTest) for (let i = 2; i < process.argv.length; i += 2) {
  const key = { "--url": "url", "--output": "output" }[process.argv[i]];
  assert(key && process.argv[i + 1] && !process.argv[i + 1].startsWith("--") && !seen.has(key),
    "Usage: node local/test_microduck_theme.mjs --output NEW_DIRECTORY [--url LOCAL_URL] | --self-test");
  seen.add(key); options[key] = process.argv[i + 1];
}
const url = new URL(options.url);
assert(["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) && ["http:", "https:"].includes(url.protocol)
  && !url.username && !url.password, "Only credential-free local HTTP(S) URLs are permitted");
const hash = value => createHash("sha256").update(value).digest("hex");

// Self-contained browser function: solid CSS text colors only, not a claim about
// rasterized glyphs, graphics, disabled controls, all states, or overall WCAG conformance.
function cssTextAudit() {
  const colorCanvas = document.createElement("canvas"); colorCanvas.width = colorCanvas.height = 1;
  const painter = colorCanvas.getContext("2d", { willReadFrequently: true });
  function color(value) {
    if (!CSS.supports("color", value)) return null;
    painter.clearRect(0, 0, 1, 1); painter.fillStyle = value; painter.fillRect(0, 0, 1, 1);
    const rgba = Array.from(painter.getImageData(0, 0, 1, 1).data); rgba[3] /= 255; return rgba;
  }
  function blend(fg, bg) { return fg.slice(0, 3).map((c, i) => c * fg[3] + bg[i] * (1 - fg[3])); }
  function luminance(rgb) {
    const channels = rgb.slice(0, 3).map(c => { const v = c / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  }
  function label(element) {
    return `${element.tagName.toLowerCase()}${element.id ? `#${element.id}` : ""}${[...element.classList].filter(c => !c.startsWith("svelte-")).slice(0, 3).map(c => `.${c}`).join("")}`;
  }
  function visibleRect(rect, element) {
    let left = Math.max(0, rect.left), right = Math.min(innerWidth, rect.right);
    let top = Math.max(0, rect.top), bottom = Math.min(innerHeight, rect.bottom);
    for (let parent = element; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent), box = parent.getBoundingClientRect();
      if (style.display === "none" || style.visibility !== "visible" || style.contentVisibility === "hidden" || Number(style.opacity) === 0) return false;
      if (parent.tagName === "DETAILS" && !parent.open && !parent.querySelector(":scope > summary")?.contains(element)) return false;
      if (/(auto|scroll|hidden|clip)/.test(style.overflowX)) { left = Math.max(left, box.left); right = Math.min(right, box.right); }
      if (/(auto|scroll|hidden|clip)/.test(style.overflowY)) { top = Math.max(top, box.top); bottom = Math.min(bottom, box.bottom); }
    }
    return right - left > 1 && bottom - top > 1;
  }
  const checked = [], skipped = [];
  function inspect(element, text, rectangles, pseudo = null) {
    text = text.replace(/\s+/g, " ").trim();
    if (!text || !rectangles.some(rect => visibleRect(rect, element))) return;
    const description = { element: label(element), text: text.slice(0, 120), ...(pseudo ? { pseudo } : {}) };
    const skip = reason => skipped.push({ ...description, reason });
    if (!/[A-Za-z0-9]/.test(text)) { skip("icon_or_color_glyph"); return; }
    if (element.closest("svg,canvas")) { skip("graphics_not_css_text"); return; }
    if (element.closest(":disabled,[aria-disabled=true]")) { skip("disabled_control"); return; }
    const foregroundStyle = getComputedStyle(element, pseudo);
    if (pseudo && Number(foregroundStyle.opacity) !== 1) { skip("pseudo_element_opacity"); return; }
    const foreground = color(foregroundStyle.color);
    if (!foreground) { skip("unsupported_foreground_color"); return; }
    let background = [255, 255, 255];
    const ancestors = [];
    for (let parent = element; parent; parent = parent.parentElement) ancestors.push(parent);
    let opaqueAboveCanvas = false;
    for (const ancestor of ancestors) {
      const style = getComputedStyle(ancestor), bg = color(style.backgroundColor);
      if (ancestor.matches(".stage") && ancestor.querySelector("canvas") && !opaqueAboveCanvas) {
        skip("canvas_backdrop_not_solid_css"); return;
      }
      if (bg?.[3] === 1) opaqueAboveCanvas = true;
      if (Number(style.opacity) !== 1 || style.filter !== "none" || style.mixBlendMode !== "normal") {
        skip("opacity_filter_or_blend"); return;
      }
      if (style.backgroundImage !== "none") { skip("background_image_or_gradient"); return; }
    }
    for (const ancestor of ancestors.reverse()) {
      const bg = color(getComputedStyle(ancestor).backgroundColor);
      if (!bg) { skip("unsupported_background_color"); return; }
      background = blend(bg, background);
    }
    const renderedForeground = blend(foreground, background);
    const a = luminance(renderedForeground), b = luminance(background);
    const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    const fontSize = parseFloat(foregroundStyle.fontSize);
    const fontWeight = parseInt(foregroundStyle.fontWeight, 10) || 400;
    const threshold = fontSize >= 24 || (fontSize >= 18.666 && fontWeight >= 700) ? 3 : 4.5;
    checked.push({ ...description, foreground: renderedForeground, background, ratio, threshold,
      font_size_px: fontSize, font_weight: fontWeight, passed: ratio + 0.005 >= threshold });
  }
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node, visited = 0;
  while ((node = walker.nextNode())) {
    if (++visited > 20000) throw new Error("Unexpectedly large DOM during contrast audit");
    const element = node.parentElement;
    if (!element || element.closest("script,style,option,textarea")) continue;
    const range = document.createRange(); range.selectNodeContents(node);
    inspect(element, node.textContent, Array.from(range.getClientRects()));
  }
  for (const element of document.querySelectorAll("input,textarea,select")) {
    if (element.matches('input[type=checkbox],input[type=radio],input[type=range],input[type=color],input[type=file],input[type=hidden]')) continue;
    const text = element.tagName === "SELECT" ? element.selectedOptions[0]?.textContent ?? "" : element.value;
    const rects = Array.from(element.getClientRects());
    if (text) inspect(element, text, rects);
    else if (element.placeholder) inspect(element, element.placeholder, rects, "::placeholder");
  }
  const rootStyle = getComputedStyle(document.documentElement), bodyStyle = getComputedStyle(document.body);
  const theme = Object.fromEntries(["--bg", "--panel", "--panel-hi", "--ink", "--accent", "--viewport"].map(key => {
    const css = rootStyle.getPropertyValue(key).trim(); const rgb = color(css);
    return [key, { css, rgb, luminance: rgb ? luminance(rgb) : null }];
  }));
  const bodyBackground = color(bodyStyle.backgroundColor), bodyForeground = color(bodyStyle.color);
  return { checked, failures: checked.filter(item => !item.passed), skipped,
    minimum_ratio: checked.length ? Math.min(...checked.map(item => item.ratio)) : null,
    theme, body: { background: bodyBackground, foreground: bodyForeground,
      foreground_luminance: bodyForeground ? luminance(bodyForeground) : null },
    method: "Computed CSS colors converted to sRGB with browser Canvas2D, composited over solid ancestor backgrounds; 4.5:1 normal / 3:1 large text thresholds. Visible viewport samples only." };
}

async function selfChecks(browser) {
  const page = await browser.newPage();
  await page.setContent(`<style>body{background:white;color:#111}p{margin:6px}button:disabled{color:#ddd;background:white}</style>
    <p id="good">Readable text</p><p id="bad" style="color:#aaa">Low contrast text</p>
    <button disabled>Disabled text</button><div style="background-image:linear-gradient(white,white)">Unresolved gradient</div>
    <p id="large" style="font-size:24px;color:#888">Large readable text</p>
    <details><summary>Collapsed editor</summary><textarea style="color:#aaa">Hidden low contrast</textarea></details>`);
  const audit = await page.evaluate(cssTextAudit);
  assert.equal(audit.failures.length, 1); assert.equal(audit.failures[0].element, "p#bad");
  assert(audit.checked.some(item => item.element === "p#large" && item.threshold === 3 && item.passed));
  assert(audit.skipped.some(item => item.reason === "disabled_control"));
  assert(audit.skipped.some(item => item.reason === "background_image_or_gradient"));
  assert(!audit.checked.some(item => item.text === "Hidden low contrast"));
  await page.locator("details").evaluate(element => { element.open = true; });
  assert((await page.evaluate(cssTextAudit)).failures.some(item => item.text === "Hidden low contrast"));
  await page.setContent('<div class="files" style="width:300px;height:100px;overflow:auto"><button style="margin-left:2000px">Reachable by scrolling</button></div>');
  const geometry = await layout(page);
  assert.equal(geometry.issues.length, 0); assert.equal(geometry.horizontal_scroll_regions.length, 1);
  await page.locator(".files").evaluate(element => { element.style.overflow = "hidden"; });
  assert((await layout(page)).issues.some(issue => issue.includes("Control outside horizontal viewport")));
  await page.close();
}

if (selfTest) {
  const browser = await chromium.launch({ headless: true, timeout: 30_000 });
  try { await selfChecks(browser); console.log("PASS CSS contrast checker: normal/large thresholds, low-contrast failure, disabled and unresolved-background gaps"); }
  finally { await browser.close(); }
  process.exit(0);
}
assert(options.output, "--output must name a new evidence directory");
const output = path.resolve(options.output);
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.mkdir(output);
const browser = await chromium.launch({ headless: true, timeout: 30_000 });
const report = { schema_version: 1, started_at: new Date().toISOString(), url: url.href, output,
  browser_version: browser.version(), node: process.version,
  runner_sha256: hash(await fs.readFile(new URL(import.meta.url))),
  scope: "Fresh headless browser contexts. Seven workspaces at desktop/mobile, local model assets, CSS text contrast samples, screenshots and fixed-camera robot motion. Assistant status is mocked disabled; no chat, training, checkpoint writes or provider calls.",
  limitations: ["Not a blanket WCAG conformance claim: only visible enabled DOM text with resolvable solid CSS backgrounds is measured.",
    "Disabled UI, icons/color glyphs, SVG charts, transparent overlays over canvas, gradients/filters, hover/focus/error states and native popup contents are excluded and counted.",
    "Scrollable panels are sampled at start, middle and end in both axes; dynamic or intermediate content may remain unaudited. Contained horizontal tables are reported, not treated as viewport overflow.",
    "Canvas motion establishes actual rendered robot activity with no scene props and a fixed camera, not gait/recovery correctness or GPU-compute qualification."],
  results: [], passed: false };
const button = (page, name) => page.getByRole("button", { name, exact: true });
const clock = async page => parseFloat(await page.locator(".main:not(.hidden) .transport .time").textContent());
async function pause(page) {
  if (await button(page, "Pause simulation").isVisible()) await button(page, "Pause simulation").click();
  await button(page, "Play simulation").waitFor(); await page.waitForTimeout(100);
}
async function ready(page, maker = false) {
  await page.waitForFunction(maker => {
    const stage = document.querySelector(maker ? ".maker-main:not(.hidden) .stage" : ".main:not(.hidden) .stage");
    if (stage?.querySelector(".overlay.error")) throw new Error(stage.querySelector(".overlay.error").textContent);
    return !!stage?.querySelector("canvas") && !stage.querySelector(".overlay");
  }, maker, { timeout: 120_000 });
}
async function shot(page, result, suffix) {
  const file = `${result.viewport.name}-${suffix}.png`;
  const data = await page.screenshot({ path: path.join(output, file), animations: "disabled", timeout: 30_000 });
  result.screenshots.push({ file, sha256: hash(data) });
}
async function frame(page, result, suffix) {
  const file = `${result.viewport.name}-${suffix}.png`;
  const png = await page.locator(".main:not(.hidden) .stage canvas").screenshot({ path: path.join(output, file), timeout: 30_000 });
  const meta = await sharp(png).metadata();
  assert(meta.width >= 160 && meta.height >= 180, "Canvas must provide a usable scene viewport of at least 160x180px");
  const maker = suffix === "motion-maker-canvas";
  const region = { left: 8, top: maker ? 8 : 60, width: meta.width - 16, height: meta.height - (maker ? 16 : 92) };
  assert(region.width >= 100 && region.height >= 50, "Canvas too small for scene-only analysis");
  const { data, info } = await sharp(png).extract(region).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const colors = new Set(), sums = [0, 0, 0], squares = [0, 0, 0]; let light = 0;
  for (let i = 0; i < data.length; i += 3) {
    colors.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
    if (data[i] >= 190 && data[i + 1] >= 190 && data[i + 2] >= 190) light++;
    for (let c = 0; c < 3; c++) { sums[c] += data[i + c]; squares[c] += data[i + c] ** 2; }
  }
  const pixels = info.width * info.height;
  const stddev = squares.map((sum, i) => Math.sqrt(Math.max(0, sum / pixels - (sums[i] / pixels) ** 2)));
  const evidence = { file, sha256: hash(png), width: meta.width, height: meta.height, region,
    unique_rgb_colors: colors.size, channel_stddev: stddev, light_pixel_fraction: light / pixels };
  result.frames.push(evidence);
  assert(colors.size >= 128 && Math.max(...stddev) >= 5, "Canvas lacks detailed rendered geometry");
  assert(evidence.light_pixel_fraction >= 0.5, "Canvas is not predominantly light");
  return { data, evidence };
}
function difference(a, b) {
  assert.equal(a.data.length, b.data.length); let total = 0, changed = 0;
  for (let i = 0; i < a.data.length; i += 3) {
    let max = 0;
    for (let c = 0; c < 3; c++) { const delta = Math.abs(a.data[i + c] - b.data[i + c]); total += delta; max = Math.max(max, delta); }
    if (max > 8) changed++;
  }
  return { mean_absolute_rgb_difference: total / a.data.length, changed_pixel_fraction_gt8: changed / (a.data.length / 3) };
}
async function layout(page) {
  return page.evaluate(() => {
    const issues = [], boxes = [], horizontalScrollRegions = new Set();
    for (const selector of [".shell", ".rail", ".topbar", "footer.status", ".main:not(.hidden)", ".inspector", ".training-workspace:not(.hidden)", ".assistant-workspace:not(.hidden)", ".files", ".debug"]) {
      for (const element of document.querySelectorAll(selector)) {
        if (!element.getClientRects().length) continue;
        const r = element.getBoundingClientRect();
        boxes.push({ selector, x: r.x, y: r.y, width: r.width, height: r.height });
        if (r.left < -1 || r.top < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1) issues.push(`${selector} extends outside viewport`);
        if (element.scrollWidth > element.clientWidth + 2 && !/auto|scroll/.test(getComputedStyle(element).overflowX)) issues.push(`${selector} horizontal overflow`);
      }
    }
    if (document.documentElement.scrollWidth > innerWidth + 1 || document.documentElement.scrollHeight > innerHeight + 1) issues.push("Document overflow");
    for (const element of document.querySelectorAll("button,select,textarea,input,[role=tab]")) {
      const r = element.getBoundingClientRect();
      if (!element.getClientRects().length || r.bottom <= 0 || r.top >= innerHeight) continue;
      if (r.left < -1 || r.right > innerWidth + 1) {
        let scrollParent = null;
        for (let parent = element.parentElement; parent; parent = parent.parentElement) {
          const box = parent.getBoundingClientRect();
          if (/auto|scroll/.test(getComputedStyle(parent).overflowX) && parent.scrollWidth > parent.clientWidth + 2
            && box.left >= -1 && box.right <= innerWidth + 1) { scrollParent = parent; break; }
        }
        if (scrollParent) horizontalScrollRegions.add(scrollParent);
        else issues.push(`Control outside horizontal viewport: ${element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 60)}`);
      }
    }
    return { boxes, issues: [...new Set(issues)], horizontal_scroll_regions: [...horizontalScrollRegions].map(element => ({
      element: `${element.tagName.toLowerCase()}.${[...element.classList].filter(c => !c.startsWith("svelte-")).join(".")}`,
      client_width: element.clientWidth, scroll_width: element.scrollWidth, scroll_left: element.scrollLeft,
    })) };
  });
}
async function inspectView(page, result, name) {
  const record = { name, passed: false, samples: [], errors: [] }; result.views.push(record);
  const scrollables = await page.evaluate(() => [...document.querySelectorAll("body *")].filter(element => {
    const r = element.getBoundingClientRect(), style = getComputedStyle(element);
    return element.tagName !== "TEXTAREA" && r.width > 80 && r.height > 50 && r.bottom > 0 && r.top < innerHeight
      && ((/auto|scroll/.test(style.overflowY) && element.scrollHeight > element.clientHeight + 4)
        || (/auto|scroll/.test(style.overflowX) && element.scrollWidth > element.clientWidth + 4));
  }).map(element => ({ selector: element.tagName.toLowerCase(), top: element.scrollTop })));
  record.scrollable_panels = scrollables.length;
  for (const fraction of scrollables.length ? [0, 0.5, 1] : [0]) {
    await page.evaluate(fraction => {
      for (const element of document.querySelectorAll("body *")) {
        const r = element.getBoundingClientRect(), style = getComputedStyle(element);
        if (element.tagName !== "TEXTAREA" && r.width > 80 && r.height > 50 && r.bottom > 0 && r.top < innerHeight) {
          if (/auto|scroll/.test(style.overflowY)) element.scrollTop = (element.scrollHeight - element.clientHeight) * fraction;
          if (/auto|scroll/.test(style.overflowX)) element.scrollLeft = (element.scrollWidth - element.clientWidth) * fraction;
        }
      }
    }, fraction);
    await page.waitForTimeout(100);
    const contrast = await page.evaluate(cssTextAudit), geometry = await layout(page);
    record.samples.push({ scroll_fraction: fraction, contrast, layout: geometry });
    if (contrast.checked.length < 10) record.errors.push(`Too little CSS text coverage at ${fraction}: ${contrast.checked.length}`);
    if (contrast.failures.length) record.errors.push(`${contrast.failures.length} enabled text contrast failures at ${fraction}`);
    record.errors.push(...geometry.issues);
    for (const token of ["--bg", "--panel", "--panel-hi"]) {
      if ((contrast.theme[token].luminance ?? 0) < 0.8) record.errors.push(`${token} is not a light surface`);
    }
    if ((contrast.body.foreground_luminance ?? 1) > 0.08) record.errors.push("Body text is not dark");
    const accent = contrast.theme["--accent"].rgb;
    if (!accent || accent[0] < 180 || accent[1] < 120 || accent[2] > 140) record.errors.push("Accent is not yellow");
    if (fraction !== 0.5) await shot(page, result, `${name}${fraction === 1 ? "-lower" : ""}`);
  }
  record.errors = [...new Set(record.errors)]; record.passed = record.errors.length === 0;
  console.log(JSON.stringify({ viewport: result.viewport.name, view: name, passed: record.passed, errors: record.errors,
    contrast_failures: record.samples.flatMap(sample => sample.contrast.failures) }));
}

try {
  await selfChecks(browser);
  for (const viewport of [{ name: "desktop", width: 1360, height: 900 }, { name: "mobile", width: 390, height: 844 }]) {
    const result = { viewport, passed: false, views: [], screenshots: [], frames: [], assets: [], bundles: [], errors: [],
      page_errors: [], console_errors: [], failed_requests: [], failed_http: [], forbidden_requests: [], mocked_status_calls: 0 };
    report.results.push(result);
    const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, serviceWorkers: "block", acceptDownloads: false });
    const page = await context.newPage(); page.setDefaultTimeout(20_000);
    const bundleReads = [];
    await context.route("**/*", async route => {
      const requestUrl = new URL(route.request().url());
      if (requestUrl.origin !== url.origin && !["blob:", "data:"].includes(requestUrl.protocol)) {
        result.forbidden_requests.push({ kind: "external", url: requestUrl.href }); await route.abort(); return;
      }
      if (requestUrl.pathname === "/api/assistant/status") {
        result.mocked_status_calls++; await route.fulfill({ status: 200, contentType: "application/json", body: '{"configured":false,"model":"gpt-5.4-mini"}' }); return;
      }
      if (requestUrl.pathname.startsWith("/api/")) {
        result.forbidden_requests.push({ kind: "api", url: requestUrl.href }); await route.abort(); return;
      }
      await route.continue();
    });
    page.on("pageerror", error => result.page_errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") result.console_errors.push(message.text()); });
    page.on("requestfailed", request => result.failed_requests.push({ url: request.url(), error: request.failure()?.errorText }));
    page.on("response", response => {
      if (response.status() >= 400) result.failed_http.push({ url: response.url(), status: response.status() });
      if (/\.(wasm|onnx|xml|stl)(?:\?|$)/.test(response.url())) result.assets.push({ url: response.url(), status: response.status() });
      if (/\/assets\/index-[^/]+\.(js|css)$/.test(response.url())) bundleReads.push(response.body().then(body => {
        result.bundles.push({ url: response.url(), bytes: body.length, sha256: hash(body) });
      }).catch(error => { result.errors.push(`Could not hash app bundle: ${String(error)}`); }));
    });
    const watchdog = setTimeout(() => { result.errors.push("Viewport exceeded 12-minute safety deadline"); void context.close(); }, 720_000);
    try {
      await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await button(page, "Simulate").click(); await ready(page); await pause(page);
      await page.getByRole("tab", { name: /^Scene/ }).click();
      await page.getByLabel("Scene preset", { exact: true }).selectOption("plain"); await button(page, "Load").click();
      await page.waitForFunction(() => JSON.parse(localStorage.getItem("wicroduck.world.v1"))?.objects.length === 0 && !document.querySelector(".stage .overlay"));
      await page.getByRole("tab", { name: "Robot", exact: true }).click();
      await page.getByLabel("Follow robot", { exact: true }).check(); await page.getByLabel("Follow robot", { exact: true }).uncheck();
      await page.waitForTimeout(150);
      const before = await frame(page, result, "robot-before"), start = await clock(page);
      await button(page, "Push").click(); await button(page, "Play simulation").click();
      await page.waitForFunction(start => parseFloat(document.querySelector(".main:not(.hidden) .transport .time").textContent) >= start + 1, start, { timeout: 30_000 });
      await pause(page); const after = await frame(page, result, "robot-after");
      result.robot_motion = { ...difference(before, after), simulated_seconds: await clock(page) - start,
        scope: "Plain floor, no movable props, follow disabled before both frames; overlays excluded from pixel crop" };
      assert(result.robot_motion.changed_pixel_fraction_gt8 >= 0.001 && result.robot_motion.mean_absolute_rgb_difference >= 0.05, "No actual fixed-camera robot motion detected");
      await button(page, "Reset scene").click();
      await inspectView(page, result, "simulate");
      await page.getByRole("tab", { name: /^Scene/ }).click();
      await page.getByLabel("Scene preset", { exact: true }).selectOption("playground"); await button(page, "Load").click();
      await page.waitForFunction(() => JSON.parse(localStorage.getItem("wicroduck.world.v1"))?.objects.length === 7);
      await inspectView(page, result, "scene");
      for (const [label, name] of [["Motion maker", "motion-maker"], ["Train", "train"], ["Assistant", "assistant"], ["Files", "files"], ["Debug", "debug"]]) {
        try {
          await button(page, label).click();
          if (name === "motion-maker") {
            await ready(page, true); await frame(page, result, "motion-maker-canvas");
            await page.locator(".maker-main details").first().evaluate(element => { element.open = true; });
          }
          if (name === "assistant") await page.waitForFunction(() => !document.querySelector(".assistant-view .status-loading"));
          await page.waitForTimeout(250); await inspectView(page, result, name);
        } catch (error) {
          result.errors.push(`${name}: ${error.stack ?? error}`); await shot(page, result, `${name}-failure`).catch(() => {});
        }
      }
      for (const extension of ["wasm", "onnx", "xml", "stl"])
        if (!result.assets.some(asset => new RegExp(`\\.${extension}(?:\\?|$)`).test(asset.url))) result.errors.push(`No ${extension} model asset observed`);
      if (result.views.length !== 7) result.errors.push("Not all seven requested views were audited");
      if (result.page_errors.length || result.console_errors.length || result.failed_http.length || result.failed_requests.length || result.forbidden_requests.length)
        result.errors.push("Browser/runtime/request errors occurred; see detailed lists");
      await Promise.all(bundleReads);
      result.passed = result.errors.length === 0 && result.views.every(view => view.passed);
    } catch (error) { result.errors.push(error.stack ?? String(error)); await shot(page, result, "failure").catch(() => {}); }
    finally { clearTimeout(watchdog); await context.close(); }
  }
  report.passed = report.results.length === 2 && report.results.every(result => result.passed);
} catch (error) { report.error = error.stack ?? String(error); }
finally {
  await browser.close(); report.finished_at = new Date().toISOString();
  await fs.writeFile(path.join(output, "results.json"), `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });
}
console.log(`Theme evidence: ${path.join(output, "results.json")}`);
if (!report.passed) process.exitCode = 1;
