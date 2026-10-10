# Local verification

Date: 2026-10-10. Apple M4 Mac mini, macOS 27.0, Node 24.19.0.
This is local evidence, not cross-chip, cross-browser or hardware qualification.

## Standalone packaging

- Fresh repository assembled from the pinned Wicroduck tree and reviewed local
  changes, without importing either parent repository's Git history.
- The exact pinned Microduck submodule was checked out separately.
- Fresh `npm ci --ignore-scripts --offline` installed the locked app dependencies
  from the existing package cache; package lifecycle scripts were not executed.
- Fresh `npm run prepare-assets` reconstructed all model/runtime assets and
  downloaded the three revision-pinned policies, verifying expected size/hash.
  All stripped/full model dimension and mass comparisons passed.
- `npm run build` passed, with zero Svelte errors or warnings. Final theme
  verification targets the standalone production preview directly.
- `npm run check:standup`: all five pretrained recovery trials passed.
- The fresh standalone `npm run check` also passed all stages, including 184
  assistant/training tests and 19 controller/replay tests.
- Sixteen release-guard tests passed in isolated temporary Git repositories;
  the real staged index passed before publication, including its secret checks.
- App and optional browser-tool lockfile audits reported zero known advisories.
  Audit results are dated snapshots, not guarantees against unknown issues.
- No real API request was made during packaging/theme checks. No `.env`, user
  checkpoint, browser profile, generated asset bundle or provider key is included.

The small tracked upstream SIMD WASM kernel remains paired with its C source.
Private review only: see UPSTREAM.md for the unresolved Wicroduck license and
separate noncommercial robot-model terms. No public hosted build is released.

## Light-theme checks

The standalone production preview passed all 14 combinations of seven
workspaces (Simulate, Scene, Motion maker, Train, Assistant, Files, Debug) and
two viewports (1360x900 desktop, 390x844 mobile). Loaded JS/CSS hashes match the
artifacts below. Checks used isolated Playwright Chromium profiles and the
existing Playwright 1.62.1/sharp 0.35.5 runtime; the separately locked optional
browser-tool environment was audited, not freshly installed for this run.

- White/light surfaces, black text and Microduck-yellow controls rendered.
- 1,186 enabled solid-background CSS text measurements passed, minimum 5.3997:1.
  Disabled controls, graphics, unresolved backgrounds and other unmeasured
  states are excluded; this is not a complete accessibility audit.
- No detected layout, browser-runtime or network failures. Contained scrolling
  is allowed. Motion maker retains a 334x220 preview on the mobile viewport.
- Real model assets rendered on detailed, predominantly light canvases. With
  the camera fixed and overlays excluded, robot motion changed 6.27% of desktop
  scene pixels and 8.16% on mobile. No new policy was trained by this test.
- The assistant status was mocked; chat/provider requests were blocked.
  No user profile, saved run or real API key was used.

Receipt SHA-256: `bb729ef345fba8a1405e00e729957fe607b2e79fd47a4ce37911dbe1132be354`.
Runner SHA-256: `12a0da5f149fa354b5fae54a747adbae5d2076af66c6f1fc76fc09f63035f5f9`.

## Prior behavior checks

The preceding run-control stage passed the full check suite, including 184
assistant/training tests and 19 controller/replay tests. Actual browser workers
completed two training iterations and a two-iteration continuation in a new
checkpoint while preserving the original. Desktop/mobile approval, stale/busy
refusal, evaluation/cancel and upright saved-scene replay checks passed.
Assistant routes were mocked in isolated browser tests; their physics and
training were real. A separate live API evaluation also ran the actual local
worker after explicit approval.

The actual original jump policy (50 iterations) and a separate continuation to
1000 iterations both failed jump qualification. More upright time was not a
jump. No qualified takeoff/flight/landing was observed. User checkpoints and
private chat/evaluation screenshots are intentionally not redistributed here.

## Artifact identities

| Generated build artifact | SHA-256 |
| --- | --- |
| `index-LQko89mL.js` | `d4325d9bba9b4b2132f44c75f6bb79771bc38a7677b38716e0983ac871088246` |
| `index-BAh6vBVY.css` | `e25a4e327f151b20509cb19671ec82248a3578caf41de9dcb0b55ef86df6705a` |

Generated screenshots/logs/reports stay outside source control. The optional
[browser runners](../tests/browser/README.md) create local receipts and refuse
overwrites. Two-iteration training checks prove lifecycle integration, not a
mastered skill. Text-contrast sampling and screenshots are not a complete WCAG
audit. Run the documented checks again after changes rather than treating this
snapshot as permanent certification.
