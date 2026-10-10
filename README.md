# Microduck Playground

A local robot playground: simulate Microduck, arrange physical props, author
motion references, train browser CPU policies, and use an optional AI assistant
to prepare scenes and manage bounded training runs.

**Private development snapshot. Licensing is not cleared for public release.**
Built on [Wicroduck](https://github.com/ngxson/wicroduck) and Pollen Robotics'
Microduck assets. This is not an official Pollen Robotics product. See
[upstream provenance and license boundaries](UPSTREAM.md) before sharing.

## Run locally

Requirements: Git, Node.js 24 or newer, npm, and a recent browser supporting
WebAssembly SIMD and the origin-private filesystem (Chromium is locally tested).
First installation needs network access for dependencies and pinned assets.

```bash
git clone --recurse-submodules https://github.com/lalomorales22/microduck-playground.git
cd microduck-playground
npm audit --package-lock-only --ignore-scripts
# Review advisories before installing. No known advisories is not a guarantee.
npm ci --ignore-scripts
npm run prepare-assets
npm run dev
```

Open [127.0.0.1:8767](http://127.0.0.1:8767/). The server binds to loopback and
refuses an occupied port. Stop the existing app before starting a second copy.
Checkpoints belong to the browser and exact origin; changing hostname, port or
browser profile does not move them. Export through Files before switching.

`prepare-assets` copies runtimes from installed packages, reconstructs model
assets from the pinned submodule and downloads three hash-verified policies.
Generated files are ignored, not included in this repository. A small upstream
WASM kernel is tracked with its C source; normal builds do not need Emscripten.
No global package installation is needed.

## Workspaces

- **Simulate:** standing/recovery and walking policies; saved-policy replay;
  pause, step, reset, speed, camera and collision controls.
- **Scene:** balls, boxes, cylinders, ramps, steps, hurdles, platforms and
  dominoes with MuJoCo contacts; presets, object editing and JSON import/export.
- **Motion maker:** 14-joint references, editable keyframes and 13 templates.
- **Train:** bounded PPO, nine task objectives, motion tracking, saved scenes,
  checkpoint continuation and deterministic jump evaluation.
- **Assistant:** propose scenes/motions/settings, then separately approve
  start, resume, stop, evaluation or replay. Monitor progress locally.
- **Files:** import/export and manage browser-local checkpoints and motions.

Resume writes a new checkpoint and preserves the original. Saved replay opens
upright, stationary and paused in its saved world. A poor policy can still fall
after Play; replay never teleports it upright. Locomotion, motion and acrobatics
checkpoints run without automatic recovery assistance; stand-up policies retain
their recovery controller.

## Optional AI assistant

Simulation and training work without a key. To enable the assistant:

```bash
cp .env.example .env
chmod 600 .env
```

Enter your own `OPENAI_API_KEY` in `.env` using a local editor. Never paste a key
into chat, use a `VITE_` secret variable, or place it under `public/` or `dist/`.
The server rereads it on each request. `OPENAI_MODEL` supports `gpt-5.4-mini`
and `gpt-6-luna`; availability and billing depend on your API account. This
repository has no shared key or hosted AI service.

The sharing checkbox sends chat and bounded scene/motion/training metadata to
OpenAI. Weights and arbitrary local files are not sent. Requests incur API
usage; `store: false` is requested, not a promise of zero provider retention.
Chat and evaluation reports disappear on reload. Local progress monitoring
does not repeatedly call the model.

This assistant is not Codex and cannot edit source, run shell commands, invent
new reward implementations, read arbitrary files or control a physical robot.
Operations need explicit approval and are checked for stale settings, changed
files and busy workers. See [security](SECURITY.md).

## What is and is not proven

Physics and training use **CPU/WebAssembly SIMD**. Three.js renders with WebGL.
This app does not use CUDA-M, Metal compute or the upstream CUDA/Warp training
stack. It began as a companion experiment in [CUDA-M](https://github.com/lalomorales22/cuda-m),
but has its own repository.

Completion is not skill mastery. Jumping, somersault and backflip objectives
are experimental. Motion templates are joint targets, not learned or balanced
maneuvers. Measured 50- and 1000-iteration jump checkpoints did not qualify as
jumps. A better learning curriculum remains future work.

The jump check is one deterministic three-second trial requiring grounded
takeoff, at least 3 cm airborne rise, 60 ms contact-free flight and a half-second
supported upright landing. Other tasks remain explicitly unqualified until
corresponding checks exist. This is not robustness, hardware or sim-to-real
certification. No physical robot is connected.

## Test and build

```bash
npm run prepare-assets
npm run check
npm run check:standup
npm run check:release
npm run build
npm run preview
```

Preview runs on [127.0.0.1:8768](http://127.0.0.1:8768/), with separate browser
storage. Both dev and preview include the optional local assistant backend.
Static `dist/` hosting does not provide that backend. Public hosting is not
supported by this snapshot and needs a licensing/asset-notice review. No
deployment or CI automation is configured.

See [verification](docs/VERIFICATION.md) and [developer guidance](AGENTS.md).
