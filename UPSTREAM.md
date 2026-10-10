# Upstream provenance and publication gate

This is a **private development snapshot**, not an open-source release or a
new overall license grant. Do not make it public, redistribute a build or offer
commercial hosting until the permissions below are reviewed. Private storage
itself is not evidence of redistribution clearance.

## Wicroduck application

- Source: https://github.com/ngxson/wicroduck
- Base commit: `b0035c15c532bbb69b22a7edc1bb79b7b7430cd0`
- Checked 2026-10-10: no LICENSE/NOTICE, package license field or README grant
  exists at this pin; GitHub reports no license. Publicly readable source is
  not general permission to redistribute it. Clarify the author's terms before
  a public or distributable derivative release.
- This repository preserves upstream credits and extends the application with
  props, task/motion choices, lifecycle/evaluation fixes, an optional local AI
  assistant and a light theme. It does not claim sole authorship.
- `src/train/kernels/kernels.wasm` is the tracked upstream project kernel with
  C source and an Emscripten build script. It is not a robot checkpoint or CUDA
  binary. The project is therefore not literally a source-only archive.

## Pollen Robotics

- Submodule: https://github.com/pollen-robotics/microduck_rl
- Pin: `5bbe9637294d0c794edb185e284cfb1a77c6a0b4`
- Code: Apache-2.0, copyright 2026 Pollen Robotics. See the submodule LICENSE
  and `third_party/POLLEN-APACHE-2.0.txt`. No upstream NOTICE was found. Some
  observation, reward and actuator formulas are ports of this reference code,
  not newly originated formulas; see `NOTICE.md`.
- The [pinned README](https://github.com/pollen-robotics/microduck_rl/blob/5bbe9637294d0c794edb185e284cfb1a77c6a0b4/README.md#license)
  separately labels 3D model files **Creative Commons BY-SA-NC**, without a
  version. Do not label meshes Apache-2.0 or invent a CC version.
- Local preparation copies three XML models and 38 meshes and creates three
  stripped no-visual XML variants. Generated assets are not committed here.
  Reconstruction does not erase their upstream restrictions.

## Policies and dependencies

- [Official policies](https://huggingface.co/pollen-robotics/microduck-policies)
  declare Apache-2.0 in metadata. Downloads use revision
  `6677ec007142bbaf05123a468f6ae58d13808786`; names, sizes and SHA-256 values
  are pinned in `src/sim/policy-catalog.ts`. Weights are not committed here.
- MuJoCo 3.12.0: Apache-2.0. ONNX Runtime Web 1.29.0 and Three.js 0.180.0: MIT.
- Svelte/Vite and most tooling: MIT; TypeScript: Apache-2.0; dev-only Lightning
  CSS: MPL-2.0. Exact versions/integrity/package license declarations are in
  `package-lock.json`; actual licenses remain in the packages. This summary
  does not replace those notices.
- A public static/binary bundle needs a complete dependency and asset notice
  audit. No such distributable is published by this private snapshot.

The fresh history excludes local keys, generated assets, saved runs, logs,
browser profiles and the parent CUDA-M history. CUDA-M compiler/runtime code
is not copied into this repository.
