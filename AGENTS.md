# Microduck Playground development

Private derivative development snapshot: read UPSTREAM.md before publication.
Do not grant an overall license or make this public without resolving upstream
permissions and the asset-notice gates.

- Preserve the Svelte/TypeScript/Three.js/MuJoCo architecture and upstream credit.
- Preserve checkpoints/scenes. Resume creates a fork, not an overwrite.
- AI output is untrusted: validate schema/freshness, require explicit approval,
  keep secrets server-side, and do not add shell tools.
- CPU/WebAssembly is not CUDA/Metal training. Completion, upright posture and
  reward alone do not establish a learned jump or flip.
- Replay starts upright in the saved world. Locomotion, motion and acrobatics
  checkpoints disable automatic recovery; stand-up policies retain it.
- Use white surfaces, near-black text, restrained Microduck yellow accents,
  readable statuses and responsive utility layouts.
- Use Node 24+, pinned dependencies and `npm ci --ignore-scripts` after an
  advisory review. Never read/log real `.env` values during routine work.
- Release checks: `npm run prepare-assets`, `npm run check`,
  `npm run check:standup`, `npm run build`, `npm run check:release`.
- Browser tests use isolated profiles/mocked assistant transport. Never change
  user storage or make paid API calls for automated tests.
- Exclude dependencies, builds, generated assets, private screenshots, trained
  checkpoints, browser profiles, logs and credentials from commits.
- Inspect the staged diff. No tool attribution or co-author trailers in commits.
