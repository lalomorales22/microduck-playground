# Optional browser verification

These tools are separate from the app's runtime dependencies. They use isolated
Chromium profiles, mock assistant transport and never call a paid model. Do not
point them at a public server or at a user's browser profile.

Review current advisories before installing their pinned dependencies:

```bash
cd tests/browser
npm audit --package-lock-only --ignore-scripts
# Stop and review any new advisories before proceeding.
npm ci --ignore-scripts
npx --no-install playwright install chromium
cd ../..
npm run build
npm run preview
```

In another terminal at the repository root, use new output directories:

```bash
node tests/browser/test_microduck_assistant.mjs --url http://127.0.0.1:8768/ --viewport both --output out/assistant
node tests/browser/test_microduck_playground.mjs --url http://127.0.0.1:8768/ --output out/playground
node tests/browser/test_microduck_run_assistant.mjs --url http://127.0.0.1:8768/ --output out/run-control
node tests/browser/test_microduck_theme.mjs --url http://127.0.0.1:8768/ --output out/theme
```

The optional run-control `--checkpoint PATH` argument imports an exported
checkpoint into the disposable profile, verifies evaluation/replay and confirms
unchanged source bytes. User checkpoints are not shipped with this repository.
Short two-iteration runs exercise lifecycle behavior, not learned acrobatics.
Theme contrast sampling is not a complete accessibility certification.

Receipts, screenshots and checkpoint copies stay in ignored `out/`. Each runner
refuses to overwrite its output directory. Check results rather than assuming
a screenshot proves learning. The upstream/pretrained policies and real MuJoCo
physics remain local; only the assistant HTTP replies are mocked.
