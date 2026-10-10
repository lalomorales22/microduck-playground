# Security and privacy

This private development app is for a trusted user's loopback server. It is
not an authenticated multi-user service. Do not expose Vite with `--host
0.0.0.0`, tunnels, reverse proxies or public hosting. No physical-robot control,
arbitrary shell execution or application source editing is provided.

## Secrets and assistant

- Keep keys in a local `.env` (mode 0600 on macOS/Linux) or server environment.
  Never use `VITE_` secret names or put credentials in `public/` or `dist/`.
- `.env`, `.env.*`, `local.env`, private keys, logs and generated output are
  ignored. Only the blank `.env.example` belongs in Git.
- Assistant requests require a loopback socket, expected Host/Origin and a
  custom POST header. Configuration paths under public/output directories are
  refused. Requests and model output are bounded and schema-validated.
- Context sharing requires consent; execution requires separate approval.
  The server returns data, not executable code.
- Concurrency, input/output size, duration and request rate are bounded. API
  usage still incurs charges; those limits are not an account billing cap.
- Chat is memory-only. Checkpoints live in browser origin-private storage.
  Exports contain learned weights; keep personal exports out of Git.
- If a real key is published, revoke/rotate it with the provider. Deleting a
  file or changing repo visibility does not invalidate the key or remove copies.

## Dependency and release checks

Run `npm audit --package-lock-only --ignore-scripts` before installing; review
new advisories before `npm ci --ignore-scripts`. Then run the explicit reviewed
`prepare-assets` script. Policies are pinned and verified by size/SHA-256; the
robot source is a pinned git submodule.

The 2026-10-10 lockfile audit reported zero known vulnerabilities. This is a
dated database result, not proof that dependencies are vulnerability-free.
No automatic upgrades or third-party CI actions are configured.

`npm run check:release` scans the Git index, not arbitrary private files. It
rejects common secret formats, private/generated paths, oversized files and
unexpected symlinks. This is a focused safeguard, not comprehensive DLP.
Manually review the index and license boundaries before publishing.

Keep security reports private to the repository owner. Never include real keys,
personal checkpoints or private training context in public issues/examples.
