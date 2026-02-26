# OpenClaw Companion Notes for OpenCLI

## Workspace paths & sync
- Repo root: `/root/openclaw/workspace/repos/OpenCLI`. The workspace folder is in the `syncthing` share, so keep the repo tidy and avoid editing binaries under `opencli.egg-info` unless necessary.
- Syncthing also mirrors `/root/openclaw/workspace/memory`, so any notes you append (or I append) get copied into your personal vault. That feed is maintained by the `memory:daily-save` cron job running via the Gateway agent.

## Running inside OpenClaw
- Python is available system-wide; create a venv or rely on the workspace’s interpreter to install dependencies (`pip install -e .`).
- `opencli` uses local models and functions; point it at the portal-provided helper scripts (e.g., the ones under `/root/openclaw/workspace/tools`) if you need custom tool chains.
- Keep session logs inside `/root/openclaw/workspace/tools` or other sanctioned folders so the memory tracker can pick them up.

## Operational reminders
- Syncthing is configured to sync the entire workspace folder. There’s already a `.stignore` blocking `node_modules/`, `.git/`, and temporary files, so duplicates stay out of the shared feed.
- The portal (http://100.79.186.77:8080) is your daily control center—drop session summaries, share repo links, or upload slides directly to the portal so I can reference them from memory.
- Gateway-managed cron jobs (`morning-routine`, `healthcheck:self-heal`, `memory:daily-save`) keep the infrastructure consistent. No additional `crontab` entries are needed.

## Collaboration tips
- When machining new features, mention in the portal chat or your daily note that you touched `OpenCLI` so I can tie the change to the right context.
- I can automate PRs for this repo—just flag me with the branch to push and I’ll open the pull request.
