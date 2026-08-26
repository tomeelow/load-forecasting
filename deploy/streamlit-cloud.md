# Deploying the dashboard to Streamlit Community Cloud

Free, no container, no payment method. This is the path to use unless you have a reason
not to — Hugging Face Spaces now charges for the SDK rather than only the hardware, and
[deploy/space/README-deploy.md](space/README-deploy.md) records what that costs.

Both steps need **your** Streamlit and GitHub accounts. Creating accounts and authorising
one service against another is yours to do, not something this repository automates.

1. **Point Community Cloud at the repository.** At https://share.streamlit.io sign in with
   GitHub, choose **Create app → Deploy a public app from GitHub**, and fill in:

   | Field | Value |
   |---|---|
   | Repository | `tomeelow/load-forecasting` |
   | Branch | `main` |
   | Main file path | `src/dashboard/app.py` |

   Python version, under *Advanced settings*, must be **3.12** — `pyproject.toml` pins
   `>=3.12,<3.13` and the lockfile is resolved for it.

2. **Give it the state branch.** Still in *Advanced settings*, paste this into **Secrets**:

   ```toml
   STATE_REPO_URL = "https://github.com/tomeelow/load-forecasting.git"
   ```

   It has to sit at the **top level** of that TOML, not under a `[section]`. Community
   Cloud exports root-level secrets as environment variables, and `mirror_state` reads
   `STATE_REPO_URL` from the environment; a sectioned key reaches `st.secrets` only and
   the page will report that it is reading a local working tree it does not have.

   Despite living in the secrets box it is not a secret — it is a public URL, and it is
   written down here on purpose. Secrets is simply the only place Community Cloud lets you
   set an environment variable.

Then click deploy. First build takes a few minutes.

## Renaming a config key needs a reboot

Community Cloud reruns the script when the repository changes, but it does not re-import
modules that are already in memory. `load_config()` reads `config/config.yaml` from disk
on every call while the dataclasses it fills were defined at import time, so a commit that
*renames* a key lands as a new YAML against an old schema and the page dies with a
`TypeError` naming the key — even though the repository is perfectly consistent.

Adding or changing a *value* is fine. Renaming, adding or removing a **key** is not, and
the fix is not another commit: open the app on share.streamlit.io and use **Manage app →
⋮ → Reboot app**, which starts a fresh process that imports the new schema. This happened
on 2026-08-26 when `keep_runs_days` became `keep_runs`.

## What the host actually installs

`uv.lock` is the **first** dependency file Community Cloud looks for, ahead of `Pipfile`,
`environment.yml`, `requirements.txt` and `pyproject.toml`, and this repository has one at
its root — so the environment matches the lockfile the CI and the containers use, and
there is no second list of pinned versions to drift out of step. Do not add a
`requirements.txt`: only the first file found is used, and it would silently win.

`packages.txt` asks apt for `git`, which `mirror_state` shells out to. It is one line and
carries no comment, because every line in that file is passed to `apt-get` and a `#` would
be read as a package name.

## Why the page still has data

Nothing the dashboard plots is in the repository — `state/`, `mlruns/` and
`data/processed/` are all gitignored, and a host that builds from git can only see what
git carries. They arrive at runtime instead: `mirror_state` shallow-clones the
`pipeline-state` branch that the daily GitHub Actions loop force-pushes, and the page
refreshes it hourly. That clone is filtered and sparse, so it fetches the ~4MB the page
reads rather than the ~355MB the branch holds — see `src/dashboard/state_sync.py`.

The one exception is the backtest. `reports/audit_c_h24.csv.gz` **is** committed, because
the benchmark panel — the PSE comparison this project exists to make — plots it, and it is
not something the nightly loop produces: the audit refits at each of 26 origins and runs
on demand. It is gzipped to keep 660KB of prediction dump out of git as 199KB, and
`load_backtest` reads either form. Unlike everything mirrored, it does not refresh itself.
It is a snapshot, the panel prints the window it covers, and recomputing it means:

```bash
python -m pipelines.audit && gzip -9 -c reports/audit_c_h24.csv > reports/audit_c_h24.csv.gz
```

then committing the result.

## What it is not

The hosted page shows what the system recorded; it does not serve fresh forecasts, because
that needs a live weather call and a model in memory. Run the stack locally with
`docker compose up` for that. The page says as much at the top rather than implying
otherwise.

Free apps sleep after a period of inactivity and wake on the next request. **The scheduled
retraining keeps running on GitHub Actions regardless** — the loop does not depend on where
the dashboard is hosted, which is the point of ADR-008's state branch.
