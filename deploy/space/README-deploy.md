# Deploying the dashboard to Hugging Face Spaces

Free tier, Docker SDK, no payment method. Three steps, all of which need **your** Hugging
Face account — creating accounts and entering credentials is yours to do, not something
this repository automates.

1. **Create the Space.** At https://huggingface.co/new-space choose a name, pick
   **Docker → Blank**, hardware **CPU basic (free)**, visibility **Public**.

2. **Push these files to it.** The Space is a git repository of its own. It needs the
   package, the lockfile, the Streamlit theme, the backtest reports, and the two files
   in this directory at its root. Run this from the repository root:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name> /tmp/space
   ```

   ```bash
   cp -R src pipelines config .streamlit reports pyproject.toml uv.lock /tmp/space/ && \
     cp deploy/space/Dockerfile deploy/space/README.md /tmp/space/
   ```

   ```bash
   cd /tmp/space && git add -A && git commit -m "Deploy dashboard" && git push
   ```

   `deploy/space/README.md` lands as the Space's `README.md`, which does double duty:
   Hugging Face reads its YAML front matter to configure the Space, and the build
   backend needs *a* README present because `pyproject.toml` declares one. Omitting it
   fails the build with a hatchling error that mentions neither.

   `reports` carries the backtest the benchmark panel plots. It is gitignored, so it
   exists only on the machine that ran `python -m pipelines.audit` — if that is not this
   one, run the audit before copying or the build fails on the missing directory. That is
   the better failure: a build that stops is easier to notice than a page that quietly
   drops the PSE comparison. Unlike everything in step 3 it does not refresh itself. It is
   a snapshot, the panel prints the window it covers, and recomputing the backtest means
   re-copying and rebuilding.

3. **Point it at the state branch.** In the Space's *Settings → Variables and secrets*,
   add a **variable** (not a secret — it is a public URL):

   | Name | Value |
   |---|---|
   | `STATE_REPO_URL` | `https://github.com/tomeelow/load-forecasting.git` |

   Without it the Space renders, says it is reading the local working tree, and shows
   empty panels — which is the honest failure mode, not a broken one.

The build takes about five minutes. Free Spaces sleep after 48 hours idle and wake on the
next request, which costs a visitor roughly thirty seconds on a cold open.

## Alternatives

Render, Railway and Fly.io all host the Streamlit container on a free tier with the same
two environment variables. They sleep when idle too. Whichever you choose, **the scheduled
retraining keeps running on GitHub Actions regardless** — the live loop does not depend on
where the dashboard is hosted, which is the point of ADR-008's state branch.
