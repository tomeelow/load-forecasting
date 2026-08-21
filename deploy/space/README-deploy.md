# Deploying the dashboard to Hugging Face Spaces

Free tier, Docker SDK, no payment method. Three steps, all of which need **your** Hugging
Face account — creating accounts and entering credentials is yours to do, not something
this repository automates.

1. **Create the Space.** At https://huggingface.co/new-space choose a name, pick
   **Docker → Blank**, hardware **CPU basic (free)**, visibility **Public**.

2. **Push these files to it.** The Space is a git repository of its own. It needs the
   package, the lockfile, the Streamlit theme, and the two files in this directory at
   its root. Run this from the repository root:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name> /tmp/space
   ```

   ```bash
   cp -R src pipelines config .streamlit pyproject.toml uv.lock /tmp/space/ && \
     cp deploy/space/Dockerfile deploy/space/README.md /tmp/space/
   ```

   ```bash
   cd /tmp/space && git add -A && git commit -m "Deploy dashboard" && git push
   ```

   `deploy/space/README.md` lands as the Space's `README.md`, which does double duty:
   Hugging Face reads its YAML front matter to configure the Space, and the build
   backend needs *a* README present because `pyproject.toml` declares one. Omitting it
   fails the build with a hatchling error that mentions neither.

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
