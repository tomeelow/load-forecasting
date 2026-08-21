# Deploying the dashboard to Hugging Face Spaces

Free tier, Docker SDK, no payment method. Three steps, all of which need **your** Hugging
Face account — creating accounts and entering credentials is yours to do, not something
this repository automates.

1. **Create the Space.** At https://huggingface.co/new-space choose a name, pick
   **Docker → Blank**, hardware **CPU basic (free)**, visibility **Public**.

2. **Push these files to it.** The Space is a git repository of its own; it needs the
   project source plus the two files in this directory at its root.

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name> /tmp/space
   rsync -a --exclude .git \
       --include 'src/***' --include 'pipelines/***' --include 'config/***' \
       --include '.streamlit/***' --include 'pyproject.toml' --include 'uv.lock' \
       --exclude '*' ./ /tmp/space/
   cp deploy/space/Dockerfile deploy/space/README.md /tmp/space/
   cd /tmp/space && git add -A && git commit -m "Deploy dashboard" && git push
   ```

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
