# The Hugging Face Space image. Same package as the local stack, but with no scheduler,
# no ENTSO-E token and no MLflow server — so it mirrors the pipeline's state from the
# public `pipeline-state` branch at startup and reads that. See src/dashboard/state_sync.py.
FROM python:3.12-slim-bookworm

# libgomp1 for LightGBM (MLflow loads the champion to read its signature), git for the
# state mirror, curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

# Spaces run the container as uid 1000. Everything the app writes — the mirrored state
# tree and Streamlit's own cache — has to be under a directory that user owns.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/opt/venv/bin:/home/user/.local/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/user/venv \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY --chown=user src ./src
COPY --chown=user pipelines ./pipelines
COPY --chown=user config ./config
COPY --chown=user .streamlit ./.streamlit
RUN uv sync --locked --no-dev

ENV PATH=/home/user/venv/bin:$PATH

# Spaces expose 7860 by convention.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "src/dashboard/app.py"]
