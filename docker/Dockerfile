# One image, two services. The API and the dashboard import the same package, share the
# same feature builder and read the same state directory, so building them twice would
# double the build time to produce two identical layer stacks that differ only in their
# command. Compose picks the command; see docker-compose.yml.
FROM python:3.12-slim-bookworm

# LightGBM links against the OpenMP runtime, which is not a Python package and is not in
# the slim base. Without it `import lightgbm` fails at container start with a linker
# error that says nothing about the cause. curl is here for the healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, from the lockfile only, so editing source does not re-resolve or
# re-download anything. `--no-install-project` is what keeps this layer independent of
# the code below it.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY pipelines ./pipelines
COPY config ./config
COPY .streamlit ./.streamlit
RUN uv sync --locked --no-dev

# Written to at runtime by the pipelines and read by both services. Declared here so the
# directories exist even when nothing has mounted a volume over them yet.
RUN mkdir -p /app/data/processed /app/state /app/reports /app/monitoring

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
