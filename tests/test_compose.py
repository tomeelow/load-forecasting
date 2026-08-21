"""The Compose stack, read as data.

`docker compose up` is the setup step the README promises, so what it wires together is
part of the deliverable rather than an implementation detail. The failures worth catching
here are the quiet ones: a service pointed at a tracking store the others cannot see, a
volume dropped from one service so the dashboard reads a different prediction log than
the API writes, or a port renumbered away from what the README tells people to open.

None of this starts a container. It checks that what the file says agrees with what the
code and the documentation expect.
"""

from __future__ import annotations

import yaml

from src.config import REPO_ROOT

COMPOSE = REPO_ROOT / "docker-compose.yml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"

# Services that run the project's own code, as opposed to an off-the-shelf image.
OURS = ("api", "dashboard", "trainer")

# The paths every one of our services must see the same copy of. The prediction log lives
# under `state/` and the dataset under `data/`; a service missing either is looking at a
# different system.
SHARED = ("./data", "./state")


def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def services() -> dict:
    return compose()["services"]


def mounts(service: dict) -> set[str]:
    return {volume.split(":")[0] for volume in service.get("volumes", [])}


def test_the_stack_has_the_four_pieces_the_plan_names():
    assert set(services()) >= {"mlflow-db", "mlflow", "api", "dashboard"}


def test_every_service_of_ours_points_at_the_mlflow_server_not_at_a_local_file():
    """A container falling back to `sqlite:///mlruns` would register into its own layer."""
    for name in OURS:
        uri = services()[name]["environment"]["MLFLOW_TRACKING_URI"]
        assert uri.startswith("http://mlflow:"), f"{name} points at {uri}"


def test_the_code_honours_that_variable(monkeypatch):
    """The wiring above is decorative unless config actually reads it."""
    from src.config import load_config

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    assert load_config().mlflow.tracking_uri == "http://mlflow:5000"


def test_the_api_and_dashboard_share_the_dataset_and_the_prediction_log():
    """Otherwise the dashboard plots one system's predictions and the API writes another's."""
    for name in ("api", "dashboard"):
        assert mounts(services()[name]) >= set(SHARED), f"{name} is missing a shared volume"


def test_the_trainer_writes_where_the_others_read():
    assert mounts(services()["trainer"]) >= set(SHARED)


def test_the_dashboard_can_reach_the_reports_it_plots():
    """The backtest evidence panel reads `reports/`; the API has no use for it."""
    assert "./reports" in mounts(services()["dashboard"])


def test_the_published_ports_are_the_ones_the_readme_tells_people_to_open():
    published = {
        name: [port.split(":")[0] for port in service.get("ports", [])]
        for name, service in services().items()
    }

    assert published["api"] == ["8000"]
    assert published["dashboard"] == ["8501"]
    assert published["mlflow"] == ["5000"]


def test_mlflow_waits_for_a_database_that_is_actually_ready():
    """MLflow's first act is a schema migration, which fails against a booting Postgres."""
    assert services()["mlflow"]["depends_on"]["mlflow-db"]["condition"] == "service_healthy"


def test_our_services_wait_for_mlflow():
    for name in OURS:
        assert services()[name]["depends_on"]["mlflow"]["condition"] == "service_healthy"


def test_the_registry_and_the_artifacts_survive_a_restart():
    """A named volume is the difference between a stack and a demo that forgets."""
    volumes = set(compose()["volumes"])

    assert volumes >= {"mlflow_pg", "mlflow_artifacts"}
    assert "mlflow_pg" in mounts(services()["mlflow-db"])
    assert "mlflow_artifacts" in mounts(services()["mlflow"])


def test_the_token_is_passed_from_the_environment_and_never_written_down():
    """`.env` is not committed, so the value must come from the shell at up-time."""
    text = COMPOSE.read_text()

    assert "${ENTSOE_API_KEY:-}" in text
    assert "ENTSOE_API_KEY: " not in text.replace("ENTSOE_API_KEY: ${ENTSOE_API_KEY:-}", "")


def test_the_trainer_does_not_start_with_the_stack():
    """It is a one-shot pipeline run, not a service; `up` must not block on it."""
    assert services()["trainer"]["profiles"] == ["tools"]


def test_the_api_healthcheck_tolerates_an_empty_registry():
    """503 with no champion is the correct answer, not a container that should restart."""
    check = services()["api"]["healthcheck"]["test"][-1]

    assert "503" in check


def test_the_image_installs_the_openmp_runtime_lightgbm_needs():
    """Without libgomp1 the container builds cleanly and dies on `import lightgbm`."""
    assert "libgomp1" in DOCKERFILE.read_text()


def test_the_image_installs_from_the_lockfile():
    """An unlocked install makes the container a different environment from the tests."""
    assert "--locked" in DOCKERFILE.read_text()


def test_one_image_serves_both_services():
    """Two identical builds differing only in their command is build time for nothing."""
    api, dashboard = services()["api"], services()["dashboard"]

    assert api["build"] == dashboard["build"]
    assert api["command"] != dashboard["command"]


def test_the_image_copies_everything_the_build_backend_needs():
    """`pyproject.toml` names a readme, and hatchling refuses to install without it.

    The image never reads README.md. It still has to be in the build context, and the
    failure is a stack trace from inside a temporary build environment that says nothing
    about the Dockerfile. CI caught this one; this pins it.
    """
    for dockerfile in (DOCKERFILE, REPO_ROOT / "deploy" / "space" / "Dockerfile"):
        text = dockerfile.read_text()
        assert "README.md" in text, f"{dockerfile.name} never copies the declared readme"


def test_the_readme_is_not_excluded_from_the_build_context():
    ignored = (REPO_ROOT / ".dockerignore").read_text().splitlines()

    assert "README.md" not in ignored
    assert "*.md" not in ignored


def test_the_dependency_layer_does_not_depend_on_the_readme():
    """Editing prose must not re-resolve every dependency."""
    lines = DOCKERFILE.read_text().splitlines()
    first_sync = next(i for i, line in enumerate(lines) if "--no-install-project" in line)
    readme = next(i for i, line in enumerate(lines) if line.startswith("COPY README.md"))

    assert readme > first_sync
