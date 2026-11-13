"""The dashboard: what the system currently believes, and what it is entitled to claim.

    uv run streamlit run src/dashboard/app.py

Every figure here is read from an artifact some pipeline wrote — the MLflow registry, the
prediction log, the drift history, the ingested dataset, the backtest reports. Nothing is
hardcoded, and where evidence is missing the page says which evidence is missing rather
than leaving a blank panel or filling it from somewhere else.

Two rules the layout exists to enforce:

* **Backtest evidence and served evidence are never mixed.** The backtest has a year of
  out-of-sample hours; the prediction log has however many days this deployment has been
  running. Both are honest; averaging them together is not, and neither is quietly
  showing the fuller one under a heading about production.
* **Drift is a trend, not a lamp.** Input drift on this target fires most nights for
  reasons ADR-009 sets out, so a red/green indicator would sit permanently red and read
  as neglect. The share over time, next to the served error, is what actually informs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import REPO_ROOT, Config, load_config
from src.dashboard import data
from src.dashboard.figures import (
    drift_figure,
    forecast_figure,
    hourly_error_figure,
    importance_figure,
    rolling_error_figure,
    scoreboard_figure,
    what_if_figure,
)
from src.dashboard.state_sync import MirrorResult, mirror_state
from src.evaluation.metrics import segment_labels
from src.models.baselines import NAIVE_SEASONAL, TSO_FORECAST

TITLE = "Day-ahead load forecast — Polish bidding zone"
ADR_009 = (
    "https://github.com/tomeelow/load-forecasting/blob/main/docs/ADR.md"
    "#adr-009--drift-is-measured-against-the-same-weeks-in-previous-years"
)


@st.cache_resource(ttl=3600)
def _mirror() -> MirrorResult:
    """Pull the pipeline's state once an hour where there is no pipeline (see state_sync).

    A resource rather than data, because it writes to disk and every loader below reads
    what it wrote — running it twice concurrently would have them read a half-copied tree.
    Local runs return immediately with `enabled=False`.
    """
    return mirror_state(REPO_ROOT)


# How far the what-if slider moves the forecast temperature, either way.
WHAT_IF_RANGE_C = 8.0


@st.cache_resource(ttl=3600)
def _what_if(snapshot: str | None) -> data.WhatIf:
    """The champion's booster, loaded once per snapshot — a resource, not data.

    `st.cache_data` would pickle tens of megabytes of trees on every rerun, and a slider
    reruns the script on every move.
    """
    return data.load_what_if(load_config())


@st.cache_data(ttl=300)
def _load(loader: str, snapshot: str | None, **kwargs):
    """Re-read artifacts every five minutes; the loop writes them at most once a day."""
    cfg = load_config()
    return getattr(data, loader)(cfg, **kwargs)


def _cached(loader: str, **kwargs):
    """Read an artifact through the cache, keyed on which snapshot produced it.

    The mirrored commit is part of the key so a fresh mirror is never read through a
    cache populated from the previous snapshot. It is a plain argument rather than a
    leading-underscore one because Streamlit *excludes* underscore-prefixed arguments
    from the cache key — which would have made this do exactly nothing. Locally the
    commit is `None` and constant, so the key collapses to the loader and its arguments.
    """
    return _load(loader, _mirror().commit, **kwargs)


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="⚡", layout="wide")
    cfg = load_config()

    mirror = _mirror()
    st.title(TITLE)
    st.caption(
        "Hourly day-ahead forecasts of total Polish electricity demand, benchmarked "
        "against PSE's own published forecast. Every panel below is read from the "
        "artifacts the daily pipeline writes. " + mirror.describe()
    )
    if mirror.enabled:
        st.info(
            "This is a **hosted mirror**, not a live service. It shows what the daily "
            "GitHub Actions loop recorded — the dataset, the registry, the prediction log "
            "and the drift history — refreshed hourly from the state branch. It does not "
            "serve fresh forecasts: the what-if below re-runs the champion on the weather "
            "the loop already ingested, and nothing it draws is logged. Run the stack "
            "locally with `docker compose up` for the live service.",
            icon="ℹ️",
        )

    card = _cached("load_model_card")
    _synthetic_warning(card)

    forecast_panel(cfg, card)
    st.divider()
    what_if_panel(cfg)
    st.divider()
    benchmark_panel(cfg)
    st.divider()
    explorer_panel(cfg)
    st.divider()
    scoreboard_panel(cfg)
    st.divider()
    drift_panel(cfg)
    st.divider()
    importance_panel(card)
    st.divider()
    model_card_panel(card)


def _synthetic_warning(card: data.ModelCard) -> None:
    """Unmissable when the model on show was never trained on real demand."""
    if card.synthetic:
        st.error(
            "**The serving model was trained on SYNTHETIC data.** No ENTSO-E token was "
            "available when it was trained, so the load series, the weather and the "
            "'PSE' benchmark are all generated. Nothing on this page says anything about "
            "Polish demand until a model trained on ingested data is promoted.",
            icon="⚠️",
        )


# --------------------------------------------------------------------------------------
# 1. Forecast vs actual
# --------------------------------------------------------------------------------------


def forecast_panel(cfg: Config, card: data.ModelCard) -> None:
    st.subheader("Forecast vs actual")

    actuals = _cached("load_recent_actuals", days=10)
    served = _cached("load_served_forecast", days=3)

    if actuals.empty and served.empty:
        st.info(
            "No ingested actuals and no served forecasts yet. Run `python -m "
            "pipelines.ingest` and then `python -m pipelines.forecast`."
        )
        return

    if served.empty:
        st.caption(
            "No served forecast logged yet — the chart shows actuals and PSE's forecast "
            "only. The daily loop's `forecast` step fills this in."
        )

    st.plotly_chart(
        forecast_figure(actuals, served, cfg.data.timezone_local), use_container_width=True
    )

    if not served.empty and card.available:
        st.caption(
            f"Forecast served by `{card.name}` v{card.version} at a "
            f"{card.horizon}-hour horizon. The band is the model's own P10–P90 quantile "
            "forecast, not a confidence interval around the point estimate."
        )


# --------------------------------------------------------------------------------------
# 2. What-if: the same forecast under shifted weather
# --------------------------------------------------------------------------------------


def what_if_panel(cfg: Config) -> None:
    st.subheader("What if it were warmer or colder?")

    what_if = _what_if(_mirror().commit)
    if not what_if.available:
        st.info(f"The champion is not loaded here, so there is nothing to re-run. {what_if.error}")
        return

    baseline = what_if.forecast(cfg)
    if baseline.empty:
        st.info("The recorded history cannot support a forecast yet.")
        return

    delta = st.slider(
        "Shift the forecast temperature (°C)",
        -WHAT_IF_RANGE_C,
        WHAT_IF_RANGE_C,
        0.0,
        step=0.5,
        format="%+.1f",
    )
    shifted = what_if.forecast(cfg, delta)
    change = shifted[data.ACTUAL_COLUMN] - baseline[data.ACTUAL_COLUMN]

    st.plotly_chart(
        what_if_figure(baseline, shifted, delta, cfg.data.timezone_local),
        use_container_width=True,
    )
    columns = st.columns(3)
    columns[0].metric(
        "Mean temperature",
        f"{shifted['temp_c'].mean():.1f} °C",
        delta=f"{delta:+.1f} °C" if delta else None,
        delta_color="off",
    )
    columns[1].metric("Mean change in demand", f"{change.mean():+,.0f} MW")
    columns[2].metric("Largest hourly change", f"{change.loc[change.abs().idxmax()]:+,.0f} MW")
    st.caption(
        f"The champion re-run over the {len(baseline)} hours the loop last forecast, with "
        "the forecast temperature moved and everything else held — through the same "
        "feature builder training and serving use. This is the model's response, not "
        "physics: it has seen few days far from the season's norm, so the ends of the "
        "slider are extrapolation. Nothing here is a served prediction and none of it is "
        "logged."
    )


# --------------------------------------------------------------------------------------
# 3. Model error vs PSE error — backtest, then served
# --------------------------------------------------------------------------------------


def benchmark_panel(cfg: Config) -> None:
    st.subheader("Model error vs PSE error")

    backtest = _cached("load_backtest", horizon=cfg.model.horizons[0])
    left, right = st.columns([2, 1])

    with left:
        st.markdown("**Backtest evidence** — a full year of out-of-sample hours")
        _backtest_error(cfg, backtest)

    with right:
        st.markdown("**Served evidence** — what this deployment has actually logged")
        _served_error(cfg)


def _backtest_error(cfg: Config, backtest: data.BacktestEvidence) -> None:
    if not backtest.available:
        st.info(
            "No backtest predictions on disk. Run `python -m pipelines.audit` (or "
            "`python -m pipelines.backtest`) to produce them."
        )
        return

    st.plotly_chart(
        rolling_error_figure(backtest.rolling_mape(window_days=30), cfg.data.timezone_local),
        use_container_width=True,
    )

    overall = backtest.overall()
    model, pse = overall.loc[backtest.model_column], overall.loc[TSO_FORECAST]
    gap = model["mape"] - pse["mape"]
    columns = st.columns(3)
    columns[0].metric("Model MAPE", f"{model['mape']:.2f}%")
    columns[1].metric("PSE MAPE", f"{pse['mape']:.2f}%")
    # The delta carries the verdict rather than repeating the number with the sign
    # flipped, which is what it used to do — two figures saying one thing, in opposite
    # directions, next to each other.
    columns[2].metric(
        "Gap (model − PSE)",
        f"{gap:+.2f} pp",
        delta="model behind" if gap > 0 else "model ahead",
        delta_color="inverse" if gap > 0 else "normal",
    )

    naive = (
        f", naive seasonal {overall.loc[NAIVE_SEASONAL, 'mape']:.2f}%"
        if NAIVE_SEASONAL in overall.index
        else ""
    )
    st.caption(
        f"From the **{backtest.label}** — {len(backtest.predictions):,} out-of-sample "
        f"hours, {backtest.predictions.index.min():%Y-%m-%d} to "
        f"{backtest.predictions.index.max():%Y-%m-%d}. Model and PSE are scored on "
        f"exactly the same hours{naive}. These are backtest numbers, not production "
        "numbers — the panel on the right is production."
    )


def _served_error(cfg: Config) -> None:
    performance = _cached("load_served_performance")

    if performance.status == "empty":
        st.warning(
            f"**Accumulating — 0 predictions scored, "
            f"{performance.required} needed.**\n\n"
            f"{performance.logged} forecast(s) logged, none scored yet: a prediction can "
            "only be scored once ENTSO-E publishes the actual for its hour.",
            icon="⏳",
        )
        _served_note(cfg, performance)
        return

    if not performance.sufficient:
        st.warning(
            f"**Accumulating — {len(performance.scored)} predictions scored, "
            f"{performance.required} needed.**\n\n"
            "Production error is not reported below that threshold. A MAPE over a "
            "handful of hours is noise wearing a number's clothes.",
            icon="⏳",
        )
        _served_note(cfg, performance)
        return

    model_mape, tso_mape = performance.model_mape, performance.tso_mape
    columns = st.columns(2)
    columns[0].metric("Served MAPE", f"{model_mape:.2f}%")
    if tso_mape is not None:
        columns[1].metric("PSE, same hours", f"{tso_mape:.2f}%")
    st.caption(
        f"{len(performance.scored):,} served predictions scored against published "
        f"actuals, {performance.scored.index.min():%Y-%m-%d} to "
        f"{performance.scored.index.max():%Y-%m-%d}."
    )


def _served_note(cfg: Config, performance: data.ServedPerformance) -> None:
    st.caption(
        f"The daily loop serves roughly {cfg.model.horizons[0]} hours per run and scores "
        "them the following day, so this panel fills up over about a week of operation. "
        "It is kept separate from the backtest deliberately: padding it with backtest "
        "predictions would make the number larger and meaningless."
    )


# --------------------------------------------------------------------------------------
# 4. Backtest explorer
# --------------------------------------------------------------------------------------

# The explorer's filters: a `segment_labels` column and what the page calls it.
EXPLORER_FILTERS = {
    "season": "Season",
    "daytype": "Day type",
    "period": "Time of day",
    "holiday": "Holidays",
}


def explorer_panel(cfg: Config) -> None:
    st.subheader("Cut the backtest yourself")

    backtest = _cached("load_backtest", horizon=cfg.model.horizons[0])
    if not backtest.available:
        st.info("No backtest predictions on disk, so there is nothing to cut.")
        return

    tz, peak_hours = cfg.data.timezone_local, cfg.evaluation.peak_hours
    labels = segment_labels(backtest.predictions.index, tz=tz, peak_hours=peak_hours)
    local = backtest.predictions.index.tz_convert(tz)
    first, last = local.min().date(), local.max().date()

    start, end = st.slider("Dates", first, last, (first, last), format="YYYY-MM-DD")
    chosen = {
        kind: picked
        for column, (kind, label) in zip(
            st.columns(len(EXPLORER_FILTERS)), EXPLORER_FILTERS.items(), strict=True
        )
        if (picked := column.multiselect(label, sorted(labels[kind].unique()), placeholder="All"))
    }

    selection = backtest.select(chosen, tz=tz, peak_hours=peak_hours, start=start, end=end)
    if selection.predictions.empty:
        st.info("No backtest hours match that selection.")
        return

    overall = selection.overall()
    model, pse = overall.loc[selection.model_column], overall.loc[TSO_FORECAST]
    gap = model["mape"] - pse["mape"]
    columns = st.columns(4)
    columns[0].metric("Hours selected", f"{len(selection.predictions):,}")
    columns[1].metric("Model MAPE", f"{model['mape']:.2f}%")
    columns[2].metric("PSE MAPE", f"{pse['mape']:.2f}%")
    columns[3].metric(
        "Gap (model − PSE)",
        f"{gap:+.2f} pp",
        delta="model behind" if gap > 0 else "model ahead",
        delta_color="inverse" if gap > 0 else "normal",
    )

    left, right = st.columns([2, 1])
    left.plotly_chart(hourly_error_figure(selection.hourly_mape(tz)), use_container_width=True)
    right.dataframe(
        overall.rename(
            index={
                selection.model_column: "This model",
                TSO_FORECAST: "PSE",
                NAIVE_SEASONAL: "Naive",
            },
            columns={
                "mape": "MAPE %",
                "rmse_mw": "RMSE MW",
                "mae_mw": "MAE MW",
                "bias_mw": "Bias MW",
            },
        )[["MAPE %", "RMSE MW", "MAE MW", "Bias MW"]]
        .style.format("{:.2f}", subset=["MAPE %"])
        .format("{:+.0f}", subset=["Bias MW"])
        .format("{:.0f}", subset=["RMSE MW", "MAE MW"]),
        use_container_width=True,
    )
    st.caption(
        "Every figure is recomputed from the audited out-of-sample predictions for the "
        "hours selected, model and PSE on identical hours. A cut is a cut: the reportable "
        "number is the whole year above, and a selection of a few hundred hours can be "
        "made to say most things."
    )


# --------------------------------------------------------------------------------------
# 5. Daily scoreboard — served evidence, one day at a time
# --------------------------------------------------------------------------------------


def scoreboard_panel(cfg: Config) -> None:
    st.subheader("Daily scoreboard")

    daily = _cached("load_served_performance").daily(cfg.data.timezone_local)
    if daily.empty:
        st.info(
            "No complete day has been scored yet. A day appears here once ENTSO-E has "
            "published the actual for every hour of it."
        )
        return

    latest = daily.iloc[-1]
    won = int((daily["gap"] < 0).sum())
    columns = st.columns(3)
    columns[0].metric(f"Model, {daily.index[-1]:%Y-%m-%d}", f"{latest['model_mape']:.2f}%")
    columns[1].metric(
        "PSE, same day",
        f"{latest['pse_mape']:.2f}%",
        delta="model behind" if latest["gap"] > 0 else "model ahead",
        delta_color="inverse" if latest["gap"] > 0 else "normal",
    )
    columns[2].metric("Days the model beat PSE", f"{won} of {len(daily)}")

    st.plotly_chart(scoreboard_figure(daily), use_container_width=True)
    st.caption(
        "Served forecasts only, scored against published actuals — one bar per complete "
        "local day, below zero where the model's MAPE was lower than PSE's. One day is 24 "
        "hours and proves little either way; the tally is the part worth reading."
    )


# --------------------------------------------------------------------------------------
# 6. Drift
# --------------------------------------------------------------------------------------


def drift_panel(cfg: Config) -> None:
    st.subheader("Input drift")

    history = _cached("load_drift_history")
    threshold = cfg.monitoring.drift_share_threshold

    st.markdown(
        f"Input drift on this series fires most nights, and that is expected rather than "
        f"broken. The current window is a fortnight of one realised season judged against "
        f"a three-year seasonal reference — a narrow distribution against a wide one — so "
        f"the drifted share sits above the {threshold:g} threshold most of the time. It is "
        f"read as an **early warning**, not as a verdict: what protects production is the "
        f"promotion gate, which refuses a retrained candidate that is worse than the "
        f"model already serving. See [ADR-009]({ADR_009})."
    )

    if history.empty:
        st.info(
            "No drift checks recorded yet. `python -m pipelines.check_drift` appends one "
            "row per run to the drift history."
        )
        return

    st.plotly_chart(drift_figure(history, threshold), use_container_width=True)

    latest = history.iloc[-1]
    columns = st.columns(3)
    columns[0].metric("Latest share", f"{latest['drift_share']:.2f}")
    columns[1].metric(
        "Features drifting",
        f"{int(latest['drifted_features'])} of {int(latest['monitored_features'])}"
        if pd.notna(latest.get("monitored_features"))
        else str(int(latest["drifted_features"])),
    )
    columns[2].metric("Checks recorded", f"{len(history):,}")

    if isinstance(latest.get("drifted_names"), str) and latest["drifted_names"]:
        st.caption(f"Most recently drifting: `{latest['drifted_names'].replace(' ', '`, `')}`")

    report = _cached("latest_drift_report")
    if report is not None:
        # The name, not the path: on a hosted mirror the absolute path is a container
        # directory that means nothing to whoever opened the link.
        st.caption(f"Latest Evidently report on disk: `{report.name}`")


# --------------------------------------------------------------------------------------
# 7. Feature importance
# --------------------------------------------------------------------------------------


def importance_panel(card: data.ModelCard) -> None:
    st.subheader("What the champion leans on")

    if card.importance.empty:
        st.info("The serving run logged no feature-importance artifact.")
        return

    st.plotly_chart(importance_figure(card.importance), use_container_width=True)
    st.caption(
        "Gain, as logged by the training run that produced the serving model. The weekly "
        "lag dominating is the signature of a strongly weekly-periodic series."
    )


# --------------------------------------------------------------------------------------
# 8. Model card
# --------------------------------------------------------------------------------------


def model_card_panel(card: data.ModelCard) -> None:
    st.subheader("Model card")

    if not card.available:
        st.error(
            f"No champion is registered, so nothing is serving. {card.error or ''}",
            icon="🚫",
        )
        return

    columns = st.columns(4)
    columns[0].metric("Serving", f"{card.name} v{card.version}")
    columns[1].metric("Promoted", f"{card.promoted_at:%Y-%m-%d}" if card.promoted_at else "unknown")
    columns[2].metric("Horizon", f"{card.horizon}h" if card.horizon else "unknown")
    columns[3].metric("Training data", card.data_source)

    rows = {
        "MLflow run": card.run_id,
        "Dataset fingerprint": card.dataset_version,
        "Feature set": card.params.get("feature_set"),
        "Features": card.params.get("n_features"),
        "Boosting rounds used": card.params.get("best_iteration"),
    }
    st.markdown(
        "\n".join(
            ["| | |", "|---|---|"]
            + [f"| {name} | `{value}` |" for name, value in rows.items() if value]
        )
    )
    st.caption(
        "Holdout metrics recorded at promotion time: "
        + ", ".join(
            f"{name} {card.metrics[name]:.3f}"
            for name in ("mape", "tso_mape", "naive_mape", "rmse_mw")
            if name in card.metrics
        )
        + ". These come from the single 60-day block the gate looked at, not from the "
        "backtest above — see the benchmark panel for the reportable figure."
    )


if __name__ == "__main__":
    main()
