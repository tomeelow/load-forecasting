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
import plotly.graph_objects as go
import streamlit as st

from src.config import REPO_ROOT, Config, load_config
from src.dashboard import data
from src.dashboard.state_sync import MirrorResult, mirror_state
from src.models.baselines import NAIVE_SEASONAL, TSO_FORECAST

TITLE = "Day-ahead load forecast — Polish bidding zone"
ADR_009 = (
    "https://github.com/tomeelow/load-forecasting/blob/main/docs/ADR.md"
    "#adr-009--drift-is-measured-against-the-same-weeks-in-previous-years"
)

TEMPLATE = "plotly_white"
MODEL_COLOUR = "#2563eb"
PSE_COLOUR = "#f97316"
ACTUAL_COLOUR = "#111827"
BAND_COLOUR = "rgba(37, 99, 235, 0.18)"


@st.cache_resource(ttl=3600)
def _mirror() -> MirrorResult:
    """Pull the pipeline's state once an hour where there is no pipeline (see state_sync).

    A resource rather than data, because it writes to disk and every loader below reads
    what it wrote — running it twice concurrently would have them read a half-copied tree.
    Local runs return immediately with `enabled=False`.
    """
    return mirror_state(REPO_ROOT)


@st.cache_data(ttl=300)
def _cached(loader: str, _mirrored_at=None, **kwargs):
    """Re-read artifacts every five minutes; the loop writes them at most once a day.

    `_mirrored_at` is part of the cache key and nothing else: a fresh mirror must not be
    read through a cache populated from the previous snapshot.
    """
    cfg = load_config()
    return getattr(data, loader)(cfg, **kwargs)


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
            "serve fresh forecasts, because that needs a live weather call and a model in "
            "memory; run the stack locally with `docker compose up` for that.",
            icon="ℹ️",
        )

    card = _cached("load_model_card")
    _synthetic_warning(card)

    forecast_panel(cfg, card)
    st.divider()
    benchmark_panel(cfg)
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

    figure = go.Figure()
    if not actuals.empty:
        local = actuals.index.tz_convert(cfg.data.timezone_local)
        figure.add_trace(
            go.Scatter(
                x=local,
                y=actuals[data.ACTUAL_COLUMN],
                name="Actual load",
                line={"color": ACTUAL_COLOUR, "width": 2},
            )
        )
        if data.TSO_COLUMN in actuals.columns:
            figure.add_trace(
                go.Scatter(
                    x=local,
                    y=actuals[data.TSO_COLUMN],
                    name="PSE day-ahead",
                    line={"color": PSE_COLOUR, "width": 1.5, "dash": "dot"},
                )
            )

    if served.empty:
        st.caption(
            "No served forecast logged yet — the chart shows actuals and PSE's forecast "
            "only. The daily loop's `forecast` step fills this in."
        )
    else:
        _add_forecast_band(figure, served, cfg)

    figure.update_layout(
        template=TEMPLATE,
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="MW",
        xaxis_title=f"local time ({cfg.data.timezone_local})",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    st.plotly_chart(figure, use_container_width=True)

    if not served.empty and card.available:
        st.caption(
            f"Forecast served by `{card.name}` v{card.version} at a "
            f"{card.horizon}-hour horizon. The band is the model's own P10–P90 quantile "
            "forecast, not a confidence interval around the point estimate."
        )


def _add_forecast_band(figure: go.Figure, served: pd.DataFrame, cfg: Config) -> None:
    """The P10–P90 band, drawn as a filled region under the point forecast."""
    local = served.index.tz_convert(cfg.data.timezone_local)
    has_band = served[["p10", "p90"]].notna().all(axis=None)

    if has_band:
        figure.add_trace(
            go.Scatter(x=local, y=served["p90"], line={"width": 0}, showlegend=False, name="P90")
        )
        figure.add_trace(
            go.Scatter(
                x=local,
                y=served["p10"],
                fill="tonexty",
                fillcolor=BAND_COLOUR,
                line={"width": 0},
                name="P10–P90",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=local,
            y=served["load_mw"],
            name="Model forecast",
            line={"color": MODEL_COLOUR, "width": 2.5},
        )
    )


# --------------------------------------------------------------------------------------
# 2. Model error vs PSE error — backtest, then served
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

    rolling = backtest.rolling_mape(window_days=30)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rolling.index.tz_convert(cfg.data.timezone_local),
            y=rolling["model"],
            name="This model",
            line={"color": MODEL_COLOUR, "width": 2.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rolling.index.tz_convert(cfg.data.timezone_local),
            y=rolling["PSE"],
            name="PSE day-ahead",
            line={"color": PSE_COLOUR, "width": 2.5},
        )
    )
    figure.update_layout(
        template=TEMPLATE,
        height=320,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="30-day rolling MAPE (%)",
        xaxis_title="",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.15, "x": 0},
    )
    st.plotly_chart(figure, use_container_width=True)

    overall = backtest.overall()
    model, pse = overall.loc[backtest.model_column], overall.loc[TSO_FORECAST]
    gap = model["mape"] - pse["mape"]
    columns = st.columns(3)
    columns[0].metric("Model MAPE", f"{model['mape']:.2f}%")
    columns[1].metric("PSE MAPE", f"{pse['mape']:.2f}%")
    columns[2].metric("Gap", f"{gap:+.2f} pp", delta=f"{-gap:+.2f} pp", delta_color="normal")

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
# 3. Drift
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

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["drift_share"],
            name="Drifted share of inputs",
            mode="lines+markers",
            line={"color": MODEL_COLOUR, "width": 2},
        )
    )
    figure.add_hline(
        y=threshold,
        line={"color": PSE_COLOUR, "dash": "dash"},
        annotation_text=f"retrain trigger ({threshold:g})",
        annotation_position="top left",
    )
    figure.update_layout(
        template=TEMPLATE,
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="share of monitored inputs drifting",
        yaxis_range=[0, 1.05],
        xaxis_title="",
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)

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
        st.caption(f"Latest Evidently report: `{report}`")


# --------------------------------------------------------------------------------------
# 4. Feature importance
# --------------------------------------------------------------------------------------


def importance_panel(card: data.ModelCard) -> None:
    st.subheader("What the champion leans on")

    if card.importance.empty:
        st.info("The serving run logged no feature-importance artifact.")
        return

    top = card.importance.nlargest(12, "gain").iloc[::-1]
    figure = go.Figure(
        go.Bar(x=top["gain"], y=top["feature"], orientation="h", marker_color=MODEL_COLOUR)
    )
    figure.update_layout(
        template=TEMPLATE,
        height=380,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="total gain",
        yaxis_title="",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Gain, as logged by the training run that produced the serving model. The weekly "
        "lag dominating is the signature of a strongly weekly-periodic series."
    )


# --------------------------------------------------------------------------------------
# 5. Model card
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
