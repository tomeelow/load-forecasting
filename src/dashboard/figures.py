"""The charts, built once and used by both the dashboard and the README.

    uv run python -m src.dashboard.figures      # writes docs/images/

Keeping the figures here rather than inline in `app.py` is what makes the image in the
README *the* chart rather than a lookalike drawn for the documentation. A screenshot that
diverges from the running page is a small lie that nobody notices until an interview.

It also makes the charts renderable without a Streamlit runtime, which is how they get
exported at all: Streamlit renders through a websocket, so a headless browser pointed at
the page captures an empty shell.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
MODEL_COLOUR = "#2563eb"
PSE_COLOUR = "#f97316"
ACTUAL_COLOUR = "#111827"
BAND_COLOUR = "rgba(37, 99, 235, 0.18)"

ACTUAL_COLUMN = "load_mw"
TSO_COLUMN = "tso_forecast_mw"

IMAGES_DIR = Path("docs/images")


def forecast_figure(actuals: pd.DataFrame, served: pd.DataFrame, tz: str) -> go.Figure:
    """Recent actuals, PSE's published forecast, and the model's forecast with its band."""
    figure = go.Figure()

    if not actuals.empty:
        local = actuals.index.tz_convert(tz)
        figure.add_trace(
            go.Scatter(
                x=local,
                y=actuals[ACTUAL_COLUMN],
                name="Actual load",
                line={"color": ACTUAL_COLOUR, "width": 2},
            )
        )
        if TSO_COLUMN in actuals.columns:
            figure.add_trace(
                go.Scatter(
                    x=local,
                    y=actuals[TSO_COLUMN],
                    name="PSE day-ahead",
                    line={"color": PSE_COLOUR, "width": 1.5, "dash": "dot"},
                )
            )

    if not served.empty:
        local = served.index.tz_convert(tz)
        if served[["p10", "p90"]].notna().all(axis=None):
            figure.add_trace(
                go.Scatter(
                    x=local, y=served["p90"], line={"width": 0}, showlegend=False, name="P90"
                )
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

    figure.update_layout(
        template=TEMPLATE,
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title="MW",
        xaxis_title=f"local time ({tz})",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return figure


def rolling_error_figure(rolling: pd.DataFrame, tz: str) -> go.Figure:
    """The model's rolling error against PSE's, on one axis and the same hours."""
    local = rolling.index.tz_convert(tz)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=local,
            y=rolling["model"],
            name="This model",
            line={"color": MODEL_COLOUR, "width": 2.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=local,
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
    return figure


def drift_figure(history: pd.DataFrame, threshold: float) -> go.Figure:
    """The drifted share over time, with the trigger drawn on it — a trend, not a lamp."""
    figure = go.Figure()
    # A new deployment has one or two checks, and Plotly autoscales a single timestamp
    # down to milliseconds — an axis reading 09:07:48.0665 that looks like a bug rather
    # than like a series with one point in it. Give it a day either side to sit in.
    if len(history) < 3 and len(history):
        pad = pd.Timedelta(days=1)
        figure.update_xaxes(range=[history.index.min() - pad, history.index.max() + pad])
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
    return figure


def importance_figure(importance: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """What the champion leans on, by total gain."""
    top = importance.nlargest(top_n, "gain").iloc[::-1]
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
    return figure


def main() -> int:
    """Write the README images from whatever the pipelines have actually produced."""
    from loguru import logger

    from src.config import load_config
    from src.dashboard import data

    cfg = load_config()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    forecast = forecast_figure(
        data.load_recent_actuals(cfg, days=10),
        data.load_served_forecast(cfg, days=3),
        cfg.data.timezone_local,
    )
    forecast.write_image(IMAGES_DIR / "forecast_vs_actual.png", width=1400, height=520, scale=2)
    written.append("forecast_vs_actual.png")

    backtest = data.load_backtest(cfg, cfg.model.horizons[0])
    if backtest.available:
        rolling_error_figure(backtest.rolling_mape(30), cfg.data.timezone_local).write_image(
            IMAGES_DIR / "rolling_error_vs_pse.png", width=1400, height=420, scale=2
        )
        written.append("rolling_error_vs_pse.png")
    else:
        logger.warning("No backtest predictions; skipping the rolling-error image")

    logger.info("Wrote {} to {}", ", ".join(written), IMAGES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
