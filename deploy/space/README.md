---
title: Polish Day-Ahead Load Forecast
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Day-ahead electricity load forecasting — Polish bidding zone

Hourly day-ahead forecasts of total Polish electricity demand, benchmarked against
**PSE's own published day-ahead forecast** rather than against a naive baseline alone.

This Space is a **hosted mirror**, not a live service. It shows what the daily GitHub
Actions loop recorded — the ingested dataset, the MLflow registry, the served-prediction
log and the drift history — refreshed hourly from the project's public `pipeline-state`
branch. It does not serve fresh forecasts; that needs a live weather call and a model in
memory, which is what `docker compose up` gives you locally.

Source, architecture and the evaluation audit: https://github.com/tomeelow/load-forecasting
