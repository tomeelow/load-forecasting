"""Open-Meteo parsing and the population-weighted national combination."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import requests

from src.config import City, RequestConfig
from src.ingestion import weather_client
from src.ingestion.weather_client import national_weather, parse_hourly

VARIABLES = ("temperature_2m", "wind_speed_10m", "cloud_cover", "relative_humidity_2m")


def payload(times: list[str], **series) -> dict:
    return {"utc_offset_seconds": 0, "hourly": {"time": times, **series}}


def test_parses_hourly_response_into_utc_frame_with_our_column_names():
    response = payload(
        ["2024-05-01T00:00", "2024-05-01T01:00"],
        temperature_2m=[11.5, 12.0],
        wind_speed_10m=[3.1, 3.4],
        cloud_cover=[40, 55],
        relative_humidity_2m=[80, 78],
    )

    frame = parse_hourly(response, VARIABLES)

    assert list(frame.columns) == ["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2024-05-01 00:00", tz="UTC")
    assert frame["temp_c"].tolist() == [11.5, 12.0]


def test_non_utc_response_is_rejected():
    response = payload(["2024-05-01T02:00"], temperature_2m=[11.5])
    response["utc_offset_seconds"] = 7200

    with pytest.raises(ValueError, match="UTC"):
        parse_hourly(response, ("temperature_2m",))


def test_missing_variable_is_rejected():
    response = payload(["2024-05-01T00:00"], temperature_2m=[11.5])

    with pytest.raises(ValueError, match="missing requested variables"):
        parse_hourly(response, ("temperature_2m", "wind_speed_10m"))


def cities() -> tuple[City, ...]:
    return (
        City("Warsaw", 52.23, 21.01, 0.6),
        City("Gdansk", 54.35, 18.65, 0.4),
    )


def city_frame(temp: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-05-01", periods=len(temp), freq="1h", tz="UTC")
    return pd.DataFrame({"temp_c": temp, "wind_ms": [2.0] * len(temp)}, index=index)


def test_national_series_is_population_weighted_not_a_plain_mean():
    frames = {"Warsaw": city_frame([10.0, 10.0]), "Gdansk": city_frame([0.0, 0.0])}

    national = national_weather(frames, cities())

    assert national["temp_c"].tolist() == [6.0, 6.0]  # 0.6 * 10, not the 5.0 plain mean


def test_a_city_with_a_gap_renormalises_rather_than_punching_a_hole():
    warsaw = city_frame([10.0, 10.0])
    gdansk = city_frame([0.0, np.nan])

    national = national_weather({"Warsaw": warsaw, "Gdansk": gdansk}, cities())

    assert national["temp_c"].iloc[0] == pytest.approx(6.0)
    assert national["temp_c"].iloc[1] == pytest.approx(10.0)  # Warsaw's weight renormalised to 1


def test_the_weighted_mean_cannot_drift_outside_the_values_it_averages():
    """Seven cities all at 100% cloud must give 100%, not 100.00000000000001.

    Float division by renormalised weights overshoots by ~1e-14, which is invisible
    until a 0-100 range check rejects 4,369 hours of real data.
    """
    frames = {"Warsaw": city_frame([100.0, 0.0]), "Gdansk": city_frame([100.0, 0.0])}

    national = national_weather(frames, cities())

    assert national["temp_c"].tolist() == [100.0, 0.0]
    assert national["temp_c"].max() <= 100.0
    assert national["temp_c"].min() >= 0.0


def test_the_clip_does_not_distort_a_genuine_average():
    frames = {"Warsaw": city_frame([10.0]), "Gdansk": city_frame([0.0])}

    national = national_weather(frames, cities())

    assert national["temp_c"].iloc[0] == pytest.approx(6.0)  # 0.6 * 10, still weighted


def test_an_unweighted_city_is_rejected():
    frames = {"Warsaw": city_frame([10.0]), "Katowice": city_frame([5.0])}

    with pytest.raises(ValueError, match="No configured weight"):
        national_weather(frames, cities())


def test_configured_city_weights_are_normalised(cfg):
    assert sum(c.weight for c in cfg.weather.cities) == pytest.approx(1.0)
    assert {c.name for c in cfg.weather.cities} >= {"Warsaw", "Krakow", "Gdansk"}


# --- Retrying transient failures -------------------------------------------------
#
# Fourteen Open-Meteo calls happen on every scheduled run against a free service with
# no availability promise. The behaviour worth pinning down is not that a request
# succeeds — it is which failures are worth another attempt and which are not.


def http_response(status: int, body: dict | None = None, headers: dict | None = None):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body or {}).encode()
    response.headers.update(headers or {})
    return response


def policy(**overrides) -> RequestConfig:
    settings = {"timeout_s": 30.0, "max_attempts": 5, "backoff_s": 1.0, "backoff_max_s": 30.0}
    return RequestConfig(**{**settings, **overrides})


GOOD_BODY = {
    "utc_offset_seconds": 0,
    "hourly": {"time": ["2024-05-01T00:00"], "temperature_2m": [11.5]},
}


@pytest.fixture
def http(monkeypatch):
    """Replace the transport and the clock: record every call and every pause."""
    calls, sleeps = [], []

    def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(weather_client.time, "sleep", sleep)

    def install(*outcomes):
        remaining = list(outcomes)

        def get(url, params=None, timeout=None):
            calls.append({"url": url, "params": params, "timeout": timeout})
            outcome = remaining.pop(0) if remaining else remaining
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(weather_client.requests, "get", get)
        return calls, sleeps

    return install


def test_a_timeout_is_retried_rather_than_ending_the_run(http):
    calls, sleeps = http(requests.Timeout("read timed out"), http_response(200, GOOD_BODY))

    payload = weather_client._get("https://example/api", {"latitude": 52.23}, policy())

    assert payload == GOOD_BODY
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_a_dropped_connection_is_retried(http):
    calls, _ = http(requests.ConnectionError("connection reset"), http_response(200, GOOD_BODY))

    assert weather_client._get("https://example/api", {}, policy()) == GOOD_BODY
    assert len(calls) == 2


def test_a_server_error_is_retried(http):
    calls, _ = http(http_response(503), http_response(200, GOOD_BODY))

    assert weather_client._get("https://example/api", {}, policy()) == GOOD_BODY
    assert len(calls) == 2


def test_backoff_doubles_and_stops_at_the_ceiling(http):
    failures = [requests.Timeout("slow")] * 4
    _, sleeps = http(*failures, http_response(200, GOOD_BODY))

    weather_client._get("https://example/api", {}, policy(max_attempts=6, backoff_max_s=3.0))

    assert sleeps == [1.0, 2.0, 3.0, 3.0]  # doubling, then capped


def test_a_rate_limit_waits_as_long_as_the_server_asked(http):
    _, sleeps = http(
        http_response(429, headers={"Retry-After": "7"}), http_response(200, GOOD_BODY)
    )

    weather_client._get("https://example/api", {}, policy())

    assert sleeps == [7.0]  # the server's own number, not our backoff curve


def test_a_malformed_request_is_not_retried(http):
    """A 400 will be a 400 five times over; retrying only delays the error."""
    calls, sleeps = http(http_response(400), http_response(200, GOOD_BODY))

    with pytest.raises(requests.HTTPError):
        weather_client._get("https://example/api", {}, policy())

    assert len(calls) == 1
    assert sleeps == []


def test_giving_up_raises_the_last_failure_after_the_configured_attempts(http):
    calls, sleeps = http(*[requests.Timeout("slow")] * 3)

    with pytest.raises(requests.Timeout):
        weather_client._get("https://example/api", {}, policy(max_attempts=3))

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]  # no pause after the final attempt


def test_every_attempt_carries_the_configured_timeout(http):
    calls, _ = http(requests.Timeout("slow"), http_response(200, GOOD_BODY))

    weather_client._get("https://example/api", {}, policy(timeout_s=12.5))

    assert [c["timeout"] for c in calls] == [12.5, 12.5]


def test_one_flaky_city_does_not_lose_the_national_pull(monkeypatch, cfg):
    """The nightly loop makes seven of these calls; one hiccup must not end the run."""
    monkeypatch.setattr(weather_client.time, "sleep", lambda _: None)
    attempts = {}

    def get(url, params=None, timeout=None):
        city = params["latitude"]
        attempts[city] = attempts.get(city, 0) + 1
        if city == cfg.weather.cities[3].lat and attempts[city] == 1:
            raise requests.ConnectionError("connection reset by peer")
        return http_response(
            200,
            {
                "utc_offset_seconds": 0,
                "hourly": {
                    "time": ["2024-05-01T00:00", "2024-05-01T01:00"],
                    **{v: [10.0, 11.0] for v in cfg.weather.variables},
                },
            },
        )

    monkeypatch.setattr(weather_client.requests, "get", get)

    national = weather_client.fetch_national_archive(cfg.weather, "2024-05-01", "2024-05-01")

    assert len(national) == 2
    assert national["temp_c"].notna().all()
    assert attempts[cfg.weather.cities[3].lat] == 2


def test_the_configured_policy_actually_retries(cfg):
    """A max_attempts of 1 in config would make all of the above decorative."""
    assert cfg.weather.request.max_attempts > 1
    assert cfg.weather.request.timeout_s > 0
    assert cfg.weather.request.backoff_s > 0


def test_a_policy_that_could_never_retry_is_rejected():
    with pytest.raises(ValueError, match="max_attempts"):
        policy(max_attempts=0)


def test_a_backoff_ceiling_below_the_first_pause_is_rejected():
    with pytest.raises(ValueError, match="backoff"):
        policy(backoff_s=10.0, backoff_max_s=1.0)


# --- Chunking the archive pull ---------------------------------------------------
#
# A rebuild asks for the whole configured history. As one request per city that is a
# response the server takes minutes to begin, and every retry starts it again — which
# is how the first CI rebuild burned five attempts on a read timeout.


def test_a_long_range_is_split_into_consecutive_closed_windows():
    chunks = weather_client.date_chunks("2019-01-01", "2021-06-30", days=365)

    assert chunks[0][0] == "2019-01-01"
    assert chunks[-1][1] == "2021-06-30"
    # Consecutive and non-overlapping: each window starts the day after the last ended.
    for (_, ends), (starts, _) in zip(chunks, chunks[1:], strict=False):
        assert pd.Timestamp(starts) - pd.Timestamp(ends) == pd.Timedelta(days=1)


def test_a_range_inside_one_window_is_a_single_request():
    assert weather_client.date_chunks("2024-01-01", "2024-01-31") == [("2024-01-01", "2024-01-31")]


def test_a_backwards_range_asks_for_nothing():
    assert weather_client.date_chunks("2024-02-01", "2024-01-01") == []


def test_the_archive_pull_asks_year_by_year_and_returns_one_series(monkeypatch, cfg):
    """Seven years in one request is what timed out; seven requests of a year do not."""
    monkeypatch.setattr(weather_client.time, "sleep", lambda _: None)
    asked = []

    def get(url, params=None, timeout=None):
        asked.append((params["start_date"], params["end_date"]))
        index = pd.date_range(
            params["start_date"], f"{params['end_date']} 23:00", freq="1h", tz="UTC"
        )
        return http_response(
            200,
            {
                "utc_offset_seconds": 0,
                "hourly": {
                    "time": [t.strftime("%Y-%m-%dT%H:%M") for t in index],
                    **{v: [10.0] * len(index) for v in cfg.weather.variables},
                },
            },
        )

    monkeypatch.setattr(weather_client.requests, "get", get)

    frame = weather_client.fetch_city_archive(
        cfg.weather.cities[0], "2019-01-01", "2021-01-01", cfg.weather
    )

    assert len(asked) == 3, f"expected a request per year, got {asked}"
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    # The joins between windows are ordinary hours, not gaps.
    assert frame.index.to_series().diff().dropna().max() == pd.Timedelta(hours=1)


def test_one_slow_year_does_not_restart_the_others(monkeypatch, cfg):
    """The point of chunking: a timeout costs one window, not the whole history."""
    monkeypatch.setattr(weather_client.time, "sleep", lambda _: None)
    attempts = {}

    def get(url, params=None, timeout=None):
        window = params["start_date"]
        attempts[window] = attempts.get(window, 0) + 1
        if window == "2020-01-01" and attempts[window] == 1:
            raise requests.Timeout("archive was slow")
        index = pd.date_range(
            params["start_date"], f"{params['end_date']} 23:00", freq="1h", tz="UTC"
        )
        return http_response(
            200,
            {
                "utc_offset_seconds": 0,
                "hourly": {
                    "time": [t.strftime("%Y-%m-%dT%H:%M") for t in index],
                    **{v: [10.0] * len(index) for v in cfg.weather.variables},
                },
            },
        )

    monkeypatch.setattr(weather_client.requests, "get", get)

    weather_client.fetch_city_archive(
        cfg.weather.cities[0], "2019-01-01", "2021-01-01", cfg.weather
    )

    assert attempts["2020-01-01"] == 2  # retried
    assert attempts["2019-01-01"] == 1  # and the year before it was not re-fetched


class TestDayAheadArchive:
    """The forecast archive used to measure the train-serve weather skew.

    It is not on the ingestion path and never reaches the served dataset, but it decides
    a headline number, so what it asks for and what it returns both matter.
    """

    @staticmethod
    def responder(seen: list[dict]):
        def get(url, params=None, timeout=None):
            seen.append({"url": url, **(params or {})})
            requested = params["hourly"].split(",")
            return http_response(
                200,
                {
                    "utc_offset_seconds": 0,
                    "hourly": {
                        "time": ["2024-05-01T00:00", "2024-05-01T01:00"],
                        **{v: [7.5, 8.5] for v in requested},
                    },
                },
            )

        return get

    def test_it_asks_the_forecast_archive_not_the_observation_archive(self, monkeypatch, cfg):
        """Asking the archive endpoint would silently measure nothing at all."""
        seen = []
        monkeypatch.setattr(weather_client.requests, "get", self.responder(seen))

        weather_client.fetch_national_day_ahead(cfg.weather, "2024-05-01", "2024-05-01")

        assert {call["url"] for call in seen} == {cfg.weather.historical_forecast_url}

    def test_it_requests_the_previous_day_vintage_of_every_variable(self, monkeypatch, cfg):
        seen = []
        monkeypatch.setattr(weather_client.requests, "get", self.responder(seen))

        weather_client.fetch_national_day_ahead(cfg.weather, "2024-05-01", "2024-05-01")

        requested = seen[0]["hourly"].split(",")
        assert requested == [f"{v}_previous_day1" for v in cfg.weather.variables]

    def test_the_suffix_is_stripped_back_to_our_column_names(self, monkeypatch, cfg):
        """Otherwise the frame cannot be swapped into the dataset it is measured against."""
        monkeypatch.setattr(weather_client.requests, "get", self.responder([]))

        national = weather_client.fetch_national_day_ahead(cfg.weather, "2024-05-01", "2024-05-01")

        assert set(national.columns) == {"temp_c", "wind_ms", "cloud_cover", "humidity_pct"}

    def test_a_lead_of_less_than_a_day_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            weather_client._lead_suffix(0)

    def test_a_longer_lead_changes_the_vintage_requested(self, monkeypatch, cfg):
        import dataclasses

        seen = []
        monkeypatch.setattr(weather_client.requests, "get", self.responder(seen))
        two_days = dataclasses.replace(cfg.weather, historical_forecast_lead_days=2)

        weather_client.fetch_national_day_ahead(two_days, "2024-05-01", "2024-05-01")

        assert all(v.endswith("_previous_day2") for v in seen[0]["hourly"].split(","))

    def test_an_unconfigured_endpoint_is_refused_rather_than_guessed(self, cfg):
        import dataclasses

        without = dataclasses.replace(cfg.weather, historical_forecast_url="")

        with pytest.raises(ValueError, match="historical_forecast_url"):
            weather_client.fetch_city_day_ahead(
                cfg.weather.cities[0], "2024-05-01", "2024-05-01", without
            )
