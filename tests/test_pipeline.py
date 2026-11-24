"""Unit tests for the core logic. They use small synthetic data, so they need no dataset and run in seconds."""
import numpy as np
import pandas as pd
import pytest

from cgmacros_loader import hba1c_group
from external_loaders import MMOL_TO_MGDL, to_grid
from forecast_baselines import GRID, HORIZONS, N_LAGS, fit_predict, native_grid, rows_from_series
from forecast_ladder import MEAL_COLS, context_features, shuffled
from forecast_personal import rmse

T0 = pd.Timestamp("2026-01-01 08:00:00")


# ---------------------------------------------------------------- data preparation
def test_native_grid_recovers_real_readings_from_interpolated_minutes():
    """The CGMacros minute file is linearly interpolated between real 15-minute readings; we must recover the real ones."""
    rng = np.random.default_rng(0)
    n = 3 * 96
    readings = 100 + 30 * np.sin(np.arange(n) / 7) + rng.normal(0, 5, n)
    minutes = np.arange(0, (n - 1) * 15 + 1)
    d = pd.DataFrame({"Timestamp": T0 + pd.to_timedelta(minutes, unit="m"),
                      "libre": np.interp(minutes, 15 * np.arange(n), readings)})
    g, t0 = native_grid(d, "libre")
    assert t0 == T0
    assert len(g) == n
    assert np.allclose(g.to_numpy(), readings)
    assert set(np.diff(g.index)) == {GRID}


def test_rows_from_series_lag_order_and_targets():
    g = pd.Series(np.arange(40, dtype=float), index=np.arange(40) * GRID)      # value k at minute 15k
    rows = rows_from_series(1, "healthy", g, T0)
    first = rows[0]
    assert first["lag0"] == 7 and first["lag7"] == 0                            # lag0 = now, lag7 = 105 minutes earlier
    assert first["tmin"] == 7 * GRID
    assert first["y30"] == 9 and first["y60"] == 11 and first["y120"] == 15
    assert "y120" not in rows[-1]                                               # no target beyond the end of the recording
    assert all(f"lag{k}" in first for k in range(N_LAGS))


def test_rows_from_series_drops_windows_with_a_gap():
    values = np.arange(40, dtype=float)
    values[10] = np.nan
    rows = rows_from_series(1, "healthy", pd.Series(values, index=np.arange(40) * GRID), T0)
    bad = {i * GRID for i in range(10, 10 + N_LAGS)}                            # windows that would contain the gap
    assert not bad & {r["tmin"] for r in rows}


def test_to_grid_aligns_irregular_5_minute_readings_to_15_minute_grid():
    n = 30
    ts = T0 + pd.to_timedelta(np.arange(n) * 5 * 60 + 17, unit="s")             # readings every 5 min, at odd seconds
    vals = pd.Series(np.arange(n, dtype=float) + 100)
    t0, g = to_grid(pd.Series(ts), vals, tol_min=2.5)
    assert t0 == ts[0]
    assert list(g.index[:3]) == [0, 15, 30]
    assert g.iloc[0] == 100 and g.iloc[1] == 103 and g.iloc[2] == 106            # every third real reading


def test_unit_conversion_and_hba1c_groups():
    assert MMOL_TO_MGDL * 5.5 == pytest.approx(99.09, abs=0.01)
    assert hba1c_group(5.6) == "healthy"
    assert hba1c_group(5.7) == "prediabetes" and hba1c_group(6.4) == "prediabetes"
    assert hba1c_group(6.5) == "t2d"


# ---------------------------------------------------------------- no information from the future
def _context_fixture():
    minutes = np.arange(0, 601)
    df = pd.DataFrame({"subject": 1, "Timestamp": T0 + pd.to_timedelta(minutes, unit="m"), "HR": 70.0, "METs": 10.0,
                       "cal_activity": 1.0, "meal_type": pd.Series([None] * len(minutes), dtype=object), "meal_kcal": np.nan,
                       "carbs": np.nan, "protein": np.nan, "fat": np.nan, "fiber": np.nan})
    df.loc[100, "meal_type"] = "Lunch"                                         # one lunch logged at minute 100
    df.loc[100, ["meal_kcal", "carbs", "protein", "fat", "fiber"]] = [500.0, 60.0, 20.0, 10.0, 5.0]
    S = pd.DataFrame({"subject": 1, "group": "healthy", "tmin": [90, 100, 130, 500]})
    return df, S


def test_meal_features_use_only_meals_logged_at_or_before_the_forecast_time():
    df, S = _context_fixture()
    out = context_features(df, S)
    before, at, after, long_after = (out.iloc[i] for i in range(4))
    assert before["meal_carbs"] == 0 and before["carbs_2h"] == 0 and before["mins_since_meal"] == 360   # meal is in the future
    assert at["meal_carbs"] == 60 and at["mins_since_meal"] == 0
    assert after["carbs_2h"] == 60 and after["mins_since_meal"] == 30
    assert long_after["meal_carbs"] == 0 and long_after["carbs_2h"] == 0                                # more than 6 h / 2 h ago


def test_shuffled_control_keeps_each_persons_values_but_breaks_their_order():
    rng = np.random.default_rng(1)
    S = pd.DataFrame({"subject": np.repeat([1, 2], 50), "carbs_2h": rng.normal(size=100),
                      "meal_kcal": rng.normal(size=100), "other": np.arange(100)})
    out = shuffled(S, ["carbs_2h", "meal_kcal"], seed=3)
    for sid in (1, 2):
        a, b = S[S.subject == sid], out[out.subject == sid]
        assert sorted(a.carbs_2h) == pytest.approx(sorted(b.carbs_2h))          # same values per person
        assert not np.allclose(a.carbs_2h.to_numpy(), b.carbs_2h.to_numpy())     # different order
    assert (out["other"] == S["other"]).all()                                    # other columns untouched
    assert set(MEAL_COLS) >= {"carbs_2h", "meal_kcal"}


# ---------------------------------------------------------------- baselines and metrics
def test_persistence_and_linear_trend_baselines():
    te = pd.DataFrame({"lag0": [100.0, 380.0], "lag2": [70.0, 300.0]})
    assert list(fit_predict("persistence", None, te, 60)) == [100.0, 380.0]
    pred = fit_predict("linear_trend", None, te, 60)
    assert pred[0] == pytest.approx(160.0)                                       # slope 1 mg/dL per minute, 60 minutes
    assert fit_predict("linear_trend", None, te, 120)[1] == 400.0                # clipped to the sensor range
    assert HORIZONS == [30, 60, 120]


def test_rmse():
    assert rmse(np.array([3.0, -4.0])) == pytest.approx(np.sqrt(12.5))


def test_grouped_folds_never_split_a_person():
    people = np.arange(45)
    for rep in range(3):
        folds = np.array_split(np.random.default_rng(100 + rep).permutation(people), 5)
        assert sorted(np.concatenate(folds)) == list(people)                     # everyone is tested exactly once
        assert all(set(a).isdisjoint(b) for i, a in enumerate(folds) for j, b in enumerate(folds) if i < j)
