"""Sanity tests for the churn project.

Run from the project root after running notebooks 01, 02 and 03:
    pytest -v
They guard against the mistakes that matter most: data leakage, bad data,
invalid probabilities and a silently degraded model.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "telco_clean.csv"
SCORES = ROOT / "outputs" / "customer_churn_scores.csv"
COMPARISON = ROOT / "outputs" / "model_comparison.csv"
DECILES = ROOT / "outputs" / "decile_analysis.csv"
DRIVERS = ROOT / "outputs" / "shap_top_drivers_test.csv"

LEAKAGE_COLUMNS = {"Churn_Label", "Churn_Score", "CLTV"}
EXCLUDED_COLUMNS = {"Gender", "Country", "State", "City", "Zip_Code",
                    "Lat_Long", "Latitude", "Longitude", "Count"}


def load(path, hint):
    if not path.exists():
        pytest.skip(f"{path.name} not found - {hint}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def clean():
    return load(CLEAN, "run notebook 01 first")


@pytest.fixture(scope="module")
def scores():
    return load(SCORES, "run notebook 02 first")


# ---------- cleaned data ----------
def test_no_leakage_columns(clean):
    assert not LEAKAGE_COLUMNS & set(clean.columns)


def test_excluded_columns_are_absent(clean):
    assert not EXCLUDED_COLUMNS & set(clean.columns)


def test_one_row_per_customer(clean):
    assert len(clean) == 7043
    assert clean["CustomerID"].is_unique


def test_features_have_no_missing_values(clean):
    features = clean.drop(columns=["Churn_Reason"])
    assert features.isna().sum().sum() == 0


def test_target_is_binary_with_plausible_churn_rate(clean):
    assert set(clean["Churn_Value"].unique()) == {0, 1}
    assert 0.20 < clean["Churn_Value"].mean() < 0.35


def test_churn_reason_only_exists_for_churners(clean):
    # Churn_Reason is descriptive only; it must never be usable as a feature
    assert clean.loc[clean["Churn_Reason"].notna(), "Churn_Value"].eq(1).all()


# ---------- model outputs ----------
def test_scores_are_valid_probabilities(scores):
    assert scores["churn_probability"].between(0, 1).all()
    assert scores["churn_probability"].notna().all()


def test_scores_cover_every_customer(scores, clean):
    assert set(scores["CustomerID"]) == set(clean["CustomerID"])


def test_predictions_follow_a_single_threshold(scores):
    flagged = scores.loc[scores["predicted_churn"] == 1, "churn_probability"]
    not_flagged = scores.loc[scores["predicted_churn"] == 0, "churn_probability"]
    assert flagged.min() >= not_flagged.max()


def test_risk_bands_are_valid_and_ordered(scores):
    assert set(scores["risk_band"].unique()) <= {"Low", "Medium", "High"}
    means = scores.groupby("risk_band")["churn_probability"].mean()
    assert means["Low"] < means["Medium"] < means["High"]


def test_models_beat_random_guessing():
    results = load(COMPARISON, "run notebook 02 first")
    assert (results["test_roc_auc"] > 0.75).all()
    base_rate = 0.265
    assert results["test_pr_auc"].max() > base_rate + 0.25


def test_top_decile_has_strong_lift():
    deciles = load(DECILES, "run notebook 02 first")
    assert deciles.loc[deciles["risk_decile"] == 1, "lift"].iloc[0] > 2.0
    # lowest-risk decile should contain very few churners
    assert deciles.loc[deciles["risk_decile"] == 10, "churn_rate"].iloc[0] < 0.05


def test_shap_driver_table_is_complete():
    drivers = load(DRIVERS, "run notebook 03 first")
    assert len(drivers) == 1409          # held-out test customers
    assert drivers[["driver_1", "driver_2", "driver_3"]].notna().all().all()
