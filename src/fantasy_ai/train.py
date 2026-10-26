"""Train and evaluate a leakage-aware first-pass fantasy points model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

from src.fantasy_ai.scoring import calculate_ppr_points

from src.fantasy_ai.matchups import (
    DEFENSE_FEATURES,
    add_matchup_features,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

STAT_COLUMNS = [
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds", "receptions",
    "receiving_yards", "receiving_tds",
]

USAGE_COLUMNS = [
    "completions",
    "attempts",
    "carries",
    "targets",
]

def find_latest_raw_file() -> Path:
    files = sorted(RAW_DIR.glob("weekly_player_stats_*.parquet"))
    if not files:
        raise FileNotFoundError(
            "No raw data found. Run: python -m src.fantasy_ai.data "
            "--start-season 2018 --end-season 2026"
        )
    return files[-1]


def build_modeling_frame(stats: pd.DataFrame) -> pd.DataFrame:
    """Create lagged features; all rolling inputs are based on prior appearances."""
    frame = stats.copy()
    if "season_type" in frame.columns:
        frame = frame[
            frame["season_type"] == "REG"
        ].copy()
    required = ["season", "week", "player_id"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Required columns missing from nflverse data: {missing}")

    name_col = "player_display_name" if "player_display_name" in frame.columns else "player_name"
    if name_col not in frame.columns:
        frame["player_name"] = frame["player_id"].astype(str)
        name_col = "player_name"
    frame["player_name"] = frame[name_col].fillna(frame["player_id"].astype(str))
    if "position" not in frame.columns:
        frame["position"] = "UNK"
    frame["position"] = frame["position"].fillna("UNK").astype(str)

    # Drop preseason / non-numeric weeks and sort deterministically.
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["season", "week", "player_id"])
    frame = frame[frame["week"].between(1, 18)].copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)

    for col in STAT_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    frame["actual_ppr"] = calculate_ppr_points(frame)
    frame = frame.sort_values(["player_id", "season", "week"]).drop_duplicates(
        ["player_id", "season", "week"], keep="last"
    )

    # Game/appearance rolling features. shift(1) prevents current-week leakage.
    grouped = frame.groupby("player_id", sort=False)
    frame["prior_games"] = grouped.cumcount()
    frame["prior_ppr_mean"] = grouped["actual_ppr"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    frame["prior_ppr_std"] = grouped["actual_ppr"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=2).std()
    )
    for stat in STAT_COLUMNS:
        feature = f"prior_{stat}_mean"
        frame[feature] = grouped[stat].transform(
            lambda series: series.shift(1).rolling(5, min_periods=1).mean()
        )

    for stat in USAGE_COLUMNS:
        if stat not in frame.columns:
            continue

        frame[stat] = pd.to_numeric(
            frame[stat], errors="coerce"
        ).fillna(0.0)

        grouped = frame.groupby("player_id", sort=False)

        frame[f"prior_{stat}_last_game"] = grouped[stat].shift(1)

        frame[f"prior_{stat}_avg_3"] = grouped[stat].transform(
            lambda series: series.shift(1).rolling(
                3, min_periods=1
            ).mean()
        )

        frame[f"prior_{stat}_avg_5"] = grouped[stat].transform(
            lambda series: series.shift(1).rolling(
                5, min_periods=1
            ).mean()
        )

    # Season-relative week is known in advance; current performance stats are not features.
    frame["week_number"] = frame["week"]
    frame["season_number"] = frame["season"]
    frame = frame[frame["prior_games"] >= 1].copy()
    frame["prior_ppr_std"] = frame["prior_ppr_std"].fillna(0.0)
    return frame


def chronological_split(frame: pd.DataFrame):
    """Hold out the latest 20% of unique season-week dates."""
    dates = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    if len(dates) < 5:
        raise ValueError("Not enough unique season-week values for a chronological holdout.")
    split_index = max(1, int(len(dates) * 0.8))
    cutoff = dates.iloc[split_index]
    train_mask = (frame["season"] < cutoff["season"]) | (
        (frame["season"] == cutoff["season"]) & (frame["week"] < cutoff["week"])
    )
    test_mask = ~train_mask & (
        (frame["season"] > cutoff["season"]) |
        ((frame["season"] == cutoff["season"]) & (frame["week"] >= cutoff["week"]))
    )
    return frame.loc[train_mask].copy(), frame.loc[test_mask].copy(), cutoff


def main() -> None:
    raw_path = find_latest_raw_file()
    print(f"Reading {raw_path}")
    stats = pd.read_parquet(raw_path)
    frame = build_modeling_frame(stats)

    fantasy_positions = ["QB", "RB", "WR", "TE"]
    frame = frame[frame["position"].isin(fantasy_positions)].copy()

    frame = add_matchup_features(stats, frame)

    print("\nDefensive matchup feature coverage:")
    print(
        frame[DEFENSE_FEATURES]
        .notna()
        .mean()
        .round(3)
    )

    if frame.empty:
        raise ValueError("No modeling rows remain after feature engineering.")

    numeric_features = [
        "prior_games", "prior_ppr_mean", "prior_ppr_std",
        "week_number", "season_number",
    ] + [f"prior_{col}_mean" for col in STAT_COLUMNS]

    # Include the new workload features generated in build_modeling_frame().
    usage_features = [
        col for col in frame.columns
        if col.startswith("prior_")
        and (
            col.endswith("_last_game")
            or col.endswith("_avg_3")
            or col.endswith("_avg_5")
        )
    ]

    numeric_features += usage_features

    numeric_features += DEFENSE_FEATURES

    categorical_features = ["position"]
    train, test, cutoff = chronological_split(frame)
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

    features = numeric_features + categorical_features
    model.fit(train[features], train["actual_ppr"])
    predictions = np.clip(model.predict(test[features]), 0, None)
    mae = mean_absolute_error(test["actual_ppr"], predictions)
    rmse = np.sqrt(mean_squared_error(test["actual_ppr"], predictions))
    baseline = np.clip(test["prior_ppr_mean"].to_numpy(), 0, None)
    baseline_mae = mean_absolute_error(test["actual_ppr"], baseline)

    previous_path = MODEL_DIR / "ppr_hist_gradient_boosting.joblib"

    if previous_path.exists():
        previous_saved = joblib.load(previous_path)

        previous_model = previous_saved["model"]
        previous_features = previous_saved["features"]

        previous_predictions = np.clip(
            previous_model.predict(test[previous_features]),
            0,
            None,
        )

        previous_mae = mean_absolute_error(
            test["actual_ppr"],
            previous_predictions,
        )

        print("\nModel comparison:")
        print(f"Previous model MAE: {previous_mae:.3f}")
        print(f"Matchup model MAE:  {mae:.3f}")
        print(f"MAE improvement:    {previous_mae - mae:+.3f}")

    test_results = test[
        ["position", "actual_ppr", "prior_ppr_mean"]
    ].copy()

    test_results["model_prediction"] = predictions

    test_results["model_error"] = (
        test_results["actual_ppr"]
        - test_results["model_prediction"]
    ).abs()

    test_results["baseline_error"] = (
        test_results["actual_ppr"]
        - test_results["prior_ppr_mean"]
    ).abs()

    season_results = test[
        ["season", "position", "actual_ppr"]
    ].copy()

    season_results["prediction"] = predictions

    season_results["absolute_error"] = (
        season_results["actual_ppr"]
        - season_results["prediction"]
    ).abs()

    print("\nModel MAE by season:")
    print(
        season_results.groupby("season")["absolute_error"]
        .agg(["mean", "count"])
        .round(3)
    )

    print("\nModel MAE by season and position:")
    print(
        season_results.groupby(
            ["season", "position"]
        )["absolute_error"]
        .agg(["mean", "count"])
        .round(3)
    )

    print("\nActual PPR distribution:")
    print(
        test["actual_ppr"].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ).round(2)
    )

    print(
        "\nPercentage of test observations scoring zero:",
        round((test["actual_ppr"] == 0).mean() * 100, 2),
        "%"
    )

    print("\nModel MAE by actual scoring range:")
    diagnostic = test[["actual_ppr"]].copy()
    diagnostic["prediction"] = predictions
    diagnostic["absolute_error"] = (
        diagnostic["actual_ppr"] - diagnostic["prediction"]
    ).abs()

    diagnostic["score_range"] = pd.cut(
        diagnostic["actual_ppr"],
        bins=[-float("inf"), 0, 5, 10, 20, float("inf")],
        labels=["Zero/negative", "0–5", "5–10", "10–20", "20+"],
        include_lowest=True,
    )

    print(
        diagnostic.groupby("score_range", observed=False)[
            "absolute_error"
        ].agg(["mean", "count"]).round(2)
    )

    print("\nMAE by position:")
    print(
        test_results.groupby("position")[
            ["model_error", "baseline_error"]
        ].mean().round(3)
    )

    print("\nNumber of test observations by position:")
    print(test_results.groupby("position").size())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_DIR / "ppr_hist_gradient_boosting_matchups_season_reset.joblib")

    output = test[["season", "week", "player_id", "player_name", "position", "actual_ppr"]].copy()
    output["predicted_ppr"] = predictions.round(2)
    output["baseline_prior_mean"] = baseline.round(2)
    output["absolute_error"] = (output["actual_ppr"] - output["predicted_ppr"]).abs().round(2)
    output.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)

    summary = (
        f"Chronological holdout starts at season={int(cutoff['season'])}, week={int(cutoff['week'])}\n"
        f"Train rows: {len(train):,}\nTest rows: {len(test):,}\n"
        f"Model MAE: {mae:.3f}\nModel RMSE: {rmse:.3f}\n"
        f"Prior-average baseline MAE: {baseline_mae:.3f}\n"
        f"Saved model: {MODEL_DIR / 'ppr_hist_gradient_boosting_matchups_season_reset.joblib'}\n"
        f"Saved predictions: {REPORT_DIR / 'holdout_predictions.csv'}\n"
    )
    (REPORT_DIR / "evaluation.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
