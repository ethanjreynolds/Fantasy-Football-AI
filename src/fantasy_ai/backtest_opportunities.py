"""Rolling-origin evaluation of baseline vs opportunity fantasy PPR models.

Usage:
    python -m src.fantasy_ai.backtest_opportunities
    python -m src.fantasy_ai.backtest_opportunities --first-test-season 2023

Each evaluation season is predicted using models fitted only on earlier seasons.
The two models use the same player rows, split, and estimator hyperparameters.
No production model files are overwritten.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.fantasy_ai.train import (
    MODEL_DIR, REPORT_DIR, ROOT, build_modeling_frame, find_latest_raw_file,
)
from src.fantasy_ai.opportunity import (
    OPPORTUNITY_FEATURES, add_opportunity_features, load_snap_history,
)
from src.fantasy_ai.train_opportunities import _configure_model

POSITIONS = ("QB", "RB", "WR", "TE")
TEMPLATE_PATH = MODEL_DIR / "ppr_baseline_opportunity_comparison.joblib"


def evaluate(first_test_season: int = 2023, last_test_season: int | None = None):
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline template: {TEMPLATE_PATH}. "
            "Run python -m src.fantasy_ai.train_opportunities first."
        )

    saved = joblib.load(TEMPLATE_PATH)
    template = saved["model"]
    base_features = list(saved["features"])
    defensive = [c for c in base_features if c.startswith("opp_ppr_allowed_") or c == "opp_def_games_seen"]
    if defensive:
        raise ValueError(
            f"Baseline comparison model unexpectedly includes defensive features: {defensive}. "
            "Re-run python -m src.fantasy_ai.train_opportunities to rebuild it."
        )
    opportunity_features = list(OPPORTUNITY_FEATURES)
    candidate_features = list(dict.fromkeys(base_features + opportunity_features))

    stats = pd.read_parquet(find_latest_raw_file())
    frame = build_modeling_frame(stats)
    frame = frame.loc[frame["position"].isin(POSITIONS)].copy()

    start_season = int(frame["season"].min())
    print(f"Using saved baseline template: {TEMPLATE_PATH}")
    last_available_season = int(frame["season"].max())
    if last_test_season is None:
        last_test_season = last_available_season
    if first_test_season > last_test_season:
        raise ValueError("First test season cannot be after last test season")

    snaps = load_snap_history(
        start_season, last_available_season, ROOT / "data" / "raw"
    )
    frame = add_opportunity_features(stats, frame, snaps=snaps)
    missing = [c for c in candidate_features if c not in frame.columns]
    if missing:
        raise ValueError(f"Required feature columns are missing: {missing}")

    all_predictions = []
    for year in range(first_test_season, last_test_season + 1):
        train = frame.loc[frame["season"] < year]
        test = frame.loc[frame["season"] == year]
        if train.empty or test.empty:
            print(f"Skipping {year}: no training or test rows")
            continue

        print(f"\nEvaluating {year}: train {len(train):,}; test {len(test):,}")
        baseline = _configure_model(template)
        opportunity = _configure_model(template, add_features=opportunity_features)
        baseline.fit(train[base_features], train["actual_ppr"])
        opportunity.fit(train[candidate_features], train["actual_ppr"])

        actual = test["actual_ppr"].to_numpy()
        base_pred = np.clip(baseline.predict(test[base_features]), 0, None)
        opp_pred = np.clip(
            opportunity.predict(test[candidate_features]), 0, None
        )

        result = test[["season", "week", "player_id", "position", "actual_ppr"]].copy()
        result["baseline_prediction"] = base_pred
        result["opportunity_prediction"] = opp_pred
        result["baseline_absolute_error"] = np.abs(actual - base_pred)
        result["opportunity_absolute_error"] = np.abs(actual - opp_pred)
        all_predictions.append(result)

        print(
            f"  baseline MAE {mean_absolute_error(actual, base_pred):.3f} | "
            f"opportunity MAE {mean_absolute_error(actual, opp_pred):.3f} | "
            f"gain {mean_absolute_error(actual, base_pred) - mean_absolute_error(actual, opp_pred):+.3f}"
        )

    if not all_predictions:
        raise ValueError("No evaluation folds were generated; check season bounds")

    predictions = pd.concat(all_predictions, ignore_index=True)
    summaries = []
    for (year, position), rows in predictions.groupby(["season", "position"]):
        summaries.append({
            "season": int(year), "position": position, "rows": len(rows),
            "baseline_mae": rows["baseline_absolute_error"].mean(),
            "opportunity_mae": rows["opportunity_absolute_error"].mean(),
        })
    summary = pd.DataFrame(summaries)
    summary["mae_gain"] = summary["baseline_mae"] - summary["opportunity_mae"]

    per_year = (predictions.groupby("season")
                .agg(rows=("actual_ppr", "size"),
                     baseline_mae=("baseline_absolute_error", "mean"),
                     opportunity_mae=("opportunity_absolute_error", "mean"))
                .reset_index())
    per_year["mae_gain"] = per_year["baseline_mae"] - per_year["opportunity_mae"]

    pooled_base = predictions["baseline_absolute_error"].mean()
    pooled_opp = predictions["opportunity_absolute_error"].mean()
    pooled_rmse = np.sqrt(mean_squared_error(
        predictions["actual_ppr"], predictions["opportunity_prediction"]
    ))

    print("\nRolling-origin MAE by test season (positive gain is better):")
    print(per_year.round(3).to_string(index=False))
    print("\nRolling-origin MAE by test season and position:")
    print(summary.round(3).to_string(index=False))
    print(f"\nPooled across {len(predictions):,} out-of-time predictions:")
    print(f"  Fresh baseline MAE: {pooled_base:.3f}")
    print(f"  Opportunity MAE:    {pooled_opp:.3f}")
    print(f"  Opportunity RMSE:   {pooled_rmse:.3f}")
    print(f"  MAE gain:           {pooled_base - pooled_opp:+.3f}")
    print(f"  Improved in {(per_year['mae_gain'] > 0).sum()} of {len(per_year)} seasons")
    print("Note: the last season may be incomplete; results are not a fully untouched final test.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = (
        REPORT_DIR / "rolling_opportunity_predictions.csv",
        REPORT_DIR / "rolling_opportunity_by_position.csv",
        REPORT_DIR / "rolling_opportunity_by_season.csv",
    )
    predictions.to_csv(output_paths[0], index=False)
    summary.to_csv(output_paths[1], index=False)
    per_year.to_csv(output_paths[2], index=False)
    for path in output_paths:
        print(f"Saved: {path}")
    print("Saved template models were not modified.")
    return predictions, summary, per_year


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-test-season", type=int, default=2023)
    parser.add_argument("--last-test-season", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.first_test_season, args.last_test_season)
