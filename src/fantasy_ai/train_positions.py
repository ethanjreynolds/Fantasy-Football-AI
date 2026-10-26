
"""Evaluate separate fantasy projection models for each position."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.fantasy_ai.train import (
    MODEL_DIR,
    build_modeling_frame,
    chronological_split,
    find_latest_raw_file,
)

import pandas as pd


POSITIONS = ["QB", "RB", "WR", "TE"]
BASELINE_PATH = MODEL_DIR / "ppr_hist_gradient_boosting.joblib"
OUTPUT_DIR = MODEL_DIR / "position_models"


def main():
    # Load our best existing model and its feature definitions.
    saved = joblib.load(BASELINE_PATH)
    baseline_model = saved["model"]
    features = saved["features"]

    # Rebuild exactly the historical dataset used by the baseline.
    stats = pd.read_parquet(find_latest_raw_file())
    frame = build_modeling_frame(stats)
    frame = frame[frame["position"].isin(POSITIONS)].copy()

    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing training features: {missing}")

    train, test, cutoff = chronological_split(frame)

    print(
        f"Holdout starts: {int(cutoff['season'])} "
        f"Week {int(cutoff['week'])}"
    )

    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    # Evaluate the existing global model on the same test rows.
    baseline_predictions = np.clip(
        baseline_model.predict(test[features]), 0, None
    )

    position_predictions = pd.Series(
        index=test.index,
        dtype=float,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPosition-specific model comparison")
    print("-" * 63)
    print(
        f"{'Position':<10} {'Global MAE':>12} "
        f"{'Position MAE':>14} {'Improvement':>13}"
    )
    print("-" * 63)

    for position in POSITIONS:
        pos_train = train[train["position"] == position]
        pos_test = test[test["position"] == position]

        if pos_train.empty or pos_test.empty:
            raise ValueError(
                f"Insufficient training or test data for {position}"
            )

        # Clone the original model's architecture and hyperparameters.
        model = clone(baseline_model)

        model.fit(
            pos_train[features],
            pos_train["actual_ppr"],
        )

        predictions = np.clip(
            model.predict(pos_test[features]), 0, None
        )

        position_predictions.loc[pos_test.index] = predictions

        global_predictions = np.clip(
            baseline_model.predict(pos_test[features]), 0, None
        )

        global_mae = mean_absolute_error(
            pos_test["actual_ppr"],
            global_predictions,
        )

        position_mae = mean_absolute_error(
            pos_test["actual_ppr"],
            predictions,
        )

        improvement = global_mae - position_mae

        print(
            f"{position:<10} {global_mae:>12.3f} "
            f"{position_mae:>14.3f} {improvement:>+13.3f}"
        )

        joblib.dump(
            {
                "model": model,
                "features": features,
                "position": position,
            },
            OUTPUT_DIR / f"{position.lower()}_model.joblib",
        )

    if position_predictions.isna().any():
        raise ValueError("Some test rows were not predicted.")

    # Calculate aggregate metrics using the same holdout rows.
    baseline_mae = mean_absolute_error(
        test["actual_ppr"], baseline_predictions
    )
    overall_mae = mean_absolute_error(
        test["actual_ppr"], position_predictions
    )
    overall_rmse = np.sqrt(
        mean_squared_error(
            test["actual_ppr"], position_predictions
        )
    )

    print("\nOverall comparison")
    print(f"Global model MAE:         {baseline_mae:.3f}")
    print(f"Position models MAE:      {overall_mae:.3f}")
    print(f"Position models RMSE:     {overall_rmse:.3f}")
    print(f"MAE improvement:          {baseline_mae - overall_mae:+.3f}")
    print(f"\nSaved experimental models to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
