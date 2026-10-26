"""Compare original (non-matchup) features vs added player-opportunity features.

Both models are freshly fitted on exactly the same chronological split.
The stored global model supplies the estimator architecture, not fitted weights.

Run: python -m src.fantasy_ai.train_opportunities
     python -m src.fantasy_ai.train_opportunities --refresh-snaps
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.fantasy_ai.train import (
    ROOT, MODEL_DIR, REPORT_DIR, build_modeling_frame,
    chronological_split, find_latest_raw_file,
)
from src.fantasy_ai.opportunity import (
    OPPORTUNITY_FEATURES, add_opportunity_features, load_snap_history,
)

TEMPLATE_PATH = MODEL_DIR / "ppr_hist_gradient_boosting.joblib"
BASELINE_OUTPUT_PATH = MODEL_DIR / "ppr_baseline_opportunity_comparison.joblib"
OUTPUT_PATH = MODEL_DIR / "ppr_hist_gradient_boosting_opportunities.joblib"
REPORT_PATH = REPORT_DIR / "opportunity_model_comparison.csv"
POSITIONS = ["QB", "RB", "WR", "TE"]
# Defensive matchup features have already been tested separately.
# Exclude them in this experiment to isolate player opportunity effects.
DEFENSE_FEATURES = {
    "opp_ppr_allowed_avg_3",
    "opp_ppr_allowed_avg_5",
    "opp_def_games_seen",
}


def _configure_model(template, drop_features=(), add_features=()):
    """Clone a saved sklearn Pipeline, rebuilding its numeric input list."""
    model = clone(template)
    preprocess = model.named_steps["preprocess"]
    updated = []
    found_numeric = False
    for name, transformer, columns in preprocess.transformers:
        if name == "numeric":
            numeric = [c for c in list(columns) if c not in drop_features]
            numeric.extend(c for c in add_features if c not in numeric)
            updated.append((name, transformer, numeric))
            found_numeric = True
        else:
            updated.append((name, transformer, columns))
    if not found_numeric:
        raise ValueError("The saved model has no 'numeric' transformer")
    preprocess.set_params(transformers=updated)
    return model


def main(refresh_snaps: bool = False):
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Saved model template missing: {TEMPLATE_PATH}")
    saved = joblib.load(TEMPLATE_PATH)
    stored_features = list(saved["features"])
    template = saved["model"]

    removed = [f for f in stored_features if f in DEFENSE_FEATURES]
    base_features = [f for f in stored_features if f not in DEFENSE_FEATURES]
    all_features = list(dict.fromkeys(base_features + OPPORTUNITY_FEATURES))
    print(f"Removing prior experimental matchup features: {removed or 'none'}")
    print("Both comparison models will be trained from scratch on the same rows.")

    stats = pd.read_parquet(find_latest_raw_file())
    frame = build_modeling_frame(stats)
    frame = frame[frame["position"].isin(POSITIONS)].copy()

    start = int(pd.to_numeric(stats["season"], errors="coerce").min())
    end = int(pd.to_numeric(stats["season"], errors="coerce").max())
    snaps = load_snap_history(start, end, ROOT / "data" / "raw", refresh=refresh_snaps)
    frame = add_opportunity_features(stats, frame, snaps=snaps)
    print("\nOpportunity feature coverage:")
    print(frame[OPPORTUNITY_FEATURES].notna().mean().round(3).to_string())

    snap_coverage = frame["opportunity_prior_offense_pct_avg_3"].notna().mean()
    if snap_coverage < 0.20:
        raise ValueError(f"Snap feature coverage only {snap_coverage:.1%}; check GSIS mapping")

    train, test, cutoff = chronological_split(frame)
    print(f"\nHoldout: {int(cutoff['season'])} Week {int(cutoff['week'])}")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    missing = [c for c in all_features if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    baseline = _configure_model(template, drop_features=DEFENSE_FEATURES)
    candidate = _configure_model(
        template, drop_features=DEFENSE_FEATURES,
        add_features=OPPORTUNITY_FEATURES,
    )
    baseline.fit(train[base_features], train["actual_ppr"])
    candidate.fit(train[all_features], train["actual_ppr"])

    y = test["actual_ppr"].to_numpy()
    previous = np.clip(baseline.predict(test[base_features]), 0, None)
    predicted = np.clip(candidate.predict(test[all_features]), 0, None)
    baseline_mae = mean_absolute_error(y, previous)
    candidate_mae = mean_absolute_error(y, predicted)
    print("\nSame-holdout comparison (lower is better)")
    print(f"Fresh baseline MAE:       {baseline_mae:.3f}")
    print(f"Opportunity model MAE:    {candidate_mae:.3f}")
    print(f"MAE improvement:          {baseline_mae - candidate_mae:+.3f}")
    print(f"Opportunity model RMSE:   {np.sqrt(mean_squared_error(y, predicted)):.3f}")

    evaluation = test[["season", "week", "position", "player_id", "actual_ppr"]].copy()
    evaluation["baseline_error"] = np.abs(y - previous)
    evaluation["opportunity_error"] = np.abs(y - predicted)
    print("\nMAE by position and season:")
    print(evaluation.groupby(["season", "position"])[
        ["baseline_error", "opportunity_error"]].mean().round(3).to_string())
    print("\nMAE by position:")
    print(evaluation.groupby("position")[
        ["baseline_error", "opportunity_error"]].mean().round(3).to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": baseline, "features": base_features,
                 "trained_for_opportunity_comparison": True}, BASELINE_OUTPUT_PATH)
    joblib.dump({"model": candidate, "features": all_features,
                 "uses_opportunity_features": True,
                 "snap_cache_years": [start, end]}, OUTPUT_PATH)
    evaluation.to_csv(REPORT_PATH, index=False)
    print(f"\nSaved freshly-trained baseline: {BASELINE_OUTPUT_PATH}")
    print(f"Saved experimental model: {OUTPUT_PATH}")
    print(f"Saved per-player errors: {REPORT_PATH}")
    if candidate_mae >= baseline_mae:
        print("WARNING: Opportunity features did not improve this holdout.")
    else:
        print("Opportunity features improved this holdout; confirm with rolling validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-snaps", action="store_true")
    args = parser.parse_args()
    main(refresh_snaps=args.refresh_snaps)
