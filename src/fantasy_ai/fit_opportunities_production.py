"""Fit the selected opportunity model on all completed games before a target week.

Usage:
    python -m src.fantasy_ai.fit_opportunities_production --season 2026 --week 8

Uses the established non-matchup baseline pipeline architecture and adds the
validated opportunity feature set. Does not overwrite experiment artifacts.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.fantasy_ai.train import (
    ROOT, MODEL_DIR, build_modeling_frame, find_latest_raw_file,
)
from src.fantasy_ai.opportunity import (
    OPPORTUNITY_FEATURES, add_opportunity_features, load_snap_history,
)
from src.fantasy_ai.train_opportunities import _configure_model

POSITIONS = ("QB", "RB", "WR", "TE")
TEMPLATE_PATH = MODEL_DIR / "ppr_baseline_opportunity_comparison.joblib"
OUTPUT_PATH = MODEL_DIR / "ppr_hist_gradient_boosting_opportunities_production.joblib"


def fit_for_week(season: int, week: int, refresh_snaps: bool = False) -> Path:
    if not 1 <= week <= 18:
        raise ValueError("week must be 1..18")
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline template missing: {TEMPLATE_PATH}. "
            "Run python -m src.fantasy_ai.train_opportunities first."
        )

    # Include no results from the requested game week or any future weeks.
    raw_path = find_latest_raw_file()
    stats = pd.read_parquet(raw_path)
    stats["season"] = pd.to_numeric(stats["season"], errors="coerce")
    stats["week"] = pd.to_numeric(stats["week"], errors="coerce")
    if "season_type" in stats.columns:
        stats = stats.loc[stats["season_type"].eq("REG")].copy()
    history = stats.loc[
        (stats["season"] < season)
        | ((stats["season"] == season) & (stats["week"] < week))
    ].copy()
    if history.empty:
        raise ValueError(f"No recorded games before {season} Week {week}")

    saved = joblib.load(TEMPLATE_PATH)
    base_features = list(saved["features"])
    defense_features = [
        col for col in base_features
        if col.startswith("opp_ppr_allowed_") or col == "opp_def_games_seen"
    ]
    if defense_features:
        raise ValueError(
            f"Non-matchup baseline unexpectedly includes: {defense_features}"
        )
    features = list(dict.fromkeys(base_features + OPPORTUNITY_FEATURES))
    model = _configure_model(
        saved["model"], add_features=OPPORTUNITY_FEATURES
    )

    frame = build_modeling_frame(history)
    frame = frame.loc[frame["position"].isin(POSITIONS)].copy()
    start_season = int(history["season"].min())
    end_season = int(history["season"].max())
    snaps = load_snap_history(
        start_season, end_season, ROOT / "data" / "raw", refresh=refresh_snaps
    )
    frame = add_opportunity_features(history, frame, snaps=snaps)
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing model input features: {missing}")
    if frame.empty:
        raise ValueError("No eligible training rows after filtering")

    print(f"Training cutoff: strictly before {season} Week {week}")
    print(f"Source: {raw_path}")
    print(f"Training rows: {len(frame):,}")
    print("Rows by season:")
    print(frame.groupby("season").size().to_string())
    print("Opportunity feature coverage:")
    print(frame[OPPORTUNITY_FEATURES].notna().mean().round(3).to_string())
    if frame["opportunity_prior_offense_pct_avg_3"].notna().mean() < 0.20:
        raise ValueError("Snap data coverage is too low; check player ID mapping")

    model.fit(frame[features], frame["actual_ppr"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "uses_opportunity_features": True,
            "training_cutoff_exclusive": {"season": season, "week": week},
            "training_rows": len(frame),
            "snap_cache_years": [start_season, end_season],
        },
        OUTPUT_PATH,
    )
    print(f"Saved full-history model: {OUTPUT_PATH}")
    print("This model has not been evaluated on the requested future week.")
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--refresh-snaps", action="store_true")
    args = parser.parse_args()
    fit_for_week(args.season, args.week, args.refresh_snaps)


if __name__ == "__main__":
    main()
