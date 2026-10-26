
"""Generate weekly PPR projections using a saved fantasy football model."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import nflreadpy as nfl

from src.fantasy_ai.train import (
    ROOT,
    STAT_COLUMNS,
    USAGE_COLUMNS,
    build_modeling_frame,
    find_latest_raw_file,
)

from src.fantasy_ai.schedule import (
    load_week_matchups,
    load_player_teams,
)

from src.fantasy_ai.injuries import load_week_injuries

from src.fantasy_ai.matchups import (
    DEFENSE_FEATURES,
    add_matchup_features,
)

from src.fantasy_ai.opportunity import (
    OPPORTUNITY_FEATURES,
    add_opportunity_features,
    load_snap_history,
)

from src.fantasy_ai.availability import apply_availability

MODEL_PATH = (
    ROOT
    / "models"
    / "ppr_hist_gradient_boosting_opportunities_production.joblib"
)
REPORT_DIR = ROOT / "reports"
FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]


def create_future_rows(
    history: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Create one future row per recently active fantasy player."""

    fantasy = history[
        history["position"].isin(FANTASY_POSITIONS)
    ].copy()

    # Identify players from recent historical appearances.
    if week > 1:
        recent = fantasy[
            (fantasy["season"] == season)
            & (fantasy["week"] >= max(1, week - 4))
        ]
    else:
        # Week 1 uses late-season appearances from the previous year.
        recent = fantasy[
            (fantasy["season"] == season - 1)
            & (fantasy["week"] >= 15)
        ]

    if recent.empty:
        raise ValueError(
            "No recent player data found for the requested week. "
            "Download updated weekly statistics or select another week."
        )

    latest = (
        recent.sort_values(["season", "week"])
        .drop_duplicates("player_id", keep="last")
        .copy()
    )

    # Record when each player was last observed.
    latest["last_seen_season"] = latest["season"]
    latest["last_seen_week"] = latest["week"]

    future = latest.copy()
    future["season"] = season
    future["week"] = week

    # These are placeholder rows, not actual game statistics.
    # Zero out numeric performance data so that future results
    # cannot accidentally be treated as known.
    preserved = {
        "season", "week", "last_seen_season", "last_seen_week"
    }

    for col in future.columns:
        if (
            col not in preserved
            and pd.api.types.is_numeric_dtype(future[col])
        ):
            future[col] = 0

    # Restore numeric player IDs if the source uses them.
    future["player_id"] = latest["player_id"].values

    return future


def predict_week(season: int, week: int, top: int) -> pd.DataFrame:
    if not 1 <= week <= 18:
        raise ValueError("Week must be between 1 and 18.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train and save the gradient boosting model first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    raw_path = find_latest_raw_file()
    stats = pd.read_parquet(raw_path)

    required = {"season", "week", "player_id", "position"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")

    stats["season"] = pd.to_numeric(stats["season"], errors="coerce")
    stats["week"] = pd.to_numeric(stats["week"], errors="coerce")

    # Use only games strictly before the requested week.
    history = stats[
        (stats["season"] < season)
        | (
            (stats["season"] == season)
            & (stats["week"] < week)
        )
    ].copy()

    if history.empty:
        raise ValueError("No historical games before this week.")

    future = create_future_rows(history, season, week)

    # Append placeholder rows, then calculate leakage-aware features.
    combined = pd.concat(
        [history, future],
        ignore_index=True,
        sort=False,
    )

    modeling_frame = build_modeling_frame(combined)

    prediction_rows = modeling_frame[
        (modeling_frame["season"] == season)
        & (modeling_frame["week"] == week)
        & (modeling_frame["position"].isin(FANTASY_POSITIONS))
    ].copy()

    if prediction_rows.empty:
        raise ValueError("No eligible player rows were generated.")

    needs_defense = any(f in features for f in DEFENSE_FEATURES)
    needs_opportunity = any(f in features for f in OPPORTUNITY_FEATURES)

    if needs_defense or needs_opportunity:
        player_teams = load_player_teams(season, week)
        matchups = load_week_matchups(season, week)
        player_teams["player_id"] = player_teams["player_id"].astype(str)
        prediction_rows["player_id"] = prediction_rows["player_id"].astype(str)

        opponent_lookup = player_teams.merge(
            matchups[["team", "opponent"]],
            on="team", how="inner", validate="many_to_one",
        )
        prediction_rows = prediction_rows.drop(
            columns=["team", "recent_team", "opponent_team"],
            errors="ignore",
        )
        before = len(prediction_rows)
        prediction_rows = prediction_rows.merge(
            opponent_lookup[["player_id", "team", "opponent"]].rename(
                columns={"opponent": "opponent_team"}
            ),
            on="player_id", how="inner", validate="many_to_one",
        )
        print("Players excluded before prediction:", before - len(prediction_rows))
        if prediction_rows.empty:
            raise ValueError("No scheduled players with a verified roster team.")

        if needs_defense:
            prediction_rows = add_matchup_features(history, prediction_rows)
        if needs_opportunity:
            start = int(pd.to_numeric(stats["season"], errors="coerce").min())
            end = int(pd.to_numeric(stats["season"], errors="coerce").max())
            snaps = load_snap_history(start, end, ROOT / "data" / "raw")
            prediction_rows = add_opportunity_features(
                history, prediction_rows, snaps=snaps,
            )
            print("Opportunity feature coverage:")
            print(
                prediction_rows[OPPORTUNITY_FEATURES]
                .notna().mean().round(3).to_string()
            )


    missing_features = [
        col for col in features if col not in prediction_rows.columns
    ]
    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    predictions = np.clip(
        model.predict(prediction_rows[features]),
        0,
        None,
    )

    result = prediction_rows[
        ["player_id", "player_name", "position"]
    ].copy()

    result["season"] = season
    result["week"] = week
    result["projected_ppr"] = predictions.round(2)

    # Attach current-season player-team assignments.
    player_teams = load_player_teams(season, week)

    result["player_id"] = result["player_id"].astype(str)
    player_teams["player_id"] = player_teams["player_id"].astype(str)

    result = result.merge(
        player_teams,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    result["roster_age_weeks"] = week - result["roster_week"]

    stale_count = (
        result["roster_age_weeks"] > 2
    ).sum()

    print(f"Players with roster snapshots over 2 weeks old: {stale_count}")

    unmatched = result["team"].isna().sum()
    print(f"Players without a roster match: {unmatched}")

    # Exclude players without a verified team assignment.
    result = result.dropna(subset=["team"]).copy()

    # Match each player's team to the requested week's schedule.
    matchups = load_week_matchups(season, week)

    CURRENT_NFL_TEAMS = {
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    }

    playing_teams = set(matchups["team"])
    bye_teams = sorted(CURRENT_NFL_TEAMS - playing_teams)

    print(f"Teams without a scheduled game: {', '.join(bye_teams)}")

    # Inner join removes players on bye weeks.
    before = len(result)
    result = result.merge(
        matchups,
        on="team",
        how="inner",
        validate="many_to_one",
    )

    print(f"Players excluded because their team has no game: "
        f"{before - len(result)}")

    if result.empty:
        raise ValueError(
            "No players remain after roster and schedule filtering."
        )

    injuries = load_week_injuries(season, week)

    if not injuries.empty:
        matching_ids = (
            set(result["player_id"].astype(str))
            & set(injuries["player_id"].astype(str))
        )

        print(f"Matching player IDs: {len(matching_ids)}")

        print("\nLoaded injury designations:")
        print(injuries["injury_status"].value_counts(dropna=False))

    result = result.merge(
        injuries,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    result["injury_status"] = (
        result["injury_status"].fillna("Unknown")
    )

    result["injury_risk"] = result["injury_status"].map({
        "Out": "Unavailable",
        "Doubtful": "High",
        "Questionable": "Moderate",
        "No Game Designation": "Undetermined",
        "Unknown": "Unknown",
    })

    print("\nInjury designation summary:")
    print(result["injury_status"].value_counts())

    if injuries.empty:
        print(
            f"WARNING: No Week {week} injury reports available. "
            "All injury statuses are unknown."
        )

    result = apply_availability(
        result,
        season,
        week,
        overrides_path=(
            ROOT / "data" / "manual" / "availability_overrides.csv"
        ),
    )

    print("\nAvailability summary:")
    print(result["availability_status"].value_counts())

    print(
        "Players excluded due to unavailability:",
        int(result["exclude_from_rankings"].sum())
    )

    # Keep excluded players and their reasons in the audit.
    full_results = result.copy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full_results.to_csv(
        REPORT_DIR / f"projections_audit_{season}_week_{week}.csv",
        index=False,
    )

    excluded = result.loc[
        result["exclude_from_rankings"],
        [
            "player_name",
            "team",
            "roster_status",
            "roster_week",
            "injury_status",
            "availability_reason",
        ],
    ]

    print("\nExcluded player audit:")
    print(excluded.to_string(index=False))

    result = result[
        ~result["exclude_from_rankings"]
    ].copy()

    print(
        f"Players excluded due to Out designation: "
        f"{len(full_results) - len(result)}"
    )

    reported = result["injury_report_week"].notna().sum()
    total = len(result)

    print(
        f"Injury report coverage: "
        f"{reported}/{total} players "
        f"({reported / total:.1%})"
    )

    result["position_rank"] = (
        result.groupby("position")["projected_ppr"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    result["fantasy_rank"] = (
        result["position"]
        + result["position_rank"].astype(str)
    )

    result = result.sort_values(
        "projected_ppr", ascending=False
    ).reset_index(drop=True)

    result.insert(0, "rank", range(1, len(result) + 1))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        REPORT_DIR / f"projections_{season}_week_{week}.csv"
    )
    result.to_csv(output_path, index=False)

    print(f"\nPPR projections — {season} Week {week}")
    print(
        result[
            [
                "rank",
                "fantasy_rank",
                "player_name",
                "team",
                "opponent",
                "availability_status",
                "injury_status",
                "practice_status",
                #"injury_description",
                "projected_ppr",
            ]
        ].head(top).to_string(index=False)
    )
    print(f"\nTotal projected players: {len(result)}")
    print(f"Saved projections: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict weekly fantasy football PPR points."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--top", type=int, default=25)

    args = parser.parse_args()
    predict_week(args.season, args.week, args.top)


if __name__ == "__main__":
    main()
