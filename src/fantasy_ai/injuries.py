"""Load and inspect weekly NFL injury reports."""

import argparse

import nflreadpy as nfl
import pandas as pd


def load_week_injuries(season: int, week: int) -> pd.DataFrame:
    """Load available injury designations for a specific NFL week."""
    injuries = nfl.load_injuries(season).to_pandas()

    required = {
        "week", "gsis_id", "team",
        "report_status", "report_primary_injury",
        "practice_status",
    }
    missing = required - set(injuries.columns)
    if missing:
        raise ValueError(
            f"Missing injury columns: {sorted(missing)}"
        )

    injuries = injuries.copy()
    injuries["week"] = pd.to_numeric(
        injuries["week"], errors="coerce"
    )

    if "season_type" in injuries.columns:
        print("\nInjury season_type values:")
        print(injuries["season_type"].value_counts(dropna=False))

        injuries = injuries[
            injuries["season_type"].astype(str).str.upper().isin(
                ["REG", "REGULAR", "REGULAR SEASON"]
            )
        ].copy()

    current = injuries[
        (injuries["week"] == week)
        & injuries["gsis_id"].notna()
    ].copy()

    columns = [
        "player_id",
        "injury_status",
        "injury_description",
        "practice_status",
        "injury_report_week",
    ]

    if current.empty:
        return pd.DataFrame(columns=columns)

    current["player_id"] = current["gsis_id"].astype(str)

    current["injury_status"] = (
        current["report_status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "out": "Out",
            "doubtful": "Doubtful",
            "questionable": "Questionable",
        })
    )

    current["practice_status"] = (
        current["practice_status"]
        .astype("string")
        .str.strip()
    )

    current["injury_status"] = current["injury_status"].fillna(
        "No Game Designation"
    )

    current["injury_description"] = (
        current["report_primary_injury"]
    )
    current["injury_report_week"] = week

    # The dataset has no report timestamp. If multiple rows
    # exist for a player, retain the most restrictive designation
    # rather than guessing which entry was published last.
    priority = {
        "Out": 3,
        "Doubtful": 2,
        "Questionable": 1,
    }

    current["_priority"] = (
        current["injury_status"]
        .map(priority)
        .fillna(0)
    )

    current = (
        current.sort_values(
            ["player_id", "_priority"],
            ascending=[True, False],
            kind="stable",
        )
        .drop_duplicates("player_id", keep="first")
    )

    return current[columns].reset_index(drop=True)


def inspect_injuries(season: int, week: int) -> None:
    injuries = nfl.load_injuries(season).to_pandas()

    print("\nInjury dataset columns:")
    print(injuries.columns.tolist())

    print(f"\nTotal injury rows: {len(injuries):,}")

    print("\nAvailable weeks:")
    print(sorted(injuries["week"].dropna().unique().tolist()))

    current = injuries[injuries["week"] == week]

    print(f"\nRows for Week {week}: {len(current):,}")
    if not current.empty:
        print(current.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    inspect_injuries(args.season, args.week)


if __name__ == "__main__":
    main()