
"""NFL schedules and player-team assignments."""

import pandas as pd
import nflreadpy as nfl


def load_week_matchups(season: int, week: int) -> pd.DataFrame:
    """Return one matchup row for each team playing this week."""
    schedules = nfl.load_schedules(season).to_pandas()

    games = schedules[
        (schedules["season"] == season)
        & (schedules["week"] == week)
        & (schedules["game_type"] == "REG")
    ].copy()

    if games.empty:
        raise ValueError(
            f"No regular-season schedule found for {season} Week {week}."
        )

    required = ["home_team", "away_team", "gameday", "gametime"]
    missing = [c for c in required if c not in games.columns]
    if missing:
        raise ValueError(f"Schedule columns missing: {missing}")

    games["game_date"] = pd.to_datetime(
        games["gameday"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    games["kickoff_et"] = games["gametime"].astype("string")

    home = games[
        ["home_team", "away_team", "game_date", "kickoff_et"]
    ].rename(columns={
        "home_team": "team",
        "away_team": "opponent",
    })
    home["home_away"] = "home"

    away = games[
        ["away_team", "home_team", "game_date", "kickoff_et"]
    ].rename(columns={
        "away_team": "team",
        "home_team": "opponent",
    })
    away["home_away"] = "away"

    matchups = pd.concat([home, away], ignore_index=True)

    if matchups["team"].duplicated().any():
        raise ValueError("Duplicate teams in the weekly schedule.")

    return matchups


def load_player_teams(season: int, week: int) -> pd.DataFrame:
    """Find each player's latest available weekly roster snapshot."""
    rosters = nfl.load_rosters_weekly(season).to_pandas()

    required = ["gsis_id", "team", "week"]
    missing = [col for col in required if col not in rosters.columns]
    if missing:
        raise ValueError(f"Missing roster columns: {missing}")

    rosters = rosters.copy()
    rosters["week"] = pd.to_numeric(
        rosters["week"], errors="coerce"
    )

    rosters = rosters[
        rosters["week"].le(week)
        & rosters["gsis_id"].notna()
        & rosters["team"].notna()
    ].copy()

    if rosters.empty:
        raise ValueError(
            f"No roster snapshots for {season} through Week {week}."
        )

    rosters = (
        rosters.sort_values("week")
        .drop_duplicates("gsis_id", keep="last")
    )

    result = rosters[["gsis_id", "team", "week"]].rename(
        columns={
            "gsis_id": "player_id",
            "week": "roster_week",
        }
    )

    return result