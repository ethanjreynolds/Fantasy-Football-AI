"""Leakage-aware player opportunity features for NFL weekly PPR projections.

All features are computed exclusively from game weeks STRICTLY before the
requested season/week. This works for historical training and future inference.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Each rolling feature is based on completed games, not calendar weeks.
PLAYER_METRICS = ("targets", "carries", "attempts", "target_share", "carry_share",
                  "offense_pct", "offense_snaps")
TEAM_METRICS = ("team_targets", "team_carries", "team_pass_attempts", "team_plays_proxy")

OPPORTUNITY_FEATURES = (
    [f"opportunity_prior_{metric}_{suffix}" for metric in PLAYER_METRICS
     for suffix in ("last", "avg_3", "avg_5", "trend_3_vs_5")]
    + [f"opportunity_prior_{metric}_{suffix}" for metric in TEAM_METRICS
       for suffix in ("avg_3", "avg_5")]
    + ["opportunity_weeks_since_last_appearance", "opportunity_prior_appearances"]
)


def _normalize_teams(frame: pd.DataFrame) -> pd.Series:
    """Choose the most useful team field; nflverse schema changed over time."""
    if "team" in frame.columns:
        team = frame["team"].astype("string")
    else:
        team = pd.Series(pd.NA, index=frame.index, dtype="string")
    if "recent_team" in frame.columns:
        team = team.fillna(frame["recent_team"].astype("string"))
    return team.str.strip().replace({"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"})


def load_snap_history(start_season: int, end_season: int,
                      cache_dir: Path | str, refresh: bool = False) -> pd.DataFrame:
    """Download PFR snap data; map PFR player IDs to GSIS using nflverse players.

    Reuses a local parquet cache; use refresh=True when new weeks arrive.
    """
    import nflreadpy as nfl

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"snap_counts_gsis_{start_season}_{end_season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    seasons = list(range(start_season, end_season + 1))
    snaps = nfl.load_snap_counts(seasons).to_pandas()
    players = nfl.load_players().to_pandas()
    for col in ("pfr_player_id", "season", "week", "offense_snaps", "offense_pct"):
        if col not in snaps.columns:
            raise ValueError(f"Snap data missing {col!r}. Available: {list(snaps.columns)}")
    for col in ("gsis_id", "pfr_id"):
        if col not in players.columns:
            raise ValueError(f"Players crosswalk missing {col!r}. Available: {list(players.columns)}")
    if "game_type" in snaps.columns:
        snaps = snaps[snaps["game_type"].eq("REG")].copy()

    crosswalk = (players[["gsis_id", "pfr_id"]]
                 .dropna().rename(columns={"gsis_id": "player_id", "pfr_id": "pfr_player_id"})
                 .drop_duplicates())
    # Do not guess when a PFR ID maps to multiple GSIS IDs.
    ambiguous = crosswalk["pfr_player_id"].duplicated(keep=False)
    crosswalk = crosswalk.loc[~ambiguous]

    snaps = snaps.merge(crosswalk, on="pfr_player_id", how="left", validate="many_to_one")
    total = len(snaps)
    snaps = snaps.dropna(subset=["player_id"]).copy()
    print(f"Snap GSIS ID mapping: {len(snaps):,}/{total:,} rows ({len(snaps)/max(total,1):.1%})")
    snaps["season"] = pd.to_numeric(snaps["season"], errors="coerce")
    snaps["week"] = pd.to_numeric(snaps["week"], errors="coerce")
    snaps["offense_snaps"] = pd.to_numeric(snaps["offense_snaps"], errors="coerce")
    snaps["offense_pct"] = pd.to_numeric(snaps["offense_pct"], errors="coerce")
    # PFR snap percentages might be represented as 0..100 or 0..1.
    snaps.loc[snaps["offense_pct"] > 1, "offense_pct"] /= 100.0
    snaps["offense_pct"] = snaps["offense_pct"].clip(0, 1)
    snaps = snaps.dropna(subset=["season", "week"])
    snaps["season"] = snaps["season"].astype(int)
    snaps["week"] = snaps["week"].astype(int)
    snaps["player_id"] = snaps["player_id"].astype(str)
    snaps = (snaps.groupby(["player_id", "season", "week"], as_index=False)
             .agg(offense_snaps=("offense_snaps", "sum"),
                  offense_pct=("offense_pct", "max")))
    snaps.to_parquet(path, index=False)
    print(f"Cached GSIS-mapped snap counts: {path}")
    return snaps


def _prepare_history(stats: pd.DataFrame, snaps: pd.DataFrame | None) -> pd.DataFrame:
    required = ("player_id", "season", "week")
    if any(col not in stats for col in required):
        raise ValueError(f"Stats must include {required}")

    raw = stats.copy()
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce")
    raw["week"] = pd.to_numeric(raw["week"], errors="coerce")
    if "season_type" in raw:
        raw = raw[raw["season_type"].eq("REG")].copy()
    raw = raw.dropna(subset=list(required))
    raw = raw[raw["week"].between(1, 18)].copy()
    raw["season"] = raw["season"].astype(int)
    raw["week"] = raw["week"].astype(int)
    raw["player_id"] = raw["player_id"].astype(str)
    raw["_team"] = _normalize_teams(raw)
    for stat in ("targets", "carries", "attempts"):
        if stat not in raw:
            raise ValueError(f"Weekly player stats missing required opportunity field: {stat}")
        raw[stat] = pd.to_numeric(raw[stat], errors="coerce")

    # Align with build_modeling_frame's one-row-per-player-week convention.
    raw = (raw.sort_values(["player_id", "season", "week"])
           .drop_duplicates(["player_id", "season", "week"], keep="last")
           .copy())

    if snaps is not None and len(snaps):
        snaps = snaps[["player_id", "season", "week", "offense_snaps", "offense_pct"]].copy()
        snaps["player_id"] = snaps["player_id"].astype(str)
        snaps = snaps.drop_duplicates(["player_id", "season", "week"], keep="last")
        raw = raw.merge(snaps, on=["player_id", "season", "week"], how="left",
                        validate="one_to_one", suffixes=("", "_snap"))
        for col in ("offense_snaps", "offense_pct"):
            if f"{col}_snap" in raw:
                raw[col] = raw[f"{col}_snap"]
                raw = raw.drop(columns=f"{col}_snap")
    else:
        raw["offense_snaps"] = np.nan
        raw["offense_pct"] = np.nan

    # Denominators count ALL players on a team, not just QB/RB/WR/TE.
    totals = (raw.dropna(subset=["_team"])
              .groupby(["season", "week", "_team"], as_index=False)
              .agg(team_targets=("targets", "sum"),
                   team_carries=("carries", "sum"),
                   team_pass_attempts=("attempts", "sum")))
    totals["team_plays_proxy"] = totals["team_carries"] + totals["team_pass_attempts"]
    raw = raw.merge(totals, on=["season", "week", "_team"], how="left", validate="many_to_one")
    raw["target_share"] = (raw["targets"] / raw["team_targets"].replace(0, np.nan)).clip(0, 1)
    raw["carry_share"] = (raw["carries"] / raw["team_carries"].replace(0, np.nan)).clip(0, 1)
    raw["_key"] = raw["season"] * 100 + raw["week"]
    return raw


def _rolling_table(raw: pd.DataFrame, by: str, metrics: tuple[str, ...],
                   with_last: bool = False) -> pd.DataFrame:
    raw = raw.sort_values([by, "_key"]).copy()
    groups = raw.groupby(by, sort=False)
    raw["_appearances"] = groups.cumcount() + 1
    for metric in metrics:
        x = pd.to_numeric(raw[metric], errors="coerce")
        raw[metric] = x
        if with_last:
            raw[f"opportunity_prior_{metric}_last"] = x
        for window in (3, 5):
            raw[f"opportunity_prior_{metric}_avg_{window}"] = groups[metric].transform(
                lambda series: series.rolling(window, min_periods=1).mean()
            )
        if with_last:
            raw[f"opportunity_prior_{metric}_trend_3_vs_5"] = (
                raw[f"opportunity_prior_{metric}_avg_3"] - raw[f"opportunity_prior_{metric}_avg_5"]
            )
    return raw


def _lookup_previous(targets: pd.DataFrame, source: pd.DataFrame,
                     by: str, extra_cols: list[str]) -> pd.DataFrame:
    left = targets[["_row", "_key", by]].copy()
    right = source[["_key", by, *extra_cols]].copy()
    # merge_asof requires a global sort by the as-of key, *not* by group first.
    left = left.sort_values("_key")
    right = right.sort_values("_key")
    merged = pd.merge_asof(left, right, on="_key", by=by,
                           direction="backward", allow_exact_matches=False)
    return merged.sort_values("_row").reset_index(drop=True)


def add_opportunity_features(history: pd.DataFrame, targets: pd.DataFrame,
                             snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Append prior-only features. Training: history=raw stats, targets=model frame.

    Inference: history=raw games before target week, targets=future prediction rows.
    Future targets must have their *current* team assigned before calling.
    """
    if targets.empty:
        result = targets.copy()
        for feature in OPPORTUNITY_FEATURES:
            result[feature] = pd.Series(dtype="float64")
        return result
    raw = _prepare_history(history, snaps)
    lookup = targets[["player_id", "season", "week"]].reset_index(drop=True).copy()
    lookup["player_id"] = lookup["player_id"].astype(str)
    lookup["season"] = pd.to_numeric(lookup["season"]).astype(int)
    lookup["week"] = pd.to_numeric(lookup["week"]).astype(int)
    lookup["_team"] = _normalize_teams(targets.reset_index(drop=True))
    lookup["_key"] = lookup["season"] * 100 + lookup["week"]
    lookup["_row"] = np.arange(len(lookup))

    player = _rolling_table(raw, "player_id", PLAYER_METRICS, with_last=True)
    cols = [col for col in OPPORTUNITY_FEATURES if col.startswith("opportunity_prior_")
            and any(col.startswith(f"opportunity_prior_{m}_") for m in PLAYER_METRICS)]
    player["_last_season"] = player["season"]
    player["_last_week"] = player["week"]
    player["opportunity_prior_appearances"] = player["_appearances"]
    joined = _lookup_previous(lookup, player, "player_id", cols +
                              ["_last_season", "_last_week", "opportunity_prior_appearances"])
    # The elapsed-week gap is defined within the same NFL season only.
    joined["opportunity_weeks_since_last_appearance"] = np.where(
        joined["season"] == joined["_last_season"],
        joined["week"] - joined["_last_week"],
        np.nan,
    ) if "season" in joined.columns else np.where(
        lookup["season"].to_numpy() == joined["_last_season"].to_numpy(),
        lookup["week"].to_numpy() - joined["_last_week"].to_numpy(),
        np.nan,
    )

    # Team rolling volumes are calculated on *one* row per team/game.
    team = (raw[["season", "week", "_team", *TEAM_METRICS]]
            .dropna(subset=["_team"])
            .drop_duplicates(["season", "week", "_team"])
            .copy())
    team["_key"] = team["season"] * 100 + team["week"]
    team = _rolling_table(team, "_team", TEAM_METRICS)
    team_cols = [f"opportunity_prior_{m}_{suffix}" for m in TEAM_METRICS
                 for suffix in ("avg_3", "avg_5")]
    joined_team = _lookup_previous(lookup, team, "_team", team_cols)
    result = targets.copy()
    for col in OPPORTUNITY_FEATURES:
        if col in joined.columns:
            result[col] = joined[col].to_numpy()
        elif col in joined_team.columns:
            result[col] = joined_team[col].to_numpy()
        else:
            result[col] = np.nan
    return result
