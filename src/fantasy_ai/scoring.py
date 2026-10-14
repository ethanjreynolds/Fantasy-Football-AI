"""Fantasy scoring helpers."""

import pandas as pd


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric column, treating missing columns and null values as zero."""
    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)


def calculate_ppr_points(frame: pd.DataFrame) -> pd.Series:
    """Calculate standard PPR points from a frame of weekly NFL statistics.

    Scoring: pass yards / 25, pass TD +4, interception -2, rush/rec yards / 10,
    rush/rec TD +6, reception +1, and fumbles lost -2.
    """
    points = (
        _numeric_column(frame, "passing_yards") / 25.0
        + _numeric_column(frame, "passing_tds") * 4.0
        - _numeric_column(frame, "interceptions") * 2.0
        + _numeric_column(frame, "rushing_yards") * 0.1
        + _numeric_column(frame, "rushing_tds") * 6.0
        + _numeric_column(frame, "receiving_yards") * 0.1
        + _numeric_column(frame, "receiving_tds") * 6.0
        + _numeric_column(frame, "receptions")
    )

    # nflverse can split fumbles lost by play type. Sum each available component.
    # If a single aggregate column is available, use it as a fallback.
    fumble_columns = [
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
    ]
    present = [name for name in fumble_columns if name in frame.columns]
    if present:
        fumbles_lost = sum((_numeric_column(frame, name) for name in present))
    else:
        fumbles_lost = _numeric_column(frame, "fumbles_lost")
    points = points - fumbles_lost * 2.0
    return points.round(3)
