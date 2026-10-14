import pandas as pd

from src.fantasy_ai.scoring import calculate_ppr_points


def test_standard_ppr_scoring():
    frame = pd.DataFrame([{
        "passing_yards": 250,
        "passing_tds": 2,
        "interceptions": 1,
        "rushing_yards": 20,
        "rushing_tds": 1,
        "receptions": 5,
        "receiving_yards": 60,
        "receiving_tds": 1,
        "sack_fumbles_lost": 0,
    }])
    # 10 + 8 - 2 + 2 + 6 + 5 + 6 + 6 = 41
    assert calculate_ppr_points(frame).iloc[0] == 41.0


def test_missing_columns_default_to_zero():
    frame = pd.DataFrame([{"receptions": 3}])
    assert calculate_ppr_points(frame).iloc[0] == 3.0
