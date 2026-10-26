"""Run with: python -m pytest tests/test_matchups.py"""
import pandas as pd
import pytest

from src.fantasy_ai.matchups import add_matchup_features


def test_matchup_features_use_only_previous_games():
    history = pd.DataFrame([
        {
            'season': 2026, 'week': week, 'player_id': f'p{week}',
            'position': 'WR', 'opponent_team': 'BUF',
            'receptions': points, 'receiving_yards': 0,
        }
        for week, points in [(1, 10), (2, 20), (3, 100)]
    ])
    targets = pd.DataFrame([
        {'season': 2026, 'week': week, 'position': 'WR', 'opponent_team': 'BUF'}
        for week in [1, 2, 3, 4]
    ])
    actual = add_matchup_features(history, targets)
    assert actual.loc[0, 'opp_def_games_seen'] == 0
    assert pd.isna(actual.loc[0, 'opp_ppr_allowed_avg_3'])
    assert actual.loc[1, 'opp_ppr_allowed_avg_3'] == pytest.approx(10.0)
    assert actual.loc[2, 'opp_ppr_allowed_avg_3'] == pytest.approx(15.0)
    assert actual.loc[3, 'opp_ppr_allowed_avg_3'] == pytest.approx(130 / 3)
    assert actual.loc[2, 'opp_def_games_seen'] == 2

    changed = history.copy()
    changed.loc[changed['week'] == 3, 'receptions'] = 1000
    changed_features = add_matchup_features(changed, targets)
    assert changed_features.loc[2, 'opp_ppr_allowed_avg_3'] == pytest.approx(15.0)


def test_matchup_features_reset_each_season():
    history = pd.DataFrame([
        {
            "season": 2025,
            "week": 18,
            "player_id": "old_player",
            "position": "WR",
            "opponent_team": "BUF",
            "receptions": 30,
        },
        {
            "season": 2026,
            "week": 1,
            "player_id": "new_player",
            "position": "WR",
            "opponent_team": "BUF",
            "receptions": 10,
        },
    ])

    targets = pd.DataFrame([
        {
            "season": 2026,
            "week": 1,
            "position": "WR",
            "opponent_team": "BUF",
        },
        {
            "season": 2026,
            "week": 2,
            "position": "WR",
            "opponent_team": "BUF",
        },
    ])

    result = add_matchup_features(history, targets)

    # Week 1 has no previous defensive games this season.
    assert pd.isna(
        result.loc[0, "opp_ppr_allowed_avg_3"]
    )
    assert result.loc[0, "opp_def_games_seen"] == 0

    # Week 2 should only see Week 1 of 2026.
    assert result.loc[1, "opp_ppr_allowed_avg_3"] == pytest.approx(10.0)
    assert result.loc[1, "opp_def_games_seen"] == 1
