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