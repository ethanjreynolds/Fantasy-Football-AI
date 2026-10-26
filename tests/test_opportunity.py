"""Run with: python -m pytest tests/test_opportunity.py -v"""
import numpy as np
import pandas as pd
import pytest
from src.fantasy_ai.opportunity import add_opportunity_features, OPPORTUNITY_FEATURES


def sample():
    return pd.DataFrame([
        dict(player_id='A', season=2026, week=1, team='KC', targets=2, carries=4, attempts=0),
        dict(player_id='B', season=2026, week=1, team='KC', targets=6, carries=6, attempts=30),
        dict(player_id='C', season=2026, week=1, team='LV', targets=5, carries=1, attempts=30),
        # Week 2: A does not play, but KC does.
        dict(player_id='B', season=2026, week=2, team='KC', targets=10, carries=2, attempts=25),
        dict(player_id='C', season=2026, week=2, team='LV', targets=3, carries=1, attempts=28),
        dict(player_id='A', season=2026, week=3, team='KC', targets=4, carries=8, attempts=0),
        dict(player_id='B', season=2026, week=3, team='KC', targets=4, carries=12, attempts=27),
    ])


def test_target_and_carry_share_and_team_volume():
    history = sample()
    target = pd.DataFrame([dict(player_id='A', season=2026, week=2, team='KC')])
    row = add_opportunity_features(history, target).iloc[0]
    assert row['opportunity_prior_target_share_last'] == pytest.approx(2 / 8)
    assert row['opportunity_prior_carry_share_last'] == pytest.approx(4 / 10)
    assert row['opportunity_prior_team_targets_avg_3'] == pytest.approx(8)
    assert row['opportunity_prior_team_pass_attempts_avg_3'] == pytest.approx(30)
    assert row['opportunity_weeks_since_last_appearance'] == 1


def test_no_current_or_future_week_leakage():
    h = sample()
    target = pd.DataFrame([dict(player_id='A', season=2026, week=3, team='KC')])
    base = add_opportunity_features(h, target)
    mutated = h.copy()
    mutated.loc[mutated['week'] == 3, ['targets', 'carries']] = 100000
    mutated = pd.concat([mutated, pd.DataFrame([
        dict(player_id='A', season=2026, week=4, team='KC', targets=9999, carries=9999, attempts=99)
    ])], ignore_index=True)
    changed = add_opportunity_features(mutated, target)
    pd.testing.assert_frame_equal(base[OPPORTUNITY_FEATURES], changed[OPPORTUNITY_FEATURES])
    # Latest appearance Week 1; prior team volume sees Weeks 1 and 2.
    row = base.iloc[0]
    assert row['opportunity_prior_team_targets_avg_3'] == pytest.approx((8 + 10) / 2)
    assert row['opportunity_weeks_since_last_appearance'] == 2


def test_future_rows_and_snap_features():
    snaps = pd.DataFrame([
        dict(player_id='A', season=2026, week=1, offense_snaps=55, offense_pct=0.75),
        dict(player_id='A', season=2026, week=3, offense_snaps=40, offense_pct=0.6),
    ])
    target = pd.DataFrame([dict(player_id='A', season=2026, week=4, team='KC')])
    row = add_opportunity_features(sample(), target, snaps=snaps).iloc[0]
    assert row['opportunity_prior_offense_pct_last'] == pytest.approx(0.6)
    assert row['opportunity_prior_offense_pct_avg_3'] == pytest.approx(0.675)
    assert row['opportunity_prior_offense_snaps_last'] == 40
    assert row['opportunity_weeks_since_last_appearance'] == 1


def test_unknown_snap_data_is_missing_not_zero():
    target = pd.DataFrame([dict(player_id='A', season=2026, week=4, team='KC')])
    row = add_opportunity_features(sample(), target, snaps=None).iloc[0]
    assert np.isnan(row['opportunity_prior_offense_pct_avg_3'])


def test_season_boundary_has_no_fake_week_gap():
    history = pd.DataFrame([
        dict(player_id='A', season=2025, week=18, team='KC', targets=5, carries=2, attempts=0)
    ])
    target = pd.DataFrame([dict(player_id='A', season=2026, week=1, team='KC')])
    row = add_opportunity_features(history, target).iloc[0]
    assert np.isnan(row['opportunity_weeks_since_last_appearance'])
    assert row['opportunity_prior_appearances'] == 1
