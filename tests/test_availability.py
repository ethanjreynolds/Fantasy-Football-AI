import pandas as pd
from src.fantasy_ai.availability import apply_availability


def test_roster_reserve_and_prior_week_inactive():
    rows = pd.DataFrame([
        {'player_id':'1','player_name':'One','team':'PHI', 'roster_status':'RES','roster_week':7,'injury_status':'Unknown'},
        {'player_id':'2','player_name':'Two','team':'PIT', 'roster_status':'INA','roster_week':7,'injury_status':'Unknown'},
        {'player_id':'3','player_name':'Three','team':'MIN', 'roster_status':'RES','roster_week':4,'injury_status':'Unknown'},
    ])
    out = apply_availability(rows, 2026, 8)
    assert out.loc[0, 'exclude_from_rankings']
    assert not out.loc[1, 'exclude_from_rankings']
    assert out.loc[2, 'availability_status'] == 'Roster review'


def test_manual_source_and_date(tmp_path):
    rows = pd.DataFrame([
        {'player_id':'1','player_name':'Jalen Hurts','team':'PHI','injury_status':'Unknown'},
        {'player_id':'2','player_name':'Aaron Jones','team':'MIN','injury_status':'Unknown'},
    ])
    overrides = tmp_path / 'notes.csv'
    overrides.write_text(
        'season,week,player_id,player_name,team,status,reason,source_url,as_of\n'
        '2026,8,,Jalen Hurts,PHI,Out,Team statement,https://example.com/a,2026-10-19\n'
        '2026,8,,Aaron Jones,MIN,Injury concern,Reported knee,https://example.com/b,2026-10-26\n', encoding='utf-8')
    early = apply_availability(rows, 2026, 8, overrides, as_of='2026-10-20')
    assert early.loc[0, 'exclude_from_rankings']
    assert early.loc[1, 'availability_status'] == 'Unverified'
    late = apply_availability(rows, 2026, 8, overrides, as_of='2026-10-26')
    assert late.loc[1, 'availability_status'] == 'Injury concern'
    assert not late.loc[1, 'exclude_from_rankings']
