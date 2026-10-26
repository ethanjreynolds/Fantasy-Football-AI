"""Diagnostic: confirm recent player-stat and teammate usage availability."""
import argparse
import pandas as pd
from src.fantasy_ai.train import find_latest_raw_file

PLAYERS = ['Jalen Hurts', 'Aaron Jones', 'Jordan Mason', 'Jaylen Warren', 'Rico Dowdle']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--week', type=int, default=8)
    args = parser.parse_args()
    raw = pd.read_parquet(find_latest_raw_file())
    current = raw.loc[
        pd.to_numeric(raw['season'], errors='coerce').eq(args.season)
        & pd.to_numeric(raw['week'], errors='coerce').lt(args.week)
    ].copy()
    if 'season_type' in current:
        current = current.loc[current['season_type'].eq('REG')]
    name = 'player_display_name' if 'player_display_name' in current else 'player_name'
    if current.empty:
        raise ValueError('No prior-season data available for requested week')
    print('Available player-stat weeks:', sorted(current['week'].dropna().unique().tolist()))
    selected = current.loc[current[name].astype(str).str.casefold().isin([p.casefold() for p in PLAYERS])]
    cols = [c for c in ['week', name, 'recent_team', 'team', 'position', 'carries', 'targets', 'receptions', 'rushing_yards', 'receiving_yards'] if c in selected]
    print('\nRecent teammate usage (latest recorded games):')
    if selected.empty:
        print('No matching players found; check player naming in source')
    else:
        print(selected.sort_values([name, 'week'])[cols].to_string(index=False))
    if args.week - 1 not in set(pd.to_numeric(current['week'], errors='coerce')):
        print('\nWARNING: prior week is not in this local player-stats file. Update raw data before expecting teammate returns to influence forecasts.')
    else:
        print('\nPrior week exists in the file, but confirm each team/game is complete before relying on it.')


if __name__ == '__main__':
    main()
