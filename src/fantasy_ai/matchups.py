"""Leakage-safe defensive fantasy points allowed features."""
import numpy as np
import pandas as pd
from src.fantasy_ai.scoring import calculate_ppr_points

FANTASY_POSITIONS = ('QB', 'RB', 'WR', 'TE')
DEFENSE_FEATURES = [
    'opp_ppr_allowed_avg_3',
    'opp_ppr_allowed_avg_5',
    'opp_def_games_seen',
]
TEAM_ALIASES = {'LAR': 'LA', 'STL': 'LA', 'OAK': 'LV', 'SD': 'LAC'}

def _standardize_team(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.upper().replace(TEAM_ALIASES)


def add_matchup_features(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Attach prior-game defense-versus-position statistics to target rows.

    Defense observations are computed from `history`. For each target at (season,
    week), only defense games with strictly *earlier* season-week may match.
    """
    required_history = {'season', 'week', 'player_id', 'position', 'opponent_team'}
    required_targets = {'season', 'week', 'position', 'opponent_team'}
    for label, data, required in (
        ('history', history, required_history),
        ('targets', targets, required_targets),
    ):
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f'{label} missing columns: {sorted(missing)}')

    src = history.copy()
    if 'season_type' in src.columns:
        src = src[src['season_type'].astype('string').str.upper().eq('REG')].copy()
    src['season'] = pd.to_numeric(src['season'], errors='coerce')
    src['week'] = pd.to_numeric(src['week'], errors='coerce')
    src = src.dropna(subset=['season', 'week', 'player_id', 'opponent_team'])
    src = src[src['week'].between(1, 18)].copy()
    src = src[src['position'].isin(FANTASY_POSITIONS)].copy()
    src['season'] = src['season'].astype(int)
    src['week'] = src['week'].astype(int)
    src['opponent_team'] = _standardize_team(src['opponent_team'])
    src = (src.sort_values(['season', 'week']).drop_duplicates(['season', 'week', 'player_id'], keep='last'))
    if src.empty:
        output = targets.copy()
        for col in DEFENSE_FEATURES:
            output[col] = 0 if col == 'opp_def_games_seen' else np.nan
        return output

    # Make scoring use the same columns and zero-filling as training.
    stat_cols = [
        'passing_yards', 'passing_tds', 'interceptions',
        'rushing_yards', 'rushing_tds', 'receptions',
        'receiving_yards', 'receiving_tds',
    ]
    for col in stat_cols:
        if col not in src.columns:
            src[col] = 0.0
        src[col] = pd.to_numeric(src[col], errors='coerce').fillna(0.0)
    src['_ppr'] = calculate_ppr_points(src)

    weekly = (src.groupby(['season', 'week', 'opponent_team', 'position'], as_index=False).agg(ppr_allowed=('_ppr', 'sum')))
    # Include zero point totals for a position if a defense played but
    # nobody at that position recorded a fantasy stat.
    games = src[['season', 'week', 'opponent_team']].drop_duplicates()
    positions = pd.DataFrame({'position': list(FANTASY_POSITIONS)})
    game_position = games.merge(positions, how='cross')
    game_position = game_position.merge(
        weekly, on=['season', 'week', 'opponent_team', 'position'], how='left',
        validate='one_to_one',
    )
    game_position['ppr_allowed'] = game_position['ppr_allowed'].fillna(0.0)
    game_position = game_position.sort_values(
        ['opponent_team', 'position', 'season', 'week']
    )

    groups = game_position.groupby(
        ["season", "opponent_team", "position"],
        sort=False,
    )
    
    for n in (3, 5):
        game_position[f'opp_ppr_allowed_avg_{n}'] = groups['ppr_allowed'].transform(
            lambda s: s.rolling(n, min_periods=1).mean()
        )
    game_position['opp_def_games_seen'] = groups.cumcount() + 1
    game_position['game_key'] = game_position['season'] * 100 + game_position['week']

    lookup = targets[['season', 'week', 'position', 'opponent_team']].copy()
    lookup['_row'] = np.arange(len(lookup))
    lookup['opponent_team'] = _standardize_team(lookup['opponent_team'])
    lookup['game_key'] = (
        pd.to_numeric(lookup['season'], errors='raise').astype(int) * 100
        + pd.to_numeric(lookup['week'], errors='raise').astype(int)
    )

    # pd.merge_asof requires sorting by the merge key, globally.
    lookup = lookup.sort_values('game_key', kind='stable')
    right = game_position[
        ['game_key', 'season', 'opponent_team', 'position', *DEFENSE_FEATURES]
    ]
    right = right.sort_values('game_key', kind='stable')
    joined = pd.merge_asof(
        lookup,
        right,
        on='game_key',
        by=['season', 'opponent_team', 'position'],
        direction='backward',
        allow_exact_matches=False,
    ).sort_values('_row')

    output = targets.copy()
    for col in DEFENSE_FEATURES:
        output[col] = joined[col].to_numpy()
    output['opp_def_games_seen'] = output['opp_def_games_seen'].fillna(0).astype(int)
    return output
