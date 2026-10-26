"""Combine injury reports, weekly roster statuses and dated manual notes.

This is a filtering/annotation layer, not a numeric fantasy projection adjuster.
Manual notes must be documented with as-of date and source URL.
"""
from pathlib import Path
import pandas as pd

OVERRIDE_FIELDS = [
    'season', 'week', 'player_id', 'player_name', 'team',
    'status', 'reason', 'source_url', 'as_of',
]
ALLOWED_MANUAL = {'Out', 'Questionable', 'Doubtful', 'Injury concern', 'Role concern', 'Unverified'}


def _normalize_name(value):
    return ' '.join(str(value).casefold().strip().split())


def apply_availability(
    projections: pd.DataFrame,
    season: int,
    week: int,
    overrides_path: str | Path | None = None,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Return projections with status and an explicit exclusion flag.

    Expected input fields: player_id, player_name, team, injury_status,
    plus roster_status and roster_week when available.

    Statuses from prior-week reserve rosters may be provisional. Manual
    overrides support situations that injury-practice feeds miss.

    as_of restricts overrides by their publication date for historical use.
    A production backtest must also use as-of snapshots for all other feeds.
    """
    required = {'player_id', 'player_name', 'team'}
    missing = required - set(projections.columns)
    if missing:
        raise ValueError(f'Missing availability keys: {sorted(missing)}')

    result = projections.copy()
    if result['player_id'].astype(str).duplicated().any():
        raise ValueError('Availability rows must have one record per player_id')

    result['availability_status'] = 'Unverified'
    result['availability_reason'] = 'No confirmed current-week availability information'
    result['availability_source'] = ''

    # Game designation, not practice participation. Unknown != healthy.
    if 'injury_status' in result:
        statuses = result['injury_status'].fillna('').astype(str).str.strip().str.casefold()
        recognized = {'out': 'Out', 'doubtful': 'Doubtful', 'questionable': 'Questionable'}
        for raw, display in recognized.items():
            flag = statuses.eq(raw)
            result.loc[flag, 'availability_status'] = display
            result.loc[flag, 'availability_reason'] = 'Current-week injury report game designation'

    # 'RES' in a fresh roster indicates on reserve; 'CUT' no longer on the team.
    # Do NOT treat last week's 'INA' as inactive this coming week.
    if 'roster_status' in result:
        code = result['roster_status'].astype('string').str.strip().str.upper()
        age = (week - pd.to_numeric(result['roster_week'], errors='coerce')) if 'roster_week' in result else pd.Series(pd.NA, index=result.index, dtype='Float64')
        fresh = age.between(0, 1).fillna(False)
        reserve = code.isin(['RES', 'CUT']).fillna(False)
        confirmed = reserve & fresh
        result.loc[confirmed, 'availability_status'] = 'Out'
        result.loc[confirmed, 'availability_reason'] = 'Latest weekly roster lists player as reserve or cut; recheck for activation'
        old_reserve = reserve & ~fresh
        # An old reserve code deserves a flag, not an automatic exclusion.
        result.loc[old_reserve & result['availability_status'].eq('Unverified'), 'availability_status'] = 'Roster review'
        result.loc[old_reserve, 'availability_reason'] = 'Stale reserve/cut roster entry; verify current team status'

    if overrides_path is not None:
        path = Path(overrides_path)
        if path.exists():
            notes = pd.read_csv(path, dtype=str, keep_default_na=False)
            missing_fields = set(OVERRIDE_FIELDS) - set(notes.columns)
            if missing_fields:
                raise ValueError(f'Override CSV columns missing: {sorted(missing_fields)}')
            notes = notes.loc[
                notes['season'].eq(str(season)) & notes['week'].eq(str(week))
            ].copy()
            if as_of is not None:
                cutoff = pd.Timestamp(as_of)
                pub = pd.to_datetime(notes['as_of'], errors='coerce')
                if pub.isna().any():
                    raise ValueError('Every matching override must have a valid as_of date')
                notes = notes.loc[pub <= cutoff]

            for row in notes.itertuples(index=False):
                if row.status not in ALLOWED_MANUAL:
                    raise ValueError(f'Unsupported availability status: {row.status!r}')
                if not row.source_url or not row.as_of or not row.reason:
                    raise ValueError(f'Override requires source, as_of, reason: {row.player_name}')

                # Prefer a stable GSIS ID. Fall back to exact name+team, only if unique.
                if str(row.player_id).strip():
                    flag = result['player_id'].astype(str).eq(str(row.player_id).strip())
                else:
                    flag = (
                        result['player_name'].map(_normalize_name).eq(_normalize_name(row.player_name))
                        & result['team'].astype(str).str.upper().eq(row.team.strip().upper())
                    )
                if flag.sum() != 1:
                    raise ValueError(
                        f'Override matched {int(flag.sum())} players for '
                        f'{row.player_name} / {row.team}; use a valid GSIS player_id'
                    )
                result.loc[flag, 'availability_status'] = row.status
                result.loc[flag, 'availability_reason'] = row.reason
                result.loc[flag, 'availability_source'] = row.source_url

    result['exclude_from_rankings'] = result['availability_status'].eq('Out')
    return result
