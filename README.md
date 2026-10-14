# Fantasy Football AI

A Python-first fantasy football analytics project. The initial milestone downloads weekly NFL player statistics from nflverse, calculates PPR points, creates leakage-safe historical features, trains a baseline machine-learning model, and reports a chronological holdout evaluation.

## Features in v0.1

- Pull weekly player statistics with [`nflreadpy`](https://github.com/nflverse/nflreadpy).
- Calculate standard PPR fantasy points from passing, rushing, and receiving stats.
- Create rolling player averages using *prior games only*.
- Train a Random Forest baseline.
- Evaluate using a chronological holdout split and MAE/RMSE.
- Save a model and a CSV of historical predictions.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Download seasons 2018 through 2026 (the current season may be incomplete):

```powershell
python -m src.fantasy_ai.data --start-season 2018 --end-season 2026
```

Train and evaluate:

```powershell
python -m src.fantasy_ai.train
```

The first data download requires internet and can take a few minutes. `nflreadpy` returns Polars data; this project converts it to pandas for modeling.

## Project layout

```text
fantasy-football-ai/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/fantasy_ai/
│   ├── __init__.py
│   ├── scoring.py
│   ├── data.py
│   └── train.py
├── tests/
│   └── test_scoring.py
├── requirements.txt
└── README.md
```

## PPR scoring assumptions

- Passing yards: 1 point per 25 yards
- Passing TD: 4 points
- Interception: -2 points
- Rushing/receiving yards: 0.1 point per yard
- Rushing/receiving TD: 6 points
- Reception: 1 point
- Fumble lost: -2 points

This is a configurable baseline, not a claim that one scoring system fits every league. Adjust `calculate_ppr_points` for your league's rules.

## Modeling notes

- The target is actual weekly PPR points.
- Rolling features are grouped by stable player ID and shifted by one game before calculating averages, so the target week's stats are not used as inputs.
- The holdout is chronological, not randomly shuffled.
- Players with only one historical appearance have no previous-game feature row and are excluded from model training until we add a principled rookie/low-data fallback.
- This is an early baseline, not yet a production-quality weekly projection system. A future version should add snap counts, targets/carries, opponent defense, injury status, uncertainty intervals, and a rolling week-by-week backtest.

## Data attribution

Player statistics are sourced from the nflverse open-data ecosystem. Check the source project's current data documentation and licensing before redistributing data or building a public hosted service.
