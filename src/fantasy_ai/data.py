"""Download weekly player statistics and create a local CSV cache."""

import argparse
from pathlib import Path

import nflreadpy as nfl


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def download_weekly_stats(start_season: int, end_season: int) -> Path:
    """Download nflverse weekly player stats and save a raw parquet snapshot."""
    seasons = list(range(start_season, end_season + 1))
    print(f"Downloading weekly NFL player stats for {seasons} ...")
    stats = nfl.load_player_stats(seasons, summary_level="week")
    frame = stats.to_pandas()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output = RAW_DIR / f"weekly_player_stats_{start_season}_{end_season}.parquet"
    frame.to_parquet(output, index=False)
    print(f"Saved {len(frame):,} rows to {output}")
    print("Available columns:", ", ".join(frame.columns[:30]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=2026)
    args = parser.parse_args()
    if args.start_season > args.end_season:
        parser.error("--start-season must be <= --end-season")
    download_weekly_stats(args.start_season, args.end_season)


if __name__ == "__main__":
    main()
