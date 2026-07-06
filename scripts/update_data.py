"""Refresh every data file the Streamlit app (app.py) reads.

Run manually after each new match result:

    python scripts/update_data.py

This is the ONLY part of the project that calls the OddsPapi API (via
src.odds, transparently cached under data/odds_cache/). app.py itself
never touches the network -- it only reads the JSON files this script
writes under data/app/, so viewing the deployed app costs zero API
requests no matter how many times it's opened.

Writes:
    data/app/model_ratings.json    -- current Elo top teams
    data/app/backtest.json         -- model vs market on played knockout fixtures
    data/app/live_predictions.json -- model vs market on upcoming knockout fixtures

data/app/ is committed to git (see the !data/app/ exception in
.gitignore) since the deployed app needs these files to exist.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results
from src.elo import compute_elo_ratings, top_teams
from src.evaluation import (
    KNOCKOUT_START_DATE,
    ROUND_OF_16_END_DATE,
    ROUND_OF_16_START_DATE,
    backtest_knockout_fixtures,
    find_unplayed_fixtures_in_window,
    generate_live_predictions,
    save_backtest_json,
    save_predictions_json,
)

APP_DATA_DIR = Path("data/app")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat()

    print("Loading match results...")
    matches = load_results()

    print("Computing Elo ratings...")
    df_elo, current_ratings = compute_elo_ratings(matches)
    ratings_path = APP_DATA_DIR / "model_ratings.json"
    # Route through pandas' own JSON serializer: it converts numpy float64
    # dtypes to native JSON numbers, which plain dict-then-json.dumps can't.
    ratings_records = json.loads(top_teams(current_ratings, n=30).to_json(orient="records"))
    _write_json(
        ratings_path,
        {
            "generated_at": generated_at,
            "as_of_date": str(df_elo["date"].max().date()),
            "ratings": ratings_records,
        },
    )
    print(f"  wrote {ratings_path}")

    print("Backtesting the model against the market on already-played knockout fixtures...")
    print("(refits the goals model per fixture date -- this takes a couple of minutes)")
    comparison, summary = backtest_knockout_fixtures(matches, start_date=KNOCKOUT_START_DATE)
    backtest_path = APP_DATA_DIR / "backtest.json"
    save_backtest_json(comparison, summary, str(backtest_path), generated_at=generated_at)
    print(f"  wrote {backtest_path}")
    print(
        f"  model Brier={summary['model_brier']:.4f} vs market Brier={summary['market_brier']:.4f} | "
        f"model beat market on {summary['n_model_beats_market']}/{summary['n_fixtures']} fixtures"
    )

    print("Generating live predictions for upcoming round-of-16 fixtures...")
    unplayed = find_unplayed_fixtures_in_window(matches, ROUND_OF_16_START_DATE, ROUND_OF_16_END_DATE)
    live_records, _, live_as_of = generate_live_predictions(matches, unplayed)
    live_path = APP_DATA_DIR / "live_predictions.json"
    save_predictions_json(live_records, str(live_path), live_as_of, generated_at=generated_at)
    print(f"  wrote {live_path} ({len(live_records)} fixtures)")

    print("\nDone. Commit data/app/*.json to publish these results to the app.")


if __name__ == "__main__":
    main()
