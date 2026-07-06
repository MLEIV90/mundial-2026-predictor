"""Backtesting and live-prediction tracking: model vs market vs reality.

This module answers two questions:

1. On knockout fixtures that have already been played, was the goals
   model's P(advance) closer to the truth than the (de-vigged) betting
   market's implied P(advance)? ``backtest_knockout_fixtures`` walks
   forward through already-played knockout matches, refitting the goals
   model as-of each fixture's own date (so no fixture's own result, or
   any later match, ever informs its prediction), and scores both the
   model and the market against what actually happened.

2. For knockout fixtures that haven't been played yet, what does the
   model currently say, and how does that compare to the market right
   now? ``generate_live_predictions`` produces those numbers and
   ``save_predictions_json`` timestamps and commits them to a file, so
   they're a verifiable pre-match record rather than a claim made after
   the fact.

Knockout ties have no draws
----------------------------
A tied fixture's 90-minute score alone doesn't say who advanced -- that
was decided by extra time or a penalty shootout, which isn't in
results.csv. ``_resolve_actual_winner`` recovers it without needing a
separate shootouts feed: in a single-elimination bracket, the team that
actually advanced is the one that shows up in a later match. If neither
(or both) team reappears later, the tie can't be resolved from the data
available and that fixture is skipped from the backtest, with a warning.

Market probability of advancing
---------------------------------
The betting market only prices 90-minute 1X2, not "advances the
knockout tie" -- there is no such market pre-match. As a simple,
transparent approximation, a drawn 90 minutes is treated as a coin flip
for who ultimately advances:

    P(team advances) ~= P(team wins in 90') + 0.5 * P(draw in 90')

Brier score and log loss
--------------------------
Both are computed against binary "did the home team actually advance"
labels. Lower is better for both; ``brier_score`` is the mean squared
error between predicted probability and outcome (bounded in [0, 1]),
``log_loss`` is the mean negative log-likelihood of the outcome under
the predicted probability (unbounded, penalizes confident wrong calls
much more heavily).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.elo import compute_elo_ratings
from src.knockout import (
    DEFAULT_ET_FACTOR,
    DEFAULT_PENALTY_WIN_PROB,
    advance_probability,
)
from src.model import DEFAULT_HALF_LIFE_DAYS, GoalsModel, fit_poisson_model
from src.odds import find_world_cup_fixtures, get_market_probabilities_for_teams

# Empirically determined from the current results.csv structure (no explicit
# "round" column exists): the first Round-of-32 match kicks off 2026-06-28,
# and the Round-of-16 window runs 2026-07-04 to 2026-07-06. See the project
# notebook for how these were derived (matches-per-day breaks in the schedule).
KNOCKOUT_START_DATE = "2026-06-28"
ROUND_OF_16_START_DATE = "2026-07-04"
ROUND_OF_16_END_DATE = "2026-07-06"


def brier_score(y_true, p_pred) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    return float(np.mean((p_pred - y_true) ** 2))


def log_loss(y_true, p_pred, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of binary outcomes under predicted probabilities."""
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred)))


def _resolve_actual_winner(
    df: pd.DataFrame, date: pd.Timestamp, home_team: str, away_team: str, home_score: float, away_score: float
) -> Optional[str]:
    """Return which team actually advanced a knockout tie.

    A clear 90-minute winner is unambiguous. A draw means the tie was
    decided by extra time/penalties, which isn't recorded directly --
    but in a single-elimination bracket, whichever of the two teams
    plays again in a later match is the one that advanced. Returns None
    if that can't be determined (neither or both teams reappear later).
    """
    if home_score > away_score:
        return home_team
    if away_score > home_score:
        return away_team

    later = df[df["date"] > date]
    home_plays_again = ((later["home_team"] == home_team) | (later["away_team"] == home_team)).any()
    away_plays_again = ((later["home_team"] == away_team) | (later["away_team"] == away_team)).any()

    if home_plays_again and not away_plays_again:
        return home_team
    if away_plays_again and not home_plays_again:
        return away_team
    return None


def backtest_knockout_fixtures(
    df: pd.DataFrame,
    start_date: str = KNOCKOUT_START_DATE,
    end_date: Optional[str] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    et_factor: float = DEFAULT_ET_FACTOR,
    penalty_win_prob: float = DEFAULT_PENALTY_WIN_PROB,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Backtest the goals model against the betting market on already-played
    knockout fixtures in ``[start_date, end_date]``.

    For each fixture, the goals model is refit with ``as_of_date`` equal to
    that fixture's own date, so no fixture's own result -- or any later
    fixture's -- ever informs its own prediction (fixtures sharing a date
    share one fit, since ``date < as_of_date`` excludes same-day matches
    from each other identically either way). Market odds are the de-vigged
    Pinnacle closing line via ``src.odds``.

    Returns ``(comparison_df, summary)`` where ``summary`` has
    ``n_fixtures``, ``n_skipped`` (unresolved shootouts or missing market
    odds), ``model_brier``/``market_brier``, ``model_log_loss``/
    ``market_log_loss``, and ``n_model_beats_market`` (fixtures where the
    model's squared error was strictly lower than the market's).
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date) if end_date is not None else df["date"].max()

    df_elo, _ = compute_elo_ratings(df)  # leakage-free pre-match Elo for every played match

    window = df_elo[(df_elo["date"] >= start_date) & (df_elo["date"] <= end_date)].sort_values("date")

    finished_fixtures = find_world_cup_fixtures(status_id=2)  # cached after first call

    model_cache: Dict[pd.Timestamp, GoalsModel] = {}
    rows: List[dict] = []
    n_skipped = 0

    for _, fixture in window.iterrows():
        winner = _resolve_actual_winner(
            df, fixture["date"], fixture["home_team"], fixture["away_team"],
            fixture["home_score"], fixture["away_score"],
        )
        if winner is None:
            n_skipped += 1
            if verbose:
                print(
                    f"Skipping {fixture['home_team']} vs {fixture['away_team']} "
                    f"({fixture['date'].date()}): can't resolve shootout winner from data."
                )
            continue

        market = get_market_probabilities_for_teams(
            finished_fixtures, fixture["home_team"], fixture["away_team"]
        )
        if market is None:
            n_skipped += 1
            if verbose:
                print(
                    f"Skipping {fixture['home_team']} vs {fixture['away_team']} "
                    f"({fixture['date'].date()}): no matching OddsPapi fixture found."
                )
            continue

        as_of = fixture["date"]
        if as_of not in model_cache:
            model_cache[as_of] = fit_poisson_model(df_elo, as_of_date=as_of, half_life_days=half_life_days)
        model = model_cache[as_of]

        adv = advance_probability(
            model, fixture["home_team"], fixture["away_team"],
            fixture["home_elo_pre"], fixture["away_elo_pre"],
            neutral=bool(fixture["neutral"]), et_factor=et_factor, penalty_win_prob=penalty_win_prob,
        )

        p_market_home_advances = market.p_home + 0.5 * market.p_draw
        y_home_advanced = 1 if winner == fixture["home_team"] else 0

        rows.append(
            {
                "date": fixture["date"].date(),
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "winner": winner,
                "p_model_home_advances": adv.p_a_advances,
                "p_market_home_advances": p_market_home_advances,
                "y_home_advanced": y_home_advanced,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No backtestable fixtures found (all skipped or none in range).")

    model_sq_err = (result_df["p_model_home_advances"] - result_df["y_home_advanced"]) ** 2
    market_sq_err = (result_df["p_market_home_advances"] - result_df["y_home_advanced"]) ** 2
    result_df["model_beat_market"] = model_sq_err < market_sq_err

    summary = {
        "n_fixtures": len(result_df),
        "n_skipped": n_skipped,
        "model_brier": brier_score(result_df["y_home_advanced"], result_df["p_model_home_advances"]),
        "market_brier": brier_score(result_df["y_home_advanced"], result_df["p_market_home_advances"]),
        "model_log_loss": log_loss(result_df["y_home_advanced"], result_df["p_model_home_advances"]),
        "market_log_loss": log_loss(result_df["y_home_advanced"], result_df["p_market_home_advances"]),
        "n_model_beats_market": int(result_df["model_beat_market"].sum()),
    }
    return result_df, summary


def find_unplayed_fixtures_in_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Fixtures with a missing score (not yet played) within a date window."""
    window = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
    return window[window["home_score"].isna()].sort_values("date").reset_index(drop=True)


def generate_live_predictions(
    df: pd.DataFrame,
    fixtures_to_predict: pd.DataFrame,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    et_factor: float = DEFAULT_ET_FACTOR,
    penalty_win_prob: float = DEFAULT_PENALTY_WIN_PROB,
) -> Tuple[List[dict], GoalsModel, pd.Timestamp]:
    """Predict P(advance) for not-yet-played knockout fixtures, model + market.

    Fits the goals model as-of "now" (one day past the latest played
    match, so it uses every available result and nothing else -- there
    is no future data to leak against fixtures that haven't happened).
    ``fixtures_to_predict`` needs ``date``/``home_team``/``away_team``/
    ``neutral`` columns (see ``find_unplayed_fixtures_in_window``).

    Returns ``(records, model, as_of_date)``. Each record has the model's
    P(advance) for both teams, plus the market's current de-vigged
    P(advance) when a matching OddsPapi fixture is found (``None``
    otherwise, e.g. if there's no API key or no name match).

    results.csv is a live feed and can lag reality: a fixture can still
    show a missing score there while the match has already finished
    elsewhere. This is caught from the market side instead of guessed at
    from dates -- if OddsPapi's market for a fixture has already closed
    (``get_fixture_market_probabilities`` falls back to the historical,
    "closing line" price because live odds are gone), that fixture is
    dropped from the results with a warning, since it is no longer a
    genuine pre-match prediction.
    """
    df_elo, current_ratings = compute_elo_ratings(df)
    as_of_date = df_elo["date"].max() + pd.Timedelta(days=1)
    model = fit_poisson_model(df_elo, as_of_date=as_of_date, half_life_days=half_life_days)

    odds_fixtures = find_world_cup_fixtures()  # all statuses, cached after first call

    records: List[dict] = []
    for _, fixture in fixtures_to_predict.iterrows():
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        neutral = bool(fixture["neutral"])
        home_elo_pre = current_ratings.get(home_team, 1500.0)
        away_elo_pre = current_ratings.get(away_team, 1500.0)

        adv = advance_probability(
            model, home_team, away_team, home_elo_pre, away_elo_pre,
            neutral=neutral, et_factor=et_factor, penalty_win_prob=penalty_win_prob,
        )

        record = {
            "date": str(pd.Timestamp(fixture["date"]).date()),
            "home_team": home_team,
            "away_team": away_team,
            "neutral": neutral,
            "home_elo_pre": home_elo_pre,
            "away_elo_pre": away_elo_pre,
            "model_p_home_advances": adv.p_a_advances,
            "model_p_away_advances": adv.p_b_advances,
            "market_p_home_advances": None,
            "market_p_away_advances": None,
            "market_source": None,
        }

        market = get_market_probabilities_for_teams(odds_fixtures, home_team, away_team)
        if market is not None and "closing line" in market.source:
            print(
                f"Skipping {home_team} vs {away_team}: market has already closed "
                "(fixture appears to have been decided, even though results.csv "
                "still shows it as unplayed)."
            )
            continue
        if market is not None:
            record["market_p_home_advances"] = market.p_home + 0.5 * market.p_draw
            record["market_p_away_advances"] = market.p_away + 0.5 * market.p_draw
            record["market_source"] = market.source

        records.append(record)

    return records, model, as_of_date


def save_predictions_json(
    records: List[dict], path: str, as_of_date: pd.Timestamp, generated_at: Optional[str] = None
) -> None:
    """Write timestamped predictions to ``path`` as a verifiable pre-match record."""
    payload = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "model_as_of_date": str(pd.Timestamp(as_of_date).date()),
        "predictions": records,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    from src.data import load_results

    matches = load_results()

    comparison, summary = backtest_knockout_fixtures(matches)
    print(comparison.to_string(index=False))
    print()
    print(summary)

    unplayed = find_unplayed_fixtures_in_window(matches, ROUND_OF_16_START_DATE, ROUND_OF_16_END_DATE)
    print(f"\n{len(unplayed)} unplayed Round-of-16 fixtures found.")
    live_records, _, live_as_of = generate_live_predictions(matches, unplayed)
    save_predictions_json(live_records, "predictions/round_of_16_live.json", live_as_of)
    print("Saved predictions/round_of_16_live.json")
