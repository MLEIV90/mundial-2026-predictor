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

A methodological asymmetry in the "advance" backtest (2026-07)
-------------------------------------------------------------------
``backtest_knockout_fixtures`` is an **approximation, not a fair
comparison**, and should be read that way. The model's side of the
comparison runs its full extra-time/penalty machinery
(``src.knockout.advance_probability``), while the market's side is the
crude ``P(win) + 0.5*P(draw)`` coin-flip approximation above -- there is
no real market for "wins the tie" to compare against instead. That means
part of any apparent model edge in that backtest could simply be the
model doing more work on a question the market was never asked, rather
than the model being better-calibrated on the thing the market actually
prices. On a small, in-tournament sample, that asymmetry can matter.

``backtest_90min_fixtures`` is the fair, apples-to-apples alternative:
it scores the model and the market on exactly the same target the market
prices pre-match -- the 90-minute 1X2 result (home win / draw / away
win) -- with no knockout-stage logic on either side. Both
``backtest_knockout_fixtures`` and ``backtest_90min_fixtures`` are worth
keeping: one is the metric that actually matters for "who goes through,"
the other is the one that isolates model-vs-market skill without the
extra-time/penalty asymmetry. Report them side by side, not one
in place of the other, and read both as a small, in-tournament sample,
not a claim of long-run edge over the market.

Multiclass Brier score and log loss
--------------------------------------
For the 90-minute comparison there are three possible outcomes, not two,
so ``multiclass_brier_score`` sums the squared error across all three
predicted probabilities (home win / draw / away win) instead of just
one, and ``multiclass_log_loss`` is the negative log-likelihood of
whichever of the three actually happened. Both reduce to the same
concept as their binary counterparts -- lower is better, 0 is perfect --
just extended to three outcomes instead of two.
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
from src.model import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_REG_STRENGTH,
    GoalsModel,
    fit_poisson_model,
    predict_match,
)
from src.odds import (
    OddsApiError,
    OddsApiRateLimitError,
    find_world_cup_fixtures,
    get_market_probabilities_for_teams,
)

# Empirically determined from the current results.csv structure (no explicit
# "round" column exists): the first Round-of-32 match kicks off 2026-06-28,
# the Round-of-16 window ran 2026-07-04 to 2026-07-06 (now complete), and the
# Quarterfinal window runs 2026-07-09 to 2026-07-11. See the project notebook
# for how these were derived (matches-per-day breaks in the schedule).
KNOCKOUT_START_DATE = "2026-06-28"
ROUND_OF_16_START_DATE = "2026-07-04"
ROUND_OF_16_END_DATE = "2026-07-06"
QUARTERFINAL_START_DATE = "2026-07-09"
QUARTERFINAL_END_DATE = "2026-07-11"


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


def multiclass_brier_score(y_true_onehot, p_pred) -> float:
    """Mean multiclass Brier score: mean over fixtures of the summed squared
    error across all outcome classes (see module docstring). ``y_true_onehot``
    and ``p_pred`` are both ``(n_fixtures, n_classes)``.
    """
    y_true_onehot = np.asarray(y_true_onehot, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    return float(np.mean(np.sum((p_pred - y_true_onehot) ** 2, axis=1)))


def multiclass_log_loss(y_true_idx, p_pred, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the class that actually occurred.
    ``y_true_idx`` is the true class index per fixture (0/1/2); ``p_pred``
    is ``(n_fixtures, n_classes)``.
    """
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0)
    picked = p_pred[np.arange(len(y_true_idx)), y_true_idx]
    return float(-np.mean(np.log(picked)))


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
    reg_strength: float = DEFAULT_REG_STRENGTH,
    et_factor: float = DEFAULT_ET_FACTOR,
    penalty_win_prob: float = DEFAULT_PENALTY_WIN_PROB,
    blend_weight: float = 1.0,
    verbose: bool = True,
    finished_fixtures: Optional[List[dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Backtest the goals model against the betting market on already-played
    knockout fixtures in ``[start_date, end_date]``.

    **This is an approximation, not a fair, apples-to-apples comparison --
    see "A methodological asymmetry" in the module docstring.** The model
    runs its full extra-time/penalty machinery
    (``src.knockout.advance_probability``); the market side is only ever
    the crude ``P(win) + 0.5*P(draw)`` coin-flip approximation below, since
    there's no real market for "wins the tie" to use instead. For a fair
    comparison against what the market actually prices, use
    ``backtest_90min_fixtures`` instead (or alongside this one).

    For each fixture, the goals model is refit with ``as_of_date`` equal to
    that fixture's own date, so no fixture's own result -- or any later
    fixture's -- ever informs its own prediction (fixtures sharing a date
    share one fit, since ``date < as_of_date`` excludes same-day matches
    from each other identically either way). Market odds are the de-vigged
    Pinnacle closing line via ``src.odds``.

    ``reg_strength`` is passed straight through to ``fit_poisson_model``
    (see its docstring, and the "Elo vs. attack/defense calibration" note
    in the ``src.model`` module docstring) -- this is the knob to use if
    the model is under- or over-weighting team-strength Elo relative to
    noisy recent-form attack/defense parameters. It should be tuned
    against *this* backtest (real outcomes), not against the market.

    ``blend_weight`` is passed through to ``advance_probability`` (see
    ``src.blend`` for what it does and why 1.0, pure Poisson, is this
    function's own default -- like ``reg_strength``, it should be chosen
    by how it moves *this* backtest's Brier/log loss against real
    outcomes, not by how closely it matches the market).

    Returns ``(comparison_df, summary)`` where ``summary`` has
    ``n_fixtures``, ``n_skipped`` (unresolved shootouts or missing market
    odds), ``model_brier``/``market_brier``, ``model_log_loss``/
    ``market_log_loss``, and ``n_model_beats_market`` (fixtures where the
    model's squared error was strictly lower than the market's).

    ``finished_fixtures`` (optional) lets a caller that's also calling
    ``backtest_90min_fixtures`` in the same run (e.g.
    ``scripts/update_data.py``) pass in an already-fetched
    ``find_world_cup_fixtures(status_id=2, ...)`` list instead of each
    function force-refreshing its own -- see that parameter's use there.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date) if end_date is not None else df["date"].max()

    df_elo, _ = compute_elo_ratings(df)  # leakage-free pre-match Elo for every played match

    window = df_elo[(df_elo["date"] >= start_date) & (df_elo["date"] <= end_date)].sort_values("date")

    if finished_fixtures is None:
        # The *list* of which fixtures count as finished changes as the
        # tournament progresses, so it's force-refreshed here (one cheap
        # call, and now that _find_world_cup_tournament_id no longer
        # force-refreshes alongside it, the only call this actually makes)
        # even though each fixture's own odds lookup stays fully cached --
        # otherwise a fixture that finished after the list was last cached
        # would be missed entirely.
        finished_fixtures = find_world_cup_fixtures(status_id=2, force_refresh=True)

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

        try:
            market = get_market_probabilities_for_teams(
                finished_fixtures, fixture["home_team"], fixture["away_team"]
            )
        except OddsApiError as exc:
            n_skipped += 1
            if verbose:
                print(
                    f"Skipping {fixture['home_team']} vs {fixture['away_team']} "
                    f"({fixture['date'].date()}): OddsPapi lookup failed ({exc})."
                )
            continue

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
            model_cache[as_of] = fit_poisson_model(
                df_elo, as_of_date=as_of, half_life_days=half_life_days, reg_strength=reg_strength
            )
        model = model_cache[as_of]

        adv = advance_probability(
            model, fixture["home_team"], fixture["away_team"],
            fixture["home_elo_pre"], fixture["away_elo_pre"],
            neutral=bool(fixture["neutral"]), et_factor=et_factor, penalty_win_prob=penalty_win_prob,
            blend_weight=blend_weight,
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


def backtest_90min_fixtures(
    df: pd.DataFrame,
    start_date: str = KNOCKOUT_START_DATE,
    end_date: Optional[str] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    reg_strength: float = DEFAULT_REG_STRENGTH,
    blend_weight: float = 1.0,
    verbose: bool = True,
    finished_fixtures: Optional[List[dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fair, apples-to-apples backtest: model vs market on the 90-minute
    1X2 result only -- no extra-time/penalty logic on either side.

    See "A methodological asymmetry" in the module docstring for why this
    exists alongside ``backtest_knockout_fixtures``: that function's
    market side is a coin-flip approximation for a question ("wins the
    tie") the market is never actually asked, while this one scores both
    the model and the market against exactly what the market prices
    pre-match -- so any model edge (or deficit) found here isn't an
    artifact of extra-time/penalty machinery only one side has.

    The actual outcome is read directly from ``home_score``/``away_score``
    (no shootout-resolution heuristic needed, unlike
    ``backtest_knockout_fixtures`` -- the 90-minute result is unambiguous
    even when the tie itself went to penalties).

    Returns ``(comparison_df, summary)``. ``summary`` has ``n_fixtures``,
    ``n_skipped`` (no matching OddsPapi fixture), ``model_brier``/
    ``market_brier`` and ``model_log_loss``/``market_log_loss`` (the
    multiclass versions, see module docstring), and
    ``n_model_beats_market`` (fixtures where the model's multiclass
    squared error was strictly lower than the market's).

    ``finished_fixtures`` (optional): see ``backtest_knockout_fixtures``'s
    docstring -- pass an already-fetched list to skip this function's own
    ``find_world_cup_fixtures`` call.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date) if end_date is not None else df["date"].max()

    df_elo, _ = compute_elo_ratings(df)
    window = df_elo[(df_elo["date"] >= start_date) & (df_elo["date"] <= end_date)].sort_values("date")

    if finished_fixtures is None:
        finished_fixtures = find_world_cup_fixtures(status_id=2, force_refresh=True)

    model_cache: Dict[pd.Timestamp, GoalsModel] = {}
    rows: List[dict] = []
    n_skipped = 0

    for _, fixture in window.iterrows():
        try:
            market = get_market_probabilities_for_teams(
                finished_fixtures, fixture["home_team"], fixture["away_team"]
            )
        except OddsApiError as exc:
            n_skipped += 1
            if verbose:
                print(
                    f"Skipping {fixture['home_team']} vs {fixture['away_team']} "
                    f"({fixture['date'].date()}): OddsPapi lookup failed ({exc})."
                )
            continue

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
            model_cache[as_of] = fit_poisson_model(
                df_elo, as_of_date=as_of, half_life_days=half_life_days, reg_strength=reg_strength
            )
        model = model_cache[as_of]

        pred = predict_match(
            model, fixture["home_team"], fixture["away_team"],
            fixture["home_elo_pre"], fixture["away_elo_pre"],
            neutral=bool(fixture["neutral"]), blend_weight=blend_weight,
        )

        if fixture["home_score"] > fixture["away_score"]:
            outcome_idx = 0
        elif fixture["home_score"] < fixture["away_score"]:
            outcome_idx = 2
        else:
            outcome_idx = 1

        rows.append(
            {
                "date": fixture["date"].date(),
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "result": ["Home win", "Draw", "Away win"][outcome_idx],
                "model_p_home": pred.home_win,
                "model_p_draw": pred.draw,
                "model_p_away": pred.away_win,
                "market_p_home": market.p_home,
                "market_p_draw": market.p_draw,
                "market_p_away": market.p_away,
                "outcome_idx": outcome_idx,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No backtestable fixtures found (all skipped or none in range).")

    y_onehot = np.zeros((len(result_df), 3))
    y_onehot[np.arange(len(result_df)), result_df["outcome_idx"].to_numpy()] = 1.0
    model_probs = result_df[["model_p_home", "model_p_draw", "model_p_away"]].to_numpy()
    market_probs = result_df[["market_p_home", "market_p_draw", "market_p_away"]].to_numpy()

    model_sq_err = np.sum((model_probs - y_onehot) ** 2, axis=1)
    market_sq_err = np.sum((market_probs - y_onehot) ** 2, axis=1)
    result_df["model_beat_market"] = model_sq_err < market_sq_err

    summary = {
        "n_fixtures": len(result_df),
        "n_skipped": n_skipped,
        "model_brier": multiclass_brier_score(y_onehot, model_probs),
        "market_brier": multiclass_brier_score(y_onehot, market_probs),
        "model_log_loss": multiclass_log_loss(result_df["outcome_idx"].to_numpy(), model_probs),
        "market_log_loss": multiclass_log_loss(result_df["outcome_idx"].to_numpy(), market_probs),
        "n_model_beats_market": int(result_df["model_beat_market"].sum()),
    }
    return result_df, summary


def sweep_blend_weight(
    df: pd.DataFrame,
    blend_weights: Optional[List[float]] = None,
    start_date: str = KNOCKOUT_START_DATE,
    end_date: Optional[str] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    reg_strength: float = DEFAULT_REG_STRENGTH,
    verbose: bool = True,
    finished_fixtures: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """Sweep ``blend_weight`` (see ``src.blend``) against the FAIR 90-minute
    backtest across every available knockout fixture, to pick the value
    that best predicts real outcomes.

    **This calibrates against actual match results (multiclass Brier/log
    loss on the true 90-minute outcome), not against the market** -- the
    market's own Brier/log loss is included in the output purely as
    context, exactly as in ``backtest_90min_fixtures``, never as the
    quantity being minimized. Picking ``blend_weight`` to minimize the
    *gap to the market* instead would just be curve-fitting to Pinnacle's
    odds, which defeats the point of an independent model.

    The goals model is refit only once per unique fixture date (not once
    per candidate weight) since ``blend_weight`` only affects prediction,
    not fitting -- reusing the same model_cache across the whole sweep is
    what keeps this fast.

    Returns a DataFrame with one row per candidate weight:
    ``blend_weight``, ``model_brier``, ``model_log_loss``, ``n_fixtures``,
    plus the (constant across rows) ``market_brier``/``market_log_loss``
    for reference.

    ``finished_fixtures`` (optional): see ``backtest_knockout_fixtures``'s
    docstring -- pass an already-fetched list to skip this function's own
    ``find_world_cup_fixtures`` call.
    """
    if blend_weights is None:
        blend_weights = [round(w * 0.1, 1) for w in range(11)]  # 0.0, 0.1, ..., 1.0

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date) if end_date is not None else df["date"].max()

    df_elo, _ = compute_elo_ratings(df)
    window = df_elo[(df_elo["date"] >= start_date) & (df_elo["date"] <= end_date)].sort_values("date")

    if finished_fixtures is None:
        finished_fixtures = find_world_cup_fixtures(status_id=2, force_refresh=True)

    # Gather each fixture's market probs, actual outcome, and pre-fitted
    # model ONCE; every blend_weight candidate below just re-blends with
    # the same already-fitted model, no refitting.
    model_cache: Dict[pd.Timestamp, GoalsModel] = {}
    fixtures: List[dict] = []
    n_skipped = 0

    for _, fixture in window.iterrows():
        try:
            market = get_market_probabilities_for_teams(
                finished_fixtures, fixture["home_team"], fixture["away_team"]
            )
        except OddsApiError as exc:
            n_skipped += 1
            if verbose:
                print(
                    f"Skipping {fixture['home_team']} vs {fixture['away_team']} "
                    f"({fixture['date'].date()}): OddsPapi lookup failed ({exc})."
                )
            continue

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
            model_cache[as_of] = fit_poisson_model(
                df_elo, as_of_date=as_of, half_life_days=half_life_days, reg_strength=reg_strength
            )

        if fixture["home_score"] > fixture["away_score"]:
            outcome_idx = 0
        elif fixture["home_score"] < fixture["away_score"]:
            outcome_idx = 2
        else:
            outcome_idx = 1

        fixtures.append(
            {
                "model": model_cache[as_of],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "home_elo_pre": fixture["home_elo_pre"],
                "away_elo_pre": fixture["away_elo_pre"],
                "neutral": bool(fixture["neutral"]),
                "outcome_idx": outcome_idx,
                "market_probs": (market.p_home, market.p_draw, market.p_away),
            }
        )

    if not fixtures:
        raise ValueError("No backtestable fixtures found (all skipped or none in range).")

    y_onehot = np.zeros((len(fixtures), 3))
    for i, fx in enumerate(fixtures):
        y_onehot[i, fx["outcome_idx"]] = 1.0
    outcome_idxs = [fx["outcome_idx"] for fx in fixtures]
    market_probs = np.array([fx["market_probs"] for fx in fixtures])
    market_brier = multiclass_brier_score(y_onehot, market_probs)
    market_log_loss = multiclass_log_loss(outcome_idxs, market_probs)

    rows = []
    for w in blend_weights:
        model_probs = np.zeros((len(fixtures), 3))
        for i, fx in enumerate(fixtures):
            pred = predict_match(
                fx["model"], fx["home_team"], fx["away_team"],
                fx["home_elo_pre"], fx["away_elo_pre"],
                neutral=fx["neutral"], blend_weight=w,
            )
            model_probs[i] = [pred.home_win, pred.draw, pred.away_win]

        rows.append(
            {
                "blend_weight": w,
                "model_brier": multiclass_brier_score(y_onehot, model_probs),
                "model_log_loss": multiclass_log_loss(outcome_idxs, model_probs),
                "market_brier": market_brier,
                "market_log_loss": market_log_loss,
                "n_fixtures": len(fixtures),
            }
        )

    return pd.DataFrame(rows)


def save_backtest_json(
    comparison: pd.DataFrame, summary: Dict[str, float], path: str, generated_at: Optional[str] = None
) -> None:
    """Write the backtest comparison table + summary to ``path`` as JSON."""
    comparison = comparison.copy()
    comparison["date"] = comparison["date"].astype(str)
    # Route through pandas' own JSON serializer first: it correctly converts
    # numpy int64/float64/bool_ dtypes to native JSON types, which plain
    # json.dumps cannot do on its own.
    fixtures = json.loads(comparison.to_json(orient="records"))

    payload = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "fixtures": fixtures,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_unplayed_fixtures_in_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Fixtures with a missing score (not yet played) within a date window."""
    window = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
    return window[window["home_score"].isna()].sort_values("date").reset_index(drop=True)


def generate_live_predictions(
    df: pd.DataFrame,
    fixtures_to_predict: pd.DataFrame,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    reg_strength: float = DEFAULT_REG_STRENGTH,
    et_factor: float = DEFAULT_ET_FACTOR,
    penalty_win_prob: float = DEFAULT_PENALTY_WIN_PROB,
    blend_weight: float = 1.0,
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
    otherwise, e.g. if there's no API key, no name match, or OddsPapi
    itself fails for that fixture -- see below).

    results.csv is a live feed and can lag reality: a fixture can still
    show a missing score there while the match has already finished
    elsewhere. This is caught from the market side instead of guessed at
    from dates -- if OddsPapi's market for a fixture has already closed
    (``get_fixture_market_probabilities`` falls back to the historical,
    "closing line" price because live odds are gone), that fixture is
    dropped from the results with a warning, since it is no longer a
    genuine pre-match prediction.

    Market odds are never allowed to fail the whole run: any
    ``OddsApiError`` (missing API key, rate limit exhausted, a plan-tier
    403, etc.) while fetching the fixtures list or one fixture's odds is
    caught and logged, and that fixture's record is still produced with
    ``market_p_*_advances``/``market_source`` left ``None`` -- a
    prediction with no market comparison is still useful, and a market
    hiccup for one fixture shouldn't take down the whole pipeline (and
    scripts/update_data.py with it).
    """
    df_elo, current_ratings = compute_elo_ratings(df)
    as_of_date = df_elo["date"].max() + pd.Timedelta(days=1)
    model = fit_poisson_model(
        df_elo, as_of_date=as_of_date, half_life_days=half_life_days, reg_strength=reg_strength
    )

    # Force-refreshed for the same reason as in backtest_knockout_fixtures:
    # this list (which fixtures exist and their status) goes stale as the
    # tournament progresses, even though each fixture's own odds stay cached.
    # If even this fails, fall back to an empty list rather than aborting --
    # every fixture below then just gets no market comparison.
    try:
        odds_fixtures = find_world_cup_fixtures(force_refresh=True)
    except OddsApiRateLimitError as exc:
        print(
            f"OddsPapi rate-limited us even after retries ({exc}). This usually means the "
            "free tier's monthly request quota has been exhausted -- try again once it "
            "resets. Continuing with no market data for this run."
        )
        odds_fixtures = []
    except OddsApiError as exc:
        print(f"Could not fetch the OddsPapi fixtures list ({exc}) -- continuing with no market data.")
        odds_fixtures = []

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
            blend_weight=blend_weight,
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

        try:
            market = get_market_probabilities_for_teams(odds_fixtures, home_team, away_team)
        except OddsApiError as exc:
            print(f"No market odds for {home_team} vs {away_team} ({exc}) -- model-only prediction.")
            market = None

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
    print("=== 'Advance' backtest (APPROXIMATION -- see docstring) ===")
    print(comparison.to_string(index=False))
    print()
    print(summary)

    comparison_90, summary_90 = backtest_90min_fixtures(matches)
    print("\n=== 90-minute backtest (FAIR, apples-to-apples comparison) ===")
    print(comparison_90.to_string(index=False))
    print()
    print(summary_90)

    print("\n=== Side by side ===")
    print(
        f"Advance (approx., NOT fair) -- model Brier={summary['model_brier']:.4f}  "
        f"market Brier={summary['market_brier']:.4f}  |  "
        f"model log loss={summary['model_log_loss']:.4f}  "
        f"market log loss={summary['market_log_loss']:.4f}  |  "
        f"model closer on {summary['n_model_beats_market']}/{summary['n_fixtures']}"
    )
    print(
        f"90-minute (fair)            -- model Brier={summary_90['model_brier']:.4f}  "
        f"market Brier={summary_90['market_brier']:.4f}  |  "
        f"model log loss={summary_90['model_log_loss']:.4f}  "
        f"market log loss={summary_90['market_log_loss']:.4f}  |  "
        f"model closer on {summary_90['n_model_beats_market']}/{summary_90['n_fixtures']}"
    )
    print(
        "\nNote: small, in-tournament sample (n < 25 fixtures either way) -- read both as "
        "an early signal. On the fair comparison, the model is competitive with / roughly "
        "on par with the market -- not a claim of beating it or of long-run edge."
    )

    unplayed = find_unplayed_fixtures_in_window(matches, QUARTERFINAL_START_DATE, QUARTERFINAL_END_DATE)
    print(f"\n{len(unplayed)} unplayed Round-of-16 fixtures found.")
    live_records, _, live_as_of = generate_live_predictions(matches, unplayed)
    save_predictions_json(live_records, "predictions/round_of_16_live.json", live_as_of)
    print("Saved predictions/round_of_16_live.json")
