"""Knockout-stage layer: converts match-level goal rates into P(advances).

In a World Cup knockout tie there are no draws: a 90-minute draw goes to
30 minutes of extra time, and if that is still level, a penalty shootout
decides the winner. This module turns ``src.model``'s regulation-time
score matrix into a single "who advances" probability by adding those
two extra stages on top.

    P(A advances) = P(A wins in 90')
                  + P(draw in 90') * P(A wins extra time)
                  + P(draw in 90') * P(level after ET) * P(A wins penalties)

Extra time
----------
The 30 extra minutes are modeled as more goals from the same two Poisson
processes as the full match, just scaled down to roughly the fraction of
a match they represent: ``lambda_et = lambda_90 * et_factor``, with
``et_factor`` defaulting to 0.33 (30 minutes is 1/3 of 90). The same
Dixon-Coles rho from the fitted model is reused, since low-scoring
periods are exactly where that correction matters most. This gives
P(A wins ET) / P(B wins ET) / P(still level after ET).

Penalties
---------
If the tie is still level after extra time, it is decided by a penalty
shootout. There is no goals model for that, so it is treated as a coin
flip by default: ``penalty_win_prob`` (probability team A wins the
shootout) defaults to 0.5. It is a free, adjustable parameter so it can
later be replaced with a calibrated estimate (e.g. derived from historic
shootout data such as shootouts.csv) without changing anything else here.

All of this only ever consumes pre-match Elo ratings and a fitted
``GoalsModel`` -- it adds no new sources of leakage on top of what
``src.elo``/``src.model`` already guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.model import GoalsModel, expected_goal_rates, score_matrix_and_outcomes

DEFAULT_ET_FACTOR = 0.33
DEFAULT_PENALTY_WIN_PROB = 0.5


@dataclass
class AdvanceResult:
    """P(advances) for both sides of a knockout tie, with a full breakdown.

    Each stage's fields are *unconditional* probabilities (i.e. already
    multiplied through by the probability of reaching that stage), so
    ``p_a_win_90 + p_a_win_et + p_a_win_penalties == p_a_advances`` and
    ``p_a_advances + p_b_advances == 1``.
    """

    team_a: str
    team_b: str

    p_a_advances: float
    p_b_advances: float

    p_a_win_90: float
    p_draw_90: float
    p_b_win_90: float

    p_a_win_et: float
    p_b_win_et: float
    p_level_after_et: float

    p_a_win_penalties: float
    p_b_win_penalties: float


def advance_probability(
    model: GoalsModel,
    team_a: str,
    team_b: str,
    elo_a: float,
    elo_b: float,
    neutral: bool = True,
    et_factor: float = DEFAULT_ET_FACTOR,
    penalty_win_prob: float = DEFAULT_PENALTY_WIN_PROB,
    max_goals: int = 10,
) -> AdvanceResult:
    """Compute P(team_a advances) and P(team_b advances) from a knockout tie.

    Parameters
    ----------
    model:
        A fitted ``src.model.GoalsModel``.
    team_a, team_b:
        The two teams in the tie. ``team_a`` fills the "home" slot of the
        underlying goals model (only relevant if ``neutral`` is False).
    elo_a, elo_b:
        Pre-match Elo ratings for each team (e.g. from
        ``src.elo.compute_elo_ratings``).
    neutral:
        Whether the venue is neutral (the World Cup default -- most
        knockout matches are neutral-venue). Passed straight through to
        the goals model, which zeroes out home advantage when True.
    et_factor:
        Fraction of a full match that extra time represents, applied to
        scale down both teams' goal rates for the extra-time stage.
        Default 0.33 (30 of 90 minutes).
    penalty_win_prob:
        Probability team_a wins a penalty shootout, used only if the tie
        is still level after extra time. Default 0.5 (coin flip).
    max_goals:
        Max goals per side considered when building the regulation and
        extra-time score matrices.

    Returns
    -------
    An ``AdvanceResult`` with ``p_a_advances``/``p_b_advances`` (summing
    to 1) plus the full stage-by-stage breakdown.
    """
    if not 0.0 <= penalty_win_prob <= 1.0:
        raise ValueError(f"penalty_win_prob must be in [0, 1], got {penalty_win_prob}")
    if et_factor <= 0.0:
        raise ValueError(f"et_factor must be positive, got {et_factor}")

    lambda_a_90, lambda_b_90 = expected_goal_rates(model, team_a, team_b, elo_a, elo_b, neutral)
    _, p_a_win_90, p_draw_90, p_b_win_90 = score_matrix_and_outcomes(
        lambda_a_90, lambda_b_90, model.rho, max_goals=max_goals
    )

    lambda_a_et = lambda_a_90 * et_factor
    lambda_b_et = lambda_b_90 * et_factor
    _, p_a_win_et_given_draw, p_level_given_draw, p_b_win_et_given_draw = score_matrix_and_outcomes(
        lambda_a_et, lambda_b_et, model.rho, max_goals=max_goals
    )

    p_a_win_et = p_draw_90 * p_a_win_et_given_draw
    p_b_win_et = p_draw_90 * p_b_win_et_given_draw
    p_level_after_et = p_draw_90 * p_level_given_draw

    p_a_win_penalties = p_level_after_et * penalty_win_prob
    p_b_win_penalties = p_level_after_et * (1.0 - penalty_win_prob)

    p_a_advances = p_a_win_90 + p_a_win_et + p_a_win_penalties
    p_b_advances = p_b_win_90 + p_b_win_et + p_b_win_penalties

    return AdvanceResult(
        team_a=team_a,
        team_b=team_b,
        p_a_advances=p_a_advances,
        p_b_advances=p_b_advances,
        p_a_win_90=p_a_win_90,
        p_draw_90=p_draw_90,
        p_b_win_90=p_b_win_90,
        p_a_win_et=p_a_win_et,
        p_b_win_et=p_b_win_et,
        p_level_after_et=p_level_after_et,
        p_a_win_penalties=p_a_win_penalties,
        p_b_win_penalties=p_b_win_penalties,
    )


if __name__ == "__main__":
    import pandas as pd

    from src.data import load_results
    from src.elo import compute_elo_ratings
    from src.model import fit_poisson_model

    matches = load_results()
    matches_with_elo, current_ratings = compute_elo_ratings(matches)

    as_of = matches_with_elo["date"].max() + pd.Timedelta(days=1)
    model = fit_poisson_model(matches_with_elo, as_of_date=as_of)

    team_a, team_b = "Argentina", "Brazil"
    result = advance_probability(
        model,
        team_a,
        team_b,
        current_ratings.get(team_a, 1500.0),
        current_ratings.get(team_b, 1500.0),
    )
    print(
        f"{team_a} vs {team_b} (knockout): "
        f"P({team_a} advances)={result.p_a_advances:.1%}, "
        f"P({team_b} advances)={result.p_b_advances:.1%}"
    )
