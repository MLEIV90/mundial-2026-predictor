"""Blend the Dixon-Coles goals model with pure Elo to correct favorite bias.

Why this exists
----------------
The Poisson goals model (``src.model``) regularizes team strength toward
Elo via ``reg_strength`` (see that module's "Calibration finding" section),
but even after that fix it stays conservative on clear favorites: it gives
France ~39% to beat Morocco in regulation, while the market (Pinnacle)
gives ~60%. The per-team attack/defense parameters -- even shrunk -- pull
mismatched games back toward a draw more than a strength gap this size
warrants. Pure Elo, using the classical expected-score formula, captures
that gap directly and doesn't have this bias.

**This module is a principled ensemble of two legitimate signals, not an
attempt to replicate bookmaker odds.** The blend weight is chosen (and
should be re-tuned) by backtesting against real match outcomes -- see
``src.evaluation.sweep_blend_weight`` -- never by minimizing the gap to
the market directly. The market is used only as a diagnostic in that
sweep, to see how the model's own Brier/log-loss compares in context,
not as the thing being fit.

The blend
---------
    P_final = w * P_poisson + (1 - w) * P_elo

for each of P(home win), P(draw), P(away win) independently. Both
components already sum to 1, so the blend does too automatically (a
convex combination), no renormalization needed. ``blend_weight=1.0``
recovers the pure-Poisson model exactly, kept available throughout
``src.model``/``src.knockout`` for comparison.

Calibration finding (2026-07-07)
---------------------------------
``sweep_blend_weight`` was run over ``blend_weight in [0.0, 0.1, ..., 1.0]``
against the FAIR 90-minute 1X2 backtest across all 22 knockout fixtures
with cached market odds (``data/app/backtest_90min.json``). The model's
own multiclass Brier score against the *actual* 90-minute outcome
increased smoothly and monotonically from 0.365 at ``blend_weight=0.0``
to 0.413 at ``blend_weight=1.0`` -- i.e. pure Elo (0.0) minimized Brier
on this sample, with no interior optimum.

**``DEFAULT_BLEND_WEIGHT`` is deliberately set to 0.5, not 0.0.** Taking
the raw sweep minimizer at face value would mean concluding, from 22
fixtures, that the Poisson goals model adds no value at all to the 1X2
outcome -- a boundary result on a sample this small deserves more
skepticism than an interior one would, since it's just as consistent
with "this particular knockout bracket happened to favor Elo" as with
"the Poisson model has no signal here." Both signals are legitimately
informative (Elo from long-run team strength, Poisson from actual
goal-scoring dynamics plus the Dixon-Coles low-score correlation), so an
even 50/50 blend is the principled choice absent stronger evidence for
a corner solution -- it captures most of the favorite-bias correction
the sweep points toward without fully discarding the goals model.

Re-run the sweep (``scripts/calibrate_blend_weight.py``) as more results
come in, and revisit this value once the sample is large enough that a
boundary optimum (if it persists) can be trusted.

Deriving win/draw/loss from Elo
----------------------------------
The classical Elo "expected score" is

    E = 1 / (1 + 10^(-elo_diff / 400))

which represents P(win) + 0.5 * P(draw) under the classical (chess-derived)
assumption that a win and two draws are equivalent. To split that into
P(win)/P(draw)/P(loss) separately, this module models the draw
probability as a fraction of a fitted baseline rate that peaks when the
match is a toss-up and vanishes as it becomes lopsided:

    P(draw) = base_draw_rate * 4 * E * (1 - E)

``4*E*(1-E)`` is 1 exactly at E=0.5 (an even match) and falls smoothly to
0 as E -> 0 or E -> 1 (a near-certainty), which matches the well-known
empirical pattern that draws are rare in blowouts. ``base_draw_rate`` is
not a free/tuned knob -- it's the actual (time-decay-weighted) draw rate
observed in the same training window used to fit the goals model (see
``src.model.fit_poisson_model``), so it moves with the data rather than
being hand-picked.

P(win) is then set to preserve the classical expected-score identity
exactly (``P(win) + 0.5*P(draw) = E``), which also guarantees
P(win)+P(draw)+P(loss) sums to 1 and both are non-negative for any
``base_draw_rate`` in a sane range (which historical international
football draw rates, ~20-28%, comfortably are):

    P(win) = E - 0.5 * P(draw)
    P(loss) = 1 - P(win) - P(draw)

Home advantage in the Elo comparison, for non-neutral matches, is a
fixed, commonly-used convention (``ELO_HOME_ADVANTAGE = 100`` rating
points added to the home team before computing ``elo_diff``) -- not
fitted, since pure Elo has no notion of a home-advantage parameter of its
own the way the goals model does.
"""

from __future__ import annotations

from typing import Tuple

DEFAULT_BLEND_WEIGHT = 0.5
ELO_HOME_ADVANTAGE = 100.0


def elo_match_probabilities(
    elo_home: float,
    elo_away: float,
    base_draw_rate: float,
    neutral: bool = True,
    home_advantage_elo: float = ELO_HOME_ADVANTAGE,
) -> Tuple[float, float, float]:
    """Pure-Elo P(home win), P(draw), P(away win) for one match.

    See the module docstring for the derivation. ``base_draw_rate`` should
    be the (time-decay-weighted) empirical draw rate from the same
    training window as the goals model being blended with --
    ``GoalsModel.base_draw_rate`` after fitting.
    """
    effective_home_elo = elo_home + (0.0 if neutral else home_advantage_elo)
    elo_diff = effective_home_elo - elo_away

    expected = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    p_draw = base_draw_rate * 4.0 * expected * (1.0 - expected)
    p_home = expected - 0.5 * p_draw
    p_away = 1.0 - p_home - p_draw
    return p_home, p_draw, p_away


def blend_outcome_probabilities(
    poisson_probs: Tuple[float, float, float],
    elo_probs: Tuple[float, float, float],
    blend_weight: float = DEFAULT_BLEND_WEIGHT,
) -> Tuple[float, float, float]:
    """Convex-combine (P(home), P(draw), P(away)) from the two signals.

    ``blend_weight=1.0`` returns ``poisson_probs`` unchanged;
    ``blend_weight=0.0`` returns pure Elo. The default (0.5, an even
    blend) is a deliberate choice, not the sweep's raw minimizer -- see
    the "Calibration finding" section of the module docstring for why.
    """
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError(f"blend_weight must be in [0, 1], got {blend_weight}")
    return tuple(
        blend_weight * p_poisson + (1.0 - blend_weight) * p_elo
        for p_poisson, p_elo in zip(poisson_probs, elo_probs)
    )
