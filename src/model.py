"""Dixon-Coles style bivariate Poisson goals model.

This is the project's baseline match-level model: it predicts the full
distribution of a match's scoreline, not just win/draw/loss, by modeling
home and away goals as (correlated) Poisson random variables.

Rate equations
--------------
For a match between a home team H and an away team A, with
``elo_diff = home_elo_pre - away_elo_pre`` (the pre-match Elo gap from
``src.elo.compute_elo_ratings``):

    log(lambda_home) = intercept + attack[H] + defense[A]
                        + home_advantage * (not neutral)
                        + beta_elo * elo_diff
    log(lambda_away) = intercept + attack[A] + defense[H]
                        - beta_elo * elo_diff

``attack``/``defense`` are per-team strength parameters, ``intercept`` is
the shared baseline scoring rate, ``home_advantage`` is a fixed bonus
applied to the home side (set to 0 automatically for ``neutral`` matches,
e.g. most FIFA World Cup games), and ``beta_elo`` lets the current Elo gap
push the expected scoreline on top of the teams' long-run attack/defense
levels.

``attack``/``defense`` are only identified up to a shared additive shift
(raising every team's attack by c and lowering every defense by c leaves
every match's rates unchanged), so an L2 penalty (``reg_strength``) is
added on both during fitting. This pins down that shift and, as a side
effect, shrinks teams with little match history toward the average team
(attack=defense=0) -- a sensible default for sparsely observed sides. A
team that never appears in the training window falls back to
attack=defense=0 (an average team) at prediction time.

Calibration finding: Elo vs. attack/defense (2026-07)
--------------------------------------------------------
With the old default (``reg_strength=0.01``), the model gave France only
32% to beat Morocco in regulation despite a ~150-point Elo edge, against
60% from the market (Pinnacle) -- a 28-point gap. The cause: Morocco's
defense parameter had fit to an exceptionally strong recent scoring
record (a handful of low-scoring matches this tournament, weighted
heavily by the time-decay), which is real signal but from a very small,
noisy sample -- and it was swamping the Elo term's contribution, which
is estimated from the team's *entire* Elo history and is comparatively
well-calibrated.

``reg_strength`` was swept from 0.01 to 50 and scored against
already-played knockout fixtures (``src.evaluation.backtest_knockout_fixtures``,
scored against actual outcomes, *not* against the market -- see that
function's docstring for why the market must not be the tuning target).
0.01 (Brier 0.1522) was the worst value tested; **10.0 was the best**
(Brier 0.1416, log loss 0.4501) among the candidates tried, which is why
it's the default now -- **this Brier score is only a comparison of
``reg_strength`` values against each other, never a comparison against
the market's own Brier score** (see the paragraph below for that
comparison, done properly). The curve isn't perfectly smooth (20
fixtures is a small backtest), but the improvement from roughly 1.0
upward is consistent enough to trust: 3.0 -> 0.1436, 8.0 -> 0.1429,
10.0 -> 0.1416, 20.0 -> 0.1449, 50.0 -> 0.1480 (degrading again).

Note: ``backtest_knockout_fixtures`` is itself only an *approximation* of
a model-vs-market comparison, not a fair one (see its docstring and
``src.evaluation``'s module docstring for why) -- so the numbers above
say which ``reg_strength`` is best *among themselves*, not that the
model beats the market. On the fair, apples-to-apples 90-minute 1X2
backtest (``backtest_90min_fixtures``, at the default
``blend_weight=0.5`` -- see ``src.blend``'s "Calibration finding"
section for why the sweep's raw minimizer of 0.0 was deliberately not
used as-is), the model is competitive with / roughly on par with the
market -- Brier 0.3840 vs the market's 0.4068 over 22 fixtures,
essentially a tie on this small in-tournament sample, **not** evidence
of being superior to it.

What raising ``reg_strength`` does *not* do: fitted ``beta_elo`` turns
out to be essentially constant (~0.0019) across that entire range -- it
is not being "crowded out" by attack/defense in some fixable
multicollinearity sense. The improvement comes entirely from shrinking
attack/defense toward 0, which lets the (unchanged) Elo term matter
*relatively* more. At the new default, France's chance of beating
Morocco rises from 32% to 39.5% -- a real improvement, but well short of
the market's 60%. Pushing further (``reg_strength=50``) narrows it a bit
more (47%) but makes the aggregate backtest *worse* than at 10.0, because
by then the model is discarding genuine, currently-informative team form
(e.g. a team on a hot streak Elo hasn't caught up to yet) along with the
noise. Fully closing this one anecdote would mean overfitting to it (or
to the market) at the expense of every other matchup, which is exactly
what ``reg_strength`` should not be tuned to do.

Leakage-free fitting
---------------------
``fit_poisson_model`` takes an explicit ``as_of_date`` and restricts
training data to matches strictly before it -- nothing at or after that
date (including its own goal-difference-derived features) ever informs
the fit. Combined with ``src.elo.compute_elo_ratings``'s own leakage-free
pre-match ratings, the whole pipeline (Elo -> goals model -> prediction)
never uses information that would not have been available at the time of
the match being predicted.

Time-decay weighting
---------------------
Each training match is weighted by recency:

    weight = exp(-ln(2) * age_in_days / half_life_days)

so a match exactly ``half_life_days`` old (730 days / 24 months by
default) counts half as much as a match played today, one two half-lives
old counts a quarter as much, and so on. ``half_life_days`` is fully
adjustable. Matches whose weight falls below ``min_weight`` are dropped
before fitting purely as a numerical optimization (they are, by
construction, too decayed to move the likelihood in any measurable way).

Dixon-Coles low-score correction
----------------------------------
Plain independent Poisson models underestimate how often low scorelines
(0-0, 1-0, 0-1, 1-1) actually occur, because in practice they are
slightly correlated. Following Dixon & Coles (1997), the joint
probability of scoreline (x, y) is corrected by a multiplier tau:

    tau(0,0) = 1 - lambda_home * lambda_away * rho
    tau(0,1) = 1 + lambda_home * rho
    tau(1,0) = 1 + lambda_away * rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1                          for every other scoreline

``rho`` is a free, adjustable parameter. Setting ``rho = 0`` makes
``tau`` identically 1 for every scoreline, which reduces the model
exactly to independent Poisson -- so the same code path can be used to
fit and compare both variants (pass ``fit_rho=False, rho=0.0`` for the
independent-Poisson baseline, or ``fit_rho=True`` to estimate rho).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from src.blend import blend_outcome_probabilities, elo_match_probabilities

DEFAULT_HALF_LIFE_DAYS = 730.0
DEFAULT_REG_STRENGTH = 10.0
DEFAULT_MIN_WEIGHT = 1e-6
TAU_FLOOR = 1e-6

REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral",
    "home_elo_pre",
    "away_elo_pre",
)


def time_decay_weight(age_days, half_life_days: float = DEFAULT_HALF_LIFE_DAYS):
    """Return exp(-ln(2) * age_days / half_life_days), vectorized."""
    age_days = np.asarray(age_days, dtype=float)
    return np.exp(-np.log(2.0) * age_days / half_life_days)


def dixon_coles_tau(x, y, lambda_home, lambda_away, rho: float):
    """Vectorized Dixon-Coles low-score correction tau(x, y).

    ``rho = 0`` returns 1 everywhere, i.e. independent Poisson.
    """
    if rho == 0.0:
        return np.ones_like(np.asarray(lambda_home, dtype=float))

    x = np.asarray(x)
    y = np.asarray(y)
    lambda_home = np.asarray(lambda_home, dtype=float)
    lambda_away = np.asarray(lambda_away, dtype=float)

    tau = np.ones_like(lambda_home)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lambda_home[m00] * lambda_away[m00] * rho
    tau[m01] = 1.0 + lambda_home[m01] * rho
    tau[m10] = 1.0 + lambda_away[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


@dataclass
class GoalsModel:
    """A fitted Dixon-Coles-style bivariate Poisson goals model.

    ``base_draw_rate`` is the time-decay-weighted empirical draw rate in
    this model's own training window -- not a modeling parameter of the
    Poisson model itself, but carried on the fitted model so
    ``src.blend`` can derive pure-Elo win/draw/loss probabilities from
    the same data window without needing the raw match history passed
    around separately.
    """

    teams: List[str]
    attack: Dict[str, float]
    defense: Dict[str, float]
    intercept: float
    home_advantage: float
    beta_elo: float
    rho: float
    half_life_days: float
    as_of_date: pd.Timestamp
    n_matches_used: int
    converged: bool
    base_draw_rate: float

    def get_attack(self, team: str) -> float:
        """Attack strength for ``team``, or 0.0 (average team) if unseen."""
        return self.attack.get(team, 0.0)

    def get_defense(self, team: str) -> float:
        """Defense strength for ``team``, or 0.0 (average team) if unseen."""
        return self.defense.get(team, 0.0)


def _check_required_columns(matches: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in matches.columns]
    if missing:
        raise KeyError(
            f"matches is missing required column(s) {missing}. "
            "Pass the DataFrame returned by src.elo.compute_elo_ratings."
        )


def fit_poisson_model(
    matches: pd.DataFrame,
    as_of_date,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    rho: float = 0.0,
    fit_rho: bool = True,
    reg_strength: float = DEFAULT_REG_STRENGTH,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_iter: int = 2000,
) -> GoalsModel:
    """Fit the bivariate Poisson goals model by maximum likelihood.

    Only matches with ``date`` strictly before ``as_of_date`` are used --
    this is the leakage-free boundary: to predict a match, call this with
    ``as_of_date`` equal to that match's date (or earlier).

    Parameters
    ----------
    matches:
        Match history with the columns produced by
        ``src.elo.compute_elo_ratings`` (``home_elo_pre``/``away_elo_pre``
        included).
    as_of_date:
        Cutoff date (exclusive). Training uses ``matches["date"] < as_of_date``.
    half_life_days:
        Time-decay half-life in days (see module docstring).
    rho:
        Dixon-Coles correlation parameter. Used as-is if ``fit_rho`` is
        False, otherwise used as the optimizer's starting value.
    fit_rho:
        If True (default), rho is estimated jointly with the other
        parameters. If False, rho is held fixed at the given value --
        pass ``rho=0.0, fit_rho=False`` for the plain independent-Poisson
        baseline.
    reg_strength:
        L2 penalty strength on the attack/defense parameters -- both for
        identifiability and to shrink noisy, small-sample team strength
        toward the (better-calibrated) Elo-driven estimate. See "Calibration
        finding" in the module docstring for why the default is 10.0, not a
        smaller "just enough for identifiability" value.
    min_weight:
        Matches whose time-decay weight falls below this are dropped
        before fitting (numerical optimization only, see module
        docstring).
    max_iter:
        Maximum L-BFGS-B iterations.

    Returns
    -------
    A fitted ``GoalsModel``.
    """
    _check_required_columns(matches)
    as_of_date = pd.Timestamp(as_of_date)

    train = matches.loc[matches["date"] < as_of_date].copy()
    if train.empty:
        raise ValueError(f"No matches strictly before {as_of_date.date()} to fit on.")

    age_days = (as_of_date - train["date"]).dt.days.to_numpy(dtype=float)
    weight = time_decay_weight(age_days, half_life_days)

    keep = weight >= min_weight
    train = train.loc[keep]
    weight = weight[keep]
    if train.empty:
        raise ValueError(
            "All matches were filtered out by min_weight; lower min_weight or "
            "raise half_life_days."
        )

    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    team_index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = train["home_team"].map(team_index).astype(np.int64).to_numpy()
    away_idx = train["away_team"].map(team_index).astype(np.int64).to_numpy()
    x = train["home_score"].to_numpy(dtype=float)
    y = train["away_score"].to_numpy(dtype=float)
    elo_diff = (train["home_elo_pre"] - train["away_elo_pre"]).to_numpy(dtype=float)
    home_flag = (~train["neutral"].to_numpy(dtype=bool)).astype(float)

    # Time-decay-weighted draw rate in this same training window, carried on
    # the fitted model for src.blend's pure-Elo derivation (see GoalsModel).
    base_draw_rate = float(np.average((x == y).astype(float), weights=weight))

    use_correction = fit_rho or (rho != 0.0)
    n_free = 3 + (1 if fit_rho else 0) + 2 * n_teams

    def unpack(theta):
        intercept, home_adv, beta_elo = theta[0], theta[1], theta[2]
        offset = 3
        if fit_rho:
            rho_ = theta[3]
            offset = 4
        else:
            rho_ = rho
        attack = theta[offset : offset + n_teams]
        defense = theta[offset + n_teams :]
        return intercept, home_adv, beta_elo, rho_, attack, defense

    def neg_log_likelihood_and_grad(theta):
        intercept, home_adv, beta_elo, rho_, attack, defense = unpack(theta)

        log_lh = intercept + attack[home_idx] + defense[away_idx] + home_adv * home_flag + beta_elo * elo_diff
        log_la = intercept + attack[away_idx] + defense[home_idx] - beta_elo * elo_diff
        log_lh = np.clip(log_lh, -20.0, 20.0)
        log_la = np.clip(log_la, -20.0, 20.0)
        lh = np.exp(log_lh)
        la = np.exp(log_la)

        poisson_ll = x * log_lh - lh - gammaln(x + 1.0) + y * log_la - la - gammaln(y + 1.0)

        g_home = weight * (x - lh)
        g_away = weight * (y - la)
        d_rho = 0.0
        log_tau = np.zeros_like(lh)

        if use_correction:
            m00 = (x == 0) & (y == 0)
            m01 = (x == 0) & (y == 1)
            m10 = (x == 1) & (y == 0)
            m11 = (x == 1) & (y == 1)

            tau = np.ones_like(lh)
            tau[m00] = 1.0 - lh[m00] * la[m00] * rho_
            tau[m01] = 1.0 + lh[m01] * rho_
            tau[m10] = 1.0 + la[m10] * rho_
            tau[m11] = 1.0 - rho_
            tau = np.clip(tau, TAU_FLOOR, None)
            log_tau = np.log(tau)

            dlogtau_dlh = np.zeros_like(lh)
            dlogtau_dla = np.zeros_like(la)
            dlogtau_drho_terms = np.zeros_like(lh)

            dlogtau_dlh[m00] = -la[m00] * rho_ / tau[m00]
            dlogtau_dla[m00] = -lh[m00] * rho_ / tau[m00]
            dlogtau_drho_terms[m00] = -lh[m00] * la[m00] / tau[m00]

            dlogtau_dlh[m01] = rho_ / tau[m01]
            dlogtau_drho_terms[m01] = lh[m01] / tau[m01]

            dlogtau_dla[m10] = rho_ / tau[m10]
            dlogtau_drho_terms[m10] = la[m10] / tau[m10]

            dlogtau_drho_terms[m11] = -1.0 / tau[m11]

            g_home = g_home + weight * dlogtau_dlh * lh
            g_away = g_away + weight * dlogtau_dla * la
            d_rho = np.sum(weight * dlogtau_drho_terms)

        ll_i = poisson_ll + log_tau
        logL = np.sum(weight * ll_i)
        reg = reg_strength * (np.sum(attack**2) + np.sum(defense**2))
        nll = -logL + reg

        d_intercept = np.sum(g_home + g_away)
        d_home_adv = np.sum(g_home * home_flag)
        d_beta_elo = np.sum(elo_diff * (g_home - g_away))

        attack_grad = np.zeros(n_teams)
        defense_grad = np.zeros(n_teams)
        np.add.at(attack_grad, home_idx, g_home)
        np.add.at(attack_grad, away_idx, g_away)
        np.add.at(defense_grad, away_idx, g_home)
        np.add.at(defense_grad, home_idx, g_away)

        grad = np.empty(n_free)
        grad[0] = -d_intercept
        grad[1] = -d_home_adv
        grad[2] = -d_beta_elo
        offset = 3
        if fit_rho:
            grad[3] = -d_rho
            offset = 4
        grad[offset : offset + n_teams] = -attack_grad + 2 * reg_strength * attack
        grad[offset + n_teams :] = -defense_grad + 2 * reg_strength * defense

        return nll, grad

    theta0 = np.zeros(n_free)
    theta0[1] = 0.1  # mild positive initial home-advantage guess

    bounds = [(None, None)] * n_free
    if fit_rho:
        bounds[3] = (-0.9, 0.9)

    result = minimize(
        neg_log_likelihood_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    intercept, home_adv, beta_elo, rho_fitted, attack, defense = unpack(result.x)

    return GoalsModel(
        teams=teams,
        attack=dict(zip(teams, attack.tolist())),
        defense=dict(zip(teams, defense.tolist())),
        intercept=float(intercept),
        home_advantage=float(home_adv),
        beta_elo=float(beta_elo),
        rho=float(rho_fitted),
        half_life_days=half_life_days,
        as_of_date=as_of_date,
        n_matches_used=len(train),
        converged=bool(result.success),
        base_draw_rate=base_draw_rate,
    )


def expected_goal_rates(
    model: GoalsModel,
    home_team: str,
    away_team: str,
    home_elo_pre: float,
    away_elo_pre: float,
    neutral: bool = False,
):
    """Return (lambda_home, lambda_away) implied by ``model`` for one match."""
    elo_diff = home_elo_pre - away_elo_pre
    home_flag = 0.0 if neutral else 1.0

    log_lambda_home = (
        model.intercept
        + model.get_attack(home_team)
        + model.get_defense(away_team)
        + model.home_advantage * home_flag
        + model.beta_elo * elo_diff
    )
    log_lambda_away = (
        model.intercept
        + model.get_attack(away_team)
        + model.get_defense(home_team)
        - model.beta_elo * elo_diff
    )
    return float(np.exp(log_lambda_home)), float(np.exp(log_lambda_away))


@dataclass
class MatchPrediction:
    """Full-scoreline prediction for one match."""

    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_win: float
    draw: float
    away_win: float
    score_matrix: pd.DataFrame  # rows = home goals, columns = away goals


def score_matrix_and_outcomes(lambda_home: float, lambda_away: float, rho: float, max_goals: int = 10):
    """Build the Dixon-Coles corrected, renormalized score matrix for a pair
    of goal rates, and the outcome probabilities derived from it.

    Returns ``(matrix, p_home_win, p_draw, p_away_win)`` where ``matrix`` is
    a ``(max_goals + 1, max_goals + 1)`` numpy array (rows = home goals,
    columns = away goals) that sums to 1. Reused by both ``predict_match``
    (for full-time scorelines) and ``src.knockout`` (for extra-time
    scorelines, with scaled-down goal rates).
    """
    goals = np.arange(max_goals + 1)
    pmf_home = poisson.pmf(goals, lambda_home)
    pmf_away = poisson.pmf(goals, lambda_away)
    matrix = np.outer(pmf_home, pmf_away)

    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    tau = dixon_coles_tau(
        home_grid,
        away_grid,
        np.full_like(matrix, lambda_home),
        np.full_like(matrix, lambda_away),
        rho,
    )
    matrix = np.clip(matrix * tau, 0.0, None)
    matrix = matrix / matrix.sum()

    p_home_win = float(matrix[home_grid > away_grid].sum())
    p_draw = float(matrix[home_grid == away_grid].sum())
    p_away_win = float(matrix[home_grid < away_grid].sum())

    return matrix, p_home_win, p_draw, p_away_win


def predict_match(
    model: GoalsModel,
    home_team: str,
    away_team: str,
    home_elo_pre: float,
    away_elo_pre: float,
    neutral: bool = False,
    max_goals: int = 10,
    blend_weight: float = 1.0,
) -> MatchPrediction:
    """Predict the full scoreline distribution for one match.

    Builds the (Dixon-Coles corrected, renormalized) joint probability
    matrix for scorelines 0..``max_goals`` on each side, then derives
    P(home win) / P(draw) / P(away win) and each side's expected goals
    directly from that matrix.

    ``blend_weight`` (default 1.0, i.e. pure Poisson) optionally blends
    the outcome probabilities with pure Elo via ``src.blend`` -- see that
    module for why (the Poisson model is conservative on clear
    favorites) and for the explicit framing of this as bias correction,
    not market-fitting. Only ``home_win``/``draw``/``away_win`` are
    blended; ``home_xg``/``away_xg``/``score_matrix`` remain the
    Poisson model's own output, since pure Elo has no notion of expected
    goals or a scoreline distribution to blend with.
    """
    lambda_home, lambda_away = expected_goal_rates(
        model, home_team, away_team, home_elo_pre, away_elo_pre, neutral
    )

    matrix, home_win, draw, away_win = score_matrix_and_outcomes(
        lambda_home, lambda_away, model.rho, max_goals=max_goals
    )

    if blend_weight < 1.0:
        elo_probs = elo_match_probabilities(home_elo_pre, away_elo_pre, model.base_draw_rate, neutral)
        home_win, draw, away_win = blend_outcome_probabilities(
            (home_win, draw, away_win), elo_probs, blend_weight
        )

    goals = np.arange(max_goals + 1)
    home_xg = float(np.sum(matrix.sum(axis=1) * goals))
    away_xg = float(np.sum(matrix.sum(axis=0) * goals))

    score_matrix = pd.DataFrame(
        matrix,
        index=pd.Index(goals, name="home_goals"),
        columns=pd.Index(goals, name="away_goals"),
    )

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        home_xg=home_xg,
        away_xg=away_xg,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        score_matrix=score_matrix,
    )


if __name__ == "__main__":
    from src.data import load_results
    from src.elo import compute_elo_ratings

    matches = load_results()
    matches_with_elo, current_ratings = compute_elo_ratings(matches)

    as_of = matches_with_elo["date"].max() + pd.Timedelta(days=1)
    model = fit_poisson_model(matches_with_elo, as_of_date=as_of)
    print(
        f"Fitted on {model.n_matches_used} matches "
        f"(rho={model.rho:.3f}, converged={model.converged})"
    )

    home, away = "Argentina", "Brazil"
    prediction = predict_match(
        model,
        home,
        away,
        current_ratings.get(home, 1500.0),
        current_ratings.get(away, 1500.0),
        neutral=True,
    )
    print(
        f"{home} vs {away}: xG {prediction.home_xg:.2f}-{prediction.away_xg:.2f} | "
        f"H {prediction.home_win:.1%} D {prediction.draw:.1%} A {prediction.away_win:.1%}"
    )
