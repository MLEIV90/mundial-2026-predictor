"""Leakage-free Elo rating engine for international football.

Design
------
Ratings are computed by walking through the full match history in
chronological order, once, and maintaining a running "current rating" per
team in a plain dict. For every match, in order:

1. Each team's CURRENT rating (i.e. as it stood strictly before this match)
   is read out and stored as that match's ``home_elo_pre`` / ``away_elo_pre``.
2. Only after being recorded, both ratings are updated using the actual
   result of *this* match.

Because a match's pre-match ratings are always read before that same
match's result is applied to the ratings dict, no match's outcome -- and no
later match -- can ever influence its own pre-match rating. This is what
makes the ratings safe to use as model features: at "prediction time" for
match N, only information from matches 1..N-1 has been used.

K-factor (match importance)
----------------------------
The K-factor controls how much a single match can move a team's rating.
It is derived from the ``tournament`` column, in tiers loosely modeled on
https://www.eloratings.net/about :

    K = 60   FIFA World Cup (the final tournament itself)
    K = 50   Major continental / intercontinental championship finals:
             UEFA Euro, Copa America, African Cup of Nations, AFC Asian Cup,
             CONCACAF Championship, Gold Cup, Confederations Cup,
             Oceania Nations Cup
    K = 40   Any qualifier for the tournaments above (or for other
             competitions) -- matched via "qualification" in the name
    K = 30   Everything else not covered above (regional cups, games,
             minor invitational tournaments, ...)
    K = 20   Friendly

Home advantage
---------------
Home advantage is modeled as a fixed number of Elo points added to the
home team's rating, but only for the purpose of computing the *expected*
result -- it never changes a team's stored rating. It is a fully
adjustable parameter (``home_advantage``) and is automatically switched
off (treated as 0) for any match flagged ``neutral`` in results.csv, since
neither side is playing at home in that case (most FIFA World Cup matches,
for example).

Goal-difference weighting
--------------------------
The rating update is scaled by a goal-difference multiplier, following
the same approach as eloratings.net, so that big wins move ratings more
than narrow ones:

    G = 1                    if goal difference is 0 or 1
    G = 1.5                  if goal difference is 2
    G = (11 + goal diff) / 8 if goal difference is 3 or more
"""

from typing import Callable, Dict, Tuple

import pandas as pd

INITIAL_RATING = 1500.0
DEFAULT_HOME_ADVANTAGE = 100.0

K_WORLD_CUP = 60.0
K_MAJOR_TOURNAMENT = 50.0
K_QUALIFICATION = 40.0
K_OTHER = 30.0
K_FRIENDLY = 20.0

WORLD_CUP_TOURNAMENTS = {"FIFA World Cup"}

MAJOR_TOURNAMENTS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Championship",
    "Gold Cup",
    "Confederations Cup",
    "Oceania Nations Cup",
}


def get_k_factor(tournament: str) -> float:
    """Return the K-factor for a given ``tournament`` name.

    See the module docstring for the full tier breakdown.
    """
    if tournament in WORLD_CUP_TOURNAMENTS:
        return K_WORLD_CUP
    if tournament in MAJOR_TOURNAMENTS:
        return K_MAJOR_TOURNAMENT
    if "qualification" in tournament:
        return K_QUALIFICATION
    if tournament == "Friendly":
        return K_FRIENDLY
    return K_OTHER


def _goal_diff_multiplier(goal_diff: float) -> float:
    """Return the eloratings.net-style goal-difference multiplier."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0


def compute_elo_ratings(
    df: pd.DataFrame,
    initial_rating: float = INITIAL_RATING,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    k_factor_func: Callable[[str], float] = get_k_factor,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute rolling, leakage-free Elo ratings over the full match history.

    Parameters
    ----------
    df:
        Match results with at least the columns ``date``, ``home_team``,
        ``away_team``, ``home_score``, ``away_score``, ``tournament`` and
        ``neutral`` (as produced by ``src.data.load_results``).
    initial_rating:
        Rating assigned to a team the first time it appears in the data.
    home_advantage:
        Elo points added to the home team's rating when computing the
        expected result. Set to 0 to disable home advantage entirely.
        Automatically ignored (treated as 0) for matches where
        ``neutral`` is True.
    k_factor_func:
        Function mapping a tournament name to its K-factor. Defaults to
        ``get_k_factor``.

    Returns
    -------
    A tuple ``(result_df, final_ratings)`` where ``result_df`` is ``df``
    (restricted to already-played matches, see below) sorted
    chronologically with two new columns, ``home_elo_pre`` and
    ``away_elo_pre`` (each team's rating immediately before that match),
    and ``final_ratings`` is a dict mapping team name to its rating after
    the very last match in the data.

    Notes
    -----
    results.csv also lists scheduled fixtures that have not been played
    yet (``home_score`` / ``away_score`` is missing). Those rows carry no
    result to learn from, so they are dropped before computing ratings.
    """
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    ratings: Dict[str, float] = {}
    home_elo_pre = []
    away_elo_pre = []

    for row in df.itertuples(index=False):
        home_rating = ratings.get(row.home_team, initial_rating)
        away_rating = ratings.get(row.away_team, initial_rating)

        home_elo_pre.append(home_rating)
        away_elo_pre.append(away_rating)

        effective_home_advantage = 0.0 if row.neutral else home_advantage

        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_rating - (home_rating + effective_home_advantage)) / 400.0)
        )
        expected_away = 1.0 - expected_home

        if row.home_score > row.away_score:
            actual_home, actual_away = 1.0, 0.0
        elif row.home_score < row.away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        goal_diff = abs(row.home_score - row.away_score)
        k = k_factor_func(row.tournament) * _goal_diff_multiplier(goal_diff)

        ratings[row.home_team] = home_rating + k * (actual_home - expected_home)
        ratings[row.away_team] = away_rating + k * (actual_away - expected_away)

    result = df.copy()
    result["home_elo_pre"] = home_elo_pre
    result["away_elo_pre"] = away_elo_pre
    return result, ratings


def top_teams(ratings: Dict[str, float], n: int = 20) -> pd.DataFrame:
    """Return the top ``n`` teams by current rating as a sorted DataFrame."""
    return (
        pd.DataFrame(sorted(ratings.items(), key=lambda item: item[1], reverse=True)[:n],
                     columns=["team", "elo"])
    )


if __name__ == "__main__":
    from src.data import load_results

    matches = load_results()
    matches_with_elo, current_ratings = compute_elo_ratings(matches)
    print(top_teams(current_ratings, n=20))
