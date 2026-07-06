"""Monte-Carlo simulation of the remaining World Cup knockout bracket.

Simulates the tournament from the current round of 16 through to the
final, ``n_simulations`` times, to estimate each remaining team's
probability of reaching the quarterfinals, semifinals, the final, and
winning the title outright.

The bracket
-----------
``build_remaining_bracket`` hardcodes the actual remaining bracket as of
the round of 16: QF1 (France vs Morocco) and QF4 (Norway vs England) are
already decided; the other four quarterfinalists are still to be
determined by the round-of-16 ties (Portugal/Spain, USA/Belgium,
Argentina/Egypt, Switzerland/Colombia). It's a tree of ``Tie`` nodes,
where each side is either a confirmed team name or another (still to be
played) ``Tie`` whose winner fills that slot.

How a single simulation run works
-----------------------------------
Each run walks the tree bottom-up: a leaf is just a team name; a ``Tie``
resolves its two sides first (recursively), then samples a winner from
``src.knockout.advance_probability`` using each team's *current* Elo
rating (ratings are held fixed for the whole simulation -- this models
"if the bracket played out from today's strength levels," not a
day-by-day Elo evolution). Both teams that entered a quarterfinal,
semifinal, or final tie are credited with "reaching" that stage for this
run; the final's winner is additionally credited as champion.

Performance: memoizing repeated matchups
-------------------------------------------
A team's Elo doesn't change between simulation runs, so
``advance_probability`` for a given pair of teams returns the exact same
numbers every time it's asked -- only the random draw differs. Rather
than recomputing it up to ``n_simulations`` times, each unique
``(team_a, team_b, neutral)`` matchup is computed once and cached; with
only a few dozen possible matchups across this bracket, this turns what
would be tens of thousands of score-matrix computations into a couple of
dozen, making 20,000 runs take a fraction of a second instead of minutes.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

import pandas as pd

from src.knockout import AdvanceResult, advance_probability
from src.model import GoalsModel

DEFAULT_N_SIMULATIONS = 20000
DEFAULT_SEED = 42

_STAGE_PREFIXES = {
    "QF": "quarterfinal",
    "SF": "semifinal",
    "Final": "final",
}


@dataclass
class Tie:
    """One knockout match in the bracket tree.

    ``home``/``away`` are each either a confirmed team name (``str``) or
    another, still-to-be-played ``Tie`` whose winner fills that slot.
    """

    label: str
    home: "Node"
    away: "Node"
    neutral: bool = True


Node = Union[str, Tie]


def _stage_of(label: str) -> Optional[str]:
    """Map a Tie's label to the stage it represents, for probability
    reporting. Round-of-16 ties (labels matching none of these) aren't
    reported individually -- only their winners, once they feed into a
    QF/SF/Final tie, are.
    """
    for prefix, stage in _STAGE_PREFIXES.items():
        if label.startswith(prefix):
            return stage
    return None


def build_remaining_bracket() -> Tie:
    """The actual remaining 2026 World Cup bracket, as of the round of 16."""
    return Tie(
        "Final",
        home=Tie(
            "SF1",
            home=Tie("QF1", home="France", away="Morocco", neutral=True),
            away=Tie(
                "QF2",
                home=Tie("R16-Portugal-Spain", home="Portugal", away="Spain", neutral=True),
                away=Tie("R16-USA-Belgium", home="United States", away="Belgium", neutral=False),
                neutral=True,
            ),
            neutral=True,
        ),
        away=Tie(
            "SF2",
            home=Tie(
                "QF3",
                home=Tie("R16-Argentina-Egypt", home="Argentina", away="Egypt", neutral=True),
                away=Tie("R16-Switzerland-Colombia", home="Switzerland", away="Colombia", neutral=True),
                neutral=True,
            ),
            away=Tie("QF4", home="Norway", away="England", neutral=True),
            neutral=True,
        ),
        neutral=True,
    )


def _resolve(
    node: Node,
    model: GoalsModel,
    current_ratings: Dict[str, float],
    rng: random.Random,
    reached_this_run: Dict[str, Set[str]],
    prob_cache: Dict[Tuple[str, str, bool], AdvanceResult],
) -> str:
    if isinstance(node, str):
        return node

    team_a = _resolve(node.home, model, current_ratings, rng, reached_this_run, prob_cache)
    team_b = _resolve(node.away, model, current_ratings, rng, reached_this_run, prob_cache)

    stage = _stage_of(node.label)
    if stage is not None:
        reached_this_run.setdefault(team_a, set()).add(stage)
        reached_this_run.setdefault(team_b, set()).add(stage)

    key = (team_a, team_b, node.neutral)
    adv = prob_cache.get(key)
    if adv is None:
        elo_a = current_ratings.get(team_a, 1500.0)
        elo_b = current_ratings.get(team_b, 1500.0)
        adv = advance_probability(model, team_a, team_b, elo_a, elo_b, neutral=node.neutral)
        prob_cache[key] = adv

    winner = team_a if rng.random() < adv.p_a_advances else team_b

    if node.label == "Final":
        reached_this_run.setdefault(winner, set()).add("champion")

    return winner


def simulate_tournament(
    model: GoalsModel,
    current_ratings: Dict[str, float],
    bracket: Optional[Tie] = None,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Run a Monte-Carlo simulation of the remaining bracket.

    Returns a DataFrame with one row per team still alive in the bracket
    and columns ``p_quarterfinal``/``p_semifinal``/``p_final``/
    ``p_champion``, sorted by ``p_champion`` descending. Teams already
    confirmed for a stage (e.g. France and Morocco are already
    quarterfinalists) show probability 1.0 for that stage, as expected.
    """
    bracket = bracket or build_remaining_bracket()
    rng = random.Random(seed)
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prob_cache: Dict[Tuple[str, str, bool], AdvanceResult] = {}

    for _ in range(n_simulations):
        reached_this_run: Dict[str, Set[str]] = {}
        _resolve(bracket, model, current_ratings, rng, reached_this_run, prob_cache)
        for team, stages in reached_this_run.items():
            for stage in stages:
                counts[team][stage] += 1

    rows = []
    for team, stage_counts in counts.items():
        rows.append(
            {
                "team": team,
                "p_quarterfinal": stage_counts.get("quarterfinal", 0) / n_simulations,
                "p_semifinal": stage_counts.get("semifinal", 0) / n_simulations,
                "p_final": stage_counts.get("final", 0) / n_simulations,
                "p_champion": stage_counts.get("champion", 0) / n_simulations,
            }
        )

    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)


def bracket_to_dict(node: Node) -> dict:
    """Serialize the bracket tree to a plain (JSON-able) dict."""
    if isinstance(node, str):
        return {"team": node}
    return {
        "label": node.label,
        "neutral": node.neutral,
        "home": bracket_to_dict(node.home),
        "away": bracket_to_dict(node.away),
    }


def save_simulation_json(
    results: pd.DataFrame,
    bracket: Tie,
    path: str,
    as_of_date,
    n_simulations: int,
    generated_at: Optional[str] = None,
) -> None:
    """Write the simulation results + bracket structure to ``path`` as JSON."""
    payload = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(pd.Timestamp(as_of_date).date()),
        "n_simulations": n_simulations,
        "bracket": bracket_to_dict(bracket),
        "results": json.loads(results.to_json(orient="records")),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import time

    import pandas as pd

    from src.data import load_results
    from src.elo import compute_elo_ratings
    from src.model import fit_poisson_model

    matches = load_results()
    df_elo, current_ratings = compute_elo_ratings(matches)
    as_of_date = df_elo["date"].max() + pd.Timedelta(days=1)
    model = fit_poisson_model(df_elo, as_of_date=as_of_date)

    t0 = time.time()
    results = simulate_tournament(model, current_ratings)
    print(f"Simulated {DEFAULT_N_SIMULATIONS} tournaments in {time.time() - t0:.2f}s")
    print(results.to_string(index=False))
