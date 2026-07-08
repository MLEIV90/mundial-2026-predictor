# Mundial 2026 Predictor

[🇪🇸 Español](README.md) · **🇬🇧 English**

A probabilistic match-prediction model for the 2026 FIFA World Cup (knockout
stage), built with a Risk Analytics approach: calibration, leakage-free
backtesting, and an honest comparison against the betting market as a benchmark.

**▶️ Run locally** — see the "Running it" section below.

## What it does

For each knockout tie, it estimates the probability that each team advances, and
simulates the remaining bracket to produce each team's title probability.
Predictions are compared against Pinnacle's closing line (the sharpest
bookmaker) using Brier score and log-loss.

The goal is not to "beat the market", but to build an interpretable model of my
own that is **on par with** the market and whose limits are measured and
documented.

## Headline result (fair 90-minute backtest)

Apples-to-apples comparison on the 90-minute 1X2 result, which is what the
market actually prices:

| | Brier score | Log-loss |
|---|---|---|
| **Model** | 0.3915 | 0.7076 |
| **Market (Pinnacle)** | 0.4089 | 0.7261 |

Across 24 knockout fixtures, the model was more accurate than the market on 15.
**Honest reading:** this is a small, in-tournament sample, so it's an early
signal that the model is **competitive / on par with Pinnacle**, not proof of a
durable edge. It is not a system for beating the bookmakers.

## How it works (layered architecture)

- **Data** (`src/data.py`): downloads and validates the international match
  history (martj42 dataset, ~49,500 matches since 1872).
- **Elo** (`src/elo.py`): strength rating computed **leakage-free** (each match
  uses only the pre-match rating). Variable K by importance (World Cup >
  qualifiers > friendlies) and home advantage switched off at neutral venues.
- **Goals model** (`src/model.py`): bivariate Poisson (Dixon-Coles style) with
  Elo-driven goal rates, weighted by **recency** (time-decay, 24-month
  half-life) so recent football dominates.
- **Knockout** (`src/knockout.py`): converts the 90-minute result into an
  advance probability, modeling extra time and penalties (50/50 coin flip).
- **Blend** (`src/blend.py`): ensembles the Poisson model with pure Elo,
  correcting a Poisson bias that underrated clear favorites (see below).
- **Simulation** (`src/simulation.py`): Monte-Carlo of the remaining bracket
  (20,000 tournaments) for title probabilities.
- **Market** (`src/odds.py`): pulls OddsPapi odds, removes the margin (vig), and
  derives Pinnacle's implied probability. Locally cached to conserve the free
  quota.
- **Evaluation** (`src/evaluation.py`): model-vs-market backtest with Brier and
  log-loss, leakage-free (the model is refit as of each fixture's date).

## A finding from the process

During development the model rated Morocco a favorite over France despite France
having 150 more Elo points. Checking against the market (Pinnacle had France at
60%) confirmed the goals model was **over-reacting to recent form** and
underrating the strength gap. It was fixed with an Elo+Poisson blend layer,
calibrated against actual results. This cycle — distrusting a result, testing it
against a benchmark, and correcting the bias — is the methodological core of the
project.

## Design decisions

- **Elo over the FIFA ranking.** Elo updates match by match and is a better
  predictor; the FIFA ranking updates in windows (frozen before the tournament)
  and was designed to seed draws, not to predict.
- **Poisson / boosting over deep learning.** On low-volume tabular data,
  interpretable goals models perform as well or better than a neural net, with
  more transparency.
- **Blend calibrated by judgment (0.5), not to the small-sample optimum.**
  Calibration over 24 fixtures favored pure Elo, but a balanced blend was chosen
  to avoid overfitting a fragile result.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads pre-computed data from `data/app/*.json` and **never** calls the
API, so it runs with no configuration. To refresh with the latest results and
odds:

```bash
python scripts/update_data.py
```

That script is the only one that calls OddsPapi (via `src/odds.py`, cached under
`data/odds_cache/` to conserve the free quota). It requires `ODDSPAPI_API_KEY`
in a `.env` file (see `.env.example`). Run it after each new match and commit the
updated `data/app/*.json` files.

## Limitations and future work

- **Small sample:** the backtest covers the in-tournament knockout stage (~24
  fixtures). Results are early signals, not evidence of a long-run edge.
- **Hardcoded bracket:** the bracket structure is updated by hand each round; a
  more robust version would derive it automatically from results.
- **Team-level data only:** no squad value (Transfermarkt) or player-level data,
  which the literature suggests improve prediction.
- **Manual refresh:** could be automated with GitHub Actions.

## Stack

Python · pandas · numpy · scipy · statsmodels · scikit-learn · Streamlit ·
OddsPapi API · Elo · bivariate Poisson (Dixon-Coles) · Monte-Carlo

## Data

- [International football results 1872–2026 (martj42)](https://github.com/martj42/international_results)
- Odds: [OddsPapi](https://oddspapi.io/) (Pinnacle closing line)

---

*Note: betting carries real financial risk. This is a modeling and portfolio
project, not betting advice.*
