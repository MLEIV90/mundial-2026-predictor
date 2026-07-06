"""Mundial 2026 Predictor -- Streamlit app.

Reads exclusively from the pre-computed JSON files under ``data/app/``
(written by ``scripts/update_data.py``) for the Live Predictions and Track
Record tabs -- this app never calls the OddsPapi API itself, so viewing it
costs zero API requests no matter how many times it's opened. The Match
Predictor tab loads the goals model and Elo ratings directly from ``src``
(downloading results.csv and fitting the model once, cached for the
session) since that only needs match results, not betting odds.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.elo import compute_elo_ratings
from src.knockout import advance_probability
from src.model import fit_poisson_model, predict_match

APP_DATA_DIR = Path("data/app")

st.set_page_config(page_title="Mundial 2026 Predictor", page_icon="⚽", layout="wide")


def _load_json(name: str) -> dict | None:
    path = APP_DATA_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


@st.cache_resource(show_spinner="Fitting the goals model (first load only)...")
def _load_model_and_ratings():
    from src.data import load_results

    matches = load_results()
    df_elo, current_ratings = compute_elo_ratings(matches)
    as_of_date = df_elo["date"].max() + pd.Timedelta(days=1)
    model = fit_poisson_model(df_elo, as_of_date=as_of_date)
    return model, current_ratings


st.title("⚽ Mundial 2026 Predictor")
st.caption(
    "A Dixon-Coles goals model with Elo-adjusted team strength, benchmarked "
    "against the betting market."
)

ratings_data = _load_json("model_ratings.json")
if ratings_data is not None:
    with st.sidebar:
        st.subheader("Current Elo Top 10")
        st.caption(f"As of {ratings_data['as_of_date']}")
        ratings_df = pd.DataFrame(ratings_data["ratings"][:10])
        ratings_df["elo"] = ratings_df["elo"].round(1)
        st.dataframe(ratings_df, hide_index=True, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Live Predictions", "Track Record", "Match Predictor"])

# --------------------------------------------------------------------------
# Tab 1: Live Predictions
# --------------------------------------------------------------------------
with tab1:
    st.header("Upcoming Knockout Fixtures")

    live_data = _load_json("live_predictions.json")
    if live_data is None:
        st.warning("No live predictions found. Run `python scripts/update_data.py` first.")
    elif not live_data["predictions"]:
        st.info("No upcoming knockout fixtures right now -- check back after the next round is drawn.")
    else:
        st.caption(
            f"Model fit as of {live_data['model_as_of_date']} · "
            f"generated {live_data['generated_at']}"
        )

        for pred in live_data["predictions"]:
            home, away = pred["home_team"], pred["away_team"]
            model_home, model_away = pred["model_p_home_advances"], pred["model_p_away_advances"]
            market_home, market_away = pred["market_p_home_advances"], pred["market_p_away_advances"]

            with st.container(border=True):
                st.subheader(f"{home} vs {away}")
                st.caption(f"{pred['date']} · {'Neutral venue' if pred['neutral'] else 'Home advantage applies'}")

                col_model, col_market = st.columns(2)
                with col_model:
                    st.markdown("**Model**")
                    st.metric(f"{home} advances", _fmt_pct(model_home))
                    st.metric(f"{away} advances", _fmt_pct(model_away))
                with col_market:
                    st.markdown("**Market** (de-vigged Pinnacle)")
                    st.metric(f"{home} advances", _fmt_pct(market_home))
                    st.metric(f"{away} advances", _fmt_pct(market_away))

                if market_home is None:
                    st.info("No market odds available for this fixture yet.")
                else:
                    model_favors_home = model_home >= 0.5
                    market_favors_home = market_home >= 0.5
                    if model_favors_home == market_favors_home:
                        st.success("Model and market agree on the favorite.")
                    else:
                        st.warning(
                            f"⚠️ Disagreement: model favors "
                            f"{home if model_favors_home else away}, market favors "
                            f"{home if market_favors_home else away}."
                        )

# --------------------------------------------------------------------------
# Tab 2: Track Record
# --------------------------------------------------------------------------
with tab2:
    st.header("Backtest: Model vs Market")

    backtest_data = _load_json("backtest.json")
    if backtest_data is None:
        st.warning("No backtest found. Run `python scripts/update_data.py` first.")
    else:
        st.info(
            "\U0001f4cb **Preliminary, in-tournament backtest.** This covers only the "
            f"{backtest_data['summary']['n_fixtures']} knockout fixtures played so far "
            "(Round of 32 onward). Small sample -- read the numbers as an early signal, "
            "not a final verdict."
        )

        summary = backtest_data["summary"]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Brier score (lower is better)",
            f"{summary['model_brier']:.4f}",
            delta=f"{summary['model_brier'] - summary['market_brier']:+.4f} vs market",
            delta_color="inverse",
        )
        col2.metric(
            "Log loss (lower is better)",
            f"{summary['model_log_loss']:.4f}",
            delta=f"{summary['model_log_loss'] - summary['market_log_loss']:+.4f} vs market",
            delta_color="inverse",
        )
        col3.metric(
            "Beat the market",
            f"{summary['n_model_beats_market']} / {summary['n_fixtures']}",
        )
        st.caption(f"Market Brier: {summary['market_brier']:.4f} · Market log loss: {summary['market_log_loss']:.4f}")

        st.subheader("Fixture-by-fixture")
        fixtures_df = pd.DataFrame(backtest_data["fixtures"])
        fixtures_df = fixtures_df.rename(
            columns={
                "date": "Date",
                "home_team": "Home",
                "away_team": "Away",
                "winner": "Advanced",
                "p_model_home_advances": "Model P(Home)",
                "p_market_home_advances": "Market P(Home)",
                "model_beat_market": "Model beat market",
            }
        )
        fixtures_df["Model P(Home)"] = fixtures_df["Model P(Home)"].map(lambda v: f"{v:.1%}")
        fixtures_df["Market P(Home)"] = fixtures_df["Market P(Home)"].map(lambda v: f"{v:.1%}")
        display_cols = ["Date", "Home", "Away", "Advanced", "Model P(Home)", "Market P(Home)", "Model beat market"]
        st.dataframe(fixtures_df[display_cols], hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3: Match Predictor
# --------------------------------------------------------------------------
with tab3:
    st.header("Predict Any Matchup")
    st.caption("Uses the current goals model and Elo ratings directly -- no betting odds involved.")

    model, current_ratings = _load_model_and_ratings()
    teams = sorted(current_ratings.keys())

    default_a = teams.index("Argentina") if "Argentina" in teams else 0
    default_b = teams.index("Brazil") if "Brazil" in teams else min(1, len(teams) - 1)

    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox("Team A (home slot)", teams, index=default_a)
    with col_b:
        team_b = st.selectbox("Team B (away slot)", teams, index=default_b)
    neutral = st.checkbox("Neutral venue", value=True)

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        elo_a = current_ratings.get(team_a, 1500.0)
        elo_b = current_ratings.get(team_b, 1500.0)

        match_pred = predict_match(model, team_a, team_b, elo_a, elo_b, neutral=neutral)
        adv = advance_probability(model, team_a, team_b, elo_a, elo_b, neutral=neutral)

        st.subheader("If this were a knockout tie")
        col1, col2 = st.columns(2)
        col1.metric(f"{team_a} advances", f"{adv.p_a_advances:.1%}")
        col2.metric(f"{team_b} advances", f"{adv.p_b_advances:.1%}")

        st.subheader("Regulation time (90')")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{team_a} win", f"{match_pred.home_win:.1%}")
        col2.metric("Draw", f"{match_pred.draw:.1%}")
        col3.metric(f"{team_b} win", f"{match_pred.away_win:.1%}")

        st.metric("Expected goals", f"{match_pred.home_xg:.2f} - {match_pred.away_xg:.2f}")

        matrix = match_pred.score_matrix
        flat_idx = matrix.values.argmax()
        most_likely_home, most_likely_away = divmod(flat_idx, matrix.shape[1])
        st.metric(
            "Most likely scoreline",
            f"{team_a} {most_likely_home} - {most_likely_away} {team_b}",
            help=f"P(this exact scoreline) = {matrix.values.max():.1%}",
        )

        with st.expander("Full score matrix (rows = home goals, columns = away goals)"):
            st.dataframe(matrix.style.format("{:.1%}"), use_container_width=True)
