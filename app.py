"""Mundial 2026 Predictor -- Streamlit app.

Reads exclusively from the pre-computed JSON files under ``data/app/``
(written by ``scripts/update_data.py``) for the Live Predictions, Track
Record, and Title Odds tabs -- this app never calls the OddsPapi API
itself, so viewing it costs zero API requests no matter how many times
it's opened. The Match Predictor tab (and the bracket visual) load the
goals model and Elo ratings directly from ``src`` (downloading
results.csv and fitting the model once, cached for the session) since
that only needs match results, not betting odds.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.elo import compute_elo_ratings
from src.knockout import advance_probability
from src.model import fit_poisson_model, predict_match

APP_DATA_DIR = Path("data/app")

# Categorical slots from the project's validated palette (see the dataviz
# skill's references/palette.md): blue = primary/favorite, red = a second
# point of emphasis, muted gray = everything else.
COLOR_BLUE = "#2a78d6"
COLOR_RED = "#e34948"
COLOR_GRAY = "#c3c2b7"
COLOR_TEXT_SECONDARY = "#52514e"

st.set_page_config(page_title="Mundial 2026 Predictor", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-weight: 500; color: #52514e; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _tie_probs(model, current_ratings: dict, home: str, away: str, neutral: bool):
    elo_home = current_ratings.get(home, 1500.0)
    elo_away = current_ratings.get(away, 1500.0)
    return advance_probability(model, home, away, elo_home, elo_away, neutral=neutral)


# Confirmed round-of-16 ties and quarterfinals (see src/simulation.py for the
# full bracket definition used by the Monte Carlo simulation).
R16_TIES = [
    ("Portugal", "Spain", True),
    ("United States", "Belgium", False),
    ("Argentina", "Egypt", True),
    ("Switzerland", "Colombia", True),
]
QF_SLOTS = [
    ("QF1", "France", "Morocco", True, True),
    ("QF2", "Winner: Portugal / Spain", "Winner: USA / Belgium", True, False),
    ("QF3", "Winner: Argentina / Egypt", "Winner: Switzerland / Colombia", True, False),
    ("QF4", "Norway", "England", True, True),
]


def render_bracket(model, current_ratings: dict) -> None:
    st.subheader("Remaining Bracket")
    st.caption(
        "Confirmed ties show the model's P(advance); undetermined slots show the pending matchup."
    )

    col_r16, col_qf, col_sf, col_final = st.columns(4)

    with col_r16:
        st.markdown("**Round of 16**")
        for home, away, neutral in R16_TIES:
            adv = _tie_probs(model, current_ratings, home, away, neutral)
            with st.container(border=True):
                st.markdown(f"{home} &nbsp;**{adv.p_a_advances:.0%}**", unsafe_allow_html=True)
                st.markdown(f"{away} &nbsp;**{adv.p_b_advances:.0%}**", unsafe_allow_html=True)

    with col_qf:
        st.markdown("**Quarterfinals**")
        for label, home, away, neutral, confirmed in QF_SLOTS:
            with st.container(border=True):
                st.caption(label)
                if confirmed:
                    adv = _tie_probs(model, current_ratings, home, away, neutral)
                    st.markdown(f"{home} &nbsp;**{adv.p_a_advances:.0%}**", unsafe_allow_html=True)
                    st.markdown(f"{away} &nbsp;**{adv.p_b_advances:.0%}**", unsafe_allow_html=True)
                else:
                    st.caption(home)
                    st.caption(away)

    with col_sf:
        st.markdown("**Semifinals**")
        for label, desc in [("SF1", "Winner QF1 vs Winner QF2"), ("SF2", "Winner QF3 vs Winner QF4")]:
            with st.container(border=True):
                st.caption(label)
                st.caption(desc)

    with col_final:
        st.markdown("**Final**")
        with st.container(border=True):
            st.caption("Final")
            st.caption("Winner SF1 vs Winner SF2")


# --------------------------------------------------------------------------
# Load everything the header / KPI row needs up front
# --------------------------------------------------------------------------
ratings_data = _load_json("model_ratings.json")
live_data = _load_json("live_predictions.json")
backtest_data = _load_json("backtest.json")
sim_data = _load_json("simulation.json")

st.title("⚽ Mundial 2026 Predictor")
st.caption(
    "A Dixon-Coles goals model with Elo-adjusted team strength, benchmarked "
    "against the betting market."
)

kpi1, kpi2, kpi3 = st.columns(3)
if sim_data is not None and sim_data["results"]:
    top_row = max(sim_data["results"], key=lambda r: r["p_champion"])
    kpi1.metric("Title favorite", top_row["team"], f"{top_row['p_champion']:.1%} to win")
else:
    kpi1.metric("Title favorite", "n/a")

if backtest_data is not None:
    s = backtest_data["summary"]
    verdict = "ahead of" if s["model_brier"] < s["market_brier"] else "behind"
    kpi2.metric(
        "Model vs market (Brier)",
        f"{s['model_brier']:.4f}",
        f"{verdict} market by {abs(s['model_brier'] - s['market_brier']):.4f}",
        delta_color="off",
    )
else:
    kpi2.metric("Model vs market", "n/a")

if live_data is not None and live_data["predictions"]:
    next_fixture = live_data["predictions"][0]
    kpi3.metric(
        "Next fixture",
        f"{next_fixture['home_team']} vs {next_fixture['away_team']}",
        next_fixture["date"],
        delta_color="off",
    )
else:
    kpi3.metric("Next fixture", "n/a")

if ratings_data is not None:
    with st.sidebar:
        st.subheader("Current Elo Top 10")
        st.caption(f"As of {ratings_data['as_of_date']}")
        ratings_df = pd.DataFrame(ratings_data["ratings"][:10])
        ratings_df["elo"] = ratings_df["elo"].round(1)
        st.dataframe(ratings_df, hide_index=True, use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Live Predictions", "Track Record", "Match Predictor", "Title Odds"]
)

# --------------------------------------------------------------------------
# Tab 1: Live Predictions
# --------------------------------------------------------------------------
with tab1:
    bracket_model, bracket_ratings = _load_model_and_ratings()
    render_bracket(bracket_model, bracket_ratings)

    st.divider()
    st.header("Upcoming Knockout Fixtures")

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

# --------------------------------------------------------------------------
# Tab 4: Title Odds
# --------------------------------------------------------------------------
with tab4:
    st.header("Title Odds -- Monte Carlo Simulation")

    if sim_data is None:
        st.warning("No simulation found. Run `python scripts/update_data.py` first.")
    else:
        st.caption(
            f"{sim_data['n_simulations']:,} simulated tournaments · as of {sim_data['as_of_date']} · "
            f"generated {sim_data['generated_at']}"
        )

        sim_df = (
            pd.DataFrame(sim_data["results"])
            .sort_values("p_champion", ascending=False)
            .reset_index(drop=True)
        )
        top_team = sim_df.iloc[0]["team"]
        top_p = sim_df.iloc[0]["p_champion"]

        ratings_lookup = {r["team"]: r["elo"] for r in ratings_data["ratings"]} if ratings_data else {}
        remaining_teams = set(sim_df["team"])
        elo_leader = (
            max(remaining_teams, key=lambda t: ratings_lookup.get(t, 0.0)) if ratings_lookup else None
        )

        if elo_leader is not None:
            elo_leader_p = float(sim_df.set_index("team")["p_champion"].get(elo_leader, 0.0))
            if elo_leader == top_team:
                st.info(
                    f"\U0001f3c6 **{elo_leader}** has both the highest Elo rating among the remaining "
                    f"teams and the simulation's best title odds ({elo_leader_p:.1%}) -- Elo and the "
                    "bracket simulation agree here."
                )
            else:
                st.info(
                    f"\U0001f3c6 **{elo_leader}** has the highest Elo rating among the remaining teams, "
                    f"but the simulation favors **{top_team}** for the title ({top_p:.1%} vs "
                    f"{elo_leader_p:.1%} for {elo_leader}) -- the highest-Elo team isn't always the "
                    "simulation's favorite."
                )

        def _highlight(team: str) -> str:
            if team == top_team:
                return "Simulation favorite"
            if team == elo_leader:
                return "Highest Elo"
            return "Other"

        chart_df = sim_df.copy()
        chart_df["Highlight"] = chart_df["team"].apply(_highlight)

        color_scale = alt.Scale(
            domain=["Simulation favorite", "Highest Elo", "Other"],
            range=[COLOR_BLUE, COLOR_RED, COLOR_GRAY],
        )

        team_order = chart_df["team"].tolist()  # already sorted by p_champion descending

        bars = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("p_champion:Q", title="P(win the title)", axis=alt.Axis(format="%")),
                y=alt.Y("team:N", sort=team_order, title=None),
                color=alt.Color("Highlight:N", scale=color_scale, legend=alt.Legend(title=None)),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("p_champion:Q", title="P(champion)", format=".1%"),
                ],
            )
        )
        labels = bars.mark_text(align="left", dx=3, color="#0b0b0b").encode(
            text=alt.Text("p_champion:Q", format=".1%")
        )
        st.altair_chart(
            (bars + labels).properties(height=32 * len(chart_df) + 40),
            use_container_width=True,
        )

        st.subheader("Full stage-by-stage probabilities")
        display_df = sim_df.rename(
            columns={
                "team": "Team",
                "p_quarterfinal": "Reach QF",
                "p_semifinal": "Reach SF",
                "p_final": "Reach Final",
                "p_champion": "Win Title",
            }
        )
        for col in ["Reach QF", "Reach SF", "Reach Final", "Win Title"]:
            display_df[col] = display_df[col].map(lambda v: f"{v:.1%}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
