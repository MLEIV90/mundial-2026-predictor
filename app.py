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

from src.blend import DEFAULT_BLEND_WEIGHT
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
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; box-shadow: 0 1px 3px rgba(11,11,11,0.06); }
h1, h2, h3 { letter-spacing: -0.01em; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.app-banner {
  background: linear-gradient(135deg, rgba(42,120,214,0.10) 0%, rgba(42,120,214,0.015) 65%);
  border: 1px solid rgba(42,120,214,0.14);
  border-radius: 16px;
  padding: 22px 28px;
  margin-bottom: 20px;
}
.app-banner-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; color: #0b0b0b; line-height: 1.2; }
.app-banner-subtitle { font-size: 0.95rem; color: #52514e; margin-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-banner"><div class="app-banner-title">⚽ Mundial 2026 Predictor</div>'
    '<div class="app-banner-subtitle">A Dixon-Coles goals model with Elo-adjusted team strength, '
    "benchmarked against the betting market.</div></div>",
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
    return advance_probability(
        model, home, away, elo_home, elo_away, neutral=neutral, blend_weight=DEFAULT_BLEND_WEIGHT
    )


# Confirmed round-of-16 ties and quarterfinals (see src/simulation.py for the
# full bracket definition used by the Monte Carlo simulation).
R16_TIES = [
    ("Portugal", "Spain", True),
    ("United States", "Belgium", False),
    ("Argentina", "Egypt", True),
    ("Switzerland", "Colombia", True),
]
# Only QF1 and QF4 are confirmed matchups (both teams already through);
# QF2/QF3 depend on the still-open round-of-16 ties above.
CONFIRMED_QF = {
    0: ("QF1", "France", "Morocco", True),
    3: ("QF4", "Norway", "England", True),
}

# Bracket geometry: every position is computed once, in Python, rather than
# leaned on CSS flexbox auto-layout -- for a fixed-shape tree like this one,
# explicit pixel math is far more reliable than fighting flexbox/grid for
# perfectly-met connector lines.
CARD_W = 208
CARD_H = 62
COL_GAP = 64
ROW_STEP = 86  # center-to-center spacing between adjacent Round of 16 cards


def _bracket_geometry():
    """Compute (x, y_center) for every card and the connector line segments.

    Only R16 pairs -> QF2/QF3 and QF -> SF -> Final have a real "two
    children, one parent" relationship in what's rendered (QF1 and QF4 are
    already-decided ties with no round-of-16 match shown, so they get no
    incoming connector) -- geometry follows that shape exactly rather than
    assuming a perfectly symmetric bracket.
    """
    r16_y = [CARD_H / 2 + i * ROW_STEP for i in range(4)]
    qf2_y = (r16_y[0] + r16_y[1]) / 2
    qf3_y = (r16_y[2] + r16_y[3]) / 2
    qf1_y = qf2_y - ROW_STEP
    qf4_y = qf3_y + ROW_STEP
    sf1_y = (qf1_y + qf2_y) / 2
    sf2_y = (qf3_y + qf4_y) / 2
    final_y = (sf1_y + sf2_y) / 2

    all_y = r16_y + [qf1_y, qf2_y, qf3_y, qf4_y, sf1_y, sf2_y, final_y]
    shift = CARD_H / 2 - min(all_y)

    col_x = [i * (CARD_W + COL_GAP) for i in range(4)]

    return {
        "r16_y": [y + shift for y in r16_y],
        "qf_y": [qf1_y + shift, qf2_y + shift, qf3_y + shift, qf4_y + shift],
        "sf_y": [sf1_y + shift, sf2_y + shift],
        "final_y": final_y + shift,
        "col_x": col_x,
        "total_h": max(all_y) - min(all_y) + CARD_H,
        "total_w": col_x[-1] + CARD_W,
    }


def _connector_html(y1: float, y2: float, x_from: float, gap: float) -> str:
    """Elbow connector from two child cards (at y1, y2) into one parent card
    (vertically centered between them) -- two stubs in, a vertical bar, one
    stub out.
    """
    elbow_x = x_from + gap / 2
    parent_y = (y1 + y2) / 2
    top, bottom = min(y1, y2), max(y1, y2)
    return f"""
    <div class="bx-conn-h" style="top:{y1 - 1:.1f}px; left:{x_from:.1f}px; width:{gap / 2:.1f}px;"></div>
    <div class="bx-conn-h" style="top:{y2 - 1:.1f}px; left:{x_from:.1f}px; width:{gap / 2:.1f}px;"></div>
    <div class="bx-conn-v" style="top:{top:.1f}px; left:{elbow_x - 1:.1f}px; height:{bottom - top:.1f}px;"></div>
    <div class="bx-conn-h" style="top:{parent_y - 1:.1f}px; left:{elbow_x:.1f}px; width:{gap / 2:.1f}px;"></div>
    """


def _decided_card_html(x: float, y: float, label: str, home: str, away: str, p_home: float, p_away: float) -> str:
    home_fav = p_home >= p_away
    home_cls = "bx-team bx-fav" if home_fav else "bx-team"
    away_cls = "bx-team bx-fav" if not home_fav else "bx-team"
    # The bar is a two-segment split (home share, away share) where the
    # favored team's segment is always the accent color, so the bar and the
    # bold/accent-colored name always point at the same team.
    home_color = COLOR_BLUE if home_fav else COLOR_GRAY
    away_color = COLOR_BLUE if not home_fav else COLOR_GRAY
    return f"""
    <div class="bx-card bx-has-fav" style="left:{x:.1f}px; top:{y - CARD_H / 2:.1f}px; width:{CARD_W}px; height:{CARD_H}px;">
      <div class="bx-label">{label}</div>
      <div class="{home_cls}"><span>{home}</span><span class="bx-pct">{p_home:.0%}</span></div>
      <div class="{away_cls}"><span>{away}</span><span class="bx-pct">{p_away:.0%}</span></div>
      <div class="bx-bar">
        <div class="bx-bar-seg" style="width:{p_home * 100:.1f}%; background:{home_color};"></div>
        <div class="bx-bar-seg" style="width:{p_away * 100:.1f}%; background:{away_color};"></div>
      </div>
    </div>
    """


def _pending_card_html(x: float, y: float, label: str, line1: str, line2: str) -> str:
    return f"""
    <div class="bx-card bx-pending" style="left:{x:.1f}px; top:{y - CARD_H / 2:.1f}px; width:{CARD_W}px; height:{CARD_H}px;">
      <div class="bx-label">{label}</div>
      <div class="bx-team bx-tbd">{line1}</div>
      <div class="bx-team bx-tbd">{line2}</div>
    </div>
    """


BRACKET_CSS = f"""
<style>
.bx-wrap {{ overflow-x: auto; padding: 8px 4px 20px 4px; }}
.bx-headers {{ display: flex; gap: {COL_GAP}px; margin-bottom: 12px; min-width: max-content; }}
.bx-headers > div {{ width: {CARD_W}px; font-weight: 700; font-size: 0.8rem; color: {COLOR_BLUE};
  text-transform: uppercase; letter-spacing: 0.06em; padding-left: 2px; }}
.bx-canvas {{ position: relative; min-width: max-content; }}
.bx-col-bg {{ position: absolute; top: -8px; background: rgba(42,120,214,0.025);
  border-radius: 14px; }}
.bx-card {{ position: absolute; background: #ffffff; border: 1px solid rgba(11,11,11,0.08);
  border-left: 3px solid {COLOR_GRAY}; border-radius: 9px; padding: 7px 11px;
  box-shadow: 0 2px 5px rgba(11,11,11,0.06); box-sizing: border-box; transition: box-shadow 0.15s ease; }}
.bx-card.bx-has-fav {{ border-left: 3px solid {COLOR_BLUE}; }}
.bx-card.bx-pending {{ background: #f9f9f7; border: 1px dashed #c3c2b7; border-left: 3px dashed #c3c2b7;
  box-shadow: none; }}
.bx-label {{ font-size: 0.68rem; font-weight: 700; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 3px;
  text-transform: uppercase; letter-spacing: 0.03em; }}
.bx-team {{ display: flex; justify-content: space-between; font-size: 0.87rem; line-height: 1.4; color: #0b0b0b; }}
.bx-team.bx-fav {{ font-weight: 700; color: {COLOR_BLUE}; }}
.bx-team.bx-tbd {{ color: #898781; font-style: italic; font-weight: 400; }}
.bx-pct {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
.bx-bar {{ margin-top: 5px; height: 5px; border-radius: 3px; overflow: hidden; display: flex; }}
.bx-bar-seg {{ height: 100%; }}
.bx-bar-seg:first-child {{ margin-right: 2px; }}
.bx-conn-h {{ position: absolute; height: 2.5px; background: #b7b6ac; border-radius: 2px; }}
.bx-conn-v {{ position: absolute; width: 2.5px; background: #b7b6ac; border-radius: 2px; }}
</style>
"""


def render_bracket(model, current_ratings: dict) -> None:
    st.subheader("Remaining Bracket")
    st.caption(
        "Confirmed ties show the model's P(advance) with the favored team highlighted; "
        "undetermined slots (dashed) show the pending matchup."
    )

    geo = _bracket_geometry()
    col_x = geo["col_x"]

    parts = [BRACKET_CSS]
    parts.append(
        '<div class="bx-wrap"><div class="bx-headers">'
        '<div>Round of 16</div><div>Quarterfinals</div><div>Semifinals</div><div>Final</div>'
        "</div>"
    )
    parts.append(f'<div class="bx-canvas" style="height:{geo["total_h"]:.0f}px; width:{geo["total_w"]:.0f}px;">')

    # Subtle background band per round column, added first so cards and
    # connectors (added after) stack visually on top of them.
    band_h = geo["total_h"] + 16
    for x in col_x:
        parts.append(
            f'<div class="bx-col-bg" style="left:{x - 10:.1f}px; width:{CARD_W + 20}px; height:{band_h:.0f}px;"></div>'
        )

    # Round of 16
    for i, (home, away, neutral) in enumerate(R16_TIES):
        adv = _tie_probs(model, current_ratings, home, away, neutral)
        parts.append(
            _decided_card_html(
                col_x[0], geo["r16_y"][i], f"R16-{i + 1}", home, away, adv.p_a_advances, adv.p_b_advances
            )
        )

    # Quarterfinals
    for i in (0, 3):
        label, home, away, neutral = CONFIRMED_QF[i]
        adv = _tie_probs(model, current_ratings, home, away, neutral)
        parts.append(
            _decided_card_html(col_x[1], geo["qf_y"][i], label, home, away, adv.p_a_advances, adv.p_b_advances)
        )

    parts.append(_pending_card_html(col_x[1], geo["qf_y"][1], "QF2", "Winner: Portugal / Spain", "Winner: USA / Belgium"))
    parts.append(_pending_card_html(col_x[1], geo["qf_y"][2], "QF3", "Winner: Argentina / Egypt", "Winner: Switzerland / Colombia"))

    # Semifinals + Final (fully undetermined)
    parts.append(_pending_card_html(col_x[2], geo["sf_y"][0], "SF1", "Winner QF1", "Winner QF2"))
    parts.append(_pending_card_html(col_x[2], geo["sf_y"][1], "SF2", "Winner QF3", "Winner QF4"))
    parts.append(_pending_card_html(col_x[3], geo["final_y"], "Final", "Winner SF1", "Winner SF2"))

    # Connectors: R16 pairs -> QF2/QF3, QF -> SF, SF -> Final
    card_right = col_x[0] + CARD_W
    parts.append(_connector_html(geo["r16_y"][0], geo["r16_y"][1], card_right, COL_GAP))
    parts.append(_connector_html(geo["r16_y"][2], geo["r16_y"][3], card_right, COL_GAP))

    card_right = col_x[1] + CARD_W
    parts.append(_connector_html(geo["qf_y"][0], geo["qf_y"][1], card_right, COL_GAP))
    parts.append(_connector_html(geo["qf_y"][2], geo["qf_y"][3], card_right, COL_GAP))

    card_right = col_x[2] + CARD_W
    parts.append(_connector_html(geo["sf_y"][0], geo["sf_y"][1], card_right, COL_GAP))

    parts.append("</div></div>")

    # Markdown treats any line indented >=4 spaces as a code block, which
    # would otherwise render this HTML as literal text instead of parsing
    # it -- the card/connector helpers above are written as pretty-printed,
    # indented triple-quoted strings for readability, so strip that
    # indentation back out line-by-line before handing it to st.markdown.
    html = "\n".join(line.strip() for line in "".join(parts).splitlines())
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Load everything the header / KPI row needs up front
# --------------------------------------------------------------------------
ratings_data = _load_json("model_ratings.json")
live_data = _load_json("live_predictions.json")
backtest_data = _load_json("backtest.json")
backtest_90_data = _load_json("backtest_90min.json")
sim_data = _load_json("simulation.json")

kpi1, kpi2, kpi3 = st.columns(3)
if sim_data is not None and sim_data["results"]:
    top_row = max(sim_data["results"], key=lambda r: r["p_champion"])
    kpi1.metric("Title favorite", top_row["team"], f"{top_row['p_champion']:.1%} to win")
else:
    kpi1.metric("Title favorite", "n/a")

if backtest_90_data is not None:
    s = backtest_90_data["summary"]
    diff = abs(s["model_brier"] - s["market_brier"])
    kpi2.metric(
        "Model vs market (fair, 90')",
        f"{s['model_brier']:.4f}",
        f"competitive with market (Δ {diff:.4f})",
        delta_color="off",
        help=(
            "Brier score on the 90-minute 1X2 result -- the fair, apples-to-apples "
            "comparison. On this small in-tournament sample the model is roughly on "
            "par with Pinnacle's closing line, not beating it. See the Track Record tab."
        ),
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
    st.info(
        "\U0001f4cb **Preliminary, in-tournament backtest.** Small sample (fewer than 25 "
        "knockout fixtures played so far) -- read every number on this tab as an early "
        "signal, not a claim of long-run edge over the market."
    )

    st.subheader("Fair comparison: 90-minute 1X2")
    st.caption(
        "Model vs market on exactly what the market prices pre-match -- the 90-minute "
        "result (home win / draw / away win) -- with no extra-time/penalty logic on "
        "either side. **On this sample the model is competitive with / roughly on par "
        "with Pinnacle's closing line -- this is not a case of the model beating the "
        "market.**"
    )

    if backtest_90_data is None:
        st.warning("No 90-minute backtest found. Run `python scripts/update_data.py` first.")
    else:
        summary_90 = backtest_90_data["summary"]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Brier score (lower is better)",
            f"{summary_90['model_brier']:.4f}",
            delta=f"{summary_90['model_brier'] - summary_90['market_brier']:+.4f} vs market",
            delta_color="inverse",
        )
        col2.metric(
            "Log loss (lower is better)",
            f"{summary_90['model_log_loss']:.4f}",
            delta=f"{summary_90['model_log_loss'] - summary_90['market_log_loss']:+.4f} vs market",
            delta_color="inverse",
        )
        col3.metric(
            "Fixtures model was closer on",
            f"{summary_90['n_model_beats_market']} / {summary_90['n_fixtures']}",
        )
        st.caption(
            f"Market Brier: {summary_90['market_brier']:.4f} · "
            f"Market log loss: {summary_90['market_log_loss']:.4f} · "
            "small in-tournament sample, not a claim of long-run edge."
        )

        with st.expander("Fixture-by-fixture (90-minute)"):
            fixtures_90_df = pd.DataFrame(backtest_90_data["fixtures"])
            fixtures_90_df = fixtures_90_df.rename(
                columns={
                    "date": "Date",
                    "home_team": "Home",
                    "away_team": "Away",
                    "result": "Result",
                    "model_p_home": "Model P(Home)",
                    "model_p_draw": "Model P(Draw)",
                    "model_p_away": "Model P(Away)",
                    "market_p_home": "Market P(Home)",
                    "market_p_draw": "Market P(Draw)",
                    "market_p_away": "Market P(Away)",
                    "model_beat_market": "Model closer",
                }
            )
            for col in ["Model P(Home)", "Model P(Draw)", "Model P(Away)", "Market P(Home)", "Market P(Draw)", "Market P(Away)"]:
                fixtures_90_df[col] = fixtures_90_df[col].map(lambda v: f"{v:.1%}")
            display_cols_90 = [
                "Date", "Home", "Away", "Result",
                "Model P(Home)", "Model P(Draw)", "Model P(Away)",
                "Market P(Home)", "Market P(Draw)", "Market P(Away)",
                "Model closer",
            ]
            st.dataframe(fixtures_90_df[display_cols_90], hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Approximation: knockout advancement")
    st.caption(
        "⚠️ **Not an apples-to-apples comparison** -- the model runs its full "
        "extra-time/penalty machinery, while the market side is only ever the crude "
        "P(win) + 0.5*P(draw) coin-flip approximation, since there's no real market for "
        "\"wins the tie.\" Part of any model edge below could simply be the model doing "
        "more work on a question the market was never asked. Kept for reference "
        "alongside the fair comparison above, not in place of it."
    )

    if backtest_data is None:
        st.warning("No backtest found. Run `python scripts/update_data.py` first.")
    else:
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
            "Fixtures model was closer on",
            f"{summary['n_model_beats_market']} / {summary['n_fixtures']}",
        )
        st.caption(f"Market Brier: {summary['market_brier']:.4f} · Market log loss: {summary['market_log_loss']:.4f}")

        with st.expander("Fixture-by-fixture (advance approximation)"):
            fixtures_df = pd.DataFrame(backtest_data["fixtures"])
            fixtures_df = fixtures_df.rename(
                columns={
                    "date": "Date",
                    "home_team": "Home",
                    "away_team": "Away",
                    "winner": "Advanced",
                    "p_model_home_advances": "Model P(Home)",
                    "p_market_home_advances": "Market P(Home)",
                    "model_beat_market": "Model closer",
                }
            )
            fixtures_df["Model P(Home)"] = fixtures_df["Model P(Home)"].map(lambda v: f"{v:.1%}")
            fixtures_df["Market P(Home)"] = fixtures_df["Market P(Home)"].map(lambda v: f"{v:.1%}")
            display_cols = ["Date", "Home", "Away", "Advanced", "Model P(Home)", "Market P(Home)", "Model closer"]
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
        team_a = st.selectbox("Team A", teams, index=default_a)
    with col_b:
        team_b = st.selectbox("Team B", teams, index=default_b)

    host_options = ["Neutral (no host advantage)", f"{team_a} is the host", f"{team_b} is the host"]
    host_choice = st.selectbox(
        "Host nation playing at home?",
        host_options,
        index=0,
        help=(
            "World Cup knockout matches are at neutral venues by default -- picking a "
            "team here isn't about which dropdown it's in, only whether that specific "
            "team is the actual tournament host playing on home soil."
        ),
    )
    use_blend = st.checkbox(
        "Apply Elo correction (recommended)",
        value=True,
        help=(
            "Corrects the Poisson goals model's conservatism on clear favorites "
            "using pure Elo win/draw/loss probabilities -- see src/blend.py. "
            "Calibrated by backtesting against actual results (not the market): "
            "on the 22 knockout fixtures backtested so far, full Elo minimized "
            "prediction error, so that's the default. Uncheck to see the "
            "pure-Poisson prediction for comparison."
        ),
    )
    blend_weight = DEFAULT_BLEND_WEIGHT if use_blend else 1.0

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        elo_a = current_ratings.get(team_a, 1500.0)
        elo_b = current_ratings.get(team_b, 1500.0)

        # Home advantage must follow actual host status, never dropdown slot --
        # so team_b is only ever passed as the model's "home" team when team_b
        # is explicitly the selected host; results are swapped back afterward
        # so "Team A"/"Team B" always refer to the dropdowns, not the model's
        # internal home/away slots.
        b_is_host = host_choice == host_options[2]
        neutral = host_choice == host_options[0]
        if b_is_host:
            model_home, model_away, elo_home, elo_away = team_b, team_a, elo_b, elo_a
        else:
            model_home, model_away, elo_home, elo_away = team_a, team_b, elo_a, elo_b

        match_pred = predict_match(
            model, model_home, model_away, elo_home, elo_away, neutral=neutral, blend_weight=blend_weight
        )
        adv = advance_probability(
            model, model_home, model_away, elo_home, elo_away, neutral=neutral, blend_weight=blend_weight
        )

        if b_is_host:
            p_a_advances, p_b_advances = adv.p_b_advances, adv.p_a_advances
            a_win, draw, b_win = match_pred.away_win, match_pred.draw, match_pred.home_win
            a_xg, b_xg = match_pred.away_xg, match_pred.home_xg
            matrix = match_pred.score_matrix.T
        else:
            p_a_advances, p_b_advances = adv.p_a_advances, adv.p_b_advances
            a_win, draw, b_win = match_pred.home_win, match_pred.draw, match_pred.away_win
            a_xg, b_xg = match_pred.home_xg, match_pred.away_xg
            matrix = match_pred.score_matrix
        matrix = matrix.rename_axis(index="Team A goals", columns="Team B goals")

        st.subheader("If this were a knockout tie")
        col1, col2 = st.columns(2)
        col1.metric(f"{team_a} advances", f"{p_a_advances:.1%}")
        col2.metric(f"{team_b} advances", f"{p_b_advances:.1%}")

        st.subheader("Regulation time (90')")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{team_a} win", f"{a_win:.1%}")
        col2.metric("Draw", f"{draw:.1%}")
        col3.metric(f"{team_b} win", f"{b_win:.1%}")

        st.metric("Expected goals", f"{a_xg:.2f} - {b_xg:.2f}")

        flat_idx = matrix.values.argmax()
        most_likely_a, most_likely_b = divmod(flat_idx, matrix.shape[1])
        st.metric(
            "Most likely scoreline",
            f"{team_a} {most_likely_a} - {most_likely_b} {team_b}",
            help=f"P(this exact scoreline) = {matrix.values.max():.1%}",
        )

        with st.expander("Full score matrix (rows = Team A goals, columns = Team B goals)"):
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
