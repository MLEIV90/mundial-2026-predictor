# Mundial 2026 Predictor

A machine learning project to predict outcomes of the 2026 FIFA World Cup.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads pre-computed data from `data/app/*.json` and never calls the
OddsPapi API itself. Those files are committed to the repo, so the app runs
out of the box -- to refresh them with the latest results and odds:

```bash
python scripts/update_data.py
```

This is the only script that calls OddsPapi (via `src/odds.py`, cached under
`data/odds_cache/` to conserve the free-tier quota). It requires
`ODDSPAPI_API_KEY` set in a `.env` file (see `.env.example`) and takes a
couple of minutes, since it refits the goals model once per already-played
knockout fixture for the backtest. Run it after each new match, then commit
the updated `data/app/*.json` files to publish the results to the app.
