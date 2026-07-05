"""Download and validate international football results data."""

import io

import pandas as pd
import requests

RESULTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def load_results(url: str = RESULTS_CSV_URL) -> pd.DataFrame:
    """Download results.csv from the source repo and load it into a DataFrame."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), parse_dates=["date"])
    return df


def run_sanity_checks(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the results DataFrame and print a summary."""
    invalid_dates = df["date"].isna().sum()
    assert invalid_dates == 0, f"Found {invalid_dates} invalid/missing dates"

    negative_goals = df[(df["home_score"] < 0) | (df["away_score"] < 0)]
    assert negative_goals.empty, f"Found {len(negative_goals)} rows with negative goals"

    duplicates = df.duplicated()
    assert duplicates.sum() == 0, f"Found {duplicates.sum()} duplicate rows"

    print("Sanity checks passed: valid dates, no negative goals, no duplicate rows.")
    print(f"Number of matches: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    results = load_results()
    run_sanity_checks(results)
