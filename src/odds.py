"""World Cup 1X2 betting odds from the OddsPapi API, with local caching.

The free OddsPapi tier allows 250 requests/month, so this module treats
the local cache (``data/odds_cache/``) as the source of truth and only
ever calls the API for data it doesn't already have on disk. Re-running a
notebook that only reads already-cached fixtures costs zero requests.

Authentication
--------------
The API key is read from the ``ODDSPAPI_API_KEY`` environment variable,
loaded via ``python-dotenv`` from a ``.env`` file in the project root
(see ``.env.example``). It is never hardcoded, and it is only required
when an actual network call is needed -- a fully cached lookup works
without a key at all.

Endpoints used (see https://oddspapi.io/en/docs)
--------------------------------------------------
- ``GET /v4/tournaments`` -- resolve the World Cup's ``tournamentId`` for
  soccer (``sportId=10``).
- ``GET /v4/fixtures``    -- list fixtures for that tournament.
- ``GET /v4/odds``        -- pre-game odds for one fixture, across all
  bookmakers that cover it. The match-result ("1X2") market is market id
  ``101``, with outcome ids ``101``/``102``/``103`` for home/draw/away.
  Only has data for fixtures that haven't kicked off yet -- once a match
  starts (or finishes) this returns ``hasOdds: False`` with no prices.
- ``GET /v4/historical-odds`` -- fallback for already-started/finished
  fixtures: the full time series of price updates per outcome, capped to
  3 bookmaker slugs per call (defaults to Pinnacle only here). This
  series keeps going into live play (and past full time), so the closing
  (final pre-kickoff) price is the last entry *before* the fixture's
  kickoff time, not simply the last entry overall -- see
  ``_closing_price``.

De-vigging
----------
A bookmaker's quoted decimal odds imply a probability of ``1 / odds`` per
outcome, but these three raw probabilities always sum to slightly more
than 1 -- the excess is the bookmaker's margin ("vig" or "overround").
``devig_1x2`` removes it by simply renormalizing the three raw
probabilities so they sum to exactly 1. Pinnacle is used when available
since it is a low-margin, sharp book widely used as a market-consensus
proxy; otherwise the raw odds are averaged across whichever bookmakers
quote a complete 1X2 price for that fixture, and *then* de-vigged.

Rate limiting
-------------
Every actual network call is throttled to a guaranteed minimum interval
(``MIN_REQUEST_INTERVAL_SECONDS``) since the *start* of the previous real
request, tracked in a module-level timestamp rather than a fixed
post-request sleep -- a fixed sleep only bounds the gap between the *end*
of one call and the *start* of the next, so bursts of several call sites
in a row (e.g. resolving a fixture, then its odds, then its historical
odds as a fallback) can still land requests closer together than the
API's own per-request limit if there's any variance in processing time.
This never applies to cache hits.

On an HTTP 429 (rate limited), ``_request`` retries automatically: it
reads how long the API wants us to wait (from the ``Retry-After`` header
or a ``retryAfter``/``retryMs`` field in the response body, whichever is
present), sleeps at least that long, and tries again, up to
``MAX_RETRY_ATTEMPTS`` times with the wait floor growing exponentially
attempt over attempt (in case the reported wait is unhelpfully small or
the limit is being hit repeatedly for other reasons). Only once every
attempt is exhausted does it raise ``OddsApiRateLimitError``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.oddspapi.io"
SPORT_ID_SOCCER = 10
MATCH_WINNER_MARKET_ID = "101"
OUTCOME_HOME = "101"
OUTCOME_DRAW = "102"
OUTCOME_AWAY = "103"

# OddsPapi's own limit is ~0.87s between requests; a little margin above
# that absorbs clock/scheduling jitter (a bare 0.88s margin was tight
# enough to occasionally 429 in CI -- see _throttle()).
MIN_REQUEST_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30

MAX_RETRY_ATTEMPTS = 5
RETRY_BACKOFF_BASE_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 30.0

CACHE_DIR = Path("data/odds_cache")

# Timestamp (time.monotonic()) of the start of the last real HTTP request
# made to OddsPapi, across every call site -- see _throttle().
_last_request_time: Optional[float] = None


class OddsApiError(Exception):
    """Base error for anything that goes wrong talking to OddsPapi."""


class OddsApiAuthError(OddsApiError):
    """Raised when ODDSPAPI_API_KEY is missing."""


class OddsApiRateLimitError(OddsApiError):
    """Raised on an HTTP 429 (monthly quota or rate limit exceeded)."""


class FixtureNotFoundError(OddsApiError):
    """Raised when the API has no record of a requested fixture."""


def has_api_key() -> bool:
    """Return True if ODDSPAPI_API_KEY is set (loading .env first)."""
    load_dotenv()
    return bool(os.environ.get("ODDSPAPI_API_KEY"))


def _get_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("ODDSPAPI_API_KEY")
    if not api_key:
        raise OddsApiAuthError(
            "ODDSPAPI_API_KEY is not set. Create a .env file in the project "
            "root with ODDSPAPI_API_KEY=<your key> (see .env.example)."
        )
    return api_key


def _throttle() -> None:
    """Block, if needed, until at least MIN_REQUEST_INTERVAL_SECONDS has
    elapsed since the start of the last real HTTP request, then record
    "now" as the new last-request time.

    Tracking this in a module-level timestamp (rather than sleeping a
    fixed amount after every call) enforces the gap between *every* pair
    of consecutive requests regardless of which function issued them or
    how much processing happened in between.
    """
    global _last_request_time
    now = time.monotonic()
    if _last_request_time is not None:
        remaining = MIN_REQUEST_INTERVAL_SECONDS - (now - _last_request_time)
        if remaining > 0:
            time.sleep(remaining)
            now = time.monotonic()
    _last_request_time = now


def _parse_retry_after(response: requests.Response) -> float:
    """Best-effort extraction of how long OddsPapi wants us to wait before
    retrying a 429, in seconds. Checks the standard ``Retry-After`` header
    first, then common response-body field names (seconds or
    milliseconds); falls back to MIN_REQUEST_INTERVAL_SECONDS if none of
    them are present or parseable.
    """
    header_value = response.headers.get("Retry-After")
    if header_value is not None:
        try:
            return max(float(header_value), 0.0)
        except ValueError:
            pass

    try:
        body = response.json()
    except ValueError:
        body = {}

    for key in ("retryAfter", "retry_after"):
        value = body.get(key) if isinstance(body, dict) else None
        if isinstance(value, (int, float)):
            return max(float(value), 0.0)

    for key in ("retryMs", "retry_ms"):
        value = body.get(key) if isinstance(body, dict) else None
        if isinstance(value, (int, float)):
            return max(float(value) / 1000.0, 0.0)

    return MIN_REQUEST_INTERVAL_SECONDS


def _request(path: str, params: dict) -> dict:
    """GET against the OddsPapi API, with the API key attached, a
    guaranteed minimum interval before every request (see ``_throttle``),
    and automatic retry with backoff on a 429 (rate limited).
    """
    query = dict(params)
    query["apiKey"] = _get_api_key()

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        _throttle()
        response = requests.get(f"{BASE_URL}{path}", params=query, timeout=REQUEST_TIMEOUT_SECONDS)

        if response.status_code == 429:
            if attempt == MAX_RETRY_ATTEMPTS:
                raise OddsApiRateLimitError(
                    f"Rate limited by OddsPapi calling {path} after {MAX_RETRY_ATTEMPTS} attempts."
                )
            # Wait at least as long as the API told us to (if it told us),
            # with a floor that grows exponentially attempt over attempt --
            # never a busy-loop, and never depends only on a possibly-tiny
            # reported wait. Never logs `query`/the API key, only the path
            # and how long we're backing off.
            reported_wait = _parse_retry_after(response)
            backoff_floor = min(
                RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SECONDS
            )
            wait_seconds = max(reported_wait, backoff_floor)
            print(
                f"  Rate limited by OddsPapi calling {path} "
                f"(attempt {attempt}/{MAX_RETRY_ATTEMPTS}) -- backing off {wait_seconds:.2f}s."
            )
            time.sleep(wait_seconds)
            continue

        if response.status_code == 404:
            raise FixtureNotFoundError(f"Not found calling {path} with {params}: {response.text}")
        if not response.ok:
            # requests' default HTTPError message embeds the full request URL,
            # which would leak the API key (passed as a query param) into logs
            # and tracebacks -- raise a redacted error instead, with no
            # exception chaining back to anything that holds the URL.
            raise OddsApiError(
                f"OddsPapi request to {path} with {params} failed: HTTP {response.status_code}"
            ) from None
        return response.json()


def _cache_get(cache_key: str):
    path = CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _cache_put(cache_key: str, data) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def fetch_fixture_odds(fixture_id: str, force_refresh: bool = False) -> dict:
    """Fetch raw pre-game odds for one fixture, transparently cached.

    Every response is cached to ``data/odds_cache/{fixture_id}.json``. If
    that file already exists, it is returned directly and the API is
    never called (and no API key is required). Pass ``force_refresh=True``
    to bypass the cache and re-fetch from the API.
    """
    cache_key = f"fixture_{fixture_id}"
    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    data = _request("/v4/odds", {"fixtureId": fixture_id, "oddsFormat": "decimal", "verbosity": 3})
    _cache_put(cache_key, data)
    return data


def _find_world_cup_tournament_id(tournament_name: str = "World Cup", force_refresh: bool = False) -> int:
    """Resolve the tournamentId for ``tournament_name`` among soccer tournaments.

    The tournament list is cheap and slow-changing, so it is cached too
    (under ``data/odds_cache/tournaments_soccer.json``) even though the
    caching contract emphasized in this module is per-fixture.
    """
    cache_key = "tournaments_soccer"
    tournaments = None if force_refresh else _cache_get(cache_key)
    if tournaments is None:
        tournaments = _request("/v4/tournaments", {"sportId": SPORT_ID_SOCCER})
        _cache_put(cache_key, tournaments)

    for tournament in tournaments:
        if tournament.get("tournamentName") == tournament_name:
            return tournament["tournamentId"]

    raise OddsApiError(
        f"No soccer tournament named {tournament_name!r} found via /v4/tournaments."
    )


def find_world_cup_fixtures(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    status_id: Optional[int] = None,
    force_refresh: bool = False,
) -> List[dict]:
    """List World Cup fixtures (soccer, tournamentName "World Cup").

    Parameters
    ----------
    from_date, to_date:
        Optional ISO 8601 date/datetime bounds passed straight through to
        ``/v4/fixtures``.
    status_id:
        Optional fixture status filter (0=not started, 1=live,
        2=finished, 3=cancelled).
    force_refresh:
        Bypass the cache and re-fetch from the API.

    Returns
    -------
    A list of raw fixture dicts as returned by ``/v4/fixtures``, each
    with at least ``fixtureId``, ``participant1Name``, ``participant2Name``
    and ``startTime``.
    """
    tournament_id = _find_world_cup_tournament_id(force_refresh=force_refresh)

    params = {"sportId": SPORT_ID_SOCCER, "tournamentId": tournament_id}
    if from_date is not None:
        params["from"] = from_date
    if to_date is not None:
        params["to"] = to_date
    if status_id is not None:
        params["statusId"] = status_id

    cache_key = f"fixtures_{tournament_id}_{from_date}_{to_date}_{status_id}"
    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    data = _request("/v4/fixtures", params)
    _cache_put(cache_key, data)
    return data


# Known team-name spelling differences between results.csv (martj42) and
# OddsPapi's participant names, keyed by the results.csv spelling (lowercase).
TEAM_NAME_ALIASES = {
    "united states": "usa",
    "dr congo": "congo dr",
}


def _normalize_team_name(name: str) -> str:
    lower = name.lower()
    return TEAM_NAME_ALIASES.get(lower, lower)


def find_fixture_id_by_teams(fixtures: List[dict], home_team: str, away_team: str) -> Optional[str]:
    """Find a fixtureId in a list of fixtures by team names.

    Matches case-insensitively, applying a small known-alias table
    (``TEAM_NAME_ALIASES``) for spelling differences between results.csv
    and OddsPapi's participant names (e.g. "United States" vs. "USA").
    This is still a best-effort match, not a fuzzy one -- check the
    fixture list manually if it returns None.
    """
    home_norm = _normalize_team_name(home_team)
    away_norm = _normalize_team_name(away_team)
    for fixture in fixtures:
        if (
            _normalize_team_name(fixture.get("participant1Name", "")) == home_norm
            and _normalize_team_name(fixture.get("participant2Name", "")) == away_norm
        ):
            return fixture.get("fixtureId")
    return None


def get_market_probabilities_for_teams(
    fixtures: List[dict], home_team: str, away_team: str, force_refresh: bool = False
) -> Optional["MarketOdds"]:
    """Look up a fixture by team names within an already-fetched fixtures
    list (see ``find_world_cup_fixtures``) and de-vig its 1X2 odds.

    Returns None if no fixture in ``fixtures`` matches ``home_team``/
    ``away_team`` (e.g. a team-name spelling mismatch between data
    sources), so callers can skip gracefully instead of crashing.
    """
    fixture_id = find_fixture_id_by_teams(fixtures, home_team, away_team)
    if fixture_id is None:
        return None
    return get_fixture_market_probabilities(fixture_id, force_refresh=force_refresh)


def devig_1x2(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    """Convert decimal 1X2 odds into de-vigged implied probabilities.

    Each raw implied probability is ``1 / odds``; these three always sum
    to slightly more than 1 because of the bookmaker's margin. Dividing
    each by that sum removes the margin and yields probabilities that
    sum to exactly 1.
    """
    raw_home = 1.0 / home_odds
    raw_draw = 1.0 / draw_odds
    raw_away = 1.0 / away_odds
    total = raw_home + raw_draw + raw_away
    return raw_home / total, raw_draw / total, raw_away / total


def _outcome_price(bookmaker: dict, outcome_id: str) -> Optional[float]:
    market = bookmaker.get("markets", {}).get(MATCH_WINNER_MARKET_ID)
    if not market:
        return None
    outcome = market.get("outcomes", {}).get(outcome_id)
    if not outcome:
        return None
    player = outcome.get("players", {}).get("0")
    if not player:
        return None
    return player.get("price")


def _extract_1x2_odds(raw: dict) -> Tuple[float, float, float, str]:
    """Pick 1X2 odds from a raw ``/v4/odds`` response.

    Prefers Pinnacle when it has a complete 1X2 price; otherwise averages
    the raw decimal odds across every bookmaker that does.

    Returns ``(home_odds, draw_odds, away_odds, source)`` where ``source``
    is ``"pinnacle"`` or ``"average of N bookmakers"``.
    """
    bookmaker_odds: Dict[str, dict] = raw.get("bookmakerOdds", {})

    pinnacle = bookmaker_odds.get("pinnacle")
    if pinnacle is not None:
        home = _outcome_price(pinnacle, OUTCOME_HOME)
        draw = _outcome_price(pinnacle, OUTCOME_DRAW)
        away = _outcome_price(pinnacle, OUTCOME_AWAY)
        if home is not None and draw is not None and away is not None:
            return home, draw, away, "pinnacle"

    homes, draws, aways = [], [], []
    for bookmaker in bookmaker_odds.values():
        home = _outcome_price(bookmaker, OUTCOME_HOME)
        draw = _outcome_price(bookmaker, OUTCOME_DRAW)
        away = _outcome_price(bookmaker, OUTCOME_AWAY)
        if home is not None and draw is not None and away is not None:
            homes.append(home)
            draws.append(draw)
            aways.append(away)

    if not homes:
        raise OddsApiError("No bookmaker in this response has a complete 1X2 price.")

    n = len(homes)
    return sum(homes) / n, sum(draws) / n, sum(aways) / n, f"average of {n} bookmakers"


def fetch_fixture_historical_odds(
    fixture_id: str, bookmakers: str = "pinnacle", force_refresh: bool = False
) -> dict:
    """Fetch the historical odds time series for one fixture.

    ``/v4/odds`` only has data for fixtures that haven't kicked off yet --
    for an already-played fixture it returns ``hasOdds: False`` with no
    bookmaker data. This is the fallback for that case: it returns every
    price update recorded for the fixture, including ones made during
    live play -- see ``_closing_price`` for how the actual closing
    (pre-kickoff) price is picked out of that series.

    ``bookmakers`` is a comma-separated list of at most 3 bookmaker slugs
    (an OddsPapi limit on this endpoint) and defaults to Pinnacle only,
    matching this module's bookmaker preference and keeping the call
    cheap. Cached to
    ``data/odds_cache/fixture_historical_{fixture_id}_{bookmakers}.json``.
    """
    cache_key = f"fixture_historical_{fixture_id}_{bookmakers}"
    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    data = _request("/v4/historical-odds", {"fixtureId": fixture_id, "bookmakers": bookmakers})
    _cache_put(cache_key, data)
    return data


def _closing_price(bookmaker: dict, outcome_id: str, kickoff: str) -> Optional[float]:
    """Last recorded price strictly before ``kickoff`` (ISO 8601, UTC).

    ``/v4/historical-odds`` keeps recording price updates as the market
    stays open into live play, so the *last* snapshot overall is an
    in-play (or post-match, settled) price, not a pre-game one -- it must
    be filtered to snapshots before kickoff first. ISO 8601 UTC timestamps
    with "Z" sort lexicographically the same as chronologically, so a
    plain string comparison is enough.
    """
    market = bookmaker.get("markets", {}).get(MATCH_WINNER_MARKET_ID)
    if not market:
        return None
    outcome = market.get("outcomes", {}).get(outcome_id)
    if not outcome:
        return None
    snapshots = outcome.get("players", {}).get("0")
    if not snapshots:
        return None
    pre_kickoff = [s for s in snapshots if s.get("createdAt", "") < kickoff]
    if not pre_kickoff:
        return None
    return pre_kickoff[-1].get("price")


def _extract_1x2_odds_historical(raw: dict, kickoff: str) -> Tuple[float, float, float, str]:
    """Pick closing (pre-kickoff) 1X2 odds from a raw ``/v4/historical-odds`` response.

    Same Pinnacle-first, else-average logic as ``_extract_1x2_odds``, but
    reading the last pre-``kickoff`` price from each outcome's time series
    instead of a single current price.
    """
    bookmakers: Dict[str, dict] = raw.get("bookmakers", {})

    pinnacle = bookmakers.get("pinnacle")
    if pinnacle is not None:
        home = _closing_price(pinnacle, OUTCOME_HOME, kickoff)
        draw = _closing_price(pinnacle, OUTCOME_DRAW, kickoff)
        away = _closing_price(pinnacle, OUTCOME_AWAY, kickoff)
        if home is not None and draw is not None and away is not None:
            return home, draw, away, "pinnacle (closing line)"

    homes, draws, aways = [], [], []
    for bookmaker in bookmakers.values():
        home = _closing_price(bookmaker, OUTCOME_HOME, kickoff)
        draw = _closing_price(bookmaker, OUTCOME_DRAW, kickoff)
        away = _closing_price(bookmaker, OUTCOME_AWAY, kickoff)
        if home is not None and draw is not None and away is not None:
            homes.append(home)
            draws.append(draw)
            aways.append(away)

    if not homes:
        raise OddsApiError(
            "No requested bookmaker has a complete pre-kickoff historical 1X2 "
            "price for this fixture. Try passing a broader `bookmakers` list."
        )

    n = len(homes)
    return sum(homes) / n, sum(draws) / n, sum(aways) / n, f"average of {n} bookmakers (closing line)"


@dataclass
class MarketOdds:
    """De-vigged 1X2 market probabilities for one fixture."""

    fixture_id: str
    home_team: str
    away_team: str
    date: str
    source: str
    home_odds: float
    draw_odds: float
    away_odds: float
    p_home: float
    p_draw: float
    p_away: float


def get_fixture_market_probabilities(fixture_id: str, force_refresh: bool = False) -> MarketOdds:
    """Fetch (from cache if possible) and de-vig 1X2 odds for one fixture.

    Tries the live pre-game odds first (``/v4/odds``); if the fixture has
    already kicked off and that endpoint has nothing (``hasOdds: False``),
    falls back to the closing line from ``/v4/historical-odds``.

    Returns a ``MarketOdds`` with the teams, kickoff date, which
    bookmaker(s) the odds came from, the raw decimal odds used, and the
    de-vigged market-implied P(home)/P(draw)/P(away).
    """
    raw = fetch_fixture_odds(fixture_id, force_refresh=force_refresh)

    if raw.get("bookmakerOdds"):
        home_odds, draw_odds, away_odds, source = _extract_1x2_odds(raw)
    else:
        historical = fetch_fixture_historical_odds(fixture_id, force_refresh=force_refresh)
        home_odds, draw_odds, away_odds, source = _extract_1x2_odds_historical(
            historical, kickoff=raw.get("startTime", "")
        )

    p_home, p_draw, p_away = devig_1x2(home_odds, draw_odds, away_odds)

    return MarketOdds(
        fixture_id=raw.get("fixtureId", fixture_id),
        home_team=raw.get("participant1Name", ""),
        away_team=raw.get("participant2Name", ""),
        date=raw.get("startTime", ""),
        source=source,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
    )


if __name__ == "__main__":
    if not has_api_key():
        print("ODDSPAPI_API_KEY is not set -- add it to a .env file to run this demo.")
    else:
        fixtures = find_world_cup_fixtures(status_id=2)  # finished matches
        print(f"Found {len(fixtures)} finished World Cup fixtures.")
        if fixtures:
            first = fixtures[0]
            odds = get_fixture_market_probabilities(first["fixtureId"])
            print(
                f"{odds.home_team} vs {odds.away_team} ({odds.source}): "
                f"P(home)={odds.p_home:.1%} P(draw)={odds.p_draw:.1%} P(away)={odds.p_away:.1%}"
            )
