"""Unit tests for src.odds's rate-limit handling: the guaranteed minimum
interval between requests (_throttle) and automatic retry-with-backoff on
an HTTP 429 (_request). Uses a mocked requests.get and a mocked
time.sleep, so these make no real network calls and cost no real wall
-clock time or OddsPapi quota.

Run with:

    python -m unittest tests/test_odds.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.odds as odds


def _rate_limited_response(retry_after_seconds: float = 0.10, retry_ms: float = 100) -> MagicMock:
    """A 429 response shaped like the one OddsPapi actually returned in
    CI: "Please wait 0.10 seconds ... retryAfter: 0.10 seconds,
    retryMs: 100" -- a real reported wait well under the ~0.87s
    per-request floor, consistent with a separate requests-per-window
    limit on top of the per-request one.
    """
    response = MagicMock()
    response.status_code = 429
    response.ok = False
    response.headers = {}
    body = {
        "error": f"You are being rate limited. Please wait {retry_after_seconds} seconds "
        "before making another request",
        "retryAfter": retry_after_seconds,
        "retryMs": retry_ms,
    }
    response.json.return_value = body
    response.text = str(body)
    return response


def _success_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.ok = True
    response.json.return_value = payload
    response.text = str(payload)
    return response


def _forbidden_response() -> MagicMock:
    """A 403, shaped like /v4/odds on a plan that doesn't include it."""
    response = MagicMock()
    response.status_code = 403
    response.ok = False
    response.headers = {}
    response.json.return_value = {"error": "This endpoint is not available on your plan."}
    response.text = "Forbidden"
    return response


def _historical_odds_payload(kickoff: str, home_price=2.0, draw_price=3.5, away_price=3.8) -> dict:
    """A /v4/historical-odds body with a single pre-kickoff Pinnacle snapshot."""
    snapshot_time = "2026-07-01T00:00:00Z"

    def _outcome(price):
        return {"players": {"0": [{"createdAt": snapshot_time, "price": price}]}}

    return {
        "bookmakers": {
            "pinnacle": {
                "markets": {
                    odds.MATCH_WINNER_MARKET_ID: {
                        "outcomes": {
                            odds.OUTCOME_HOME: _outcome(home_price),
                            odds.OUTCOME_DRAW: _outcome(draw_price),
                            odds.OUTCOME_AWAY: _outcome(away_price),
                        }
                    }
                }
            }
        }
    }


class RequestRetryTests(unittest.TestCase):
    def setUp(self):
        odds._last_request_time = None
        os.environ["ODDSPAPI_API_KEY"] = "test-key-not-real"

    @patch("src.odds.time.sleep", return_value=None)
    @patch("src.odds.requests.get")
    def test_retries_on_429_then_succeeds(self, mock_get, mock_sleep):
        """The exact regression this guards against: a 429 on the first
        attempt must NOT raise immediately -- it must back off and retry,
        succeeding once the API stops rate-limiting us.
        """
        mock_get.side_effect = [
            _rate_limited_response(),
            _rate_limited_response(),
            _success_response({"ok": True}),
        ]

        result = odds._request("/v4/fake", {"a": 1})

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get.call_count, 3)

    @patch("src.odds.time.sleep", return_value=None)
    @patch("src.odds.requests.get")
    def test_raises_only_after_exhausting_all_retries(self, mock_get, mock_sleep):
        mock_get.return_value = _rate_limited_response()

        with self.assertRaises(odds.OddsApiRateLimitError) as ctx:
            odds._request("/v4/fake", {"a": 1})

        self.assertEqual(mock_get.call_count, odds.MAX_RETRY_ATTEMPTS)
        # Never leaks the API key or query string into the error message.
        self.assertNotIn("test-key-not-real", str(ctx.exception))
        self.assertNotIn("apiKey", str(ctx.exception))

    @patch("src.odds._throttle", return_value=None)
    @patch("src.odds.time.sleep", return_value=None)
    @patch("src.odds.requests.get")
    def test_backoff_grows_exponentially(self, mock_get, mock_sleep, mock_throttle):
        # _throttle is a no-op here so this test isolates the backoff
        # sequence itself -- ThrottleTests covers _throttle in isolation.
        # (With a mocked time.sleep that doesn't actually advance the
        # clock, a real _throttle would add its own extra wait on every
        # retry too, since it can't tell time actually passed.)
        mock_get.return_value = _rate_limited_response()

        with self.assertRaises(odds.OddsApiRateLimitError):
            odds._request("/v4/fake", {"a": 1})

        # One backoff sleep per retried attempt (the final, non-retried
        # attempt raises without sleeping), each at least the exponential
        # floor (1s, 2s, 4s, 8s) since the reported wait (0.10s) is
        # smaller than all of them.
        sleep_durations = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertEqual(len(sleep_durations), odds.MAX_RETRY_ATTEMPTS - 1)
        for i, waited in enumerate(sleep_durations):
            expected_floor = odds.RETRY_BACKOFF_BASE_SECONDS * (2**i)
            self.assertGreaterEqual(waited, expected_floor - 1e-9)


class AccessErrorFallbackTests(unittest.TestCase):
    """Covers the /v4/odds-returns-403 scenario: OddsPapi's free tier
    doesn't include live/current odds, only historical -- these confirm
    get_fixture_market_probabilities falls back cleanly instead of
    propagating the 403 and crashing the caller.
    """

    def setUp(self):
        odds._last_request_time = None
        os.environ["ODDSPAPI_API_KEY"] = "test-key-not-real"
        # get_fixture_market_probabilities caches to disk -- point CACHE_DIR
        # at a fresh temp directory per test so these never read/write the
        # real data/odds_cache/, and a leftover file from one test can't
        # mask the mocked requests.get in another.
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        patcher = patch("src.odds.CACHE_DIR", Path(tmpdir.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("src.odds.time.sleep", return_value=None)
    @patch("src.odds.requests.get")
    def test_request_raises_access_error_on_403(self, mock_get, mock_sleep):
        mock_get.return_value = _forbidden_response()

        with self.assertRaises(odds.OddsApiAccessError):
            odds._request("/v4/odds", {"fixtureId": "id123"})

        # A 403 is not retried like a 429 -- it's a plan limitation, not a
        # transient rate limit, so it should fail on the very first call.
        self.assertEqual(mock_get.call_count, 1)

    @patch("src.odds.time.sleep", return_value=None)
    @patch("src.odds.requests.get")
    def test_falls_back_to_historical_odds_on_403(self, mock_get, mock_sleep):
        kickoff = "2026-07-09T18:00:00Z"
        mock_get.side_effect = [
            _forbidden_response(),  # /v4/odds -- 403, not on this plan
            _success_response(_historical_odds_payload(kickoff)),  # /v4/historical-odds fallback
        ]

        result = odds.get_fixture_market_probabilities(
            "id123", fixture_meta={"fixtureId": "id123", "startTime": kickoff}
        )

        self.assertEqual(mock_get.call_count, 2)
        self.assertIn("closing line", result.source)
        self.assertAlmostEqual(result.p_home + result.p_draw + result.p_away, 1.0, places=9)

    @patch("src.odds.time.sleep", return_value=None)
    @patch("src.odds.requests.get")
    def test_403_without_any_kickoff_source_fails_clearly_not_silently(self, mock_get, mock_sleep):
        # With no fixture_meta AND a 403 (so /v4/odds never returns its own
        # startTime either), there's genuinely no kickoff time available to
        # filter the historical snapshots to a pre-game price -- this must
        # raise a clear OddsApiError, not silently return a wrong (e.g.
        # in-play or post-match) price. In practice this doesn't happen on
        # the real call path: get_market_probabilities_for_teams always
        # supplies fixture_meta from the fixtures list it already has.
        mock_get.side_effect = [
            _forbidden_response(),
            _success_response(_historical_odds_payload("2026-07-09T18:00:00Z")),
        ]

        with self.assertRaises(odds.OddsApiError):
            odds.get_fixture_market_probabilities("id123")


class ThrottleTests(unittest.TestCase):
    def setUp(self):
        odds._last_request_time = None

    @patch("src.odds.time.sleep", return_value=None)
    def test_throttle_sleeps_for_the_remaining_interval(self, mock_sleep):
        odds._last_request_time = time.monotonic()
        odds._throttle()

        mock_sleep.assert_called_once()
        (waited,) = mock_sleep.call_args.args
        self.assertGreater(waited, 0)
        self.assertLessEqual(waited, odds.MIN_REQUEST_INTERVAL_SECONDS)

    def test_throttle_does_not_sleep_on_first_call(self):
        with patch("src.odds.time.sleep") as mock_sleep:
            odds._throttle()
            mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
