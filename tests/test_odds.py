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
