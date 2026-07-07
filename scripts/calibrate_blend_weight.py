"""One-off calibration script: sweep blend_weight (see src/blend.py) against
the FAIR 90-minute 1X2 backtest across every available knockout fixture with
cached market odds, and report which value minimizes the model's own
multiclass Brier score against actual results.

This is NOT tuning the model to match bookmaker odds -- the market's Brier/
log loss is reported purely as context, never as the quantity being
minimized. See src.evaluation.sweep_blend_weight for details.

Run:
    python scripts/calibrate_blend_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results
from src.evaluation import KNOCKOUT_START_DATE, sweep_blend_weight

matches = load_results()
sweep = sweep_blend_weight(matches, start_date=KNOCKOUT_START_DATE)

print()
print(sweep.to_string(index=False))
print()

best = sweep.loc[sweep["model_brier"].idxmin()]
print(
    f"Best blend_weight = {best['blend_weight']:.1f} "
    f"(model Brier={best['model_brier']:.4f}, model log-loss={best['model_log_loss']:.4f}, "
    f"n_fixtures={int(best['n_fixtures'])})"
)
print(
    f"Market (context, not the target): Brier={best['market_brier']:.4f}, "
    f"log-loss={best['market_log_loss']:.4f}"
)
print(
    "\nHonest framing: this is a small in-tournament sample "
    f"({int(best['n_fixtures'])} knockout fixtures), so treat this as 'on par with "
    "the market', not 'beats the market'."
)
