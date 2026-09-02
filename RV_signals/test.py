"""
Implied-vs-realized smile panel.

Run from anywhere:   python RV_signals/test.py
"""

import sys
from pathlib import Path

# Resolve the package dir off this file, not the cwd, so this runs the same from
# the repo root, from RV_signals/, or from an IDE run button.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / 'smile_formation'))

import pandas as pd

from dataset import FXVolDataset
from SABR import REALIZED_WINDOWS, smile_vs_realized_panel

pd.set_option('display.width', 240)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 200)


PAIR   = 'AUDCAD'
TENORS = ['1W', '2W', '1M', '2M', '3M']

# {tenor: (short_bd, long_bd)} — realized lookbacks in business days.
# Defaults live in SABR.REALIZED_WINDOWS; override here to experiment.
WINDOWS = REALIZED_WINDOWS

# 5y of percentile history + the longest realized window + holiday buffer.
DAYS = 366 * 5 + int(max(max(w) for w in WINDOWS.values()) * 7 / 5) + 90


ds  = FXVolDataset.build(pairs=[PAIR], days=DAYS)
out = smile_vs_realized_panel(PAIR, TENORS, windows=WINDOWS,
                              dataset=ds)

panel = out['panel']

print(f"\n=== {PAIR} — full panel ===")
print(panel.round(4).to_string())

