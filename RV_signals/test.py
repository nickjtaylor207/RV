"""
Volga / vanna signal harness.

Run from anywhere:   python RV_signals/test.py
"""

import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / 'smile_formation'))


import pandas as pd

from dataset import FXVolDataset
from SABR import REALIZED_WINDOWS, smile_vs_realized_panel
from signals import (DecisionConfig, VolgaVannaCarry, aggregate_tenors, decide,
                     latest_decisions, orthogonality_report)

pd.set_option('display.width', 240)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_colwidth', 90)


PAIR   = 'AUDCAD'
TENORS = ['1W', '2W', '1M', '2M', '3M']

WINDOWS = REALIZED_WINDOWS
# 7y: percentile history + longest realized window + the score's own rank
# burn-in (see the Burn-in note in signals.py) + holiday buffer.
DAYS = 366 * 7 + int(max(max(w) for w in WINDOWS.values()) * 7 / 5) + 90
ds   = FXVolDataset.build(pairs=[PAIR], days=DAYS)

SHOW_PANEL = True


# ══════════════════════════════════════════════════════════════════════════════
# The snapshot panel — levels and percentiles as of today
# ══════════════════════════════════════════════════════════════════════════════

if SHOW_PANEL:
    panel = smile_vs_realized_panel(PAIR, TENORS, windows=WINDOWS,
                                    dataset=ds)['panel']
    print(f"\n=== {PAIR} | implied vs realized panel ===")
    print(panel.round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — the carry producer (full score history)
# ══════════════════════════════════════════════════════════════════════════════

prod = VolgaVannaCarry(windows=WINDOWS)
res  = prod.compute(PAIR, TENORS, ds, verbose=True)

sig, ctx = res['signals'], res['context']

print('\n=== score history ===')
print(f"  {len(sig)} rows | {sig['date'].min().date()} -> {sig['date'].max().date()}")
print('\n  score distribution by risk:')
print(sig.groupby('risk')['score']
         .describe(percentiles=[.1, .5, .9]).round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Orthogonality — two signals, or one wearing two hats?
# ══════════════════════════════════════════════════════════════════════════════

ortho = orthogonality_report(sig, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# Tenor aggregation — is it a regime, or one tenor?
# ══════════════════════════════════════════════════════════════════════════════

agg = aggregate_tenors(sig)
print('\n=== aggregated across tenors (latest) ===')
print(agg.sort_values('date').groupby('risk').tail(1).round(3).to_string(index=False))
print("""
  high |score|, low dispersion  -> genuine regime, trade outright
  high |score|, high dispersion -> one tenor is the outlier, trade a calendar""")


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — decide: gates, dead band, hysteresis, target
# ══════════════════════════════════════════════════════════════════════════════

# Per-risk caps, in each Greek's OWN units. Set these STRESS-EQUIVALENT: pick a
# reference joint spot/vol shock and size each so a full position loses the same
# amount under it. The 1.0 placeholders below just reproduce the old
# dimensionless behaviour -- replace with your real notionals.
CAPS = {'volga': 1.0,    # ccy per vol-pt^2
        'vanna': 1.0}    # ccy per vol-pt per spot %

cfg = DecisionConfig(entry_band=0.40, exit_band=0.20,
                     cap=CAPS, size_by_confidence=True)
dec = decide(sig, ctx, config=cfg)

print('\n=== decisions: current book ===')
print(latest_decisions(dec)[['pair', 'tenor', 'risk', 'score', 'confidence',
                             'target', 'cap', 'state', 'reason']].to_string(index=False))

print('\n=== state mix over history (sanity: not all one state) ===')
print(dec.groupby(['risk', 'state']).size().unstack(fill_value=0).to_string())

print('\n=== why rows were gated ===')
g = dec[dec['state'] == 'gated']
print(g['gates_failed'].value_counts().to_string() if len(g) else '  none')

print('\n=== turnover (position changes per year, per tenor/risk) ===')
turn = (dec.assign(sgn=dec['state'].map({'long': 1, 'short': -1}).fillna(0))
           .sort_values('date')
           .groupby(['risk', 'tenor'])['sgn']
           .apply(lambda s: s.diff().ne(0).sum() / max(1, len(s)) * 252))
print(turn.round(1).to_string())
