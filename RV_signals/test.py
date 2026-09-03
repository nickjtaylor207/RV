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
                     latest_decisions, orthogonality_report,
                     vanna_quality_report)

pd.set_option('display.width', 240)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_colwidth', 90)


PAIR   = 'USDJPY'
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
print("""
  std near 0.577 (= uniform on [-1,1]) means the rank transform is healthy;
  much lower and scores clump near zero and the entry band never triggers.
  |mean| above ~0.15 is expanding-rank drift, not an economic tilt.""")




# ══════════════════════════════════════════════════════════════════════════════
# Confidence decomposition — which of the five terms is binding?
# ══════════════════════════════════════════════════════════════════════════════

print('\n=== confidence terms (latest per tenor/risk) ===')
CONF_COLS = ['tenor', 'risk', 'window_agreement', 'horizon_concord',
             'tenor_reliability', 'level_support', 'n_long']
print(ctx.sort_values('date').groupby(['tenor', 'risk']).tail(1)[CONF_COLS]
         .round(3).to_string(index=False))
print("""
  confidence = window_agreement x (0.5 + 0.5*horizon_concord)
               x tenor_reliability x sample_adequacy x level_support
  conf == tenor_reliability  -> limited only by window length; nothing to fix
  horizon_concord near 0     -> one horizon carries the whole gap
  level_support at 0.40      -> impl/real rho levels are inside the noise
                                (vanna only; always 1.0 for volga)""")




# ══════════════════════════════════════════════════════════════════════════════
# Orthogonality — two signals, or one wearing two hats?
# ══════════════════════════════════════════════════════════════════════════════

ortho = orthogonality_report(sig, ctx)




# ══════════════════════════════════════════════════════════════════════════════
# Vanna quality — is rho measurable on this pair at all?
# ══════════════════════════════════════════════════════════════════════════════

vq = vanna_quality_report(sig, ctx)




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
print("""
  A long/short skew is mostly the SHORT-ONLY gate arms converting shorts to
  flat, not a claim the Greek is usually cheap. Read it with the vetoes below.""")

print('\n=== why rows were gated ===')
g = dec[dec['state'] == 'gated']
print(g['gates_failed'].value_counts().to_string() if len(g) else '  none')
print("""
  rho_ident is vanna-only and should track frac_rho_small above.
  'level' fires on floor (shorts) AND ceiling (longs, signed risks only).
  'fit' reads rmse_vp; if it never fires on any pair, check that column.""")

print('\n=== turnover (position changes per year, per tenor/risk) ===')
turn = (dec.assign(sgn=dec['state'].map({'long': 1, 'short': -1}).fillna(0))
           .sort_values('date')
           .groupby(['risk', 'tenor'])['sgn']
           .apply(lambda s: s.diff().ne(0).sum() / max(1, len(s)) * 252))
print(turn.round(1).to_string())












# --------------------------------------------------------------





# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path('RV_signals/smile_formation').resolve()))

# import numpy as np
# import pandas as pd
# from dataset import FXVolDataset
# from SABR    import REALIZED_WINDOWS
# from signals import VolgaVannaCarry

# TENORS = ['1W', '2W', '1M', '2M', '3M']
# DAYS   = 366 * 7 + int(max(max(w) for w in REALIZED_WINDOWS.values()) * 7 / 5) + 90


# def rho_sign_report(pair, tenors=TENORS, entry_band=0.40):
#     """
#     Is the vanna leg trustworthy on this pair? Three failure modes:
#       A. rho sits near zero      -> weakly identified in the fit, RMSE won't show it
#       B. rho changed sign        -> percentile history spans two regimes
#       C. level gap and pct gap disagree in sign -> the score is a rank artefact
#     """
#     ds  = FXVolDataset.build(pairs=[pair], days=DAYS)
#     res = VolgaVannaCarry(windows=REALIZED_WINDOWS).compute(pair, tenors, ds)
#     sig, ctx = res['signals'], res['context']
#     if sig.empty:
#         print(f'{pair}: no history'); return None

#     key = ['date', 'pair', 'tenor', 'risk']
#     v = (sig[sig['risk'] == 'vanna'][key + ['raw', 'score']]
#          .merge(ctx[ctx['risk'] == 'vanna'], on=key, how='left')
#          .sort_values(['tenor', 'date']))

#     # levels ARE comparable for rho (both are correlations) -- unlike nu
#     v['level_gap'] = v['impl_level'] - v['real_level']
#     v['traded']    = v['score'].abs() >= entry_band
#     v['disagree']  = np.sign(v['level_gap']) != np.sign(v['raw'])

#     rows = []
#     for tenor, g in v.groupby('tenor'):
#         t = g[g['traded']]
#         rows.append({
#             'tenor': tenor,
#             # -- A: identifiability
#             'mean_|rho|':     g['impl_level'].abs().mean(),
#             'frac_|rho|<.10': (g['impl_level'].abs() < 0.10).mean(),
#             # -- B: regime
#             'frac_impl>0':    (g['impl_level'] > 0).mean(),
#             'frac_real>0':    (g['real_level'] > 0).mean(),
#             'impl_crossings': int((np.sign(g['impl_level']).diff() != 0).sum() - 1),
#             # -- C: does the rank agree with the level?
#             'disagree_all':   g['disagree'].mean(),
#             'disagree_traded': t['disagree'].mean() if len(t) else np.nan,
#             # -- the unguarded tail: long vanna at a realized-rho ceiling
#             'long_at_pct>95': ((g['score'] > 0) & (g['real_level_pct'] > 95)).mean(),
#             'short_at_pct<5': ((g['score'] < 0) & (g['real_level_pct'] < 5)).mean(),
#         })

#     out = pd.DataFrame(rows).set_index('tenor')
#     print(f'\n=== {pair} | vanna sign diagnostics ===')
#     print(out.round(3).to_string())
#     print(f"""
#   frac_|rho|<.10   > 0.25  -> rho weakly identified much of the time; RMSE is blind to it
#   impl_crossings   > ~5    -> percentile history mixes skew regimes
#   disagree_traded  > 0.20  -> 1-in-5 traded days the level gap contradicts the rank
#   long_at_pct>95           -> currently UNGATED (floor gate is short-only)""")
#     return out


# for p in ['AUDCAD', 'USDJPY', 'EURUSD', 'AUDUSD', 'USDCAD']:
#     rho_sign_report(p)