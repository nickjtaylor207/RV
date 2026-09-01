"""
THE PHASE 1 GATE.

Run the SAME trade through the old Delta_Hedged engine and the new
Systematic_RV engine, and compare them day by day, bucket by bucket.

WHY THIS MATTERS MORE THAN ANYTHING ELSE IN PHASE 1
---------------------------------------------------
Right now the old stack is the only trusted reference that exists. Once Phase 2
adds netted hedging, option transaction costs and multi-expiry rolling, the two
engines are no longer computing the same thing and this comparison becomes
impossible. This is the one and only window in which a silent arithmetic error
in the rewrite can be caught cheaply.

So: get this to pass before building anything on top of it. If a bucket
disagrees, the rewrite is wrong -- the old engine's formulas are the ones that
produced every result recorded so far.

WHAT SHOULD MATCH EXACTLY (to floating point)
---------------------------------------------
    option_pnl, theta_pnl, gamma_pnl, vega_pnl, vanna_pnl, volga_pnl,
    gamma_pnl_be, vanna_pnl_be, volga_pnl_be, and the greek exposures.

Both engines price off the same dataset, the same SABR surface and the same
start-of-day convention, and core/greeks.py's standardisation was derived to
reproduce the old inline formulas term for term.

WHAT IS EXPECTED TO DIFFER, AND WHY
-----------------------------------
 1. recon_resid. The old engine's residual was portfolio-level and netted the
    HEDGE P&L into the reconciliation. The new one is per-position and
    hedge-free, including the first-order delta term instead. Different
    quantity, same purpose. Compare `delta_pnl + hedge_pnl` against the old
    engine's implied delta term if you want to tie them out.

 2. Terminal-unwind transaction cost. The old engine charged none at natural
    expiry but full cost on an early exit. Set
    `charge_tc_on_expiry_unwind=False` to reproduce that.

 3. Hedge carry rates. The old engine used the option's remaining-tenor rates
    for an overnight spot hedge. `carry_tenor_days` reproduces it. Leave the
    new default for real work -- it is the more defensible convention.

HOW TO READ THE OUTPUT
----------------------
The comparison prints, per bucket: the summed value from each engine, the
absolute difference, and the max per-day absolute difference. A bucket is
CLEAN if max daily |diff| is below `tol` in base-ccy terms. Anything else is
printed with the worst offending dates so you can go straight to them.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --- make the OLD stack importable. It uses flat imports (`from pricer import
#     ...`), so its own directory must be on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # ...\Backtesting
_OLD  = os.path.join(_ROOT, 'Delta_Hedged')
_NEW  = os.path.dirname(_HERE)                            # ...\Systematic_RV
for _p in (_OLD, _NEW):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Buckets that MUST agree, and the old engine's column name for each.
BUCKET_MAP = {
    'option_pnl':   'option_pnl',
    'theta_pnl':    'theta_pnl',
    'gamma_pnl':    'gamma_pnl',
    'vega_pnl':     'vega_pnl',
    'vanna_pnl':    'vanna_pnl',
    'volga_pnl':    'volga_pnl',
    'gamma_pnl_be': 'gamma_pnl_be',
    'vanna_pnl_be': 'vanna_pnl_be',
    'volga_pnl_be': 'volga_pnl_be',
    'hedge_pnl':    'hedge_pnl',
    'hedge_carry':  'hedge_carry',
}

# Exposure columns. The new names are the standardised ones; the old engine's
# `vega_1vp` is directly comparable, and `volga_1vp` / `vanna_1vp` need the
# factor that core/greeks.py's as_trader_units() applies.
EXPO_MAP = {
    'net_vega_1vp':  'vega_1vp',
}


def compare(pair: str = 'USDJPY',
            tenor: str = '1M',
            call_delta: float = 0.25,
            put_delta: float = -0.25,
            direction: int = -1,
            notional: float = 10_000_000,
            entry_days_back: int = 120,
            spot_tc: float = 0.0001,
            history_days: int = 500,
            rtol: float = 1e-9,
            atol: float = 1e-6,
            legacy: bool = True,
            verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run one strangle through both engines and diff them.

    legacy
        True  -- turn ON every old-engine convention (half-step strike
                 bootstrap, end-of-period nu/rho, remaining-tenor carry rates,
                 no expiry-unwind TC). This is the mode in which the two
                 engines should agree to floating point. Use it to PROVE the
                 rewrite is arithmetically correct.
        False -- run the new engine as it is actually meant to run. The diffs
                 you then see are the deliberate improvements, and their size
                 tells you how much each old convention was costing you.

    Tolerance is RELATIVE (rtol, scaled by the bucket's own magnitude) with an
    absolute floor (atol). An absolute 1e-6 on numbers of order 1e5 flags pure
    floating-point reassociation as a failure, which is useless.

    Returns {'old', 'new', 'diff', 'strikes', 'result'} so you can drill in.
    """
    # ------------------------------------------------------------------ #
    # OLD ENGINE
    # ------------------------------------------------------------------ #
    from backtest_MLeg import LegSpec, run_backtest_multi_leg          # noqa: E402
    from exit_hedge_logic import HoldToExpiry, DailyHedge              # noqa: E402

    old_legs = [LegSpec('call', call_delta, direction, notional),
                LegSpec('put',  put_delta,  direction, notional)]

    if verbose:
        print('=' * 74)
        print(f'RECONCILIATION  {pair} {tenor} strangle '
              f'({call_delta:+.2f}/{put_delta:+.2f}) dir={direction:+d}')
        print('=' * 74)
        print('\n--- running OLD engine (Delta_Hedged) ---')

    old_df, _leg_dfs, old_sum, old_leg_sums = run_backtest_multi_leg(
        legs=old_legs, pair=pair, tenor=tenor,
        entry_days_back=entry_days_back, history_days=history_days,
        tc_fraction=spot_tc, verbose=False,
        exit_rule=HoldToExpiry(), hedge_rule=DailyHedge())

    entry_date = pd.Timestamp(old_sum['entry_date'])
    exit_date  = pd.Timestamp(old_sum.get('exit_date') or old_df.index[-1])
    if verbose:
        print(f'    entry {entry_date.date()}  exit {exit_date.date()}  '
              f'{len(old_df)} rows  net {old_sum["net_pnl"]:,.2f}')

    # ------------------------------------------------------------------ #
    # NEW ENGINE -- identical trade, matched conventions
    # ------------------------------------------------------------------ #
    from market.dataset import FXVolDataset                            # noqa: E402
    from engine.loop import EngineConfig, HoldStatic, run              # noqa: E402
    from book.position import LegRequest                               # noqa: E402
    from core.conventions import TENOR_DAYS                            # noqa: E402

    if verbose:
        print('\n--- running NEW engine (Systematic_RV) ---')

    ds = FXVolDataset.build(pairs=[pair], days=history_days)

    new_legs = [LegRequest('call', direction, notional, tenor,
                           target_delta=call_delta, sleeve='wings', tag='wing_c'),
                LegRequest('put',  direction, notional, tenor,
                           target_delta=put_delta,  sleeve='wings', tag='wing_p')]

    cfg = EngineConfig(
        pairs=[pair],
        start=entry_date, end=exit_date,
        hedge_fraction=1.0,
        spot_tc=spot_tc,
        flatten_at_end=False,                  # old engine does not flatten
        carry_tenor_days=(None if legacy else 30.0),
        legacy_strike_halfstep=legacy,
        legacy_nu_rho_at_end=legacy,
        charge_tc_on_expiry_unwind=not legacy,
        verbose=False)

    res = run(HoldStatic(pair, new_legs, tenor), ds, cfg)
    new_df = res.daily
    if verbose:
        print(f'    {len(new_df)} rows  net {new_df["pnl"].sum():,.2f}')

    # ------------------------------------------------------------------ #
    # STRIKE CHECK -- if the strikes differ, nothing else can match
    # ------------------------------------------------------------------ #
    srows = []
    for i, (_, t) in enumerate(res.trades.iterrows()):
        ols  = old_leg_sums[i]
        oK   = float(ols['strike'])
        ovol = float(ols['sigma_entry']) / 100.0        # stored in percent
        srows.append({
            'leg':      t['tag'],
            'old_K':    oK,
            'new_K':    t['strike'],
            'dK_bp':    (t['strike'] / oK - 1.0) * 1e4,
            'old_vol':  ovol,
            'new_vol':  t['entry_vol'],
            'dvol_bp':  (t['entry_vol'] - ovol) * 1e4,
        })
    strikes = pd.DataFrame(srows)

    if verbose:
        print('\n--- STRUCK STRIKES (if these differ, every bucket below is moot) ---')
        with pd.option_context('display.width', 160,
                               'display.float_format', lambda v: f'{v:,.6f}'):
            print(strikes.to_string(index=False))
        worst_k = strikes.dK_bp.abs().max()
        print(f'    max |dK| {worst_k:.6f} bp   '
              f'max |dvol| {strikes.dvol_bp.abs().max():.6f} bp of vol')
        if worst_k > 1e-6:
            print('    -> STRIKES DIFFER. Fix the strike/vol fixed point before')
            print('       reading anything below. Under legacy=True this should be ~0;')
            print('       under legacy=False a non-zero value here is EXPECTED and is')
            print('       the size of the old half-step bootstrap error.')

    # ------------------------------------------------------------------ #
    # DIFF
    # ------------------------------------------------------------------ #
    old_idx = pd.DatetimeIndex(old_df.index).normalize()
    new_idx = pd.DatetimeIndex(new_df.index).normalize()
    common  = old_idx.intersection(new_idx)

    o = old_df.copy();  o.index = old_idx
    n = new_df.copy();  n.index = new_idx
    o, n = o.loc[common], n.loc[common]

    if verbose:
        print(f'\n--- {len(common)} common dates '
              f'(old {len(old_idx)}, new {len(new_idx)}) ---')
        if len(old_idx) != len(new_idx):
            print('    WARNING: row counts differ. Usually the entry date: the '
                  'old engine\n    skips it, the new one records an open-only '
                  'row. Check the edges.')

    rows = []
    for new_col, old_col in BUCKET_MAP.items():
        if old_col not in o.columns or new_col not in n.columns:
            rows.append({'bucket': new_col, 'status': 'MISSING'})
            continue
        a, b = o[old_col].astype(float), n[new_col].astype(float)
        d = (b - a)
        scale = max(a.abs().max(), b.abs().max(), 1.0)
        rel   = d.abs().max() / scale
        rows.append({
            'bucket':     new_col,
            'old_total':  a.sum(),
            'new_total':  b.sum(),
            'total_diff': b.sum() - a.sum(),
            'max_abs_d':  d.abs().max(),
            'max_rel_d':  rel,
            'worst_date': d.abs().idxmax().date() if len(d) else None,
            'status':     'OK' if (rel < rtol or d.abs().max() < atol) else 'MISMATCH',
        })
    diff = pd.DataFrame(rows)

    if verbose:
        print('\n--- BUCKET COMPARISON ---')
        with pd.option_context('display.width', 160,
                               'display.float_format', lambda v: f'{v:,.6f}'):
            print(diff.to_string(index=False))

        bad = diff[diff.status == 'MISMATCH']
        if bad.empty:
            print('\n  *** ALL BUCKETS RECONCILE. The rewrite reproduces the old')
            print('      engine exactly. Phase 1 gate PASSED -- proceed to Phase 2. ***')
        else:
            print(f'\n  *** {len(bad)} BUCKET(S) MISMATCH -- DO NOT BUILD ON THIS ***')
            for _, r in bad.iterrows():
                c_new, c_old = r.bucket, BUCKET_MAP[r.bucket]
                sub = pd.DataFrame({'old': o[c_old], 'new': n[c_new]})
                sub['diff'] = sub.new - sub.old
                print(f'\n  {r.bucket}: worst 5 days')
                print(sub.reindex(sub['diff'].abs().sort_values(
                    ascending=False).index).head(5).to_string())

        print('\n--- EXPECTED DIFFERENCES (not failures) ---')
        if 'recon_resid' in o.columns:
            print(f'  recon_resid  old {o["recon_resid"].sum():>14,.2f}   '
                  f'new {n["recon_resid"].sum():>14,.2f}')
            print('     Different by design: the old one nets hedge P&L into the')
            print('     reconciliation; the new one is per-position and includes')
            print('     the first-order delta term instead.')
        print(f'  new delta_pnl {n["delta_pnl"].sum():>14,.2f}  '
              f'+ hedge_pnl {n["hedge_pnl"].sum():>14,.2f}  '
              f'= {n["delta_pnl"].sum() + n["hedge_pnl"].sum():>14,.2f}')
        print('     Should be near zero under a full daily hedge -- that IS the')
        print('     hedge working. What is left over is gamma.')

    return {'old': o, 'new': n, 'diff': diff, 'strikes': strikes,
            'result': res, 'old_summary': old_sum}




# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python recon/reconcile.py
# Requires a live Bloomberg connection AND the Delta_Hedged folder present.
#
# TWO PASSES, and they answer different questions:
#
#   PASS 1  legacy=True   Every old convention turned ON. The two engines
#                         should agree to floating point. THIS IS THE GATE --
#                         it proves the rewrite is arithmetically correct.
#
#   PASS 2  legacy=False  The new engine as it is meant to run. Every diff is
#                         a DELIBERATE improvement, and its size answers "how
#                         much was that old convention costing me?" Nothing
#                         here is a failure; read it as a report.
# ====================================================================== #


# if __name__ == '__main__':
#     import os as _os, sys as _sys
#     _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

#     CASES = [
#         dict(pair='USDJPY', tenor='1M', call_delta=+0.25, put_delta=-0.25,
#              direction=-1, entry_days_back=120),                 # baseline
#         dict(pair='USDJPY', tenor='1M', call_delta=+0.10, put_delta=-0.10,
#              direction=-1, entry_days_back=120),                 # deep wings
#         dict(pair='USDJPY', tenor='1M', call_delta=+0.25, put_delta=-0.25,
#              direction=+1, entry_days_back=120),                 # LONG
#         dict(pair='USDJPY', tenor='3M', call_delta=+0.25, put_delta=-0.25,
#              direction=-1, entry_days_back=200),                 # longer tenor
#         dict(pair='EURUSD', tenor='1M', call_delta=+0.25, put_delta=-0.25,
#              direction=-1, entry_days_back=120),                 # USD as quote
#     ]

#     def label(c):
#         return f"{c['pair']} {c['tenor']} {c['call_delta']:+.2f} dir{c['direction']:+d}"

#     # ---------------- PASS 1 : THE GATE ------------------------------ #
#     print('#' * 74)
#     print('# PASS 1: legacy=True -- must reconcile to floating point')
#     print('#' * 74)
#     verdict = {}
#     for c in CASES:
#         r   = compare(**c, legacy=True)
#         bad = r['diff'][r['diff'].status == 'MISMATCH']
#         verdict[label(c)] = list(bad.bucket)
#         print()
#         print('>>> ' + label(c) + ': ' +
#               ('PASS' if bad.empty else 'FAIL ' + str(list(bad.bucket))))
#         print()

#     print('=' * 74)
#     print('PASS 1 SUMMARY')
#     print('=' * 74)
#     for k, v in verdict.items():
#         print(f'  {k:<34} ' + ('PASS' if not v else 'FAIL ' + str(v)))
#     if not any(verdict.values()):
#         print()
#         print('  *** PHASE 1 GATE PASSED. The rewrite reproduces the old engine')
#         print('      exactly under its own conventions. Proceed to Phase 2. ***')

#     # ---------------- PASS 2 : QUANTIFY THE DIVERGENCE ---------------- #
#     print()
#     print('#' * 74)
#     print('# PASS 2: legacy=False -- quantify the deliberate divergences')
#     print('#' * 74)
#     impact = []
#     for c in CASES:
#         r = compare(**c, legacy=False, verbose=False)
#         d = r['diff'].set_index('bucket')
#         impact.append({
#             'case':         label(c),
#             'volga_be_old': d.loc['volga_pnl_be', 'old_total'],
#             'volga_be_new': d.loc['volga_pnl_be', 'new_total'],
#             'vanna_be_old': d.loc['vanna_pnl_be', 'old_total'],
#             'vanna_be_new': d.loc['vanna_pnl_be', 'new_total'],
#             'carry_old':    d.loc['hedge_carry', 'old_total'],
#             'carry_new':    d.loc['hedge_carry', 'new_total'],
#             'max_dK_bp':    r['strikes'].dK_bp.abs().max(),
#         })
#     imp = pd.DataFrame(impact)
#     with pd.option_context('display.width', 200,
#                            'display.float_format', lambda v: f'{v:,.2f}'):
#         print(imp.to_string(index=False))
#     print()
#     print('  HOW TO READ THIS')
#     print('  volga_be / vanna_be : the old numbers were built from END-of-period')
#     print('     nu/rho, a look-ahead. The new ones are the honest measurement of')
#     print('     the convexity and skew premium. If the gap is large, every')
#     print('     _be-based conclusion drawn from the old stack needs revisiting.')
#     print('  carry              : the old engine took carry rates at the option')
#     print('     remaining tenor, which shrinks daily. New uses a fixed short')
#     print('     tenor. Small, but it drifts systematically as a trade ages.')
#     print('  max_dK_bp          : how far the old half-step strike bootstrap sat')
#     print('     from the true fixed point. Watch this GROW toward 10d/5d -- that')
#     print('     is the error you were carrying exactly where this strategy lives.')

    # ---------------- drill into a single day ------------------------- #
    # r = compare(pair='USDJPY', tenor='1M', legacy=True)
    # o, n = r['old'], r['new']
    # d = pd.Timestamp('2026-04-30')
    # print(o.loc[d, ['spot', 'option_pnl', 'gamma_pnl', 'vega_pnl']])
    # print(n.loc[d, ['option_pnl', 'gamma_pnl', 'vega_pnl']])
    # print(r['result'].positions.query('date == @d.date()').T)























