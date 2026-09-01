"""
THE PHASE 2 GATE: at what transaction cost does the premium disappear?

This is not a backtest. It is a feasibility test, and it is the most important
number you will produce before committing to Phases 3-7.

THE QUESTION
------------
The strategy premise is that FX vol surfaces embed a harvestable risk premium
in the wings (convexity / volga) and the skew (vanna). Phase 1 can measure that
premium -- `volga_pnl_be` and `vanna_pnl_be` are exactly it.

But Phase 1 charged nothing to trade an option, and a rolled wing-selling book
pays the wing spread over and over. So the premium is only real if:

    gross premium harvested  >  cost of repeatedly putting the trade on

This script sweeps `OptionCostModel.scale` from 0 (free, the Phase 1 world) up
to several multiples of assumed G10 interbank levels, and finds the level at
which net P&L crosses zero. Compare that break-even to what you believe you
would actually pay.

    break-even scale >> 1   the premium survives realistic costs. Proceed.
    break-even scale ~  1   marginal. The strategy is a cost-execution problem
                            before it is an alpha problem, and Phase 3's sizer
                            and Phase 6's roll cadence become the whole game.
    break-even scale <  1   the premise does not survive. Stop and rethink the
                            structure (wider wings? longer tenors? fewer rolls?)
                            before building anything on top.

WHY IT SWEEPS A SCALE RATHER THAN QUOTING ONE NUMBER
----------------------------------------------------
You chose the parametric cost route, so the absolute spreads in book/costs.py
are assumptions, not measurements. Any single P&L figure derived from them
inherits that uncertainty. A break-even LEVEL does not: it says "the strategy
needs costs to be below X", and you can judge X against your own execution
without trusting my defaults at all.

WHAT IT DELIBERATELY DOES NOT ANSWER
------------------------------------
  * Whether the premium is there in the first place across a long sample. This
    runs one window of one pair per call; loop it yourself.
  * Whether a SIGNAL improves it. Phase 5/6. Selling wings unconditionally is
    the floor, not the strategy.
  * Whether continuous vega hedging is affordable -- that hedge is an OPTION
    hedge and pays this same spread on every rebalance. Phase 4 will run
    through the same cost model and can be measured here once it exists.
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from market.dataset import FXVolDataset
from book.costs import OptionCostModel
from book.position import LegRequest
from engine.loop import EngineConfig, RollingStructure, HoldStatic, run


def strangle(wing: float, direction: int = -1, notional: float = 10_000_000,
             tenor: str = '1M') -> List[LegRequest]:
    """A symmetric strangle at +/- `wing` delta."""
    return [LegRequest('call', direction, notional, tenor,
                       target_delta=+wing, sleeve='wings', tag='wing_c'),
            LegRequest('put',  direction, notional, tenor,
                       target_delta=-wing, sleeve='wings', tag='wing_p')]


def sweep(pair: str = 'USDJPY',
          tenor: str = '1M',
          wing: float = 0.25,
          roll_days: int = 5,
          start: Optional[str] = None,
          end: Optional[str] = None,
          history_days: int = 900,
          scales=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
          spot_tc: float = 0.0001,
          dataset=None,
          verbose: bool = True) -> pd.DataFrame:
    """
    Run the same rolled short-strangle book at several cost levels.

    Everything except the cost model is held fixed, so every difference between
    rows is cost and nothing else.
    """
    ds = dataset if dataset is not None else FXVolDataset.build(pairs=[pair],
                                                                days=history_days)
    legs = strangle(wing, direction=-1, tenor=tenor)

    rows = []
    for sc in scales:
        cfg = EngineConfig(pairs=[pair], start=start, end=end,
                           spot_tc=spot_tc,
                           cost_model=OptionCostModel(scale=sc),
                           verbose=False)
        res = run(RollingStructure(pair, legs, tenor, roll_days=roll_days), ds, cfg)
        d = res.daily
        rows.append({
            'scale':        sc,
            'net_pnl':      d.pnl.sum(),
            'option_pnl':   d.option_pnl.sum(),
            'option_tc':    d.option_tc.sum(),
            'spot_tc':      d.hedge_tc.sum(),
            'gamma_be':     d.gamma_pnl_be.sum(),
            'vanna_be':     d.vanna_pnl_be.sum(),
            'volga_be':     d.volga_pnl_be.sum(),
            'legs':         len(res.trades),
            'peak_open':    int(d.n_open.max()),
            'volga_per_unit': res.pnl_per_unit_greek('volga_pnl_be', 'volga_1vp'),
            'vanna_per_unit': res.pnl_per_unit_greek('vanna_pnl_be', 'vanna_1pct_1vp'),
        })
        if verbose:
            r = rows[-1]
            print(f"  scale {sc:>4.2f}  net {r['net_pnl']:>13,.0f}   "
                  f"optTC {r['option_tc']:>11,.0f}   "
                  f"volga_be {r['volga_be']:>11,.0f}")

    out = pd.DataFrame(rows)
    out.attrs['pair'], out.attrs['tenor'] = pair, tenor
    out.attrs['wing'], out.attrs['roll_days'] = wing, roll_days
    return out


def breakeven_scale(sw: pd.DataFrame, col: str = 'net_pnl') -> float:
    """
    Linear interpolation of the cost scale at which `col` crosses zero.

    Cost is monotone in scale, so if the strategy is profitable at scale 0 the
    curve crosses at most once. Returns +inf if it never crosses (survives
    every cost level tested) and NaN if it is already negative when free --
    which means the problem is the strategy, not the cost.
    """
    x, y = sw['scale'].to_numpy(float), sw[col].to_numpy(float)
    if y[0] <= 0:
        return np.nan
    below = np.where(y <= 0)[0]
    if len(below) == 0:
        return np.inf
    i = below[0]
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return x0 + (x1 - x0) * y0 / (y0 - y1)


def report(sw: pd.DataFrame) -> None:
    """Print the sweep and the verdict."""
    pair, tenor = sw.attrs.get('pair', '?'), sw.attrs.get('tenor', '?')
    wing, roll  = sw.attrs.get('wing', '?'), sw.attrs.get('roll_days', '?')

    print()
    print('=' * 78)
    print(f'BREAK-EVEN STUDY  {pair} {tenor}  short {wing:.0%}-delta strangle, '
          f'rolled every {roll} business days')
    print('=' * 78)
    with pd.option_context('display.width', 200,
                           'display.float_format', lambda v: f'{v:,.2f}'):
        print(sw[['scale', 'net_pnl', 'option_pnl', 'option_tc', 'spot_tc',
                  'gamma_be', 'vanna_be', 'volga_be']].to_string(index=False))

    be_net   = breakeven_scale(sw, 'net_pnl')
    print()
    print(f'  legs traded {int(sw.legs.iloc[0])}, peak concurrent '
          f'{int(sw.peak_open.iloc[0])}')
    print(f'  gross (cost-free) P&L      {sw.net_pnl.iloc[0]:>14,.0f}')
    at1 = sw.loc[sw.scale == 1.0, 'option_tc']
    if len(at1):
        print(f'  option TC at scale 1.0     {at1.iloc[0]:>14,.0f}   '
              f'({at1.iloc[0] / max(abs(sw.net_pnl.iloc[0]), 1) * 100:.0f}% of gross)')

    print()
    if np.isnan(be_net):
        print('  VERDICT: negative even with FREE options. The problem is not')
        print('  cost -- this structure did not harvest a premium over this')
        print('  window at all. Check the _be buckets above: if volga_be is')
        print('  negative you were short convexity into a vol-of-vol expansion.')
    elif np.isinf(be_net):
        print('  VERDICT: survives every cost level tested. Widen the sweep')
        print('  before believing it -- and check the sample is not one benign')
        print('  window.')
    else:
        print(f'  BREAK-EVEN COST SCALE: {be_net:.2f}x assumed G10 spreads')
        if be_net < 1.0:
            print('  -> BELOW realistic costs. The premise does not survive')
            print('     execution as structured. Do NOT build Phases 3-7 on this.')
            print('     Try: wider wings, longer tenor, slower roll, hold to expiry.')
        elif be_net < 2.0:
            print('  -> MARGINAL. This is a cost-execution problem before it is')
            print('     an alpha problem. Roll cadence (Phase 6) and sizing')
            print('     efficiency (Phase 3) will dominate any signal you build.')
        else:
            print('  -> Comfortable headroom. The premium survives realistic')
            print('     costs; proceed to Phase 3.')

    print()
    print('  CAVEAT: the absolute spreads are ASSUMPTIONS (parametric model, no')
    print('  bid/ask calibration). The break-even SCALE is the robust output --')
    print('  judge it against your own execution, not against my defaults.')


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python run/breakeven_study.py
# Requires a live Bloomberg connection.
# ====================================================================== #
# if __name__ == '__main__':
#     from book.costs import OptionCostModel

#     PAIR = 'USDJPY'

#     # --- 0. Look at the cost surface you are about to assume
#     print(OptionCostModel().describe(PAIR))

#     # Build the dataset ONCE and reuse it across every sweep -- otherwise
#     # each call re-pulls Bloomberg. (The dataset is cached by (pairs, days),
#     # but passing it explicitly makes that guarantee rather than a hope.)
#     ds = FXVolDataset.build(pairs=[PAIR], days=900)

#     # --- 1. The headline sweep: 25d strangle, rolled weekly
#     print('\n--- sweeping cost scale, 25d 1M rolled every 5 days ---')
#     sw = sweep(PAIR, tenor='1M', wing=0.25, roll_days=5, dataset=ds)
#     report(sw)

#     # --- 2. Does going FURTHER out the wing help or hurt? This is the
#     #        central design question for a wing-premium strategy: the 10d is
#     #        richer in convexity terms but costs multiples more to trade.
#     print('\n\n' + '#' * 78)
#     print('# WING COMPARISON -- is the extra richness worth the extra spread?')
#     print('#' * 78)
#     wing_rows = []
#     for w in (0.35, 0.25, 0.15, 0.10):
#         s = sweep(PAIR, tenor='1M', wing=w, roll_days=5,
#                   dataset=ds, verbose=False)
#         wing_rows.append({
#             'wing':          w,
#             'gross_pnl':     s.net_pnl.iloc[0],
#             'net_at_1x':     s.loc[s.scale == 1.0, 'net_pnl'].iloc[0],
#             'tc_at_1x':      s.loc[s.scale == 1.0, 'option_tc'].iloc[0],
#             'volga_be':      s.volga_be.iloc[0],
#             'vanna_be':      s.vanna_be.iloc[0],
#             'breakeven':     breakeven_scale(s),
#         })
#     wt = pd.DataFrame(wing_rows)
#     with pd.option_context('display.width', 200,
#                            'display.float_format', lambda v: f'{v:,.2f}'):
#         print(wt.to_string(index=False))
#     print('\n  The `breakeven` column is the decision. If it FALLS as you go')
#     print('  further out the wing, the extra convexity premium is not paying')
#     print('  for the extra spread, and the sweet spot is nearer 25d than 10d.')
#     print('  That single result should set the strike range for Phase 3.')

#     # --- 3. Does rolling less often help? Cost scales with turnover, and an
#     #        expiring option settles for free while an early roll pays.
#     print('\n\n' + '#' * 78)
#     print('# ROLL CADENCE -- turnover is the other half of the cost equation')
#     print('#' * 78)
#     roll_rows = []
#     for rd in (5, 10, 21):
#         s = sweep(PAIR, tenor='1M', wing=0.25, roll_days=rd,
#                   dataset=ds, verbose=False)
#         roll_rows.append({
#             'roll_days':  rd,
#             'legs':       int(s.legs.iloc[0]),
#             'peak_open':  int(s.peak_open.iloc[0]),
#             'gross_pnl':  s.net_pnl.iloc[0],
#             'net_at_1x':  s.loc[s.scale == 1.0, 'net_pnl'].iloc[0],
#             'tc_at_1x':   s.loc[s.scale == 1.0, 'option_tc'].iloc[0],
#             'breakeven':  breakeven_scale(s),
#         })
#     rt = pd.DataFrame(roll_rows)
#     with pd.option_context('display.width', 200,
#                            'display.float_format', lambda v: f'{v:,.2f}'):
#         print(rt.to_string(index=False))
#     print('\n  Rolling less often cuts cost proportionally but also cuts')
#     print('  deployment, so gross P&L falls too. What matters is whether')
#     print('  `breakeven` improves -- that is cost efficiency per unit of')
#     print('  premium, independent of how much you deployed.')

#     # --- 4. Hold-to-expiry: zero roll cost, one entry cost per vintage.
#     #        The cheapest possible version of this trade.
#     print('\n\n--- for reference: a single hold-to-expiry vintage ---')
#     for sc in (0.0, 1.0, 2.0):
#         cfg = EngineConfig(pairs=[PAIR], cost_model=OptionCostModel(scale=sc),
#                            verbose=False)
#         r = run(HoldStatic(PAIR, strangle(0.25), '1M'), ds, cfg)
#         print(f'  scale {sc:>3.1f}:  net {r.daily.pnl.sum():>12,.0f}   '
#               f'optTC {r.daily.option_tc.sum():>10,.0f}')
#     print('\n  One entry cost, no roll cost, expiry settles free. If the rolled')
#     print('  book cannot clear its cost hurdle but this can, the answer is to')
#     print('  roll at expiry rather than on a cadence.')
