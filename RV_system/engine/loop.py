"""
The date loop - the inversion that this whole rewrite exists for.

    OLD:  for trade in trades:  for date in dates:  ...
    NEW:  for date in dates:    for position in book:  ...

That single reordering is what unlocks multi-expiry books, rolling, continuous
resizing, netted hedging and cross-sleeve attribution. None of those were
missing finance in the old stack; they were blocked by the loop ordering.

THE DAILY SEQUENCE, AND WHY IT IS IN THIS ORDER
-----------------------------------------------
  1. SNAPSHOT   build a point-in-time view per pair. Nothing downstream can see
                past `date`, structurally (market/snapshot.py).

  2. MARK       attribute yesterday -> today on the book AS IT WAS. Must precede
                any trading, or a day's P&L gets attributed to a position size
                that was not on risk for it.

  3. HEDGE P&L  P&L and carry on the spot hedge CARRIED IN. Must precede
                rebalancing, or the hedge gets a one-day look-ahead and every
                result is quietly flattered.

  4. TRADE      the strategy opens / closes / resizes. This is the ONLY step
                that changes the book's composition.

  5. REHEDGE    one net spot trade per pair against the NEW book.

  6. RECORD     append the rows.

Swapping 2 and 4, or 3 and 5, produces a backtest that looks fine and is wrong.

WHAT PHASE 1 DELIBERATELY DOES NOT DO
-------------------------------------
  * No option bid/offer. `cost_model` is a hook that Phase 2 fills. Until it
    is filled, DO NOT read these P&L numbers as achievable -- for a
    wing-selling strategy the wing spread is the dominant cost and can consume
    the entire premium.
  * No vega hedging. Phase 4. Without it a "vega-neutral" fly is neutral only
    on the day it is struck.
  * No signals, no sizing solver. Phases 5 and 3. Phase 1 ships a
    hold-a-fixed-structure strategy so the engine can be reconciled against the
    old stack before anything clever is layered on.
  * No USD conversion. Single-pair only for now; cross-pair money sums are not
    legitimate until the reporting layer converts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from core.calendar import add_tenor
from core.conventions import fx_calendar
from market.snapshot import MarketSnapshot, business_dates
from book.book import Book
from book.position import LegRequest, reset_position_ids


# ===================================================================== #
# Strategy interface
# ===================================================================== #
@dataclass
class DayContext:
    """What a strategy is handed each day. Read-only by convention."""
    date:   date
    index:  int                        # 0-based position in the date loop
    snaps:  Dict[str, MarketSnapshot]
    book:   Book
    is_first: bool
    is_last:  bool


class Strategy(Protocol):
    """
    Implement `on_date` to trade. Mutate the book directly -- open(), close(),
    resize(). Return nothing.

    Phases 3/5/6 replace the Phase-1 implementations below with:
        features -> richness z-score -> target greek vector -> sizer -> trades
    but the interface does not change.
    """
    def on_date(self, ctx: DayContext) -> None: ...


class HoldStatic:
    """
    Open one fixed structure on the first date and hold it to expiry.

    This exists for ONE reason: it is the configuration that the old
    `run_backtest_multi_leg` also produces, so it is the only strategy against
    which the new engine can be reconciled bit-for-bit. Use it to validate the
    engine, then never again.

    legs  : list of LegRequest
    tenor : resolved once, on the first date, against the FX calendar
    """
    def __init__(self, pair: str, legs: List[LegRequest], tenor: str):
        self.pair, self.legs, self.tenor = pair, legs, tenor
        self.opened = False

    def on_date(self, ctx: DayContext) -> None:
        if self.opened or self.pair not in ctx.snaps:
            return
        snap   = ctx.snaps[self.pair]
        expiry = add_tenor(snap.date, self.tenor, fx_calendar(self.pair))
        for leg in self.legs:
            ctx.book.open(leg, snap, expiry)
        self.opened = True


class RollingStructure:
    """
    Re-open a fixed structure every `roll_days` business days, letting the
    previous vintages run to expiry alongside. A stacked, continuously-rolled
    book -- the Phase-2 baseline, and the first configuration where netted
    hedging actually differs from per-trade hedging.

    max_concurrent : cap on simultaneous vintages, or None for unlimited.
    """
    def __init__(self, pair: str, legs: List[LegRequest], tenor: str,
                 roll_days: int = 5, max_concurrent: Optional[int] = None):
        self.pair, self.legs, self.tenor = pair, legs, tenor
        self.roll_days, self.max_concurrent = roll_days, max_concurrent
        self._last_open = None

    def on_date(self, ctx: DayContext) -> None:
        if self.pair not in ctx.snaps or ctx.is_last:
            return
        if self._last_open is not None and (ctx.index - self._last_open) < self.roll_days:
            return
        if self.max_concurrent is not None:
            vintages = {p.entry_date for p in ctx.book.open_positions(self.pair)}
            if len(vintages) >= self.max_concurrent:
                return
        snap   = ctx.snaps[self.pair]
        expiry = add_tenor(snap.date, self.tenor, fx_calendar(self.pair))
        for leg in self.legs:
            ctx.book.open(leg, snap, expiry)
        self._last_open = ctx.index


# ===================================================================== #
# Config and result
# ===================================================================== #
@dataclass
class EngineConfig:
    pairs:          List[str]
    start:          Optional[str] = None
    end:            Optional[str] = None
    hedge_fraction: float = 1.0        # 1.0 = full daily hedge (Phase 1)
    spot_tc:        float = 0.0001     # cost per unit of SPOT notional traded
    
    # Phase 2. None means options are FREE to trade -- Phase 1 behaviour, and
    # not achievable for a wing strategy. Pass an OptionCostModel for real work.
    cost_model:     Optional[object] = None
    flatten_at_end: bool = True
    verbose:        bool = True

    # --- reconciliation knobs -------------------------------------------
    # Which point on the yield curve to take the hedge-carry rate differential
    # from. The hedge is an OVERNIGHT spot position, so a short tenor is the
    # defensible choice.
    #
    # The old engine used the OPTION's REMAINING tenor, which shrinks every day
    # -- so its carry drifted as the trade aged (biggest divergence in the last
    # week, and ~2% over a 3M trade). `None` reproduces that by using the
    # shortest remaining tenor among the open positions in that pair.
    carry_tenor_days: Optional[float] = 30.0

    # Reproduce the old engine's half-step strike bootstrap (see
    # MarketSnapshot.solve_strike_and_vol). Reconciliation only.
    legacy_strike_halfstep: bool = False

    # Build the _be expectations from END-of-period nu/rho, as the old engine
    # did (backtest_MLeg.py:630). Reconciliation only -- it is a look-ahead.
    legacy_nu_rho_at_end: bool = False

    # The old engine charged NO transaction cost on the terminal unwind at
    # natural expiry, while charging full cost on an early exit -- an
    # asymmetry baked into every hold-to-expiry vs exit-early comparison it
    # ever produced. The new default is to charge it (more honest). Set False
    # only to reproduce old results.
    charge_tc_on_expiry_unwind: bool = True


@dataclass
class RunResult:
    """
    Three tidy frames plus the final book.

    positions : ONE ROW PER (date, position). The long format is the design
                choice that makes everything downstream a groupby -- by sleeve,
                by pair, by tenor bucket, by delta bucket -- instead of the old
                stack's wide `*_legN` columns, which cannot survive a variable
                position count.
    hedges    : one row per (date, pair) -- hedge P&L, carry, and what traded.
    daily     : the book-level roll-up. `pnl` and `equity` live here.
    """
    positions: pd.DataFrame
    hedges:    pd.DataFrame
    daily:     pd.DataFrame
    book:      Book
    config:    EngineConfig

    def by_sleeve(self, col: str = 'volga_pnl_be') -> pd.DataFrame:
        """
        Cumulative attribution of one bucket, split by sleeve.

        Reindexed onto the full daily calendar, because the first date of a run
        opens positions without marking them (there is no previous day to mark
        against) and so contributes no position rows. Without the reindex the
        frame would be one row shorter than `daily` and would not align.
        """
        p = self.positions.pivot_table(index='date', columns='sleeve',
                                       values=col, aggfunc='sum')
        return p.reindex(self.daily.index).fillna(0.0).cumsum()

    def greek_carried(self, greek: str = 'volga_1vp') -> pd.Series:
        """Mean absolute book exposure to one greek over active days."""
        g = self.positions.groupby('date')[greek].sum()
        return g

    def pnl_per_unit_greek(self, pnl_col: str = 'volga_pnl_be',
                           greek: str = 'volga_1vp') -> float:
        """
        THE metric for this strategy: premium harvested per unit of the risk
        that harvested it. Ranks sleeves and configurations far better than
        calmar, which rests on a single order statistic from one path.
        """
        tot = self.positions[pnl_col].sum()
        carried = self.greek_carried(greek).abs().mean()
        return tot / carried if carried else np.nan


# ===================================================================== #
# The loop
# ===================================================================== #
def run(strategy: Strategy, dataset, config: EngineConfig) -> RunResult:
    """
    Walk the calendar once, driving `strategy`, and return the records.

    The date grid is the UNION of every requested pair's observed spot dates.
    A pair missing on a given date is simply skipped that day rather than
    forward-filled -- inventing a mark is worse than not having one.
    """
    reset_position_ids()

    # ---- date grid
    grids = {p: business_dates(dataset, p, config.start, config.end)
             for p in config.pairs}
    all_dates = sorted(set().union(*[set(g) for g in grids.values()]))
    if not all_dates:
        raise ValueError("no dates in range for the requested pairs")

    book = Book(cost_model=config.cost_model)
    pos_rows:   List[dict] = []
    hedge_rows: List[dict] = []
    cost_rows:  List[dict] = []
    trade_rows: List[dict] = []
    daily_rows: List[dict] = []

    # per-pair previous observation, for hedge P&L and carry
    prev_state: Dict[str, dict] = {}

    if config.verbose:
        print(f"[engine] {len(all_dates)} dates, "
              f"{all_dates[0].date()} -> {all_dates[-1].date()}, "
              f"pairs={config.pairs}")

    for i, d in enumerate(all_dates):
        is_first, is_last = (i == 0), (i == len(all_dates) - 1)

        # ---- 1. SNAPSHOT -------------------------------------------------
        snaps = {p: MarketSnapshot.at(dataset, p, d,
                                      legacy_strike_halfstep=config.legacy_strike_halfstep)
                 for p, g in grids.items() if d in g}
        if not snaps:
            continue

        # ---- 2. MARK (before any trading) --------------------------------
        day_pos_rows = ([] if is_first else
                        book.mark_all(snaps, nu_rho_at_end=config.legacy_nu_rho_at_end))

        # ---- 3. HEDGE P&L on the hedge carried in (before rebalancing) ----
        day_hedge_rows = [] if is_first else book.hedge_pnl(snaps, prev_state)

        # ---- 4. TRADE -----------------------------------------------------
        ctx = DayContext(date=d.date(), index=i, snaps=snaps, book=book,
                         is_first=is_first, is_last=is_last)
        before = set(book.positions)
        strategy.on_date(ctx)
        for pid in set(book.positions) - before:
            p = book.positions[pid]
            trade_rows.append({'date': d.date(), **p.describe(),
                               'entry_premium': p.entry_premium,
                               'action': 'open'})

        # ---- 5. REHEDGE against the new book ------------------------------
        day_trades = []
        expired_today = {r['pair'] for r in day_pos_rows if r.get('expired')}
        for pair in snaps:
            if book.n_open(pair) == 0 and abs(book.hedge.get(pair, 0.0)) < 1e-9:
                continue
            frac = 1.0 if (is_last and config.flatten_at_end) else config.hedge_fraction
            tc = config.spot_tc
            if pair in expired_today and not config.charge_tc_on_expiry_unwind:
                tc = 0.0
            tr = book.rebalance_hedge(pair, fraction=frac, tc_fraction=tc)
            tr['date'] = d.date()
            day_trades.append(tr)

        # merge the rebalance info into that pair's hedge row
        by_pair = {t['pair']: t for t in day_trades}
        for hr in day_hedge_rows:
            hr.update({k: v for k, v in by_pair.get(hr['pair'], {}).items()
                       if k not in ('date', 'pair')})
        if is_first:
            day_hedge_rows = [{'date': d.date(), 'pair': t['pair'],
                               'hedge_pnl': 0.0, 'hedge_carry': 0.0,
                               'hedge_carried': 0.0, 'dt_days': 0.0,
                               'spot': snaps[t['pair']].spot, 'dS': 0.0, **t}
                              for t in day_trades]

        # ---- 6. RECORD -----------------------------------------------------
        day_cost_rows = book.drain_costs()
        pos_rows.extend(day_pos_rows)
        hedge_rows.extend(day_hedge_rows)
        cost_rows.extend(day_cost_rows)
        daily_rows.append(_roll_up(d.date(), day_pos_rows, day_hedge_rows,
                                   day_cost_rows, book))

        for pair, snap in snaps.items():
            prev_state[pair] = {'on': snap.date, 'spot': snap.spot,
                                **_carry_rates(snap, book, pair, config)}

    positions = pd.DataFrame(pos_rows)
    hedges    = pd.DataFrame(hedge_rows)
    costs     = pd.DataFrame(cost_rows)
    daily     = pd.DataFrame(daily_rows).set_index('date')
    daily['equity'] = daily['pnl'].cumsum()
    for c in ('gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be'):
        daily[c + '_cum'] = daily[c].cumsum()

    if config.verbose:
        print(f"[engine] {len(positions)} position-days, "
              f"{len(trade_rows)} legs opened, "
              f"option TC {daily['option_tc'].sum():,.0f}, "
              f"spot TC {daily['hedge_tc'].sum():,.0f}, "
              f"final equity {daily['equity'].iloc[-1]:,.2f}")

    res = RunResult(positions=positions, hedges=hedges, daily=daily,
                    book=book, config=config)
    res.trades = pd.DataFrame(trade_rows)
    res.costs  = costs
    return res


def _carry_rates(snap, book: Book, pair: str, config: EngineConfig) -> dict:
    """
    The (r_d, r_f) pair used for hedge carry over the NEXT period.

    `carry_tenor_days=None` takes the shortest remaining tenor among that
    pair's open positions, which reproduces the old engine's convention of
    using the option's own remaining tenor. With no positions open it falls
    back to 30 days.
    """
    t = config.carry_tenor_days
    if t is None:
        rem = [max((p.expiry - snap.date).days, 1) for p in book.open_positions(pair)]
        t = float(min(rem)) if rem else 30.0
    r_d, r_f = snap.rates(t)
    return {'r_d': r_d, 'r_f': r_f, 'carry_t_days': t}


_FLOW = ['option_pnl', 'delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl',
         'vanna_pnl', 'volga_pnl', 'gamma_pnl_be', 'vanna_pnl_be',
         'volga_pnl_be', 'recon_resid']
_EXPO = ['spot_1pct', 'gamma_1pct', 'vega_1vp', 'volga_1vp',
         'vanna_1pct_1vp', 'theta_1d', 'delta_hedge']


def _roll_up(d, pos_rows: List[dict], hedge_rows: List[dict],
             cost_rows: List[dict], book: Book) -> dict:
    """
    Collapse one day's position and hedge rows into the book-level record.

    FLOW columns are summed then cumsummed into equity downstream.
    EXPO columns are summed and NEVER cumsummed -- they are point-in-time
    levels. Conflating the two is the single easiest way to produce a
    nonsense equity curve; the old stack drew the same distinction and it
    still holds here.
    """
    out = {'date': d, 'n_open': book.n_open()}

    for c in _FLOW:
        out[c] = sum(r.get(c, 0.0) for r in pos_rows)
    for c in _EXPO:
        out['net_' + c] = sum(r.get(c, 0.0) for r in pos_rows)

    out['option_tc']   = sum(r.get('cost', 0.0) for r in cost_rows)
    out['hedge_pnl']   = sum(r.get('hedge_pnl', 0.0) for r in hedge_rows)
    out['hedge_carry'] = sum(r.get('hedge_carry', 0.0) for r in hedge_rows)
    out['hedge_tc']    = sum(r.get('hedge_tc', 0.0) for r in hedge_rows)
    out['net_hedge']   = sum(r.get('hedge_after', 0.0) for r in hedge_rows)

    # Option premium flows are NOT in pnl -- option_pnl is a mark-to-market
    # change, which already embeds the premium from the day the leg was struck.
    out['pnl'] = (out['option_pnl'] + out['hedge_pnl']
                  + out['hedge_carry'] - out['hedge_tc'] - out['option_tc'])
    return out


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python engine/loop.py
# Requires a live Bloomberg connection.
# ====================================================================== #
# if __name__ == '__main__':
#     import os as _os, sys as _sys
#     _sys.path.insert(0, _os.path.dirname(
#         _os.path.dirname(_os.path.abspath(__file__))))
#     from market.dataset import FXVolDataset
#
#     PAIR = 'USDJPY'
#     ds = FXVolDataset.build(pairs=[PAIR], days=400)
#
#     # ---- A: hold ONE short 25d strangle to expiry. This is the config the
#     #         old run_backtest_multi_leg also produces -- the reconciliation
#     #         target. See recon/reconcile.py.
#     legs = [LegRequest('call', -1, 10_000_000, '1M', target_delta=+0.25,
#                        sleeve='wings', tag='wing_c'),
#             LegRequest('put',  -1, 10_000_000, '1M', target_delta=-0.25,
#                        sleeve='wings', tag='wing_p')]
#     cfg = EngineConfig(pairs=[PAIR], start='2025-01-02', end='2025-03-01')
#     res = run(HoldStatic(PAIR, legs, '1M'), ds, cfg)
#
#     print('\n--- daily book, last 8 rows ---')
#     print(res.daily[['n_open', 'option_pnl', 'hedge_pnl', 'hedge_tc',
#                      'pnl', 'equity', 'net_vega_1vp', 'net_volga_1vp']].tail(8))
#
#     print('\n--- attribution over the whole run ---')
#     for c in ('option_pnl', 'hedge_pnl', 'hedge_carry', 'hedge_tc',
#               'delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl',
#               'vanna_pnl', 'volga_pnl', 'recon_resid'):
#         print(f'  {c:<14} {res.daily[c].sum():>14,.2f}')
#     print(f'  {"NET":<14} {res.daily["pnl"].sum():>14,.2f}')
#
#     print('\n--- the premium buckets (cumulative realised MINUS implied) ---')
#     for c in ('gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be'):
#         print(f'  {c:<14} {res.daily[c + "_cum"].iloc[-1]:>14,.2f}')
#
#     # ---- B: the same structure ROLLED every 5 business days. This is the
#     #         first config where netted hedging matters.
#     cfg2 = EngineConfig(pairs=[PAIR], start='2024-06-01', end='2025-06-01')
#     res2 = run(RollingStructure(PAIR, legs, '1M', roll_days=5), ds, cfg2)
#     print(f'\n--- rolling book ---')
#     print(f'  peak concurrent legs   {res2.daily.n_open.max()}')
#     print(f'  total legs opened      {len(res2.trades)}')
#     print(f'  spot TC paid           {res2.daily.hedge_tc.sum():,.2f}')
#     print(f'  final equity           {res2.daily.equity.iloc[-1]:,.2f}')
#
#     # THE metric for this strategy: premium per unit of risk carried.
#     print(f'\n  volga_be per unit |volga| carried: '
#           f'{res2.pnl_per_unit_greek("volga_pnl_be", "volga_1vp"):.4f}')
#     print(f'  vanna_be per unit |vanna| carried: '
#           f'{res2.pnl_per_unit_greek("vanna_pnl_be", "vanna_1pct_1vp"):.4f}')
#     print('\n  Read these as: base-ccy premium harvested per unit of that greek')
#     print('  held. Comparable across tenors, pairs and sleeves in a way that')
#     print('  raw P&L and calmar are not.')
#
#     # ---- C: attribution by sleeve is now a groupby, not a re-run
#     print('\n--- cumulative volga_be by sleeve ---')
#     print(res2.by_sleeve('volga_pnl_be').tail(3))
