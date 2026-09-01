"""
Book - every open position, plus the hedge state, plus the cached marks.

WHAT CHANGED VERSUS THE OLD STACK
---------------------------------
In Delta_Hedged the "book" was an accounting artifact assembled AFTER the fact
by `reporting.build_daily_book`, by superimposing trades that had each already
been run and hedged in isolation. It could describe the portfolio but could not
steer it, and crucially each trade hedged its own delta.

Here the Book is the primary object. It:

    * holds positions across arbitrary pairs and arbitrary expiries;
    * caches one PositionMark per position, rolled forward daily, so the
      surface is hit once per position per day and never twice;
    * owns the SPOT HEDGE per pair, netted across every position in that pair.

That last point is the substantive one. Netting means transaction cost is paid
on net delta, the way a desk pays it, instead of on gross delta across
overlapping trades. For a strategy earning a couple of vol points on the wings,
the difference is not a rounding error.

ORDERING INVARIANT
------------------
The daily loop must be:  MARK -> then TRADE -> then HEDGE.

    mark   : attribute yesterday->today P&L on the book as it actually was
    trade  : open / close / resize positions
    hedge  : compute net delta on the NEW book and rebalance spot

Marking after trading would attribute a day's P&L to a position size that was
not on risk for it. `Book.mark_all` refreshes the cached marks as a side
effect, so calling it is what makes `Book.greeks()` current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from core.greeks import GreekVector
from book.position import LegRequest, Position, open_position
from book.attribution import PositionMark, mark_position, take_mark, hedge_period_pnl


@dataclass
class Book:
    """
    Parameters
    ----------
    positions : pos_id -> Position, open AND recently closed (closed ones are
                retained until `sweep_closed` so the final day's row can be
                written before they disappear).
    marks     : pos_id -> PositionMark, the current state of each OPEN position.
                A position with no mark is settled/closed.
    hedge     : pair -> outstanding spot hedge notional in base ccy.
                Negative = short base. This is the position carried INTO the
                next period.
    """
    positions: Dict[int, Position]  = field(default_factory=dict)
    marks:     Dict[int, PositionMark] = field(default_factory=dict)
    hedge:     Dict[str, float]     = field(default_factory=dict)

    # Every option transaction cost incurred, appended as it happens and
    # drained by the engine loop each day. Kept here rather than returned
    # from each method so a strategy can trade freely without having to
    # remember to plumb costs back out.
    cost_log:  List[dict]           = field(default_factory=list)

    # The book OWNS the cost model. Every option trade routes through
    # Book.open / Book.close / Position.resize, so a strategy physically
    # cannot trade an option without paying the spread. That is deliberate:
    # a forgotten cost is the kind of bug that makes a backtest look good.
    cost_model: Optional[object]    = None

    # ------------------------------------------------------------------ #
    # Membership
    # ------------------------------------------------------------------ #
    def open_positions(self, pair: Optional[str] = None,
                       sleeve: Optional[str] = None) -> List[Position]:
        out = [p for p in self.positions.values()
               if p.is_open and p.pos_id in self.marks]
        if pair is not None:
            out = [p for p in out if p.pair == pair]
        if sleeve is not None:
            out = [p for p in out if p.sleeve == sleeve]
        return out

    @property
    def pairs(self) -> List[str]:
        """Pairs with at least one open position OR a residual spot hedge."""
        live = {p.pair for p in self.open_positions()}
        live |= {k for k, v in self.hedge.items() if abs(v) > 1e-9}
        return sorted(live)

    def n_open(self, pair: Optional[str] = None) -> int:
        return len(self.open_positions(pair))

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def add(self, pos: Position, snap) -> Position:
        """Put an already-filled Position into the book and take its first mark."""
        self.positions[pos.pos_id] = pos
        self.marks[pos.pos_id] = take_mark(pos, snap)
        return pos

    def open(self, req: LegRequest, snap, expiry: date,
             cost_model=None) -> Position:
        """Fill a LegRequest against `snap`, charge the spread, and add it."""
        cm = cost_model if cost_model is not None else self.cost_model
        pos = open_position(req, snap, expiry, cost_model=cm)
        if pos.cost_paid:
            self.cost_log.append({'date': snap.date, 'pair': pos.pair,
                                  'pos_id': pos.pos_id, 'sleeve': pos.sleeve,
                                  'tag': pos.tag, 'reason': 'open',
                                  'cost': pos.cost_paid,
                                  **{k: v for k, v in
                                     getattr(pos, 'entry_cost_detail', {}).items()
                                     if k in ('spread_vp', 'paid_vp', 'bound_by')}})
        return self.add(pos, snap)

    def close(self, pos_id: int, on: date, reason: str = 'closed',
              snap=None, cost_model=None) -> Position:
        """
        Close a position. Removes its mark so it stops being marked, but leaves
        it in `positions` so its final row and its trade record survive.

        COST: an EARLY close crosses the spread and is charged. Expiry is NOT
        -- an expiring option settles to intrinsic, it is not traded out of.
        That asymmetry is real, and it is a genuine reason to prefer holding to
        expiry over rolling early in a wing strategy. (Note the old engine had
        the same asymmetry on the SPOT hedge for the opposite, accidental
        reason; here it is deliberate and applies to the option.)

        This does NOT unwind the spot hedge -- the hedger sees the reduced net
        delta on the next hedge call and adjusts, paying cost on the change.
        """
        pos = self.positions[pos_id]
        cm = cost_model if cost_model is not None else self.cost_model
        if reason != 'expiry' and cm is not None and snap is not None:
            c = cm.charge(pos.option, pos.notional, snap, reason='close')
            pos.cost_paid += c['cost']
            self.cost_log.append({'date': on, 'pair': pos.pair,
                                  'pos_id': pos.pos_id, 'sleeve': pos.sleeve,
                                  'tag': pos.tag, 'reason': 'close',
                                  'cost': c['cost'], 'spread_vp': c['spread_vp'],
                                  'paid_vp': c.get('paid_vp'),
                                  'bound_by': c['bound_by']})
        pos.close(on, reason)
        self.marks.pop(pos_id, None)
        return pos

    def sweep_closed(self) -> int:
        """Drop closed positions from memory. Returns how many went."""
        gone = [k for k, p in self.positions.items() if not p.is_open]
        for k in gone:
            self.positions.pop(k)
        return len(gone)

    # ------------------------------------------------------------------ #
    # Marking -- refreshes the cached marks, so call it before reading greeks
    # ------------------------------------------------------------------ #
    def mark_all(self, snaps: Dict[str, object],
                 nu_rho_at_end: bool = False) -> List[dict]:
        """
        Advance every open position to the snapshots in `snaps` (pair -> snap)
        and return one record row per position.

        Positions that settled at expiry are closed automatically with
        exit_reason='expiry'; their final row is still returned.
        """
        rows: List[dict] = []
        settled: List[int] = []

        for pos in list(self.open_positions()):
            snap = snaps.get(pos.pair)
            if snap is None:
                continue                      # pair has no data on this date
            prev = self.marks[pos.pos_id]
            row, new_mark = mark_position(pos, prev, snap,
                                          nu_rho_at_end=nu_rho_at_end)
            rows.append(row)
            if new_mark is None:
                settled.append(pos.pos_id)
            else:
                self.marks[pos.pos_id] = new_mark

        for pid in settled:
            self.close(pid, snaps[self.positions[pid].pair].date, 'expiry')

        return rows

    def drain_costs(self) -> List[dict]:
        """Take and clear the accumulated option costs. Called once per day
        by the engine loop, which folds them into that day's P&L."""
        out, self.cost_log = self.cost_log, []
        return out

    # ------------------------------------------------------------------ #
    # Risk
    # ------------------------------------------------------------------ #
    def greeks(self, pair: Optional[str] = None,
               sleeve: Optional[str] = None) -> GreekVector:
        """
        Net book risk off the CACHED marks -- so it reflects the last call to
        `mark_all` (or `add`), not a fresh surface read.

        Aggregating across pairs is meaningful for the P&L fields (they are all
        base-ccy money for a standard move) and meaningless for `delta_hedge`,
        which GreekVector blanks to NaN. That is deliberate: see core/greeks.py.

        CAVEAT for cross-pair sums: the P&L fields are each in their OWN pair's
        base currency. Summing USDJPY (USD) with EURUSD (EUR) risk is only
        legitimate after USD conversion. Phase 1 runs single-pair, so this is
        fine; the conversion belongs in the reporting layer before any
        multi-pair risk budget is trusted.
        """
        ps = self.open_positions(pair=pair, sleeve=sleeve)
        return GreekVector.total([self.marks[p.pos_id].greeks for p in ps])

    def net_delta(self, pair: str) -> float:
        """
        Base-ccy spot notional the OPTIONS are long, before the hedge.
        Sum of premium-adjusted deltas across every open position in the pair.
        """
        return sum(self.marks[p.pos_id].greeks.delta_hedge
                   for p in self.open_positions(pair))

    def residual_delta(self, pair: str) -> float:
        """
        What is actually left on risk: option delta plus the spot hedge already
        carried. This -- not gross option delta -- is what a band-hedging rule
        must test. (The old stack's DeltaBandHedge tested gross and therefore
        either never rehedged or degenerated into a daily hedge.)
        """
        return self.net_delta(pair) + self.hedge.get(pair, 0.0)

    # ------------------------------------------------------------------ #
    # Hedging -- ONE net trade per pair
    # ------------------------------------------------------------------ #
    def hedge_pnl(self, snaps: Dict[str, object],
                  prev_state: Dict[str, dict]) -> List[dict]:
        """
        P&L and carry on the hedge CARRIED IN from the previous close, one row
        per pair.

        `prev_state[pair]` must be {'spot':, 'r_d':, 'r_f':, 'on':} as of the
        previous observation. The engine owns that; the book does not cache it,
        because the hedge is a per-pair object and positions may come and go.
        """
        rows = []
        for pair, snap in snaps.items():
            h = self.hedge.get(pair, 0.0)
            ps = prev_state.get(pair)
            if ps is None:
                continue
            dt_days = float((snap.date - ps['on']).days)
            if dt_days <= 0:
                continue
            dS = snap.spot - ps['spot']
            pnl = hedge_period_pnl(h, ps['spot'], dS, ps['r_d'], ps['r_f'], dt_days)
            rows.append({'date': snap.date, 'pair': pair,
                         'hedge_carried': h, 'dt_days': dt_days,
                         'spot': snap.spot, 'dS': dS, **pnl})
        return rows

    def rebalance_hedge(self, pair: str, fraction: float = 1.0,
                        tc_fraction: float = 0.0) -> dict:
        """
        Move the spot hedge toward flat delta and return what was traded.

        target      = -net_delta(pair)          (flat the options' delta)
        gap         = target - current
        traded      = fraction * gap
        cost        = |traded| * tc_fraction

        `fraction` in [0, 1] comes from a hedge rule: 1.0 = full daily hedge,
        0.0 = do nothing, anything between = partial. Phase 1 uses 1.0.

        Note that skipping or partially hedging only ever changes hedge P&L,
        carry and cost. It never changes option_pnl or any greek bucket -- those
        come purely from the option's own risk and the realised market move.
        """
        current = self.hedge.get(pair, 0.0)
        target  = -self.net_delta(pair)
        gap     = target - current
        traded  = fraction * gap
        self.hedge[pair] = current + traded
        return {
            'pair':          pair,
            'hedge_before':  current,
            'hedge_target':  target,
            'hedge_gap':     gap,
            'hedge_traded':  traded,
            'hedge_after':   self.hedge[pair],
            'hedge_tc':      abs(traded) * tc_fraction,
            'rehedged':      abs(traded) > 1e-9,
        }

    def flatten_hedge(self, pair: str, tc_fraction: float = 0.0) -> dict:
        """Unwind the spot hedge entirely (end of backtest, or pair exit)."""
        return self.rebalance_hedge(pair, fraction=1.0, tc_fraction=tc_fraction) \
            if abs(self.net_delta(pair)) > 0 else self._force_flat(pair, tc_fraction)

    def _force_flat(self, pair: str, tc_fraction: float) -> dict:
        current = self.hedge.get(pair, 0.0)
        self.hedge[pair] = 0.0
        return {'pair': pair, 'hedge_before': current, 'hedge_target': 0.0,
                'hedge_gap': -current, 'hedge_traded': -current,
                'hedge_after': 0.0, 'hedge_tc': abs(current) * tc_fraction,
                'rehedged': abs(current) > 1e-9}

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        lines = [f"Book: {self.n_open()} open position(s) across {len(self.pairs)} pair(s)"]
        for pair in self.pairs:
            g = self.greeks(pair)
            lines.append(f"  {pair}: {self.n_open(pair)} legs  "
                         f"vega {g.vega_1vp:>12,.0f}  volga {g.volga_1vp:>10,.0f}  "
                         f"vanna {g.vanna_1pct_1vp:>10,.0f}  theta {g.theta_1d:>10,.0f}")
            lines.append(f"        option delta {self.net_delta(pair):>14,.0f}  "
                         f"hedge {self.hedge.get(pair, 0.0):>14,.0f}  "
                         f"residual {self.residual_delta(pair):>12,.0f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Book({self.n_open()} open, pairs={self.pairs})"


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python book/book.py
# Requires a live Bloomberg connection.
# ====================================================================== #
# if __name__ == '__main__':
#     import os as _os, sys as _sys
#     _sys.path.insert(0, _os.path.dirname(
#         _os.path.dirname(_os.path.abspath(__file__))))
#     from market.dataset import FXVolDataset
#     from market.snapshot import MarketSnapshot, business_dates
#     from core.calendar import add_tenor
#     from core.conventions import fx_calendar
#
#     PAIR = 'USDJPY'
#     ds    = FXVolDataset.build(pairs=[PAIR], days=500)
#     dates = business_dates(ds, PAIR)
#     fxc   = fx_calendar(PAIR)
#
#     snap = MarketSnapshot.at(ds, PAIR, dates[-80])
#     bk   = Book()
#
#     # --- 1. Two OVERLAPPING trades of DIFFERENT tenors in one book.
#     #        The old engine could not hold this at all.
#     for tenor in ('1M', '3M'):
#         exp = add_tenor(snap.date, tenor, fxc)
#         for typ, dl in [('call', +0.25), ('put', -0.25)]:
#             bk.open(LegRequest(typ, -1, 10_000_000, tenor,
#                                target_delta=dl, sleeve='wings'), snap, exp)
#     print(bk.summary(), '\n')
#     assert bk.n_open() == 4
#     assert len({p.expiry for p in bk.open_positions()}) == 2
#     print('[OK] one book, two expiries, four legs\n')
#
#     # --- 2. NETTED hedging. Gross vs net is the whole point.
#     gross = sum(abs(bk.marks[p.pos_id].greeks.delta_hedge)
#                 for p in bk.open_positions())
#     net   = abs(bk.net_delta(PAIR))
#     print(f'  gross |delta| across legs  {gross:>15,.0f}')
#     print(f'  NET delta to hedge         {net:>15,.0f}')
#     print(f'  cost saved by netting      {(gross - net) / gross * 100:>14.1f}%')
#     assert net < gross
#     print('[OK] the book hedges net, not gross -- this is what the old')
#     print('     per-trade hedging was silently overpaying for\n')
#
#     # --- 3. Rebalance flattens delta, and reports what it traded
#     tr = bk.rebalance_hedge(PAIR, fraction=1.0, tc_fraction=0.0001)
#     for k, v in tr.items():
#         print(f'  {k:<14} {v if isinstance(v, (bool, str)) else f"{v:,.2f}"}')
#     assert abs(bk.residual_delta(PAIR)) < 1e-6
#     print('[OK] residual delta is flat after a full hedge\n')
#
#     # --- 4. Roll a few days forward: mark, then read risk
#     import pandas as pd
#     for d in dates[dates > pd.Timestamp(snap.as_of)][:5]:
#         s = MarketSnapshot.at(ds, PAIR, d)
#         rows = bk.mark_all({PAIR: s})
#         pnl  = sum(r['option_pnl'] for r in rows)
#         print(f'  {d.date()}  {len(rows)} legs marked  '
#               f'option P&L {pnl:>12,.2f}  '
#               f'residual delta {bk.residual_delta(PAIR):>13,.0f}')
#     print()
#     print(bk.summary())
#     print('\n  Residual delta drifts between hedges -- that drift IS the gamma')
#     print('  P&L the strategy is trying to be paid for.')
#
#     # --- 5. Sleeve filtering, the field that makes attribution a groupby
#     print(f"\n  wings-sleeve vega only: {bk.greeks(PAIR, sleeve='wings').vega_1vp:,.0f}")
#     print(f"  nonexistent sleeve    : {bk.greeks(PAIR, sleeve='skew').vega_1vp:,.0f} (empty -> 0)")
