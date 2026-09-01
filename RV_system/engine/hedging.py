"""
hedging.py -- PHASE 4. Generalising the hedge from a scalar delta fraction to
a greek-vector hedge with instruments.

    spot                -> hedges delta   (exists, in book.Book.rebalance_hedge)
    rolled ATM straddle -> hedges vega    (this module)

WHY THIS MODULE IS THIN
-----------------------
The sizing half of Phase 4 was built in Phase 3 without being called that. A
vega re-hedge is exactly

    solve(snap, tenor, GreekTarget(vega=0), current=<what is on risk>)

-- read the drift, trade the increment that flattens it. That is the same
`top_up` path `strategy/roller.py` uses, with a different target and a
one-column menu, which is why `current=` was built at all. What is new here is
only POLICY: what to read, which instrument, what deadband, when to re-strike.

THE ZERO-CODE BENCHMARK YOU MUST BEAT
-------------------------------------
A GreekTarget that pins `vega=0.0` -- which is the normal wing-convexity target
-- ALREADY re-neutralises vega every time the roller fires. So `roll_days=1` is
a daily vega-and-volga controller with no new code at all, and
EXPLANATION.md 10.5 has already measured it on 1M USDCAD, 200 business days:

    roll_days  option_tc  mean|vega|/budget    p90   %>0.20
            5     68,291              0.676  1.380    67.5%
            3     93,796              0.447  1.055    51.0%
            2     88,469              0.209  0.527    34.0%
            1    116,404              0.027  0.000     5.7%

Phase 4's goal is reached at rd=1 for +48k of spread. So the question this
module has to answer is NOT "does it neutralise vega" -- it is "can an ATM
straddle buy rd=1-quality neutrality at closer to rd=5 cost". Run rd=1 as the
control before reading anything here.

WHAT IT BUYS, AND WHAT IT DOES NOT
----------------------------------
10.5 also measured the residual as essentially ZERO-MEAN: signed mean 209
against a mean absolute of 2,029. So a vega hedge buys ATTRIBUTION PURITY and
VARIANCE REDUCTION, not expected return. Equity should get WORSE by roughly the
spread bill. If it improves, be suspicious before being pleased.

THE `current=` TRAP
-------------------
`sizer.solve`'s `sleeve=` argument does DOUBLE DUTY: it tags the new legs AND it
scopes what `current=<Book>` reads (see `sizer.bucket_book`). Passing
`current=book, sleeve='hedge'` would read only the hedge sleeve's own positions
-- the wing book, whose vega you are trying to flatten, would be invisible and
the hedge would size against itself, trading nothing useful forever. So the
residual is computed HERE, across an explicit list of sleeves, and handed in as
a bare GreekVector. That is legal because the hedge menu is a single tenor.

WHY TARGETING ZERO MAKES THE UNITS CANCEL
-----------------------------------------
`solve` multiplies both the carried row and the candidate columns by the same
per-tenor `sigma_scales` factor. For a target of exactly zero that factor
cancels, so "flatten vega" is invariant to `horizon_days`, to `units=`, and to
`ds.nu_rho_source` -- the ~1.8x nu remeasurement that resizes the entire book
(10.4) does not move this decision at all.

The DEADBAND is not invariant, and should not be: it is carried in normalised
units so it can be read as a fraction of the volga budget, which is the unit
10.5 measured drift in. `band = 0.20 * abs(TARGET_VOLGA)` means exactly the
0.20 threshold in the table above.

WHY AN ATM STRADDLE, AND WHY IT MUST BE RE-STRUCK
-------------------------------------------------
A straddle struck at the ATM forward is the purest vega on the menu:
volga_1vp/vega_1vp is ~0.0000 at d1 = 0, against ~+0.05 for a 25d strike. It
does not stay that way. The same straddle with 23 days left, notional held
fixed at whatever flattened 848 of raw vega when fresh, against a 524 raw volga
budget (= 3,000 normalised / w^2 = 5.73), as spot moves off its strike:

    spot move      d1    volga_1vp/vega_1vp   hedge volga_1vp   % of budget
        +0.0%   -0.01                0.0000               0.0          0.0%
        +0.5%   +0.30                0.0064               5.2          1.0%
        +1.0%   +0.60                0.0270              19.3          3.7%
        +2.0%   +1.20                0.1100              46.1          8.8%
        +3.0%   +1.80                0.2476              42.6          8.1%

A stale hedge stops being a pure vega instrument and starts injecting volga
into the sleeve the roller is holding at target -- and the roller CANNOT SEE IT,
because its own `current=` is scoped to its own sleeve. It tops out near 10% of
the budget, so this is hygiene rather than a crisis, but it is the same order as
the tracking error you would otherwise care about. `restrike_move` is the
control.

THE GATE
--------
With the hedge on, `vega_pnl` should collapse toward noise and `volga_pnl_be`
should become the dominant bucket in the convexity sleeve. If it does not, the
sleeve is not isolating what you think it is.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import pandas as pd

from core.calendar import add_tenor
from core.conventions import fx_calendar
from core.greeks import GreekVector
from strategy.sizer import GreekTarget, SizerError, solve, sigma_scales


class VegaHedge:
    """
    Flatten the book's residual vega with a freshly-struck ATM straddle.

    Drop it into the day loop after the strategy that generates the drift --
    it reads the book, it does not care who filled it. With the `Composite`
    pattern:

        strategy = Composite(exiter, roller, vhedge)

    Exit first (an early close is itself a vega jump), roller second (on roll
    days its own solve already pins vega to zero), hedge last and EVERY day --
    which is the whole point, since 10.5 shows the drift accumulates in the
    days BETWEEN rolls, being pinned to exactly zero on each roll day itself.

    Parameters
    ----------
    pair, tenor   : the pair, and the tenor of the hedge straddle. Matching the
                    sleeve's own tenor keeps the hedge in the same vega bucket
                    as the risk it is offsetting.
    read_sleeves  : whose vega is being flattened. `hedge_sleeve` is appended
                    automatically if absent -- omit it and the hedge's own
                    decaying vega is invisible to the next check, so the book
                    drifts and the hedge never notices.
    band          : deadband on |residual vega|, in NORMALISED units. Pass it
                    as a fraction of the volga budget -- 0.20 * 3_000 = 600 --
                    so the number means the same thing as 10.5's table.
    horizon_days  : MUST match the roller's, or `band` is denominated in a
                    different sigma from the target it is a fraction of.
    hedge_sleeve  : attribution tag for the hedge legs. Keeping it separate is
                    what makes `res.by_sleeve()` able to price the hedge, and
                    what the gate above needs in order to be readable.
    restrike_move : fractional spot move from a hedge leg's ENTRY spot at which
                    it is closed and re-struck at the money. None disables
                    re-striking and lets the hedge legs drift into wings.
    min_trade     : deadband on the solved gross notional, by analogy with the
                    roller's. Stops the hedge paying spread to trade noise once
                    the residual is only just outside `band`.
    cost_model    : the SAME OptionCostModel object the sizer and the Book use.
                    Passed to `solve` so the hedge is priced on the surface it
                    will be charged on.
    check_days    : business days between checks. 1 = every day, and there is
                    rarely a reason for more -- the deadband, not the cadence,
                    is the thing that should be stopping trades.
    """

    def __init__(self,
                 pair:          str,
                 tenor:         str,
                 read_sleeves:  Union[str, Sequence[str]],
                 band:          float,
                 horizon_days:  float = 7,
                 hedge_sleeve:  str = 'hedge',
                 restrike_move: Optional[float] = 0.01,
                 min_trade:     float = 250_000.0,
                 cost_model:    Optional[object] = None,
                 check_days:    int = 1):
        if band <= 0:
            raise ValueError("band must be > 0; use a large band to disable, "
                             "not zero -- zero means hedge every day.")
        self.pair          = pair
        self.tenor         = tenor
        self.read_sleeves  = ([read_sleeves] if isinstance(read_sleeves, str)
                              else list(read_sleeves))
        if hedge_sleeve not in self.read_sleeves:
            self.read_sleeves.append(hedge_sleeve)
        self.band          = float(band)
        self.horizon_days  = horizon_days
        self.hedge_sleeve  = hedge_sleeve
        self.restrike_move = restrike_move
        self.min_trade     = min_trade
        self.cost_model    = cost_model
        self.check_days    = check_days

        self.log:  List[dict] = []
        self._last: Optional[int] = None

    # ----------------------------------------------------------------- #
    # What is on risk
    # ----------------------------------------------------------------- #
    def _residual(self, ctx, snap) -> GreekVector:
        """
        Total greeks across `read_sleeves`, REPRICED on today's snap rather
        than read from `book.marks`.

        The engine marks at step 2 of the day, so the cached marks are in fact
        today's and `book.greeks()` would give the same answer. Repricing is
        still the right call: it puts the carried row on exactly the surface
        read that struck the candidate legs, which is the same reason
        `sizer.bucket_book` does not use marks either. One repricing pass over
        ~16 positions is not a cost worth optimising.
        """
        ps = [p for sl in self.read_sleeves
              for p in ctx.book.open_positions(pair=self.pair, sleeve=sl)]
        if not ps:
            return GreekVector.zero(snap.spot)
        return GreekVector.total([p.greeks(snap) for p in ps])

    def _vega_scale(self, snap) -> float:
        """The 'w' factor turning raw vega_1vp into one-sigma normalised vega."""
        expiry = add_tenor(snap.date, self.tenor, fx_calendar(self.pair))
        return sigma_scales(snap, (expiry - snap.date).days,
                            self.horizon_days)['vega']

    def _stale(self, ctx, snap) -> List:
        """Hedge legs too far off their strike to still be pure vega."""
        if self.restrike_move is None:
            return []
        return [p for p in ctx.book.open_positions(pair=self.pair,
                                                   sleeve=self.hedge_sleeve)
                if abs(snap.spot / p.entry_spot - 1.0) > self.restrike_move]

    # ----------------------------------------------------------------- #
    def on_date(self, ctx) -> None:
        if self.pair not in ctx.snaps or ctx.is_last:
            return
        if self._last is not None and (ctx.index - self._last) < self.check_days:
            return
        snap = ctx.snaps[self.pair]
        self._last = ctx.index

        w     = self._vega_scale(snap)
        gv    = self._residual(ctx, snap)
        resid = gv.vega_1vp * w

        row = {'date': snap.date, 'action': None, 'resid_vega': resid,
               'resid_raw': gv.vega_1vp, 'band': self.band,
               'restruck': 0, 'gross': 0.0, 'cost': 0.0, 'note': ''}

        stale = self._stale(ctx, snap)

        if abs(resid) <= self.band and not stale:
            row['action'] = 'in_band'
            self.log.append(row)
            return

        # --- re-strike: close the stale legs, then RE-READ. Closing is itself
        #     a vega trade, so the residual is wrong until it is recomputed.
        for p in stale:
            ctx.book.close(p.pos_id, snap.date, reason='restrike', snap=snap)
        if stale:
            row['restruck'] = len(stale)
            gv    = self._residual(ctx, snap)
            resid = gv.vega_1vp * w
            row['resid_vega'], row['resid_raw'] = resid, gv.vega_1vp

        if abs(resid) <= self.band:
            # unwinding the stale hedge was the whole trade. Common, and the
            # reason the exit spread shows up with no matching entry.
            row['action'] = 'restruck_flat'
            self.log.append(row)
            return

        target = GreekTarget(by_tenor={self.tenor: dict(vega=0.0)},
                             horizon_days = self.horizon_days,
                             units        = 'normalised')
        try:
            res = solve(
                snap, self.tenor, target,
                current         = gv,                 # bare GV: one-tenor menu.
                                                      # NOT current=ctx.book --
                                                      # see the module docstring.
                sleeve          = self.hedge_sleeve,  # tags the new legs ONLY
                allow_deltas    = ['ATM'],
                atm_as_straddle = True,               # 1 column, 1 constraint
                cost_model      = self.cost_model,
                min_notional    = 0.0,                # min_trade is the deadband;
                                                      # a 1x1 system never prunes
                                                      # anyway, this is explicit
                tag_prefix      = 'vhedge_',
            )
        except SizerError as e:
            # An ATM straddle that will not strike is a data problem, not a
            # near-cancelling solve -- but it must be visible either way.
            row['action'] = 'skipped_guard'
            row['note']   = str(e).splitlines()[0]
            self.log.append(row)
            return

        if res.gross_notional < self.min_trade:
            row['action'] = 'deadband'
            row['note']   = f'{res.gross_notional:,.0f} < min_trade'
            self.log.append(row)
            return

        res.open_into(ctx.book, snap)
        row.update(action='hedged', gross=res.gross_notional, cost=res.cost)
        self.log.append(row)

    # ----------------------------------------------------------------- #
    # The log is part of the deliverable -- same argument as the roller's
    # ----------------------------------------------------------------- #
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def report(self) -> str:
        if not self.log:
            return '[vega_hedge] never checked'
        df = self.frame()
        counts = df['action'].value_counts().to_dict()
        out_of_band = float((df['resid_vega'].abs() > self.band).mean())
        return '\n'.join([
            f"[vega_hedge band={self.band:,.0f} normalised, "
            f"restrike={self.restrike_move}] {len(df)} checks: "
            + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())),
            f"  hedge cost {df['cost'].sum():,.0f}"
            f"   gross traded {df['gross'].sum():,.0f}"
            f"   legs re-struck {int(df['restruck'].sum())}",
            f"  |resid| normalised: mean {df['resid_vega'].abs().mean():,.0f}"
            f"  p90 {df['resid_vega'].abs().quantile(0.9):,.0f}"
            f"  peak {df['resid_vega'].abs().max():,.0f}"
            f"  out-of-band {out_of_band:.1%} of checks",
            f"  signed mean {df['resid_vega'].mean():,.0f}"
            f"   <- near zero means this bought variance, not return",
        ])


# ===================================================================== #
# Composing strategies. engine.run() drives ONE object, so anything that
# wants an exit rule plus a roller plus a hedge needs this.
# ===================================================================== #
class Composite:
    """
    Run several strategies in order on each date.

    THE ORDER IS THE SEMANTICS, and there is one right answer for the usual
    three:

        Composite(exiter, roller, vhedge)

    * exiter first  -- an early close is itself a greek jump; everything
                       downstream must read the post-exit book.
    * roller second -- mode='top_up' passes current=ctx.book into the sizer, so
                       running after the exit means the solve already sees the
                       hole and refills it in the same trade step. Reverse them
                       and you top up, then close what you just paid to put on.
    * vhedge last   -- it cleans up whatever vega the first two left, and it
                       must not be reading a book that is about to change.
    """
    def __init__(self, *strategies):
        self.strategies = strategies

    def on_date(self, ctx) -> None:
        for s in self.strategies:
            s.on_date(ctx)


# ===================================================================== #
# TEST BLOCK -- uncomment and run:  python engine/hedging.py
# Requires a live Bloomberg connection.
# ===================================================================== #
# if __name__ == '__main__':
#     import os as _os, sys as _sys
#     _sys.path.insert(0, _os.path.dirname(
#         _os.path.dirname(_os.path.abspath(__file__))))
#     from market.dataset import FXVolDataset
#     from market.snapshot import business_dates
#     from book.costs import OptionCostModel
#     from strategy.roller import GreekTargetRoller
#     from engine.loop import EngineConfig, run
#
#     PAIR, TENOR, HORIZON, BUDGET = 'USDCAD', '1M', 7, -3_000.0
#     cm = OptionCostModel(scale=1.0)
#     ds = FXVolDataset.build(pairs=[PAIR], days=400)
#     dates = business_dates(ds, PAIR)
#
#     tgt = GreekTarget(by_tenor={TENOR: dict(volga=BUDGET, vega=0.0,
#                                            vanna=0.0)},
#                       horizon_days=HORIZON, units='normalised')
#     roller = GreekTargetRoller(PAIR, TENOR, lambda _s: tgt, roll_days=5,
#                                sleeve='wing_convex', mode='top_up',
#                                min_trade=1_000_000,
#                                solve_kw=dict(allow_deltas=[25, 10, 'ATM'],
#                                              cost_model=cm))
#     vh = VegaHedge(PAIR, TENOR, ['wing_convex'], band=0.20 * abs(BUDGET),
#                    horizon_days=HORIZON, cost_model=cm)
#
#     cfg = EngineConfig(pairs=[PAIR], start=dates[-200], end=dates[-5],
#                        cost_model=cm, hedge_fraction=1.0, spot_tc=0.0001)
#     res = run(Composite(roller, vh), ds, cfg)
#     print(roller.report()); print(vh.report())
#     for c in ('vega_pnl', 'volga_pnl_be', 'vanna_pnl_be', 'gamma_pnl_be'):
#         print(f'  {c:<14} {res.daily[c].sum():>12,.0f}')
