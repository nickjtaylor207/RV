"""
roller.py -- rolling a GREEK TARGET rather than a fixed structure.

WHY THIS IS NOT JUST RollingStructure WITH A SIZER BOLTED ON
-----------------------------------------------------------
`engine.loop.RollingStructure` re-opens a fixed structure every N days and lets
old vintages run to expiry alongside. With fixed notional that is coherent: you
asked for 10mm every 5 days and you get a stack of 10mm vintages.

With a greek target it is INCOHERENT. You asked for -2,500 of volga. Roll three
times with the vintages overlapping and the book carries -7,500 -- so the
number you set no longer describes the risk you hold, it describes one third of
it, and the multiple depends on your roll cadence and your tenor. Every
downstream figure (P&L per unit greek, sleeve attribution, the risk budget)
inherits that error.

So a greek target has to be maintained, not re-issued. Three modes, and the
difference between them is the whole point of this module:

    'top_up'   Read what the book already carries, subtract it from the target,
               trade only the difference. Old vintages decay towards expiry and
               each roll tops the book back up. DEFAULT, and correct.

    'replace'  Close every open position in this sleeve, then size a fresh
               structure at target. Also correct, and cleaner to reason about,
               but it pays the exit spread on every roll -- roughly doubling the
               cost line that Phase 2 says is already the binding constraint.

    'stack'    Size the full target every roll and let the vintages pile up.
               This is what you get if you ignore `current=`. It exists ONLY so
               the accumulation can be measured and compared -- do not run a
               strategy on it.

'top_up' also happens to be the mechanism Phase 4 needs. A vega re-hedge is
exactly "the book has drifted to +X vega, trade the increment that returns it
to zero", which is this same call with a different target.


THE LOG IS PART OF THE DELIVERABLE
----------------------------------
Every attempt is recorded in `self.log`, including the ones that did NOT trade.
A greek-target strategy has two new ways to quietly do nothing -- the guard
trips, or the increment falls inside the deadband -- and neither shows up in a
P&L curve. A strategy that skipped 40% of its rolls looks fine and is not the
strategy you specified. `roller.report()` prints the tally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

from strategy.sizer import GreekTarget, SizerError, solve


class GreekTargetRoller:
    """
    Maintain a greek target on a rolling basis.

    Parameters
    ----------
    pair       : the pair to trade.
    tenors     : one tenor or several -- the menu handed to the sizer each roll.
    target_fn  : snap -> GreekTarget. Constant for now:
                     lambda snap: GreekTarget(by_tenor={'3M': dict(
                         volga=-2_500, vega=0, vanna=0)})
                 In Phase 5 this becomes a function of a richness z-score and
                 nothing else in this class changes. That is the seam.
    roll_days  : business days between attempts.
    sleeve     : attribution tag, and the scope of 'replace' / 'top_up'. Only
                 positions in THIS sleeve are read or closed, so two sleeves
                 can roll independently on the same pair.
    mode       : 'top_up' | 'replace' | 'stack' -- see the module docstring.
    min_trade  : deadband. If the solved increment is smaller than this in
                 gross notional, skip rather than pay spread to trade noise.
                 The whole point of top_up is that increments get small.
    solve_kw   : passed through to sizer.solve (allow_deltas, cost_model,
                 min_notional, require_leverage, ...).
    """

    def __init__(self,
                 pair:      str,
                 tenors:    Union[str, Sequence[str]],
                 target_fn: Callable[[object], GreekTarget],
                 roll_days: int = 5,
                 sleeve:    str = 'convexity',
                 mode:      str = 'top_up',
                 min_trade: float = 1_000_000.0,
                 solve_kw:  Optional[dict] = None):
        if mode not in ('top_up', 'replace', 'stack'):
            raise ValueError("mode must be 'top_up', 'replace' or 'stack'")
        self.pair      = pair
        self.tenors    = [tenors] if isinstance(tenors, str) else list(tenors)
        self.target_fn = target_fn
        self.roll_days = roll_days
        self.sleeve    = sleeve
        self.mode      = mode
        self.min_trade = min_trade
        self.solve_kw  = dict(solve_kw or {})

        self.log:  List[dict] = []
        self._last: Optional[int] = None

    # ----------------------------------------------------------------- #
    def on_date(self, ctx) -> None:
        if self.pair not in ctx.snaps or ctx.is_last:
            return
        if self._last is not None and (ctx.index - self._last) < self.roll_days:
            return

        snap = ctx.snaps[self.pair]
        row  = {'date': snap.date, 'mode': self.mode, 'action': None,
                'legs': 0, 'gross': 0.0, 'cost': 0.0,
                'closed': 0, 'leverage': float('nan'),
                'cond': float('nan'), 'note': ''}

        # 'replace' closes first, so the solve sees a flat book.
        if self.mode == 'replace':
            for pos in list(ctx.book.open_positions(pair=self.pair,
                                                    sleeve=self.sleeve)):
                ctx.book.close(pos.pos_id, snap.date, reason='roll', snap=snap)
                row['closed'] += 1

        current = ctx.book if self.mode == 'top_up' else None

        try:
            res = solve(snap, self.tenors, self.target_fn(snap),
                        current=current, sleeve=self.sleeve, **self.solve_kw)
        except SizerError as e:
            # A guard trip is a market condition, not a bug: a wing that will
            # not strike, or a menu that cannot span the target today. No trade
            # beats a bad trade -- but it must be visible.
            row['action'] = 'skipped_guard'
            row['note']   = str(e).splitlines()[0]
            self.log.append(row)
            self._last = ctx.index
            return

        row['leverage'] = res.leverage
        row['cond']     = res.condition
        for (bucket, g, want), got in zip(res.target_rows, res.achieved):
            nm = f"{g}@{bucket or 'net'}"
            row[f'want_{nm}'] = want
            row[f'book_{nm}'] = got

        if res.gross_notional < self.min_trade:
            row['action'] = 'deadband'
            row['note']   = f"increment {res.gross_notional:,.0f} < min_trade"
            self.log.append(row)
            self._last = ctx.index
            return

        res.open_into(ctx.book, snap)
        row.update(action='traded', legs=len(res.legs),
                   gross=res.gross_notional, cost=res.cost)
        self.log.append(row)
        self._last = ctx.index

    # ----------------------------------------------------------------- #
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def report(self) -> str:
        """Tally of what happened, including the rolls that did nothing."""
        if not self.log:
            return f"[{self.mode}] no roll attempts"
        df = self.frame()
        counts = df['action'].value_counts().to_dict()
        lines = [f"[{self.mode}] {len(df)} roll attempts: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                 f"  total entry cost {df['cost'].sum():,.0f}"
                 f"   gross traded {df['gross'].sum():,.0f}"
                 f"   positions closed {int(df['closed'].sum())}"]
        skips = df[df['action'] == 'skipped_guard']
        if len(skips):
            lines.append(f"  guard trips ({len(skips)}):")
            for note in skips['note'].value_counts().head(3).items():
                lines.append(f"    {note[1]}x  {note[0]}")
        return "\n".join(lines)
