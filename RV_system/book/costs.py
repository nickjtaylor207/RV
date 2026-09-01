"""
Option transaction costs - the single most important correction in the model.

WHY THIS IS THE GATING ITEM
---------------------------
Neither the old stack nor Phase 1 charged anything to trade an option.
`tc_fraction` only ever multiplied the SPOT hedge. For a delta-hedged ATM
strategy that is survivable. For a strategy whose entire edge is a couple of
vol points harvested from the WINGS it is not, for three compounding reasons:

  1. Wing spreads are multiples of ATM spreads. A 10-delta option does not
     cost 1.4x an ATM option to trade -- it costs 2x or more in vol points,
     on top of already having less vega to amortise it against.
  2. A continuously rolled book pays it repeatedly, not once.
  3. The premium being harvested is small in absolute terms, so the cost is a
     large fraction of it rather than a rounding error.

Until this module is wired in, no P&L number out of this engine is achievable.
That is the whole point of Phase 2.

THE CONVENTION: COST IS PAID IN VOL POINTS
------------------------------------------
FX options are quoted and traded in volatility. Crossing the spread means
dealing away from mid by some number of vol points, and what that costs you in
money is:

    cost = |notional| * vega_per_vol_point * (spread_vp * crossing_fraction)

This is the right way round, and it matters. Deep wings have LITTLE vega, so a
naive "cost = x% of vega" model would make them look cheap to trade. They are
not: their vol spread is several times wider, and the wing multiplier below is
what carries that. The two effects partly offset, which is exactly why you have
to model them separately instead of assuming a flat cost on vega.

A PREMIUM FLOOR, BECAUSE VEGA GOES TO ZERO AND SPREADS DO NOT
-------------------------------------------------------------
At 5 delta and short tenors, vega collapses toward zero and a pure
vega x vol-spread cost goes to zero with it. Real market makers do not quote
that way -- they hold a minimum spread as a fraction of premium. `min_premium_frac`
imposes that floor, and the charged cost is the MAX of the two. Without it this
model systematically understates the cost of exactly the trades the strategy
wants to do most.

THESE NUMBERS ARE ASSUMPTIONS
-----------------------------
You chose the parametric route, so the defaults below are a considered guess at
G10 interbank levels, not measurements. Treat every result that depends on them
as a SENSITIVITY, not an estimate. The honest use of this module is not "what
does the strategy make" -- it is `run/breakeven_study.py`'s question:

    at what wing spread does the premium go to zero, and is that level
    above or below what I would really pay?

If the break-even sits below realistic costs, the premise is wrong and no
amount of Phase 3-7 machinery will fix it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# --------------------------------------------------------------------- #
# Defaults -- G10 interbank-ish, full bid-ask, in VOL POINTS
# --------------------------------------------------------------------- #

# 1-MONTH ATM full bid-ask, per pair. 'DEFAULT' catches anything unlisted.
BASE_VP_1M_ATM: Dict[str, float] = {
    'EURUSD':  0.20,
    'USDJPY':  0.25,
    'GBPUSD':  0.28,
    'USDCHF':  0.30,
    'AUDUSD':  0.30,
    'NZDUSD':  0.38,
    'USDCAD':  0.30,
    'DEFAULT': 0.35,
}

# Wing multiplier on the ATM spread, indexed by |spot delta|.
# Interpolated linearly in |delta| between these knots and held flat outside.
WING_DELTA_KNOTS = np.array([0.05, 0.10, 0.15, 0.25, 0.35, 0.50])
WING_MULT_KNOTS  = np.array([3.50, 2.20, 1.75, 1.40, 1.20, 1.00])

# Tenor scaling. Vol spreads widen as tenor shortens, roughly as 1/sqrt(T),
# because the premium spread a maker needs is broadly tenor-invariant while
# vega scales with sqrt(T). Clipped so 1W does not explode and 1Y does not
# collapse to nothing.
TENOR_REF_DAYS   = 30.0
TENOR_MULT_FLOOR = 0.60
TENOR_MULT_CAP   = 3.00


@dataclass
class OptionCostModel:
    """
    Parametric vol-point spread model.

    ONE interface -- `charge` -- is used by every option trade in the system:
    strategy legs, rolls, resizes, early closes, and (Phase 4) the ATM straddle
    vega hedge. Nothing gets to trade an option without going through here.

    Parameters
    ----------
    base_vp            : pair -> 1M ATM full bid-ask in vol points.
    crossing_fraction  : how much of the quoted width you actually pay.
                         0.5 = you deal at bid or ask from mid, the normal
                         assumption. 1.0 = you always pay the full width, a
                         reasonable stress case for a systematic taker.
    min_premium_frac   : floor on the cost as a fraction of the option's own
                         premium. Binds in the deep wings where vega -> 0.
                         0.03 = never pay less than 3% of premium.
    scale              : global multiplier. THE knob the break-even study
                         sweeps. 1.0 = the defaults above; 2.0 = twice as
                         expensive as assumed; 0.0 = free (Phase 1 behaviour).
    enabled            : hard off switch, for reproducing Phase 1 exactly.
    """
    base_vp:           Dict[str, float] = field(default_factory=lambda: dict(BASE_VP_1M_ATM))
    crossing_fraction: float = 0.5
    min_premium_frac:  float = 0.03
    scale:             float = 1.0
    enabled:           bool  = True

    # ------------------------------------------------------------------ #
    # The spread surface
    # ------------------------------------------------------------------ #
    def wing_mult(self, abs_delta: float) -> float:
        """Multiplier on the ATM spread for a given |spot delta|."""
        return float(np.interp(abs(abs_delta), WING_DELTA_KNOTS, WING_MULT_KNOTS))

    def tenor_mult(self, t_days: float) -> float:
        """Multiplier on the 1M spread for a given remaining tenor."""
        m = np.sqrt(TENOR_REF_DAYS / max(float(t_days), 1.0))
        return float(np.clip(m, TENOR_MULT_FLOOR, TENOR_MULT_CAP))

    def spread_vp(self, pair: str, t_days: float, abs_delta: float) -> float:
        """
        FULL quoted bid-ask in vol points for this pair / tenor / strike.

        Note this is the full width. What you PAY is
        `spread_vp * crossing_fraction`.
        """
        base = self.base_vp.get(pair, self.base_vp.get('DEFAULT', 0.35))
        return base * self.tenor_mult(t_days) * self.wing_mult(abs_delta) * self.scale

    # ------------------------------------------------------------------ #
    # The one interface every option trade goes through
    # ------------------------------------------------------------------ #
    def charge(self, option, notional: float, snap,
               sigma: Optional[float] = None,
               abs_delta: Optional[float] = None,
               reason: str = 'trade') -> dict:
        """
        Cost, in base ccy, of trading `notional` of `option` as of `snap`.

        `notional` is a POSITIVE magnitude of what was traded -- direction is
        irrelevant, you cross the spread either way, buying or selling.

        sigma / abs_delta are optional overrides. Left None they are read off
        the snapshot, which is what you want everywhere except a unit test.

        Returns a record dict rather than a bare float so the cost is auditable:
        you can always see which spread was applied and which of the two
        mechanisms (vega or premium floor) actually bound.
        """
        notional = abs(float(notional))
        if not self.enabled or notional == 0.0 or self.scale == 0.0:
            return {'cost': 0.0, 'spread_vp': 0.0, 'bound_by': 'disabled',
                    'notional_traded': notional, 'reason': reason}

        t_days = max((option.expiry - snap.date).days, 1e-6)
        st_sigma = sigma if sigma is not None else snap.smile_vol(option.K, t_days)
        r_d, r_f = snap.rates(t_days)
        S = snap.spot

        if abs_delta is None:
            abs_delta = abs(option.spot_delta(S, st_sigma, r_d, r_f, snap.date))

        vp = self.spread_vp(option.pair, t_days, abs_delta)
        paid_vp = vp * self.crossing_fraction

        # --- mechanism 1: vega x vol points. `vega_1vp` on a unit-notional,
        #     unit-direction GreekVector is already base-ccy P&L per vol point.
        g = option.greeks(S, st_sigma, r_d, r_f, snap.date,
                          notional=1.0, direction=1)
        vega_cost = notional * abs(g.vega_1vp) * paid_vp

        # --- mechanism 2: floor as a fraction of premium
        premium = notional * abs(option.value_base(S, st_sigma, r_d, r_f, snap.date))
        floor_cost = premium * self.min_premium_frac

        cost = max(vega_cost, floor_cost)
        return {
            'cost':            cost,
            'spread_vp':       vp,
            'paid_vp':         paid_vp,
            'abs_delta':       abs_delta,
            't_days':          t_days,
            'vega_cost':       vega_cost,
            'floor_cost':      floor_cost,
            'bound_by':        'premium_floor' if floor_cost > vega_cost else 'vega',
            'notional_traded': notional,
            'reason':          reason,
        }

    # ------------------------------------------------------------------ #
    def describe(self, pair: str = 'USDJPY') -> str:
        """The spread surface as a table -- read this before trusting a result."""
        lines = [f"OptionCostModel  pair={pair}  scale={self.scale:g}  "
                 f"crossing={self.crossing_fraction:g}  "
                 f"premium_floor={self.min_premium_frac:.1%}",
                 "  FULL bid-ask in vol points (you pay crossing_fraction of this)",
                 "  tenor |   50d    35d    25d    15d    10d     5d"]
        for tenor, td in [('1W', 7), ('1M', 30), ('2M', 60), ('3M', 91), ('6M', 182)]:
            row = "  ".join(f"{self.spread_vp(pair, td, d):5.3f}"
                            for d in (0.50, 0.35, 0.25, 0.15, 0.10, 0.05))
            lines.append(f"  {tenor:>5} |  {row}")
        return "\n".join(lines)


# A free model, for reproducing Phase 1 / the old engine exactly.
FREE = OptionCostModel(enabled=False)


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python book/costs.py
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
#     from core.option import FXOption

#     PAIR = 'USDJPY'
#     cm = OptionCostModel()

#     # --- 1. Look at the assumed surface BEFORE looking at any P&L.
#     print(cm.describe(PAIR))
#     print()
#     print('  Read across a row: wings cost multiples of ATM.')
#     print('  Read down a column: short tenors cost more per vol point.')
#     print('  These are ASSUMPTIONS. The break-even study is what matters.\n')

#     ds    = FXVolDataset.build(pairs=[PAIR], days=400)
#     dates = business_dates(ds, PAIR)
#     snap  = MarketSnapshot.at(ds, PAIR, dates[-40])
#     exp   = add_tenor(snap.date, '1M', fx_calendar(PAIR))
#     N     = 10_000_000

#     # --- 2. Cost of one leg, across the smile. THE key table.
#     print(f'{PAIR} 1M, {N/1e6:.0f}mm notional per leg, as of {snap.date}')
#     print('  leg     K         vol      spread  paid   cost(base)  '
#           '%prem  bound_by')
#     for tgt, typ in [(+0.50, 'call'), (+0.25, 'call'), (-0.25, 'put'),
#                      (-0.10, 'put'), (-0.05, 'put')]:
#         K, sig = snap.solve_strike_and_vol(tgt, typ, exp)
#         opt = FXOption(pair=PAIR, K=K, expiry=exp, option_type=typ)
#         c   = cm.charge(opt, N, snap, sigma=sig)
#         prem = N * abs(opt.value_base(snap.spot, sig, *snap.rates(30), snap.date))
#         print(f'  {typ[0].upper()}{abs(tgt)*100:>3.0f}d  {K:9.4f}  {sig*100:6.3f}%  '
#               f'{c["spread_vp"]:5.3f}  {c["paid_vp"]:5.3f}  '
#               f'{c["cost"]:>10,.0f}  {c["cost"]/prem*100:5.1f}  {c["bound_by"]}')
#     print()
#     print('  NOTE the %prem column. A 25d leg costs a few percent of its own')
#     print('  premium to trade; a 5d leg costs a large fraction of it. That is')
#     print('  the tax on the wing premium this strategy is trying to collect.\n')

#     # --- 3. Cost of the whole structure, round trip, vs premium received
#     for wing in (0.25, 0.10):
#         legs = []
#         for tgt, typ in [(+wing, 'call'), (-wing, 'put')]:
#             K, sig = snap.solve_strike_and_vol(tgt, typ, exp)
#             legs.append((FXOption(pair=PAIR, K=K, expiry=exp, option_type=typ), sig))
#         prem = sum(N * abs(o.value_base(snap.spot, s, *snap.rates(30), snap.date))
#                    for o, s in legs)
#         cost_in = sum(cm.charge(o, N, snap, sigma=s)['cost'] for o, s in legs)
#         print(f'  short {wing*100:.0f}d strangle: premium {prem:>10,.0f}   '
#               f'entry cost {cost_in:>9,.0f}  ({cost_in/prem*100:4.1f}% of premium)')
#     print()
#     print('  Held to expiry you pay entry cost only -- expiry settles, it does')
#     print('  not trade. A ROLLED book pays entry cost every roll, which is where')
#     print('  this actually bites. See run/breakeven_study.py.\n')

#     # --- 4. The premium floor binding in the deep wings
#     K, sig = snap.solve_strike_and_vol(-0.05, 'put', exp)
#     opt = FXOption(pair=PAIR, K=K, expiry=exp, option_type='put')
#     c_floor = OptionCostModel(min_premium_frac=0.03).charge(opt, N, snap, sigma=sig)
#     c_novs  = OptionCostModel(min_premium_frac=0.00).charge(opt, N, snap, sigma=sig)
#     print(f'  5d put: vega-only cost {c_novs["cost"]:,.0f}, '
#           f'with 3% premium floor {c_floor["cost"]:,.0f}  '
#           f'({c_floor["bound_by"]})')
#     print('  If the floor binds, a pure vega-based cost model was understating')
#     print('  the wings -- exactly where you least want to be wrong.\n')

#     # --- 5. The scale knob the break-even study sweeps
#     K, sig = snap.solve_strike_and_vol(-0.25, 'put', exp)
#     opt = FXOption(pair=PAIR, K=K, expiry=exp, option_type='put')
#     for s in (0.0, 0.5, 1.0, 2.0, 3.0):
#         c = OptionCostModel(scale=s).charge(opt, N, snap, sigma=sig)
#         print(f'  scale={s:>3.1f}  spread {c["spread_vp"]:5.3f}vp  '
#               f'cost {c["cost"]:>9,.0f}')
#     print('\n  scale=0 reproduces Phase 1 (free options) exactly.')
