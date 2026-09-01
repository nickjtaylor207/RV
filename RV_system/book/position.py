"""
Position - one option leg, alive across many dates.

THE STRUCTURAL CHANGE FROM THE OLD STACK
----------------------------------------
In Delta_Hedged a `LegSpec` was an instruction consumed once, inside a single
`run_backtest_multi_leg` call, and it died when that call returned. All legs in
one call shared one expiry, so calendar spreads were impossible and rolling was
impossible.

A `Position` here is a long-lived object with an identity (`pos_id`). It is
opened on some date, carried across the daily loop, and closed either by expiry,
by a roll, or by a risk decision. The book holds an arbitrary number of them
across arbitrary expiries and pairs. That is the whole unlock.

TWO OBJECTS, DELIBERATELY SEPARATE
----------------------------------
    LegRequest : what you ASK for -- "a 25-delta put, 1M, short, 10mm".
                 Delta-space, notional-space. No market data in it.

    Position   : what you GOT -- a struck strike, the smile vol you were struck
                 at, the premium, and an id. Created by `open_position`, which
                 is the only place market data enters.

That mirrors order -> fill, and it means a strategy can express intent without
touching a snapshot. Phase 3's sizer will emit LegRequests; the engine fills them.

SLEEVES
-------
Every position carries a `sleeve` tag: 'convexity', 'skew', 'atm', 'hedge'.
This is the field that lets you answer "did the volga sleeve actually make
money" with a groupby instead of a re-run. Tag honestly at creation -- it cannot
be reconstructed later.

INVARIANT THE ENGINE MUST HOLD
------------------------------
`notional` is mutable (positions can be resized or partially closed), and the
greeks are always computed off the CURRENT notional. Therefore the daily loop
must MARK BEFORE IT TRADES: attribute yesterday->today P&L on yesterday's book,
then resize/open/close. Doing it the other way round attributes today's P&L to
a position size that was not on risk for it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from core.greeks import GreekVector
from core.option import FXOption

# Process-wide position id counter. Ids are for joining records, not for
# ordering, so a simple counter is enough.
_POS_ID = itertools.count(1)


def reset_position_ids() -> None:
    """Restart the id counter. Call between independent backtests so record
    files from separate runs do not share ids."""
    global _POS_ID
    _POS_ID = itertools.count(1)


# --------------------------------------------------------------------- #
# What you ask for
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class LegRequest:
    """
    An instruction to open one leg, in delta space. Contains no market data, so
    it can be built by a strategy that has never seen a snapshot.

    option_type  : 'call' or 'put'
    target_delta : SIGNED spot delta to strike at. +0.25 = 25d call,
                   -0.10 = 10d put. Use `atm=True` for ATM-forward instead.
    direction    : +1 long, -1 short. Direction NEVER lives in the notional --
                   `notional` is always a positive magnitude. (Carried over
                   from the old stack; it holds at every layer here too.)
    notional     : base-ccy notional, positive.
    tenor        : tenor string ('1W','1M','3M'...) resolved against the FX
                   calendar at open time.
    sleeve       : attribution tag.
    atm          : if True, strike at ATM-forward and ignore target_delta.
    """
    option_type:  str
    direction:    int
    notional:     float
    tenor:        str
    target_delta: float = np.nan
    sleeve:       str   = 'unclassified'
    atm:          bool  = False
    tag:          str   = ''      # free-text, e.g. 'wing_c', 'body'

    def __post_init__(self):
        if self.direction not in (+1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction}")
        if self.notional < 0:
            raise ValueError("notional must be a positive magnitude; "
                             "put the sign in `direction`")
        if self.option_type not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call'/'put', got {self.option_type}")
        if not self.atm and not np.isfinite(self.target_delta):
            raise ValueError("give either target_delta or atm=True")


# --------------------------------------------------------------------- #
# What you got
# --------------------------------------------------------------------- #
@dataclass
class Position:
    """
    A live option leg.

    Contract terms (fixed at open): pair, option, entry_date, direction.
    State (mutable): notional, closed/exit fields.
    Provenance (fixed, for grouping and diagnostics): everything in the
    'entry snapshot' block.
    """
    pos_id:      int
    pair:        str
    option:      FXOption
    direction:   int
    notional:    float

    entry_date:  date
    expiry:      date
    sleeve:      str = 'unclassified'
    tag:         str = ''

    # --- entry snapshot: what the market looked like when this was struck ---
    tenor_label:      str   = ''      # '1M' -- the requested tenor, for grouping
    target_delta:     float = np.nan  # what was asked for, for delta-bucket grouping
    entry_spot:       float = np.nan
    entry_vol:        float = np.nan  # smile vol AT the struck strike
    entry_atm_vol:    float = np.nan  # pillar ATM vol, for the smile-premium read
    entry_value_base: float = np.nan  # V = P/S per unit notional at entry
    entry_premium:    float = np.nan  # SIGNED base-ccy cashflow: negative = paid

    # --- exit state ---
    exit_date:   Optional[date] = None
    exit_reason: Optional[str]  = None

    # --- transaction cost actually paid on this position, base ccy, positive.
    #     Accumulates across open / resize / early close. Expiry adds nothing:
    #     an expiring option SETTLES, it is not traded out of.
    cost_paid:   float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def signed_notional(self) -> float:
        return self.notional * self.direction

    @property
    def strike(self) -> float:
        return self.option.K

    @property
    def option_type(self) -> str:
        return self.option.option_type

    def days_to_expiry(self, on: date) -> int:
        return (self.expiry - on).days

    def is_expired(self, on: date) -> bool:
        return self.days_to_expiry(on) <= 0

    # ------------------------------------------------------------------ #
    # Valuation and risk -- always off the CURRENT notional
    # ------------------------------------------------------------------ #
    def value_base(self, snap) -> float:
        """
        Signed mark-to-market in base ccy. A short position has negative value
        (it is a liability), so `value_base` differences ARE P&L with no sign
        flip needed anywhere downstream.
        """
        st = snap.price_state(self.option.K, self.expiry)
        v  = self.option.value_base(st['S'], st['sigma'], st['r_d'], st['r_f'],
                                    snap.date)
        return v * self.signed_notional

    def settle_base(self, spot: float) -> float:
        """Signed intrinsic settlement value in base ccy, at expiry."""
        return self.option.intrinsic_base(spot) * self.signed_notional

    def greeks(self, snap) -> GreekVector:
        """Position-level GreekVector: notional and direction already applied."""
        st = snap.price_state(self.option.K, self.expiry)
        return self.option.greeks(st['S'], st['sigma'], st['r_d'], st['r_f'],
                                  snap.date,
                                  notional=self.notional, direction=self.direction)

    def greeks_from_state(self, st: dict, on: date) -> GreekVector:
        """
        Same, but reusing a price_state dict already fetched. The daily loop
        uses this so a 40-leg book does not hit the surface 40 times for the
        same (strike, expiry) it just priced.
        """
        return self.option.greeks(st['S'], st['sigma'], st['r_d'], st['r_f'], on,
                                  notional=self.notional, direction=self.direction)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def resize(self, new_notional: float, snap=None, cost_model=None) -> dict:
        """
        Change size in place.

        Returns a record containing `traded` (the CHANGE in notional, positive
        = added risk) and `cost`. You cross the spread in both directions, so
        reducing risk is charged exactly like adding it.
        """
        if new_notional < 0:
            raise ValueError("notional must stay a positive magnitude")
        traded = new_notional - self.notional
        self.notional = new_notional

        cost = 0.0
        detail = None
        if cost_model is not None and snap is not None and abs(traded) > 0:
            detail = cost_model.charge(self.option, abs(traded), snap,
                                       abs_delta=abs(self.target_delta)
                                       if np.isfinite(self.target_delta) else None,
                                       reason='resize')
            cost = detail['cost']
            self.cost_paid += cost
        return {'traded': traded, 'cost': cost, 'detail': detail}

    def close(self, on: date, reason: str) -> None:
        self.exit_date = on
        self.exit_reason = reason

    # ------------------------------------------------------------------ #
    def describe(self) -> dict:
        """Static fields for the record row. Recomputed nothing."""
        return {
            'pos_id':       self.pos_id,
            'pair':         self.pair,
            'sleeve':       self.sleeve,
            'tag':          self.tag,
            'tenor_label':  self.tenor_label,
            'option_type':  self.option_type,
            'target_delta': self.target_delta,
            'strike':       self.option.K,
            'direction':    self.direction,
            'notional':     self.notional,
            'entry_date':   self.entry_date,
            'expiry':       self.expiry,
            'entry_spot':   self.entry_spot,
            'entry_vol':    self.entry_vol,
            'entry_atm_vol': self.entry_atm_vol,
        }

    def __repr__(self) -> str:
        d = 'long ' if self.direction > 0 else 'short'
        live = '' if self.is_open else f' CLOSED({self.exit_reason})'
        return (f"Pos#{self.pos_id} {d} {self.notional / 1e6:.1f}mm {self.pair} "
                f"{self.tenor_label} {self.option_type} K={self.option.K:.4f} "
                f"exp={self.expiry} [{self.sleeve}]{live}")


# --------------------------------------------------------------------- #
# The only place market data turns a request into a position
# --------------------------------------------------------------------- #
def open_position(req: LegRequest, snap, expiry: date,
                  cost_model=None) -> Position:
    """
    Fill a LegRequest against a snapshot.

    Resolves the strike (via the snapshot's strike/smile-vol fixed point, or
    ATM-forward if req.atm), records the entry state, and returns a live
    Position.

    `cost_model` (book/costs.py) charges the vol-point spread on entry. Pass
    None -- or costs.FREE -- to reproduce Phase 1 / the old engine, which
    charged nothing to trade an option. Do not read cost-free P&L as
    achievable for a wing-selling strategy.
    """
    t_days = max((expiry - snap.date).days, 1e-6)

    if req.atm:
        from core.option import atm_forward_strike
        r_d, r_f = snap.rates(t_days)
        K = atm_forward_strike(snap.spot, r_d, r_f, expiry, snap.date,
                               pair=snap.pair)
        sigma = snap.smile_vol(K, t_days)
    else:
        K, sigma = snap.solve_strike_and_vol(req.target_delta, req.option_type, expiry)

    opt = FXOption(pair=snap.pair, K=K, expiry=expiry,
                   option_type=req.option_type,
                   S0=snap.spot, sigma0=sigma)
    r_d, r_f = snap.rates(t_days)
    opt.r_d, opt.r_f = r_d, r_f

    v_base = opt.value_base(snap.spot, sigma, r_d, r_f, snap.date)

    pos = Position(
        pos_id       = next(_POS_ID),
        pair         = snap.pair,
        option       = opt,
        direction    = req.direction,
        notional     = req.notional,
        entry_date   = snap.date,
        expiry       = expiry,
        sleeve       = req.sleeve,
        tag          = req.tag,
        tenor_label  = req.tenor,
        target_delta = req.target_delta,
        entry_spot   = snap.spot,
        entry_vol    = sigma,
        entry_atm_vol= snap.atm_vol(t_days),
        entry_value_base = v_base,
        # Signed cashflow: long pays (negative), short receives (positive).
        entry_premium = -req.direction * req.notional * v_base,
    )

    if cost_model is not None:
        c = cost_model.charge(opt, req.notional, snap, sigma=sigma,
                              abs_delta=abs(req.target_delta) if not req.atm else 0.5,
                              reason='open')
        pos.cost_paid = c['cost']
        pos.entry_cost_detail = c
    return pos


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python book/position.py
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
#     ds    = FXVolDataset.build(pairs=[PAIR], days=400)
#     dates = business_dates(ds, PAIR)
#     snap  = MarketSnapshot.at(ds, PAIR, dates[-60])
#     fxc   = fx_calendar(PAIR)
#     exp1m = add_tenor(snap.date, '1M', fxc)
#
#     # --- 1. A request is pure intent; filling it is where the market enters
#     req = LegRequest(option_type='put', target_delta=-0.25, direction=-1,
#                      notional=10_000_000, tenor='1M', sleeve='skew', tag='wing_p')
#     pos = open_position(req, snap, exp1m)
#     print(pos)
#     print(f'  struck vol   {pos.entry_vol * 100:.3f}%  '
#           f'(ATM {pos.entry_atm_vol * 100:.3f}%, '
#           f'smile premium {(pos.entry_vol - pos.entry_atm_vol) * 100:+.3f} vp)')
#     print(f'  premium      {pos.entry_premium:>15,.2f} {snap.base}  '
#           f'(positive = received, we are short)')
#     assert pos.entry_premium > 0, 'short position must RECEIVE premium'
#     print('[OK] premium sign convention\n')
#
#     # --- 2. Signed value means differences are P&L with no sign flips
#     print(f'  value_base today  {pos.value_base(snap):>15,.2f}  '
#           f'(negative = liability, we are short)')
#     assert pos.value_base(snap) < 0
#     print('[OK] signed mark-to-market\n')
#
#     # --- 3. Greeks come out already signed and sized
#     g = pos.greeks(snap)
#     print('  ', g)
#     assert g.vega_1vp < 0 and g.theta_1d > 0
#     print('[OK] short 25d put: short vega, collects theta\n')
#
#     # --- 4. Resize returns the traded amount (what Phase 2 charges cost on)
#     traded = pos.resize(6_000_000)
#     print(f'  resize 10mm -> 6mm traded {traded:,.0f}  '
#           f'(negative = risk reduced)')
#     g2 = pos.greeks(snap)
#     print(f'  vega before {g.vega_1vp:,.0f} -> after {g2.vega_1vp:,.0f} '
#           f'(ratio {g2.vega_1vp / g.vega_1vp:.3f}, expect 0.600)')
#     assert abs(g2.vega_1vp / g.vega_1vp - 0.6) < 1e-9
#     print('[OK] risk is linear in notional after a resize\n')
#
#     # --- 5. Multi-expiry is now trivially expressible -- the thing the old
#     #        engine structurally could not do.
#     exp3m = add_tenor(snap.date, '3M', fxc)
#     near = open_position(LegRequest('put', -1, 10_000_000, '1M',
#                                     target_delta=-0.25, sleeve='skew'), snap, exp1m)
#     far  = open_position(LegRequest('put', +1, 10_000_000, '3M',
#                                     target_delta=-0.25, sleeve='skew'), snap, exp3m)
#     cal = near.greeks(snap) + far.greeks(snap)
#     print(f'  1M short 25d put + 3M long 25d put (a calendar):')
#     print('  ', cal)
#     print(f'  net vega {cal.vega_1vp:,.0f} -- positive: the 3M leg has more vega')
#     assert cal.vega_1vp > 0
#     print('[OK] calendar spreads work; the old engine could not express this')
