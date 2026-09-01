"""
GreekVector - the single, consistent risk unit for the whole stack.

WHY THIS MODULE EXISTS
----------------------
In the old Delta_Hedged stack `FXOption.greeks_foreign()` returned a dict with
FIVE different scaling conventions mixed together (delta premium-adjusted,
gamma 1%-scaled, vega x0.01/S, volga completely RAW, vanna PA but unscaled).
Each consumer then re-scaled by hand -- `* 0.01 * 0.01 / prev_spot` appears at
three separate call sites in backtest_MLeg.py. That is survivable when greeks
are a display item. It is NOT survivable when you size positions on volga.

Here the scaling is applied ONCE, at construction, and every field of a
GreekVector is in the same unit:

        BASE-CURRENCY P&L, position-level (notional and direction already
        applied), for one standardised move.

Standardised moves:
        spot   : +1% of spot
        vol    : +1 vol point (0.01 in decimal vol)
        time   : +1 calendar day

Because every field is money-for-a-standard-move, GreekVectors are additive
across legs, across tenors, across pairs and across sleeves without any further
thought. That additivity is the whole point -- it is what makes a book-level
risk budget and a greek-target sizer possible.


THE BASE-CURRENCY DERIVATIVES
-----------------------------
The engine accounts in BASE (foreign) currency, so the value of one unit of
notional is  V = P / S  (P = the domestic Garman-Kohlhagen price).

Differentiating V rather than P gives a clean rule:

    * derivatives w.r.t. SPOT pick up a premium-adjustment correction,
      because S sits in the denominator of V;
    * pure VOL and TIME derivatives just divide by S.

    dV/dS      =  D_pa / S                        (D_pa = premium-adj. delta)
    d2V/dS2    =  G_raw / S  -  2 * D_pa / S^2
    dV/dsig    =  vega_raw / S
    d2V/dsig2  =  volga_raw / S
    d2V/dSdsig =  vanna_pa / S                    (vanna_pa = vanna_raw - vega_raw/S)
    dV/dt      =  theta_raw / S

    Proof of the two non-obvious ones:
        dV/dS   = D/S - P/S^2 = (D - P/S)/S = D_pa/S
        d2V/dS2 = d/dS(D_pa/S) = (dD_pa/dS)/S - D_pa/S^2
                  with dD_pa/dS = G - D_pa/S
                = G/S - 2*D_pa/S^2

NOTE: `pricer.gamma_premium_adjusted` is Wystup's Gamma_pa, which is a
DIFFERENT quantity from d2V/dS2 above. Do not substitute one for the other.
This module uses the identity, not that function.


THE TWO VIEWS
-------------
1. P&L view (the canonical fields) -- money per standard move. Additive
   everywhere. This is what the sizer, the risk budget and the attribution use.

2. Trader view (`.as_trader_units()`) -- the conventional desk numbers (delta
   in notional, gamma as delta-per-1%, vega per vol point, vanna as
   delta-per-vol-point, volga as vega-change-per-vol-point). These are for
   READING, not for arithmetic; vanna and volga come out in different units
   from each other there, which is exactly the trap this module removes from
   the P&L view.

`delta_hedge` is carried separately from the P&L fields because it is a
different kind of object: a QUANTITY (base-ccy notional of spot to trade), not
a P&L. It is additive within a pair only -- summing spot hedge notionals across
pairs is meaningless, so aggregation across pairs deliberately blanks it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


# Standardised move sizes. Change these ONLY if you want to redefine the risk
# unit for the entire stack -- everything downstream reads them from here.
SPOT_MOVE = 0.01      # 1% of spot
VOL_MOVE  = 0.01      # 1 vol point, in decimal vol
TIME_MOVE = 1.0       # 1 calendar day


@dataclass(frozen=True)
class GreekVector:
    """
    Position-level greeks in base-ccy P&L per standard move.

    Every field already includes notional and direction, so a short position
    has negative vega and a long position positive. Summing two GreekVectors
    gives the netted risk of holding both.

    Fields
    ------
    spot_1pct      : P&L for a +1% spot move                    (1st order)
    gamma_1pct     : P&L for a +1% spot move                    (2nd order, incl. the 1/2)
    vega_1vp       : P&L for a +1 vol point move                (1st order)
    volga_1vp      : P&L for a +1 vol point move                (2nd order, incl. the 1/2)
    vanna_1pct_1vp : P&L for +1% spot AND +1 vol point together (cross term)
    theta_1d       : P&L for the passage of 1 calendar day

    delta_hedge    : base-ccy notional of SPOT to sell to flatten delta.
                     A quantity, not a P&L. Within-pair additive only.
    spot           : the spot the vector was struck at (carried for unit
                     conversion and to detect cross-pair aggregation).
    """

    spot_1pct:      float = 0.0
    gamma_1pct:     float = 0.0
    vega_1vp:       float = 0.0
    volga_1vp:      float = 0.0
    vanna_1pct_1vp: float = 0.0
    theta_1d:       float = 0.0

    delta_hedge:    float = 0.0
    spot:           float = np.nan

    PNL_FIELDS = ('spot_1pct', 'gamma_1pct', 'vega_1vp',
                  'volga_1vp', 'vanna_1pct_1vp', 'theta_1d')

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_partials(cls,
                      dV_dS:      float,
                      d2V_dS2:    float,
                      dV_dsig:    float,
                      d2V_dsig2:  float,
                      d2V_dSdsig: float,
                      dV_dt:      float,
                      spot:       float,
                      notional:   float = 1.0,
                      direction:  int   = 1) -> 'GreekVector':
        """
        Build from the six BASE-CCY partial derivatives of V = P/S per unit of
        notional. This is the ONLY place the standardisation happens.
        """
        q  = notional * direction
        ds = SPOT_MOVE * spot          # absolute spot move equivalent to +1%
        dv = VOL_MOVE                  # absolute vol move equivalent to +1 vol pt

        return cls(
            spot_1pct      = q * dV_dS * ds,
            gamma_1pct     = q * 0.5 * d2V_dS2 * ds * ds,
            vega_1vp       = q * dV_dsig * dv,
            volga_1vp      = q * 0.5 * d2V_dsig2 * dv * dv,
            vanna_1pct_1vp = q * d2V_dSdsig * ds * dv,
            theta_1d       = q * dV_dt * TIME_MOVE / 365.0,
            delta_hedge    = q * dV_dS * spot,      # == D_pa * notional * direction
            spot           = spot,
        )

    @classmethod
    def zero(cls, spot: float = np.nan) -> 'GreekVector':
        return cls(spot=spot)

    # ------------------------------------------------------------------ #
    # Arithmetic -- additive by construction
    # ------------------------------------------------------------------ #
    def __add__(self, other) -> 'GreekVector':
        if other is None or (isinstance(other, (int, float)) and other == 0):
            return self
        same_spot = (np.isnan(self.spot) or np.isnan(other.spot)
                     or abs(self.spot - other.spot) < 1e-12)
        keep_spot = self.spot if not np.isnan(self.spot) else other.spot
        return GreekVector(
            spot_1pct      = self.spot_1pct      + other.spot_1pct,
            gamma_1pct     = self.gamma_1pct     + other.gamma_1pct,
            vega_1vp       = self.vega_1vp       + other.vega_1vp,
            volga_1vp      = self.volga_1vp      + other.volga_1vp,
            vanna_1pct_1vp = self.vanna_1pct_1vp + other.vanna_1pct_1vp,
            theta_1d       = self.theta_1d       + other.theta_1d,
            # delta_hedge only means something within one pair. When the spots
            # differ we are aggregating across pairs, so blank it rather than
            # silently return a meaningless number.
            delta_hedge    = (self.delta_hedge + other.delta_hedge) if same_spot else np.nan,
            spot           = keep_spot if same_spot else np.nan,
        )

    def __radd__(self, other) -> 'GreekVector':
        return self.__add__(other)

    def __mul__(self, k: float) -> 'GreekVector':
        """Scale a position's risk (e.g. resizing). Spot is unchanged."""
        return GreekVector(
            spot_1pct      = self.spot_1pct      * k,
            gamma_1pct     = self.gamma_1pct     * k,
            vega_1vp       = self.vega_1vp       * k,
            volga_1vp      = self.volga_1vp      * k,
            vanna_1pct_1vp = self.vanna_1pct_1vp * k,
            theta_1d       = self.theta_1d       * k,
            delta_hedge    = self.delta_hedge    * k,
            spot           = self.spot,
        )

    __rmul__ = __mul__

    def __neg__(self) -> 'GreekVector':
        return self * -1.0

    def __sub__(self, other: 'GreekVector') -> 'GreekVector':
        return self + (-other)

    @staticmethod
    def total(vectors: Iterable['GreekVector']) -> 'GreekVector':
        out = GreekVector.zero()
        for v in vectors:
            out = out + v
        return out

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def as_dict(self) -> Dict[str, float]:
        """The P&L fields plus delta_hedge, for writing into a record row."""
        d = {f: getattr(self, f) for f in self.PNL_FIELDS}
        d['delta_hedge'] = self.delta_hedge
        return d

    def as_array(self, keys: Iterable[str] = None) -> np.ndarray:
        """
        The P&L fields as a plain numpy array in a fixed order. This is the
        form the Phase-3 sizer's least-squares solve will consume.
        """
        keys = tuple(keys) if keys is not None else self.PNL_FIELDS
        return np.array([getattr(self, k) for k in keys], dtype=float)

    def as_trader_units(self) -> Dict[str, float]:
        """
        Conventional desk numbers, for READING ONLY.

        delta      : base-ccy spot notional to sell to flatten (== delta_hedge)
        gamma_1pct : change in that delta notional per +1% spot move
        vega_1vp   : base-ccy P&L per +1 vol point      (same as the P&L field)
        vanna_1vp  : change in delta notional per +1 vol point
        volga_1vp  : change in vega_1vp per +1 vol point
        theta_1d   : base-ccy P&L per day               (same as the P&L field)

        Note gamma/vanna come out in NOTIONAL units here while vega/volga/theta
        come out in MONEY. That mixture is precisely why this view must never be
        summed into a risk budget -- use the P&L fields for that.
        """
        S = self.spot
        ok = (S == S)   # False when NaN
        return {
            'delta':      self.delta_hedge,
            'gamma_1pct': (2.0 * self.gamma_1pct / (SPOT_MOVE * S)) if ok else np.nan,
            'vega_1vp':   self.vega_1vp,
            'vanna_1vp':  (self.vanna_1pct_1vp / (SPOT_MOVE * S)) if ok else np.nan,
            'volga_1vp':  2.0 * self.volga_1vp,
            'theta_1d':   self.theta_1d,
        }

    def __repr__(self) -> str:
        return (f"GreekVector(spot1%={self.spot_1pct:,.0f} "
                f"gamma1%={self.gamma_1pct:,.0f} "
                f"vega={self.vega_1vp:,.0f} "
                f"volga={self.volga_1vp:,.0f} "
                f"vanna={self.vanna_1pct_1vp:,.0f} "
                f"theta={self.theta_1d:,.0f} "
                f"| hedge={self.delta_hedge:,.0f})")


# ====================================================================== #
# Taylor P&L from a GreekVector and a realised move
# ====================================================================== #
def taylor_pnl(g: GreekVector,
               dS: float,
               dsigma: float,
               dt_days: float,
               prev_spot: float) -> Dict[str, float]:
    """
    Decompose a realised market move into the standard P&L buckets.

    Because every greek is already money-per-standard-move, each bucket is just
    the greek times the move expressed in STANDARD UNITS, raised to the order of
    the term. No scaling constants appear anywhere in this function -- that is
    the payoff of doing the standardisation once in from_partials().

        u = (dS / prev_spot) / SPOT_MOVE     spot move in "1% units"
        w = dsigma / VOL_MOVE                vol move in "1 vol point units"

        theta = theta_1d       * dt_days
        gamma = gamma_1pct     * u^2
        vega  = vega_1vp       * w
        volga = volga_1vp      * w^2
        vanna = vanna_1pct_1vp * u * w

    Parameters
    ----------
    dS        : absolute spot change over the period
    dsigma    : absolute change in DECIMAL vol over the period
    dt_days   : calendar days elapsed (3 across a weekend, not 1)
    prev_spot : spot at the START of the period

    Returns base-ccy P&L by bucket, plus 'taylor_total'.
    """
    u = (dS / prev_spot) / SPOT_MOVE
    w = dsigma / VOL_MOVE

    out = {
        'theta_pnl': g.theta_1d       * dt_days,
        'gamma_pnl': g.gamma_1pct     * u * u,
        'vega_pnl':  g.vega_1vp       * w,
        'volga_pnl': g.volga_1vp      * w * w,
        'vanna_pnl': g.vanna_1pct_1vp * u * w,
    }
    out['taylor_total'] = sum(out.values())
    return out


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python core/greeks.py
# ====================================================================== #
# if __name__ == '__main__':

#     S = 150.0
#     PARTIALS = dict(dV_dS=0.004, d2V_dS2=0.0002, dV_dsig=0.35,
#                     d2V_dsig2=1.2, d2V_dSdsig=0.01, dV_dt=-0.05)

#     # --- 1. Additivity: a long and a short of the SAME leg must net to zero
#     long_leg  = GreekVector.from_partials(**PARTIALS, spot=S,
#                                           notional=10_000_000, direction=+1)
#     short_leg = GreekVector.from_partials(**PARTIALS, spot=S,
#                                           notional=10_000_000, direction=-1)
#     net = long_leg + short_leg
#     print('long :', long_leg)
#     print('short:', short_leg)
#     print('net  :', net)
#     assert abs(net.vega_1vp) < 1e-9 and abs(net.volga_1vp) < 1e-9
#     print('[OK] long + short of the same leg nets to zero\n')

#     # --- 2. Risk is linear in notional, and __mul__ agrees with rebuilding
#     dbl = GreekVector.from_partials(**PARTIALS, spot=S,
#                                     notional=20_000_000, direction=+1)
#     assert abs(dbl.volga_1vp - 2 * long_leg.volga_1vp) < 1e-9
#     assert abs((long_leg * 2.0).volga_1vp - dbl.volga_1vp) < 1e-9
#     print('[OK] risk scales linearly in notional\n')

#     # --- 3. Cross-pair aggregation blanks delta_hedge instead of lying
#     other_pair = GreekVector.from_partials(
#         dV_dS=0.5, d2V_dS2=0.3, dV_dsig=0.30, d2V_dsig2=1.0,
#         d2V_dSdsig=0.02, dV_dt=-0.04, spot=1.09,
#         notional=10_000_000, direction=-1)
#     book = long_leg + other_pair
#     print('cross-pair book:', book)
#     assert np.isnan(book.delta_hedge), 'delta_hedge must NOT sum across pairs'
#     assert not np.isnan(book.vega_1vp), 'P&L fields MUST still sum across pairs'
#     print('[OK] P&L fields aggregate across pairs; delta_hedge is blanked\n')

#     # --- 4. taylor_pnl reconciles against the raw partials computed by hand
#     dS, dsig, dt = 1.5, 0.004, 3.0        # +1% spot, +0.4 vol pt, over a weekend
#     bk   = taylor_pnl(long_leg, dS=dS, dsigma=dsig, dt_days=dt, prev_spot=S)
#     q    = 10_000_000
#     hand = {
#         'theta_pnl': q * PARTIALS['dV_dt'] / 365.0 * dt,
#         'gamma_pnl': q * 0.5 * PARTIALS['d2V_dS2'] * dS ** 2,
#         'vega_pnl':  q * PARTIALS['dV_dsig'] * dsig,
#         'volga_pnl': q * 0.5 * PARTIALS['d2V_dsig2'] * dsig ** 2,
#         'vanna_pnl': q * PARTIALS['d2V_dSdsig'] * dS * dsig,
#     }
#     for k, v in hand.items():
#         print(f'  {k:<10} engine={bk[k]:>14,.2f}   hand={v:>14,.2f}')
#         assert abs(bk[k] - v) < 1e-6, k
#     print('[OK] taylor_pnl == the raw-partial formulas, exactly\n')

#     # --- 5. Trader view, for eyeballing against a Bloomberg ticket
#     print('trader units:',
#           {k: round(v, 2) for k, v in long_leg.as_trader_units().items()})











