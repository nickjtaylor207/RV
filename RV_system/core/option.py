"""
FXOption - a single European FX vanilla, plus the strike solvers.

Ported from Delta_Hedged/option.py. The pricing and the strike root-finder are
unchanged. What changed is the greek interface:

    OLD:  greeks_foreign() -> dict with five different scaling conventions,
                              which every consumer then re-scaled by hand.

    NEW:  base_ccy_partials() -> the six raw partials of V = P/S, and
          greeks()           -> a GreekVector, standardised once (core/greeks.py).

`greeks_foreign()` is deliberately NOT ported. If you find yourself wanting it,
you want `.greeks(...).as_trader_units()` instead.

All rates and vols are passed at call time, never stored, so one FXOption
object can be repriced freely at any point in a backtest loop. The only things
fixed at construction are the contract terms: pair, strike, expiry, type.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict

import numpy as np
from scipy.optimize import brentq

from core.pricer import (bs_price, delta, delta_premium_adjusted,
                         gamma, vega, volga, vanna, theta,
                         OptionTime, fwd_price, df_domestic)
from core.greeks import GreekVector
from core.conventions import fx_calendar
from core.calendar import spot_from_horizon, preceding_business_day


# Floor for any of the three time windows, so the closed forms stay finite on
# and past the expiry date. Matches the old time_to_expiry floor.
_MIN_TAU = 1e-6


@dataclass
class FXOption:
    """
    A European FX vanilla.

    Attributes
    ----------
    pair        : currency pair string e.g. 'EURUSD' (base = pair[:3])
    K           : strike
    expiry      : fixed expiry date
    option_type : 'call' or 'put'

    S0, r_d, r_f, sigma0 are inception snapshots kept for reference/reporting
    only -- nothing in the pricing path reads them.
    """
    pair:        str
    K:           float
    expiry:      date
    option_type: str = 'call'

    S0:          float = np.nan
    r_d:         float = np.nan
    r_f:         float = np.nan
    sigma0:      float = np.nan

    # Derived, cached, not part of the contract terms.
    _delivery: date = field(default=None, repr=False, compare=False)
    _tb_cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Time
    # ------------------------------------------------------------------ #
    @property
    def delivery(self) -> date:
        """
        The date the exercised option actually settles: the spot date OF THE
        EXPIRY, on this pair's settlement calendar. Fixed at construction.
        """
        if self._delivery is None:
            self._delivery = spot_from_horizon(self.expiry, fx_calendar(self.pair))
        return self._delivery

    def time_basis(self, current_date: date) -> OptionTime:
        """
        The option's three time windows as of `current_date`, in years ACT/365.

            var  = current_date -> expiry     (variance accumulates to the fixing)
            fwd  = spot date    -> delivery   (the rate differential accrues here)
            disc = current_date -> delivery   (the payoff is received here)

        These are equal only if the current-date->spot lag happens to match the
        expiry->delivery lag in CALENDAR days. Usually they do to within a day;
        a holiday cluster at either end pulls them apart. Cached per date --
        strike_from_delta's root-find reprices this option dozens of times on
        one date, and each call would otherwise walk the settlement calendar.
        """
        tb = self._tb_cache.get(current_date)
        if tb is not None:
            return tb
        fxc     = fx_calendar(self.pair)
        spot_dt = spot_from_horizon(
            preceding_business_day(current_date, fxc.cal_trade), fxc)
        tb = OptionTime(
            var  = max((self.expiry   - current_date).days / 365.0, _MIN_TAU),
            fwd  = max((self.delivery - spot_dt).days      / 365.0, _MIN_TAU),
            disc = max((self.delivery - current_date).days / 365.0, _MIN_TAU),
        )
        self._tb_cache[current_date] = tb
        return tb

    def time_to_expiry(self, current_date: date) -> float:
        """
        Remaining VARIANCE time in years, ACT/365, floored at a small epsilon.

        The floor (rather than allowing 0 or negative) keeps the closed forms
        finite on and past the expiry date. The engine must special-case actual
        expiry settlement rather than relying on this -- see book/attribution.py.

        This is only one of the option's three clocks; pricing uses
        time_basis(). Kept because plenty of callers legitimately want just
        "how long until it fixes".
        """
        return self.time_basis(current_date).var

    def is_expired(self, current_date: date) -> bool:
        return (self.expiry - current_date).days <= 0

    # ------------------------------------------------------------------ #
    # Value
    # ------------------------------------------------------------------ #
    def forward(self, S, r_d, r_f, current_date) -> float:
        """Outright forward to this option's DELIVERY date."""
        return fwd_price(S, self.time_basis(current_date), r_d, r_f)

    def price_domestic(self, S, sigma, r_d, r_f, current_date) -> float:
        """Garman-Kohlhagen present value, per unit notional, in QUOTE ccy."""
        T = self.time_basis(current_date)
        return bs_price(S, self.K, T, r_d, r_f, sigma, self.option_type)

    def value_base(self, S, sigma, r_d, r_f, current_date) -> float:
        """
        V = P / S -- the value of ONE UNIT of base-ccy notional, expressed in
        base ccy. This is the quantity the whole stack accounts in, and the
        quantity every partial in base_ccy_partials() differentiates.
        """
        return self.price_domestic(S, sigma, r_d, r_f, current_date) / S

    def intrinsic_base(self, S: float) -> float:
        """Settlement value at expiry, per unit notional, in base ccy."""
        pay = max(S - self.K, 0.0) if self.option_type == 'call' else max(self.K - S, 0.0)
        return pay / S

    # ------------------------------------------------------------------ #
    # Deltas (raw, for strike solving and smile lookups)
    # ------------------------------------------------------------------ #
    def spot_delta(self, S, sigma, r_d, r_f, current_date) -> float:
        """Unadjusted spot delta. Used to SOLVE strikes from a target delta."""
        T = self.time_basis(current_date)
        return delta(S, self.K, T, r_d, r_f, sigma, self.option_type)

    def call_delta_equivalent(self, S, sigma, r_d, r_f, current_date) -> float:
        """
        This strike's call-delta equivalent, for indexing the smile grid.

        Put-call parity on the SPOT delta gives  d_c - d_p = DF * F / S, which
        collapses to the familiar exp(-r_f*T) only when the three clocks agree.
        """
        d = self.spot_delta(S, sigma, r_d, r_f, current_date)
        if self.option_type == 'put':
            T = self.time_basis(current_date)
            d += df_domestic(T, r_d) * fwd_price(S, T, r_d, r_f) / S
        return d

    # ------------------------------------------------------------------ #
    # THE GREEK INTERFACE
    # ------------------------------------------------------------------ #
    def base_ccy_partials(self, S, sigma, r_d, r_f, current_date) -> Dict[str, float]:
        """
        The six partial derivatives of V = P/S, per unit of base-ccy notional.

        Spot derivatives carry the premium adjustment (S is in the denominator
        of V); pure vol and time derivatives are just the raw greek over S.
        See the module docstring of core/greeks.py for the derivations.

            dV/dS      = D_pa / S
            d2V/dS2    = G_raw / S - 2 * D_pa / S^2
            dV/dsig    = vega_raw / S
            d2V/dsig2  = volga_raw / S
            d2V/dSdsig = (vanna_raw - vega_raw / S) / S
            dV/dt      = theta_raw / S

        Returned unsigned and unsized -- direction and notional are applied by
        GreekVector.from_partials.

        Every identity above is a consequence of V = P/S alone, so none of them
        change when P is priced on three clocks rather than one -- only the
        raw greeks feeding them do.
        """
        T = self.time_basis(current_date)
        k = (S, self.K, T, r_d, r_f, sigma)

        D_pa  = delta_premium_adjusted(*k, self.option_type)
        G_raw = gamma(*k)
        v_raw = vega(*k)
        w_raw = volga(*k)
        x_raw = vanna(*k)
        t_raw = theta(*k, self.option_type)

        return {
            'dV_dS':      D_pa / S,
            'd2V_dS2':    G_raw / S - 2.0 * D_pa / (S * S),
            'dV_dsig':    v_raw / S,
            'd2V_dsig2':  w_raw / S,
            'd2V_dSdsig': (x_raw - v_raw / S) / S,
            'dV_dt':      t_raw / S,
        }

    def greeks(self, S, sigma, r_d, r_f, current_date,
               notional: float = 1.0, direction: int = 1) -> GreekVector:
        """
        Position-level risk as a GreekVector: base-ccy P&L per standard move,
        with notional and direction already applied.

        This is the ONLY greek accessor the engine should call.
        """
        p = self.base_ccy_partials(S, sigma, r_d, r_f, current_date)
        return GreekVector.from_partials(**p, spot=S,
                                         notional=notional, direction=direction)

    def __repr__(self) -> str:
        return (f"FXOption({self.pair} {self.option_type} "
                f"K={self.K:.4f} exp={self.expiry})")


# ====================================================================== #
# Forward and strike solvers
# ====================================================================== #
def forward_price(S: float, r_d: float, r_f: float, T: float) -> float:
    """Garman-Kohlhagen forward:  F = S * exp((r_d - r_f) * T)."""
    return S * np.exp((r_d - r_f) * T)


def atm_forward_strike(S: float, r_d: float, r_f: float,
                       expiry: date, today: date,
                       pair: str = None) -> float:
    """
    The ATM-FORWARD strike. This, not ATM-spot (K = S), is where a straddle is
    delta-neutral at inception, and is the correct ATM reference whenever the
    two legs carry meaningfully different rates.

    Pass `pair` to accrue the differential over the true spot->delivery window
    (the market forward). Without it this falls back to today->expiry, which is
    the same thing to within a day or so in a normal month but drifts when a
    holiday cluster stretches either settlement lag.
    """
    if pair is not None:
        probe = FXOption(pair=pair, K=S, expiry=expiry)
        return probe.forward(S, r_d, r_f, today)
    T = max((expiry - today).days / 365.0, 1e-6)
    return forward_price(S, r_d, r_f, T)


def strike_from_delta(pair:         str,
                      S:            float,
                      expiry:       date,
                      today:        date,
                      r_d:          float,
                      r_f:          float,
                      sigma:        float,
                      target_delta: float,
                      option_type:  str = 'call',
                      lo_mult:      float = 0.85,
                      hi_mult:      float = 1.20) -> float:
    """
    Solve for the strike whose UNADJUSTED spot delta equals target_delta under
    the supplied vol.

    target_delta is SIGNED:  +0.25 = 25d call,  -0.25 = 25d put,  ATM ~ +-0.50.

    NOTE the vol is an input, not solved for. Because the smile vol itself
    depends on the strike, the caller must iterate: guess vol -> solve K ->
    look up smile vol at K -> repeat. market/snapshot.py does that fixed point
    in one place so no caller has to remember to.

    The [0.85 * S, 1.20 * S] bracket is inherited from the old stack. It is wide
    enough for G10 out to ~1Y but WILL fail on a 5-delta wing in a high-vol EM
    pair or a long tenor -- brentq raises rather than returning a wrong strike,
    which is the behaviour we want. Widen the multipliers explicitly if needed.
    """
    def delta_error(K: float) -> float:
        probe = FXOption(pair=pair, K=K, expiry=expiry, option_type=option_type)
        return probe.spot_delta(S, sigma, r_d, r_f, today) - target_delta

    return brentq(delta_error, S * lo_mult, S * hi_mult)


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python core/option.py
# ====================================================================== #
# if __name__ == '__main__':
#     import os as _os, sys as _sys
#     _sys.path.insert(0, _os.path.dirname(
#         _os.path.dirname(_os.path.abspath(__file__))))
#     from datetime import date, timedelta

#     PAIR  = 'USDJPY'
#     TODAY = date(2025, 1, 15)
#     EXP   = TODAY + timedelta(days=30)
#     S, RD, RF, SIG = 150.0, 0.008, 0.045, 0.10       # r_d = JPY, r_f = USD
#     N = 10_000_000

#     # --- 1. Strike solving round-trips
#     for tgt, typ in [(+0.25, 'call'), (-0.25, 'put'), (-0.10, 'put')]:
#         K = strike_from_delta(PAIR, S, EXP, TODAY, RD, RF, SIG, tgt, typ)
#         opt = FXOption(pair=PAIR, K=K, expiry=EXP, option_type=typ)
#         back = opt.spot_delta(S, SIG, RD, RF, TODAY)
#         print(f'  {typ:<4} target {tgt:+.2f} -> K={K:8.4f} -> delta {back:+.4f}')
#         assert abs(back - tgt) < 1e-8
#     print('[OK] strike_from_delta inverts spot_delta\n')

#     # --- 2. The partials match numerical differentiation of value_base.
#     #     This is THE test of core/greeks.py's derivation. If it passes, the
#     #     base-ccy premium adjustment is right.
#     K25 = strike_from_delta(PAIR, S, EXP, TODAY, RD, RF, SIG, -0.25, 'put')
#     opt = FXOption(pair=PAIR, K=K25, expiry=EXP, option_type='put')
#     p   = opt.base_ccy_partials(S, SIG, RD, RF, TODAY)
#     V   = lambda s, v: opt.value_base(s, v, RD, RF, TODAY)

#     hS, hV = 1e-4 * S, 1e-5
#     num = {
#         'dV_dS':      (V(S + hS, SIG) - V(S - hS, SIG)) / (2 * hS),
#         'd2V_dS2':    (V(S + hS, SIG) - 2 * V(S, SIG) + V(S - hS, SIG)) / hS ** 2,
#         'dV_dsig':    (V(S, SIG + hV) - V(S, SIG - hV)) / (2 * hV),
#         'd2V_dsig2':  (V(S, SIG + hV) - 2 * V(S, SIG) + V(S, SIG - hV)) / hV ** 2,
#         'd2V_dSdsig': (V(S + hS, SIG + hV) - V(S + hS, SIG - hV)
#                        - V(S - hS, SIG + hV) + V(S - hS, SIG - hV)) / (4 * hS * hV),
#     }
#     print('  partial        analytic         numeric      rel.err')
#     for k, nv in num.items():
#         av  = p[k]
#         rel = abs(av - nv) / max(abs(nv), 1e-12)
#         print(f'  {k:<12} {av:>14.8f}  {nv:>14.8f}   {rel:.2e}')
#         assert rel < 1e-4, k
#     print('[OK] analytic base-ccy partials == numerical derivatives\n')

#     # --- 3. A GreekVector off a real option, both views
#     g = opt.greeks(S, SIG, RD, RF, TODAY, notional=N, direction=-1)   # SHORT 25d put
#     print('short 25d put, 10mm USD notional:')
#     print('  ', g)
#     print('   trader:', {k: round(v, 1) for k, v in g.as_trader_units().items()})
#     assert g.vega_1vp  < 0, 'short option must be short vega'
#     assert g.volga_1vp < 0, 'short option must be short volga'
#     assert g.theta_1d  > 0, 'short option must collect theta'
#     print('[OK] signs are right for a short position\n')

#     # --- 4. A vega-neutral-ish fly is short volga but flat vega:
#     #        the atom of the convexity sleeve.
#     Kc = strike_from_delta(PAIR, S, EXP, TODAY, RD, RF, SIG, +0.25, 'call')
#     Kp = strike_from_delta(PAIR, S, EXP, TODAY, RD, RF, SIG, -0.25, 'put')
#     Ka = atm_forward_strike(S, RD, RF, EXP, TODAY)
#     wing_c = FXOption(pair=PAIR, K=Kc, expiry=EXP, option_type='call')
#     wing_p = FXOption(pair=PAIR, K=Kp, expiry=EXP, option_type='put')
#     atm_c  = FXOption(pair=PAIR, K=Ka, expiry=EXP, option_type='call')
#     atm_p  = FXOption(pair=PAIR, K=Ka, expiry=EXP, option_type='put')

#     wing_v = (wing_c.greeks(S, SIG, RD, RF, TODAY, N, +1).vega_1vp
#               + wing_p.greeks(S, SIG, RD, RF, TODAY, N, +1).vega_1vp)
#     strad_v = (atm_c.greeks(S, SIG, RD, RF, TODAY, 1.0, +1).vega_1vp
#                + atm_p.greeks(S, SIG, RD, RF, TODAY, 1.0, +1).vega_1vp)
#     n_strad = wing_v / strad_v          # solve straddle notional for zero net vega

#     fly = (wing_c.greeks(S, SIG, RD, RF, TODAY, N, +1)
#            + wing_p.greeks(S, SIG, RD, RF, TODAY, N, +1)
#            + atm_c.greeks(S, SIG, RD, RF, TODAY, n_strad, -1)
#            + atm_p.greeks(S, SIG, RD, RF, TODAY, n_strad, -1))
#     print(f'vega-neutral fly (straddle notional solved = {n_strad:,.0f}):')
#     print('  ', fly)
#     assert abs(fly.vega_1vp) < 1e-6, 'fly should be vega neutral by construction'
#     assert fly.volga_1vp > 0, 'long wings / short body = LONG volga'
#     print('[OK] vega-neutral fly isolates volga\n')
#     print('    -> flip direction to be SHORT volga: that is the convexity sleeve.')
