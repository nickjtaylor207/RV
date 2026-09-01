"""
MarketSnapshot - the point-in-time view of one pair's market, as of one date.

WHY THIS MODULE EXISTS
----------------------
Trap #29 in the old stack's guide reads: "Point-in-time discipline is yours.
Nothing checks that your signal was built without look-ahead."

That is a standing invitation to a look-ahead bug, and in a wing-selling
strategy a look-ahead bug is invisible until it is expensive. This module
removes the invitation by making the discipline STRUCTURAL:

    * A MarketSnapshot is constructed for a (pair, as_of) and exposes only
      market state at or before `as_of`.
    * Everything downstream -- pricing, greeks, features, signals, sizing --
      takes a snapshot, never the raw FXVolDataset.
    * Trailing-window accessors (`history`, `atm_history`) hard-truncate at
      `as_of`, so a realised-vol or realised-nu estimator physically cannot see
      tomorrow.

The rule for the rest of the build: if a function needs market data, it takes a
MarketSnapshot. If it takes an FXVolDataset, it is a bug waiting to happen.


THE SECOND JOB: THE STRIKE/VOL FIXED POINT
------------------------------------------
Solving a strike from a target delta needs a vol; the smile vol needs a strike.
The old engine resolved that circularity with a 5-iteration fixed point written
inline in backtest_MLeg.py. Here it lives in ONE place -- `solve_strike_and_vol`
-- so no caller can forget it, and so the iteration count and convergence
tolerance are a single tunable rather than a magic number buried in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.conventions import TENOR_DAYS
from core.option import FXOption, forward_price, strike_from_delta


# Iterations for the strike <-> smile-vol fixed point. The old engine used 5.
# Kept identical so Phase 1 reconciles exactly; revisit once reconciliation
# has passed (convergence is usually reached in 2-3).
STRIKE_SOLVE_ITERS = 5
STRIKE_SOLVE_TOL   = 1e-10


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Everything the engine may know about `pair` on `as_of`, and nothing more.

    Construct via `MarketSnapshot.at(dataset, pair, as_of)`. Cheap to build --
    it holds a reference to the dataset, it does not copy any frames.

    Every vol returned is DECIMAL (0.085 = 8.5%), matching the pricer.
    """
    dataset: object                # FXVolDataset (not typed, to avoid the import cycle)
    pair:    str
    as_of:   pd.Timestamp

    # Reproduce the OLD engine's half-step strike bootstrap. See
    # solve_strike_and_vol. Reconciliation only -- leave False for real work.
    legacy_strike_halfstep: bool = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def at(cls, dataset, pair: str, as_of,
           legacy_strike_halfstep: bool = False) -> 'MarketSnapshot':
        return cls(dataset=dataset, pair=pair, as_of=pd.Timestamp(as_of),
                   legacy_strike_halfstep=legacy_strike_halfstep)

    @property
    def date(self) -> date:
        """The as-of date as a plain datetime.date, for the pricer's day counts."""
        return self.as_of.date()

    @property
    def base(self) -> str:
        """Base (foreign) ccy -- the accounting currency for this pair."""
        return self.pair[:3]

    @property
    def quote(self) -> str:
        return self.pair[3:]

    # ------------------------------------------------------------------ #
    # Point-in-time state
    # ------------------------------------------------------------------ #
    @property
    def spot(self) -> float:
        return self.dataset.get_spot(self.pair, self.as_of)

    def rates(self, t_days: float) -> Tuple[float, float]:
        """
        (r_d, r_f) for a given remaining tenor in calendar days.

        Convention, unchanged from the old stack: for pair XXXYYY, r_d is YYY's
        rate and r_f is XXX's. So for USDJPY, r_d = JPY and r_f = USD.
        """
        return self.dataset.get_rates_for_tenor(self.pair, self.as_of, t_days)

    def atm_vol(self, t_days: float) -> float:
        """ATM vol at t_days remaining, weekend-weighted variance interpolation."""
        return self.dataset.get_atm_vol(self.pair, self.as_of, t_days)

    def forward(self, t_days: float) -> float:
        """
        Outright forward to the DELIVERY date of an option expiring t_days out.

        The differential accrues over spot->delivery, not today->expiry. Those
        agree to within a day in a normal month and come apart when a holiday
        cluster stretches either settlement lag -- see core.pricer.OptionTime.
        """
        r_d, r_f = self.rates(t_days)
        return self._probe(t_days).forward(self.spot, r_d, r_f, self.date)

    def _probe(self, t_days: float) -> FXOption:
        """
        A throwaway option expiring t_days out, used purely to get at this
        pair's settlement calendar and hence the correct forward window.
        Strike is irrelevant -- only the date arithmetic is used.
        """
        expiry = self.date + pd.Timedelta(days=int(round(t_days)))
        return FXOption(pair=self.pair, K=self.spot,
                        expiry=expiry.date() if hasattr(expiry, 'date') else expiry)

    def smile_vol(self, K: float, t_days: float) -> float:
        """
        SABR smile vol at a FIXED strike. Strikes are locked at trade entry, so
        there is no circularity here -- the circularity only exists when solving
        a strike FROM a delta (see solve_strike_and_vol).
        """
        r_d, r_f = self.rates(t_days)
        F = self.forward(t_days)
        return self.dataset.get_smile_vol(self.pair, self.as_of, t_days, K, F, r_f)

    def nu_rho(self, t_days: float) -> Tuple[float, float]:
        """
        Closed-form breakeven vol-of-vol and spot/vol correlation off the
        nearest tenor pillar's quoted ATM / RR25 / BF25.

            nu_BE  = 4.0 * sqrt(BF25 / (tau * sigma_ATM))
            rho_BE = 2.5 * (RR25 / sigma_ATM)

        Phase 1 uses these only to compute the breakeven (_be) P&L buckets.

        PHASE 5 WARNING: when these become the SIGNAL rather than a diagnostic,
        the constants 4.0 / 2.5 stop being harmless. Z-score BF25/(sigma*sqrt(tau))
        and RR25/sigma directly instead -- then the constants are a pure scale
        factor and cancel out of the z-score.
        """
        return self.dataset.get_smile_nu_rho(self.pair, self.as_of, t_days)

    # ------------------------------------------------------------------ #
    # THE STRIKE / SMILE-VOL FIXED POINT
    # ------------------------------------------------------------------ #
    def solve_strike_and_vol(self,
                             target_delta: float,
                             option_type:  str,
                             expiry:       date,
                             iters:        int = STRIKE_SOLVE_ITERS,
                             tol:          float = STRIKE_SOLVE_TOL) -> Tuple[float, float]:
        """
        Resolve the circularity  strike -> smile vol -> delta -> strike.

        Start from the ATM vol, solve the strike that gives target_delta under
        it, read the smile vol at that strike, re-solve, repeat. Converges in
        2-3 passes for G10; `iters` is capped for safety and matched to the old
        engine's 5 so Phase 1 reconciles exactly.

        Parameters
        ----------
        target_delta : SIGNED spot delta. +0.25 = 25d call, -0.25 = 25d put.
        option_type  : 'call' or 'put'
        expiry       : the option's expiry date

        Returns
        -------
        (K, sigma) : the solved strike and the smile vol AT that strike, which
                     is the vol the option is priced and risk-managed on.

        SELF-CONSISTENCY, AND THE LEGACY FLAG
        -------------------------------------
        The pair returned here satisfies  sigma == smile_vol(K)  exactly: the
        strike and the vol it is priced on are the same point on the surface.

        The OLD engine did NOT do this. Its loop was

            for _ in range(5):
                K_seed = solve(sigma); sigma = smile(K_seed)
            K = solve(sigma)                       # <- one extra half-step
            return K, sigma

        so the sigma it returned was the smile vol at the PREVIOUS iterate's
        strike, not at the K it returned. Off by half a step. Harmless at 25
        delta where the smile is shallow; visible in the wings where dsigma/dK
        is steep -- which is precisely where this strategy lives.

        `legacy_strike_halfstep=True` reproduces that behaviour exactly, so
        recon/reconcile.py can prove the remaining engine is identical. Do not
        turn it on for real work: it strikes options at a vol they are not
        actually worth.

        Raises whatever brentq raises if the strike falls outside the
        [0.85 S, 1.20 S] bracket -- deliberately, rather than returning a wrong
        strike. That will happen on deep wings in high-vol pairs; widen the
        bracket explicitly in core.option.strike_from_delta when it does.
        """
        t_days = max((expiry - self.date).days, 1e-6)
        S      = self.spot
        r_d, r_f = self.rates(t_days)

        def solve(sig: float) -> float:
            return strike_from_delta(self.pair, S, expiry, self.date,
                                     r_d, r_f, sig, target_delta, option_type)

        sigma = self.atm_vol(t_days)

        if self.legacy_strike_halfstep:
            for _ in range(iters):
                s_new = self.smile_vol(solve(sigma), t_days)
                done  = abs(s_new - sigma) < 1e-8
                sigma = s_new
                if done:
                    break
            return solve(sigma), sigma

        K = np.nan
        for _ in range(iters):
            K_new     = solve(sigma)
            sigma_new = self.smile_vol(K_new, t_days)
            converged = (K == K and abs(K_new - K) < tol * S)
            K, sigma  = K_new, sigma_new
            if converged:
                break
        return K, sigma

    def price_state(self, K: float, expiry: date) -> dict:
        """
        The full pricing state for one strike/expiry as of this snapshot:
        {S, sigma, r_d, r_f, t_days}. This is the bundle every reprice needs,
        fetched once so a multi-leg book does not re-hit the dataset per leg.
        """
        t_days = max((expiry - self.date).days, 1e-6)
        r_d, r_f = self.rates(t_days)
        return {
            'S':      self.spot,
            'sigma':  self.smile_vol(K, t_days),
            'r_d':    r_d,
            'r_f':    r_f,
            't_days': t_days,
        }

    # ------------------------------------------------------------------ #
    # Trailing history -- hard-truncated at as_of
    # ------------------------------------------------------------------ #
    def spot_history(self, lookback_days: int) -> pd.Series:
        """
        Spot over the trailing window, INCLUSIVE of as_of, exclusive of
        everything after. Feeds realised-vol and realised-rho estimators.
        """
        s = self.dataset.spot.loc[:self.as_of, self.pair].dropna()
        cutoff = self.as_of - pd.Timedelta(days=int(lookback_days))
        return s.loc[cutoff:]

    def quote_history(self, tenor: str, field: str,
                      lookback_days: int) -> pd.Series:
        """
        A quoted surface series at a fixed TENOR PILLAR over the trailing
        window, converted to decimal.

        field : 'ATM', 'RR25', 'BF25', 'RR10', 'BF10', ...  (see DELTA_POINTS)

        These are CONSTANT-MATURITY series, which is what the Phase-5
        realised-nu / realised-rho estimators want. A single option's own vol
        shortens every day and so mixes term-structure roll into the
        measurement; a pillar series does not.
        """
        col = (self.pair, tenor, field)
        vs  = self.dataset.vol_surface
        if col not in vs.columns:
            avail_t = sorted({c[1] for c in vs.columns if c[0] == self.pair})
            avail_f = sorted({c[2] for c in vs.columns if c[0] == self.pair})
            raise KeyError(f"No {field} column for {self.pair} {tenor}. "
                           f"Tenors: {avail_t}  Fields: {avail_f}")
        s = vs.loc[:self.as_of, col].dropna() / 100.0     # stored in percent
        cutoff = self.as_of - pd.Timedelta(days=int(lookback_days))
        return s.loc[cutoff:]

    def atm_history(self, tenor: str, lookback_days: int) -> pd.Series:
        """Constant-maturity ATM vol history. Phase 5 builds realised nu off this."""
        return self.quote_history(tenor, 'ATM', lookback_days)

    def __repr__(self) -> str:
        return f"MarketSnapshot({self.pair} @ {self.as_of.date()})"


# ====================================================================== #
# Calendar helper -- the trading dates the engine loops over
# ====================================================================== #
def business_dates(dataset, pair: str,
                   start=None, end=None) -> pd.DatetimeIndex:
    """
    The dates the engine will step through: every date on which this pair has a
    spot observation, optionally clipped to [start, end].

    Driving the loop off observed spot (rather than a synthetic calendar) means
    the loop can never ask for a date the data does not have, and weekends and
    holidays drop out naturally. Gaps are still gaps -- the engine must scale
    theta and carry by the ACTUAL day count between consecutive rows, not by 1.
    """
    idx = dataset.spot[pair].dropna().index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return pd.DatetimeIndex(idx)


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python market/snapshot.py
# Requires a live Bloomberg connection (the FXVolDataset build hits xbbg).
# ====================================================================== #
# if __name__ == '__main__':
#     import os as _os, sys as _sys
#     _sys.path.insert(0, _os.path.dirname(
#         _os.path.dirname(_os.path.abspath(__file__))))
#     from market.dataset import FXVolDataset
#     from core.calendar import add_tenor
#     from core.conventions import fx_calendar
#
#     PAIR = 'USDJPY'
#     ds   = FXVolDataset.build(pairs=[PAIR], days=400)
#
#     dates = business_dates(ds, PAIR)
#     print(f'{len(dates)} business dates, {dates[0].date()} -> {dates[-1].date()}\n')
#
#     asof = dates[-30]
#     snap = MarketSnapshot.at(ds, PAIR, asof)
#     print(snap)
#     print(f'  spot          {snap.spot:.4f}')
#     print(f'  1M ATM vol    {snap.atm_vol(30) * 100:.3f}%')
#     print(f'  1M forward    {snap.forward(30):.4f}')
#     print(f'  rates (d,f)   {snap.rates(30)}')
#     print(f'  1M nu, rho    {snap.nu_rho(30)}\n')
#
#     # --- 1. THE POINT-IN-TIME GUARANTEE. A snapshot must never see forward.
#     hist = snap.spot_history(90)
#     assert hist.index.max() <= snap.as_of, 'LOOK-AHEAD: history ran past as_of'
#     print(f'[OK] spot_history({90}d) ends {hist.index.max().date()} '
#           f'<= as_of {snap.as_of.date()}  ({len(hist)} obs)')
#     vh = snap.atm_history('1M', 90)
#     assert vh.index.max() <= snap.as_of
#     print(f'[OK] atm_history 1M ends {vh.index.max().date()}, '
#           f'last = {vh.iloc[-1] * 100:.3f}%\n')
#
#     # --- 2. The strike/vol fixed point, across the smile
#     fxc    = fx_calendar(PAIR)
#     expiry = add_tenor(snap.date, '1M', fxc)
#     print(f'1M expiry {expiry}  ({(expiry - snap.date).days} days)')
#     print('  target    strike      smile vol    vs ATM')
#     atm = snap.atm_vol((expiry - snap.date).days)
#     for tgt, typ in [(+0.10, 'call'), (+0.25, 'call'),
#                      (-0.25, 'put'), (-0.10, 'put')]:
#         K, sig = snap.solve_strike_and_vol(tgt, typ, expiry)
#         print(f'  {typ[0].upper()}{abs(tgt) * 100:>3.0f}d   {K:9.4f}   '
#               f'{sig * 100:8.3f}%   {(sig - atm) * 100:+6.3f}')
#     print()
#
#     # --- 3. The fixed point actually converged: re-reading the smile at the
#     #        solved strike must reproduce the vol it returned.
#     K, sig = snap.solve_strike_and_vol(-0.25, 'put', expiry)
#     recheck = snap.smile_vol(K, (expiry - snap.date).days)
#     print(f'[fixed point] returned {sig:.10f}, re-read {recheck:.10f}, '
#           f'diff {abs(sig - recheck):.2e}')
#     assert abs(sig - recheck) < 1e-8, 'fixed point did not converge'
#     print('[OK] strike/vol fixed point is self-consistent\n')
#
#     # --- 4. And the solved strike really does have the target delta
#     opt   = FXOption(pair=PAIR, K=K, expiry=expiry, option_type='put')
#     r_d, r_f = snap.rates((expiry - snap.date).days)
#     print(f'[OK] solved 25d put delta = '
#           f'{opt.spot_delta(snap.spot, sig, r_d, r_f, snap.date):+.4f}')
#
#     # --- 5. price_state is the one-shot bundle the book reprices from
#     print('\nprice_state:', {k: round(v, 6) for k, v in snap.price_state(K, expiry).items()})
