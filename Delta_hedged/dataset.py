import warnings
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import QuantLib as ql
 
from data import (
    TENOR_DAYS, FWD_YIELD_TENORS, DELTA_POINTS,
    pull_dailyCCY_close, pull_fx_vol_surface,
    pull_sofr, pull_fwd_implied_yields, get_spot_days, fx_calendar
)
from vol_surface import (
    build_vol_grid, interpolate_vol_for_delta, interpolate_atm_vol,
    interpolate_spread_sqrt_t, get_sabr_vol_at_K
)
from trading_calendar import preceding_business_day, add_tenor


# (pairs, days) -> the built dataset. See FXVolDataset.build for why this exists.
_DATASET_CACHE: dict = {}


def clear_dataset_cache() -> None:
    """Drop every cached dataset (after a data refresh, or to free memory)."""
    _DATASET_CACHE.clear()


def dataset_cache_info() -> dict:
    """{'n': entries, 'keys': [(pairs, days), ...]} — what is currently held."""
    return {'n': len(_DATASET_CACHE), 'keys': list(_DATASET_CACHE)}


class FXVolDataset:
    """
    Single container for spot, vol surface, SOFR, and forward implied yields.
 
    Rate convention:
        USD leg  — flat SOFR for all tenors
        Non-USD  — forward implied yield interpolated across tenor structure
 
    Vol convention:
        All Bloomberg vol data stored in raw percent (e.g. 8.5).
        Converted to decimal at point of use in get_atm_vol / get_smile_vol.
    """

    # Ceiling for the closed-form nu_BE (vol-of-vol) returned by
    # get_smile_nu_rho — see that method's docstring for why this clip exists.
    NU_BE_MAX = 5.0

    # Shortest tenor pillar (real calendar days) get_smile_nu_rho will
    # evaluate the closed-form nu_BE/rho_BE on — excludes 'ON' (1 day), where
    # tau -> 0 blows up nu_BE = c*sqrt(Fly/(tau*sigma)). See get_smile_nu_rho's
    # docstring.
    MIN_PILLAR_DAYS = 7

    # Ravagli (2024), "Harvesting the FX skew premium" — fitted constants for
    # the closed-form nu_BE/rho_BE, validated on G10 pairs. Re-check before
    # trusting on EM/LatAm crosses.
    NU_BE_C  = 4.0
    RHO_BE_D = 2.5

    def __init__(self,
                 spot:        pd.DataFrame,
                 vol_surface: pd.DataFrame,
                 sofr:        pd.Series,
                 fwd_yields:  pd.DataFrame):
        

        self.spot        = spot.copy()
        self.vol_surface = vol_surface.copy()
        self.sofr        = sofr.copy()
        self.fwd_yields  = fwd_yields.copy()

        self.spot.index = pd.to_datetime(self.spot.index)
        self.vol_surface.index = pd.to_datetime(self.vol_surface.index)
        self.sofr.index = pd.to_datetime(self.sofr.index)
        self.fwd_yields.index = pd.to_datetime(self.fwd_yields.index)

        self.spot = self.spot.sort_index()
        self.vol_surface = self.vol_surface.sort_index()
        self.sofr = self.sofr.sort_index()
        self.fwd_yields = self.fwd_yields.sort_index()

    @classmethod
    def build(cls, pairs: List[str], days: int) -> 'FXVolDataset':
        """
        Pull all required data from Bloomberg and return a dataset.

        MEMOIZED on (pairs, days) for the life of the process. run_signal_backtest
        builds one of these per call, so a grid/sweep re-pulls the SAME data once
        per cell: a 4-column signal sweep on one pair asks for four identical
        pulls, because `dataset_days` is derived from the signal INDEX (which
        gating doesn't change), not from which days the signal fires.

        Two reasons beyond speed:
          * four pulls seconds apart could return slightly different data, so the
            columns of one table would not be strictly comparable;
          * the returned instance carries its own per-call memoized work (smile
            fits etc), which is now shared across cells too.

        The INSTANCE is shared, not copied — the same sharing that already happens
        across every trade within one backtest. Nothing downstream mutates it.
        Call clear_dataset_cache() after a data refresh, or to reclaim the memory.
        """
        key = (tuple(sorted(pairs)), int(days))
        if key not in _DATASET_CACHE:
            _DATASET_CACHE[key] = cls._build_uncached(pairs, days)
        return _DATASET_CACHE[key]

    @classmethod
    def _build_uncached(cls, pairs: List[str], days: int) -> 'FXVolDataset':
        """The actual pull. Call build() instead — see its caching note."""
        ccys = list(
            ({p[:3] for p in pairs} | {p[3:] for p in pairs}) - {'USD'}
        )
        print("Pulling spot data...")
        spot = pull_dailyCCY_close(pairs, days)
        print("Pulling vol surface...")
        vol  = pull_fx_vol_surface(pairs, days)
        print("Pulling SOFR...")
        sofr = pull_sofr(days)
        print("Pulling forward implied yields...")
        fwd_yields = pull_fwd_implied_yields(ccys, days)
        return cls(spot, vol, sofr, fwd_yields)
 
    # -------------------------------------------------------------------------
    # Pillar day helper
    # -------------------------------------------------------------------------

    def _compute_pillar_days(self, pair: str, as_of) -> dict:
        """
        Compute the actual calendar days from as_of to the real expiry for each
        tenor pillar, using the FX spot-date convention (T+1 CAD, T+2 others).
        Replaces the static TENOR_DAYS integers as interpolation x-axis nodes so
        that a '1M' pillar in February (28 days) is placed correctly vs. March (31).
        """
        as_of_date = pd.Timestamp(as_of).date()
        fxc        = fx_calendar(pair)
        entry      = preceding_business_day(as_of_date, fxc.cal_trade)
        return {
            tenor: (add_tenor(entry, tenor, fxc) - as_of_date).days
            for tenor in TENOR_DAYS
        }

    # -------------------------------------------------------------------------
    # Spot
    # -------------------------------------------------------------------------

    def get_spot(self, pair: str, as_of: datetime) -> float:
        as_of = pd.Timestamp(as_of)
        return float(self.spot.loc[:as_of, pair].dropna().iloc[-1])
 
    # -------------------------------------------------------------------------
    # Rates
    # -------------------------------------------------------------------------
 
    def get_rates_for_tenor(self, pair: str, as_of: datetime,
                             t_remaining_days: float) -> Tuple[float, float]:
        """
        Returns (r_d, r_f) for the given pair and remaining tenor.
 
        USD leg  : flat SOFR (no term structure)
        Non-USD  : forward implied yield interpolated across tenors
 
        Convention (standard FX):
            EURUSD — base=EUR (r_f), quote=USD (r_d)
            USDJPY — base=USD (r_f=SOFR), quote=JPY (r_d)
        """
        ccy_base  = pair[:3]
        ccy_quote = pair[3:]
 
        if ccy_quote == 'USD':
            r_d = float(self.sofr.loc[:as_of].iloc[-1])
        else:
            r_d = self._interpolate_fwd_yield(ccy_quote, as_of, t_remaining_days)
 
        if ccy_base == 'USD':
            r_f = float(self.sofr.loc[:as_of].iloc[-1])
        else:
            r_f = self._interpolate_fwd_yield(ccy_base, as_of, t_remaining_days)
 
        return r_d, r_f
 
    def _interpolate_fwd_yield(self, ccy: str, as_of: datetime,
                                t_remaining_days: float) -> float:
        """
        Interpolate forward implied yield for a single currency across tenors.

        Uses ql.DiscountCurve which interpolates log-linearly on discount
        factors (linear on r*T). This is theoretically correct: it implies
        piecewise-constant forward rates between pillars and guarantees no
        negative implied forwards. The old np.interp (linear on r) could
        produce rate errors of 2-10 bps at intermediate tenors on steep curves.
        """
        days_list  = []
        rates_list = []

        for tenor in FWD_YIELD_TENORS:
            days = TENOR_DAYS[tenor]
            try:
                val = self.fwd_yields.loc[:as_of, (ccy, tenor)].iloc[-1]
                if pd.notna(val):
                    days_list.append(days)
                    rates_list.append(float(val))
            except KeyError:
                continue

        if not days_list:
            raise ValueError(
                f"No forward yield data found for {ccy} as of {as_of}"
            )

        if t_remaining_days <= days_list[0]:
            return rates_list[0]
        if t_remaining_days >= days_list[-1]:
            return rates_list[-1]

        as_of_date = pd.Timestamp(as_of).date()
        today_ql   = ql.Date(as_of_date.day, as_of_date.month, as_of_date.year)
        dc         = ql.Actual365Fixed()

        # DiscountCurve requires today as the first node with DF=1.0
        ql_dates = [today_ql] + [today_ql + ql.Period(int(d), ql.Days) for d in days_list]
        dfs      = [1.0]      + [np.exp(-r * d / 365.0) for r, d in zip(rates_list, days_list)]

        curve = ql.DiscountCurve(ql_dates, dfs, dc)
        curve.enableExtrapolation()

        target = today_ql + ql.Period(int(round(t_remaining_days)), ql.Days)
        return curve.zeroRate(target, dc, ql.Continuous).rate()
 
    # -------------------------------------------------------------------------
    # Vol — ATM only
    # -------------------------------------------------------------------------
 
    def get_atm_vol(self, pair: str, as_of: datetime,
                    t_remaining_days: float,
                    pillar_days: dict = None) -> float:
        """
        ATM vol interpolated across tenors using weekend-weighted variance.
        Weekends receive a variance weight of 0.15; business-day weights are
        calibrated upward to preserve total pillar variance exactly.
        Returns decimal vol (e.g. 0.085 for 8.5%).

        pillar_days : real calendar-day node positions per tenor, computed from
                      actual expiry dates. Computed automatically from as_of if
                      not provided (the default and recommended path).
        """
        as_of = pd.Timestamp(as_of)
        if pillar_days is None:
            pillar_days = self._compute_pillar_days(pair, as_of)
        atm_cols = [(pair, t, 'ATM') for t in TENOR_DAYS
                    if (pair, t, 'ATM') in self.vol_surface.columns]
        row      = self.vol_surface.loc[:as_of, atm_cols].iloc[-1]

        vol_row = pd.Series({
            tenor: float(row[(pair, tenor, 'ATM')]) / 100.0
            for tenor in TENOR_DAYS
            if (pair, tenor, 'ATM') in row.index
            and pd.notna(row[(pair, tenor, 'ATM')])
        })
        return interpolate_atm_vol(vol_row, t_remaining_days, as_of.date(), pillar_days, pair=pair)
 
    # -------------------------------------------------------------------------
    # Vol — Full smile
    # -------------------------------------------------------------------------

    def get_smile_vol(self, pair: str, as_of: datetime,
                      t_remaining_days: float,
                      K: float,
                      F: float,
                      r_f: float) -> float:
        """
        Vol at a fixed strike K using SABR (arbitrage-free, no delta iteration).

        Architecture (unchanged from the delta-space version):
          1. ATM level  — weekend-weighted variance across tenor pillars (Wystup §1.3.4)
          2. Vol spread — SABR vol at K minus pillar ATM vol, interpolated in
                          sqrt(t) space between the two surrounding pillars (Wystup §1.3.8)
          Final vol = ATM(tr) + spread(tr)

        What changed vs the old delta-space version:
          - Old: spread_at() called interpolate_vol_for_delta(grid, option_delta)
                 option_delta depended on sigma -> circular -> 20-iteration loop
          - New: spread_at() calls get_sabr_vol_at_K(grid, K, F, T_pil, r_f)
                 K is fixed at trade entry -> no circularity -> single call

        Parameters
        ----------
        K   : fixed strike (locked in at trade entry, never changes in the loop)
        F   : forward  S * exp((r_d - r_f) * T)  for the target tenor
        r_f : foreign (base) rate, decimal — used for delta->strike inversion in SABR

        Falls back to ATM vol if smile data is unavailable.
        Returns decimal vol.
        """
        as_of_ts    = pd.Timestamp(as_of)
        tr          = int(round(t_remaining_days))

        # Compute real pillar day counts once and share across both interpolations
        pillar_days = self._compute_pillar_days(pair, as_of_ts)

        # Step 1 — ATM vol at target tenor (weekend-weighted variance, unchanged)
        sigma_atm = self.get_atm_vol(pair, as_of_ts, t_remaining_days, pillar_days)

        # Step 2 — Build smile grids at every available tenor pillar
        smile_row   = self.vol_surface.loc[:as_of_ts].iloc[-1]
        pillar_data = {}   # tenor_str -> (real_days, atm_vol_decimal, vol_grid)

        for tenor_str in TENOR_DAYS:
            real_days = pillar_days[tenor_str]
            atm_val   = smile_row.get((pair, tenor_str, 'ATM'))
            if atm_val is None or pd.isna(atm_val):
                continue
            atm_vol = float(atm_val) / 100.0

            rr, bf = {}, {}
            for d in DELTA_POINTS:
                rr_val = smile_row.get((pair, tenor_str, f'RR{d}'))
                bf_val = smile_row.get((pair, tenor_str, f'BF{d}'))
                if (rr_val is not None and bf_val is not None
                        and not pd.isna(rr_val) and not pd.isna(bf_val)):
                    rr[d] = float(rr_val) / 100.0
                    bf[d] = float(bf_val) / 100.0

            if not rr:
                continue   # no RR/BF data at this tenor

            pillar_data[tenor_str] = (real_days, atm_vol, build_vol_grid(atm_vol, rr, bf))

        if not pillar_data:
            warnings.warn(
                f"Smile data unavailable for {pair} on {as_of_ts}, using ATM vol."
            )
            return sigma_atm

        # Sort available smile pillars by tenor length
        sorted_pillars = sorted(pillar_data.items(), key=lambda x: x[1][0])
        days_list      = [v[0] for _, v in sorted_pillars]

        def spread_at(tenor_str: str) -> float:
            """SABR vol at K minus that pillar's ATM vol."""
            real_days, atm_vol, grid = pillar_data[tenor_str]
            T_pil    = max(real_days, 1) / 365.0
            # Use target-tenor F as an approximation for the pillar forward.
            # Error partially cancels in the spread (both SABR vol and ATM shift).
            vol_sabr = get_sabr_vol_at_K(grid, K, F, T_pil, r_f)
            return vol_sabr - atm_vol

        # Step 3 — Clamp or bracket, then interpolate spread in sqrt(t) space
        if tr <= days_list[0]:
            return sigma_atm + spread_at(sorted_pillars[0][0])

        if tr >= days_list[-1]:
            return sigma_atm + spread_at(sorted_pillars[-1][0])

        for i in range(len(sorted_pillars) - 1):
            t1_str, (t1_days, _, _) = sorted_pillars[i]
            t2_str, (t2_days, _, _) = sorted_pillars[i + 1]
            if t1_days <= tr <= t2_days:
                spread_tr = interpolate_spread_sqrt_t(
                    spread_at(t1_str), spread_at(t2_str),
                    t1_days, t2_days, tr
                )
                return sigma_atm + spread_tr

        return sigma_atm

    # -------------------------------------------------------------------------
    # Vol — closed-form nu/rho (vol-of-vol, spot-vol correlation) for breakeven P&L
    # -------------------------------------------------------------------------

    def get_smile_nu_rho(self, pair: str, as_of: datetime,
                         t_remaining_days: float) -> Tuple[float, float]:
        """
        Closed-form breakeven vol-of-vol (nu_BE) and spot/vol correlation
        (rho_BE), from the tenor pillar NEAREST t_remaining_days (not
        sqrt(t)-interpolated between pillars the way get_smile_vol's vol
        level/spread is) — nu/rho feed a diagnostic vanna/volga breakeven,
        not the option's actual price, so the nearest-pillar approximation
        is a deliberate simplification.

        Formula (Ravagli 2024, "Harvesting the FX skew premium", Risk.net):
            nu_BE  = NU_BE_C  * sqrt(Fly25 / (tau * sigma_ATM))
            rho_BE = RHO_BE_D * (RR25 / sigma_ATM)
        where Fly25/RR25/sigma_ATM are that pillar's 25-delta butterfly,
        25-delta risk reversal, and ATM vol (decimal), and tau is the
        pillar's time to expiry in years. NU_BE_C=4.0/RHO_BE_D=2.5 are the
        paper's fitted constants, validated on G10 pairs — re-check before
        trusting on EM/LatAm crosses.

        This reads directly off quoted market data (no numerical fit), unlike
        a per-date SABR calibration: no convergence/instability concerns, and
        no need for history-smoothing — replaces the former SABR-fitted
        nu/rho + EWMA-smoothing approach entirely.

        Unlike get_smile_vol, this does not depend on a strike K: nu/rho
        describe the whole smile's SHAPE at that pillar, not a
        strike-specific evaluation.

        Falls back to (0.0, 0.0) — i.e. no vol-of-vol/skew signal, so the
        vanna/volga breakeven collapses to the plain (non-breakeven) P&L —
        if the pillar's ATM/RR25/BF25 quotes aren't available.

        Pillar search floors at MIN_PILLAR_DAYS (excludes 'ON'): as tau -> 0,
        nu_BE = c*sqrt(Fly/(tau*sigma)) blows up algebraically. As a trade's
        t_remaining rolls under the floor in its final days, this clamps to
        the shortest still-eligible pillar rather than the unstable ON pillar.
        """
        as_of_ts    = pd.Timestamp(as_of)
        pillar_days = self._compute_pillar_days(pair, as_of_ts)
        eligible    = [t for t in TENOR_DAYS if pillar_days[t] >= self.MIN_PILLAR_DAYS] \
                      or list(TENOR_DAYS)
        nearest     = min(eligible, key=lambda t: abs(pillar_days[t] - t_remaining_days))

        smile_row = self.vol_surface.loc[:as_of_ts].iloc[-1]
        atm_val   = smile_row.get((pair, nearest, 'ATM'))
        rr_val    = smile_row.get((pair, nearest, 'RR25'))
        bf_val    = smile_row.get((pair, nearest, 'BF25'))
        if (atm_val is None or pd.isna(atm_val)
                or rr_val is None or pd.isna(rr_val)
                or bf_val is None or pd.isna(bf_val)):
            return 0.0, 0.0

        sigma0 = float(atm_val) / 100.0
        rr25   = float(rr_val) / 100.0
        fly25  = max(float(bf_val) / 100.0, 0.0)   # butterfly must be >= 0 for sqrt
        tau    = max(pillar_days[nearest], 1) / 365.0

        if sigma0 <= 0:
            return 0.0, 0.0

        nu_BE  = self.NU_BE_C  * np.sqrt(fly25 / (tau * sigma0))
        rho_BE = self.RHO_BE_D * (rr25 / sigma0)

        nu  = float(np.clip(nu_BE, 0.0, self.NU_BE_MAX))
        rho = float(np.clip(rho_BE, -0.999, 0.999))
        return nu, rho