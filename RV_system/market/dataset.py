import warnings
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import QuantLib as ql

from core.conventions import (
    TENOR_DAYS, FWD_YIELD_TENORS, DELTA_POINTS, get_spot_days, fx_calendar,
    MM_BASIS, MM_BASIS_DEFAULT
)
from market.feeds import (
    pull_dailyCCY_close, pull_fx_vol_surface,
    pull_sofr, pull_fwd_implied_yields, pull_usd_ois_curve
)
from core.vol_surface import (
    build_vol_grid, interpolate_vol_for_delta, interpolate_atm_vol,
    interpolate_spread_sqrt_t, get_sabr_vol_at_K, get_sabr_params
)
from core.calendar import preceding_business_day, add_tenor, spot_from_horizon


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
    Single container for spot, vol surface, and the rate curves.

    Rate convention:
        USD leg  — SOFR OIS par rates across FWD_YIELD_TENORS
        Non-USD  — forward implied yield across the same tenors

        Both are SIMPLE quotes on the currency's MM_BASIS, interpolated on the
        real pillar day counts and converted to continuous ACT/365 in
        _zero_rate. Bloomberg implies the forward yields off the SOFR OIS
        curve, so using the two together is what keeps S*exp((r_d-r_f)*T)
        arbitrage-free against market FX forward points.


    Vol convention:
        All Bloomberg vol data stored in raw percent (e.g. 8.5).
        Converted to decimal at point of use in get_atm_vol / get_smile_vol.
    """

    # Ceiling for the closed-form nu_BE (vol-of-vol) returned by
    # get_smile_nu_rho — see that method's docstring for why this clip exists.
    NU_BE_MAX = 5.0

    # Shortest tenor pillar (real calendar days) get_smile_nu_rho will read
    # nu/rho off, on EITHER source — excludes 'ON' (1 day). On the closed form
    # that is because tau -> 0 blows up nu_BE = c*sqrt(Fly/(tau*sigma)); on the
    # SABR path it is because a 1-day pillar's fit is noise. Below this the
    # shortest eligible pillar's values are used unchanged.
    MIN_PILLAR_DAYS = 7

    # Ravagli (2024), "Harvesting the FX skew premium" — fitted constants for
    # the closed-form nu_BE/rho_BE, validated on G10 pairs. Re-check before
    # trusting on EM/LatAm crosses.
    NU_BE_C  = 4.0
    RHO_BE_D = 2.5

    # Where get_smile_nu_rho takes (nu, rho) from.
    #
    #   'sabr'         the parameters of the SABR fit that BUILDS THE MARKS.
    #                  Default. Nothing extra is computed: get_smile_vol
    #                  already calibrates this smile on the pricing path, and
    #                  _sabr_pillar_params caches one fit per (pair, date,
    #                  tenor). Interpolated across pillars the same way every
    #                  other surface quantity is, so it does not jump.
    #
    #   'closed_form'  the Ravagli formula off BF25/RR25. Kept because it is
    #                  what every result before this change was produced with,
    #                  so it is the setting to use when reconciling against
    #                  them (see recon/reconcile.py).
    #
    # WHY THE DEFAULT MOVED. Measured on 2,583 USDJPY (date, tenor) fits, the
    # closed form tracks the calibrated nu almost exactly (corr 0.92-0.99) but
    # runs 1.28x high (sd 0.04) — and since var_expected goes as nu^2 that is
    # a 1.64x error on the quantity driving volga_pnl_be. Its rho is worse
    # than a scale error: calibrated rho decays to ~0 and turns positive by
    # 1Y while 2.5*RR25/sigma stays at -0.11, so no constant can fix it.
    #
    # Set per instance (`ds.nu_rho_source = 'closed_form'`) or per class.
    NU_RHO_SOURCE = 'sabr'

    def __init__(self,
                 spot:        pd.DataFrame,
                 vol_surface: pd.DataFrame,
                 sofr:        pd.Series,
                 fwd_yields:  pd.DataFrame,
                 usd_ois:     pd.DataFrame = None):


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

        # USD OIS joined onto the forward-implied yields, so every currency --
        # USD included -- is one column of one table read by one code path.
        # Both families are SIMPLE rates on their own MM_BASIS; the conversion
        # to continuous happens in _zero_rate, not here.
        self.usd_ois = None if usd_ois is None else usd_ois.copy()
        if self.usd_ois is not None:
            self.usd_ois.index = pd.to_datetime(self.usd_ois.index)
            self.usd_ois = self.usd_ois.sort_index()
            self.rate_curves = pd.concat([self.fwd_yields, self.usd_ois], axis=1)
            self.rate_curves = self.rate_curves.sort_index()
        else:
            # No OIS supplied — USD falls back to the flat SOFR fixing in
            # _zero_rate. Kept so an older cached/pickled build still loads.
            warnings.warn("No USD OIS curve supplied; the USD leg will use the "
                          "flat overnight SOFR fixing, which is not CIP-consistent "
                          "with the forward-implied yields.")
            self.rate_curves = self.fwd_yields

        # (pair, as_of_date) -> {'node': {...}, 'accrual': {...}}.
        # 'node'    = calendar days from as_of to each tenor's EXPIRY. The
        #             interpolation x-axis, shared with the vol surface.
        # 'accrual' = days from the pair's spot date to each tenor's DELIVERY.
        #             The window the quoted simple rate actually accrues over,
        #             so it is what the simple -> continuous conversion uses.
        # Both walk the FX calendar for ten tenors and are pure functions of
        # (pair, date). The rates path now hits this twice per
        # get_rates_for_tenor call, which the engine runs per leg per day.
        self._pillar_cache: dict = {}

        # ---- read caches ------------------------------------------------
        # Everything below is a pure function of (pair, as_of[, tenor]), so
        # caching cannot change a number -- only how many times it is
        # computed. Measured on a 120-day single-pair rolling run:
        #
        #     get_rates_for_tenor   25,398 calls    1,238 distinct   95.1% dup
        #     smile pillar grids     8,647 calls      121 distinct   98.6% dup
        #
        # The duplication is structural, not accidental: MarketSnapshot.
        # price_state asks for the same tenor's rates three times (directly,
        # inside smile_vol, and again inside forward), and every open position
        # on a date shares a handful of distinct expiries.

        # (pair, date, rounded t_days) -> (r_d, r_f).
        self._rate_cache: dict = {}

        # (pair, date) -> {tenor: (node_days, atm_vol, vol_grid)}. The quoted
        # smile, decoded once per date instead of once per strike lookup.
        self._smile_grid_cache: dict = {}

        # (pair, date, tenor) -> (nu, rho) | None, from the SABR fit at that
        # pillar. See _sabr_pillar_params for why this is a separate fit from
        # the one inside get_smile_vol.
        self._sabr_param_cache: dict = {}

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
        print("Pulling USD OIS curve...")
        usd_ois = pull_usd_ois_curve(days)
        print("Pulling forward implied yields...")
        fwd_yields = pull_fwd_implied_yields(ccys, days)
        return cls(spot, vol, sofr, fwd_yields, usd_ois)
 
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
        return self._tenor_days(pair, as_of)['node']

    def _compute_accrual_days(self, pair: str, as_of) -> dict:
        """
        Days from the pair's SPOT date to each tenor's DELIVERY date -- the
        window a quoted money-market rate for that tenor actually accrues over.

        Distinct from _compute_pillar_days, which measures as_of -> expiry.
        The two differ by roughly the T+2 settlement lag at both ends, and
        conflating them is what makes a simple rate convert to the wrong
        continuous rate.
        """
        return self._tenor_days(pair, as_of)['accrual']

    def _tenor_days(self, pair: str, as_of) -> dict:
        """Both day-count views for every tenor, computed together and cached."""
        as_of_date = pd.Timestamp(as_of).date()
        key = (pair, as_of_date)
        if key not in self._pillar_cache:
            fxc     = fx_calendar(pair)
            entry   = preceding_business_day(as_of_date, fxc.cal_trade)
            spot_dt = spot_from_horizon(entry, fxc)
            node, accrual = {}, {}
            for tenor in TENOR_DAYS:
                expiry = add_tenor(entry, tenor, fxc)
                node[tenor]    = (expiry - as_of_date).days
                accrual[tenor] = max((spot_from_horizon(expiry, fxc) - spot_dt).days, 1)
            self._pillar_cache[key] = {'node': node, 'accrual': accrual}
        return self._pillar_cache[key]

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
        Returns (r_d, r_f) as CONTINUOUS ACT/365 zero rates.

        Both legs now go down the same path: USD off the SOFR OIS curve,
        non-USD off the forward-implied yields, both interpolated on the same
        pillar grid and both converted from their quoted simple basis. USD used
        to be a flat overnight fixing with no term structure, which broke CIP
        against the forward-implied yields (~5bp at 1M, ~38bp at 1Y).

        Convention (standard FX):
            EURUSD — base=EUR (r_f), quote=USD (r_d)
            USDJPY — base=USD (r_f), quote=JPY (r_d)

        CACHED on (pair, as_of date, t_remaining_days). This is a pure lookup
        — same inputs, same curve, same answer — and it is the single largest
        cost in the date loop, because every consumer asks for the same
        handful of tenors over and over. See _rate_cache in __init__.
        """
        key = (pair, pd.Timestamp(as_of).date(), round(float(t_remaining_days), 6))
        hit = self._rate_cache.get(key)
        if hit is None:
            hit = (self._zero_rate(pair[3:], pair, as_of, t_remaining_days),
                   self._zero_rate(pair[:3], pair, as_of, t_remaining_days))
            self._rate_cache[key] = hit
        return hit

    @staticmethod
    def _simple_to_continuous(r_simple: float, accrual_days: int,
                              basis: int) -> float:
        """
        A quoted money-market rate -> the continuous ACT/365 rate the pricer wants.

            DF     = 1 / (1 + r * d/basis)
            r_cont = -ln(DF) / (d/365) = ln(1 + r*d/basis) * 365/d

        Both the OIS par rates (out to 1Y a single-payment swap, so par == zero)
        and the forward-implied yields are simple rates on their currency's
        basis. The old code fed them straight in as if they were continuous,
        which at 3.68% over 31 days is a 4.5bp error on the USD leg alone.
        """
        growth = 1.0 + r_simple * accrual_days / basis
        if growth <= 0.0:
            # Only reachable on a nonsensically negative quote; fall through to
            # the linear rate rather than take log of a non-positive number.
            return r_simple
        return float(np.log(growth) * 365.0 / accrual_days)

    def _zero_rate(self, ccy: str, pair: str, as_of: datetime,
                   t_remaining_days: float) -> float:
        """
        Continuous ACT/365 zero rate for `ccy` at `t_remaining_days`, off the
        pillar curve, for an option on `pair`.

        WHY `pair` IS AN ARGUMENT
        -------------------------
        The node x-positions come from the TRADED pair's calendar, for both
        legs. What drives the forward is the DIFFERENTIAL r_d - r_f, so the two
        legs have to be read at the same x or the differential is contaminated:
        at each pillar both currencies must return their own quoted rate, not
        one quote and one interpolation. Putting each currency on its own
        XXXUSD calendar instead would reintroduce exactly that error on any
        pair whose calendar differs from XXXUSD's -- i.e. every cross.
        """
        pillar_days  = self._compute_pillar_days(pair, as_of)
        accrual_days = self._compute_accrual_days(pair, as_of)
        basis        = MM_BASIS.get(ccy, MM_BASIS_DEFAULT)

        nodes = []
        for tenor in FWD_YIELD_TENORS:
            try:
                col = self.rate_curves.loc[:as_of, (ccy, tenor)]
            except KeyError:
                continue
            # .dropna() BEFORE .iloc[-1], not after. The last ROW at or before
            # as_of is not necessarily the last QUOTE: a currency's own holiday
            # blanks its column while the pair still trades. USOSFR is NaN on
            # Memorial Day 2025-05-26 while USDJPY spot and JPYI1M both print,
            # so taking the row would blank the entire USD curve and drop this
            # function into the flat-SOFR fallback -- silently undoing the whole
            # point of the OIS curve, on every US holiday. Carry the last real
            # quote forward instead, exactly as get_spot does.
            col = col.dropna()
            if col.empty:
                continue
            nodes.append((
                pillar_days[tenor],
                self._simple_to_continuous(float(col.iloc[-1]),
                                           accrual_days[tenor], basis),
            ))

        if not nodes:
            if ccy == 'USD':
                # Last resort: the overnight fixing, converted off a 1-day
                # accrual. Flat, so not CIP-consistent -- say so once.
                warnings.warn(
                    f"No USD OIS quotes at or before {pd.Timestamp(as_of).date()}; "
                    f"falling back to the flat overnight SOFR fixing."
                )
                r_on = float(self.sofr.loc[:as_of].dropna().iloc[-1])
                return self._simple_to_continuous(r_on, 1, MM_BASIS['USD'])
            raise ValueError(
                f"No rate curve data found for {ccy} at or before {as_of}"
            )


        # ql.DiscountCurve needs STRICTLY increasing dates. Real expiry day
        # counts are monotonic in tenor for every G10/LatAm currency across all
        # of 2026 (checked), but a long enough holiday cluster could in
        # principle collapse two pillars onto the same day. Drop rather than
        # let QuantLib throw mid-backtest, and say so.
        nodes.sort(key=lambda x: x[0])
        kept = []
        for d, r in nodes:
            if kept and d <= kept[-1][0]:
                warnings.warn(
                    f"{ccy} on {pd.Timestamp(as_of).date()}: pillar at {d}d "
                    f"collides with {kept[-1][0]}d after calendar adjustment; "
                    f"dropping the duplicate node."
                )
                continue
            kept.append((d, r))

        days_list  = [d for d, _ in kept]
        rates_list = [r for _, r in kept]

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
    # Vol — the quoted smile, decoded once per date
    # -------------------------------------------------------------------------

    def _smile_quotes(self, pair: str, as_of) -> dict:
        """
        The quoted smile at every tenor pillar for one (pair, date):

            {'pillars': {tenor: (node_days, atm_vol, vol_grid)},
             'quotes' : {tenor: {'days','atm','rr','bf'}}}

        Both views come off the SAME decode of the surface row, which is the
        point. Reading (pair, tenor, 'ATM'/'RR25'/'BF25'/...) out of the
        MultiIndex is ~100 lookups per date, and get_smile_vol used to redo
        all of them for every strike it was asked about -- 8,647 calls against
        121 distinct (pair, date) on a 120-day run.

        'pillars' is what get_smile_vol interpolates; 'quotes' is the raw
        ATM/RR/BF the closed-form nu_BE/rho_BE reads. Serving both from one
        cache is also what guarantees the two can never disagree about which
        surface row they are looking at.

        A tenor with no ATM quote, or with no RR/BF pair at any delta, is
        omitted -- same rule as before, just applied once.
        """
        key = (pair, pd.Timestamp(as_of).date())
        hit = self._smile_grid_cache.get(key)
        if hit is not None:
            return hit

        as_of_ts    = pd.Timestamp(as_of)
        pillar_days = self._compute_pillar_days(pair, as_of_ts)
        smile_row   = self.vol_surface.loc[:as_of_ts].iloc[-1]

        pillars, quotes = {}, {}
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

            pillars[tenor_str] = (real_days, atm_vol,
                                  build_vol_grid(atm_vol, rr, bf))
            quotes[tenor_str]  = {'days': real_days, 'atm': atm_vol,
                                  'rr': rr, 'bf': bf}

        hit = {'pillars': pillars, 'quotes': quotes}
        self._smile_grid_cache[key] = hit
        return hit

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

        # Step 2 — the quoted smile at every available tenor pillar. Decoded
        # once per (pair, date) and cached; the decode is ~100 MultiIndex
        # lookups and was previously repeated on every strike query.
        pillar_data = self._smile_quotes(pair, as_of_ts)['pillars']

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

    def _sabr_pillar_params(self, pair: str, as_of, tenor: str):
        """
        (nu, rho) from the SABR fit at ONE tenor pillar, or None if that pillar
        has no quotes or the calibration failed. Cached on (pair, date, tenor).

        FITTED WITH THE PILLAR'S OWN FORWARD, which is what makes it cacheable.
        get_smile_vol deliberately does not: its `spread_at` passes the TARGET
        tenor's F into every pillar's fit, on the argument (stated there) that
        the error partly cancels in the ATM-relative spread it actually uses.
        That is defensible for a spread and wrong for a shape parameter — and
        it would also make the fit depend on which option is asking, which is
        why the pricing path performs 17,225 calibrations on a 120-day run
        where one per pillar per day is ~1,080.
        """
        key = (pair, pd.Timestamp(as_of).date(), tenor)
        if key in self._sabr_param_cache:
            return self._sabr_param_cache[key]

        out     = None
        pillars = self._smile_quotes(pair, as_of)['pillars']
        entry   = pillars.get(tenor)
        if entry is not None:
            node_days, _atm, grid = entry
            T = max(node_days, 1) / 365.0
            r_d, r_f = self.get_rates_for_tenor(pair, as_of, node_days)
            F = self.get_spot(pair, as_of) * np.exp((r_d - r_f) * T)
            out = get_sabr_params(grid, F, T, r_f)

        self._sabr_param_cache[key] = out
        return out

    def _nu_rho_sabr(self, pair: str, as_of,
                     t_remaining_days: float):
        """
        (nu, rho) off the calibrated surface, interpolated across pillars, or
        None if no pillar produced a usable fit (caller falls back).

        INTERPOLATION. rho goes in sqrt(t) directly, the same scheme
        get_smile_vol uses for the smile spread. nu does NOT: its term
        structure is close to nu ~ 1/sqrt(tau), so what is interpolated is
        nu*sqrt(days), which is nearly flat across the curve — measured 0.65
        at 1W and 0.59 from 1M out on USDJPY — and the sqrt is undone at the
        end. Interpolating nu itself would cut a corner off a 1/sqrt curve.

        Outside the pillar range the nearest pillar's values are used
        unchanged, matching how interpolate_atm_vol and get_smile_vol clamp.
        Extrapolating nu*sqrt(t) inwards instead would reintroduce exactly the
        tau -> 0 blow-up that NU_BE_MAX exists to cap on the closed form.
        """
        pts = []
        for tenor, q in self._smile_quotes(pair, as_of)['quotes'].items():
            if q['days'] < self.MIN_PILLAR_DAYS:
                continue                        # excludes 'ON', as before
            p = self._sabr_pillar_params(pair, as_of, tenor)
            if p is not None:
                pts.append((q['days'], p[0], p[1]))
        if not pts:
            return None

        pts.sort()
        tr = max(float(t_remaining_days), 1.0)

        if tr <= pts[0][0]:
            return pts[0][1], float(np.clip(pts[0][2], -0.999, 0.999))
        if tr >= pts[-1][0]:
            return pts[-1][1], float(np.clip(pts[-1][2], -0.999, 0.999))

        tr_i = int(round(tr))
        for (d1, nu1, rho1), (d2, nu2, rho2) in zip(pts, pts[1:]):
            if d1 <= tr <= d2:
                s = interpolate_spread_sqrt_t(nu1 * np.sqrt(d1),
                                              nu2 * np.sqrt(d2),
                                              int(d1), int(d2), tr_i)
                nu  = float(s / np.sqrt(tr))
                rho = interpolate_spread_sqrt_t(rho1, rho2,
                                                int(d1), int(d2), tr_i)
                if not np.isfinite(nu) or nu <= 0.0:
                    return None
                return nu, float(np.clip(rho, -0.999, 0.999))
        return None

    def get_smile_nu_rho(self, pair: str, as_of: datetime,
                         t_remaining_days: float) -> Tuple[float, float]:
        """
        Vol-of-vol (nu) and spot/vol correlation (rho) at t_remaining_days.

        Dispatches on NU_RHO_SOURCE (see the class attribute for the measured
        justification of the default):

            'sabr'         the parameters of the SABR fit that builds the
                           marks, interpolated across pillars. Falls through
                           to the closed form if no pillar fit converged.
            'closed_form'  the Ravagli formula documented below — what every
                           result before this change was produced with.

        Both paths return the SAME kind of object, so nothing downstream needs
        to know which was used; the value actually applied is recorded per
        position-day as `nu_be`/`rho_be` in book/attribution.py.
        """
        source = getattr(self, 'nu_rho_source', None) or self.NU_RHO_SOURCE
        if source == 'sabr':
            hit = self._nu_rho_sabr(pair, as_of, t_remaining_days)
            if hit is not None:
                return hit
        return self._nu_rho_closed_form(pair, as_of, t_remaining_days)

    def _nu_rho_closed_form(self, pair: str, as_of: datetime,
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

        This reads directly off quoted market data with no numerical fit, so
        there are no convergence concerns and no history to smooth.

        NO LONGER THE DEFAULT — see NU_RHO_SOURCE. The reason is not
        instability, it is accuracy: measured against the SABR fit that
        actually builds the marks (2,583 USDJPY date/tenor pairs), nu_BE
        tracks it beautifully but sits 1.28x high, and rho_BE has the wrong
        term structure outright. Kept as the reconciliation reference and as
        a fit-free fallback when a pillar's calibration fails.

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