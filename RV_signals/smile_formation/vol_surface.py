import warnings
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import norm as _sabr_norm
import QuantLib as ql

from data import TENOR_DAYS, FWD_YIELD_TENORS, DELTA_POINTS, calendar_for_pair








# Call-delta axis: 0 = deep OTM call, 0.50 = ATM, 1.0 = deep ITM call.
# Put pillars sit at their same-strike call delta (delta_put + DF, DF ~= 1),
# so the smile is monotonic and continuous through the money. Feed a call
# (or call-delta-equivalent) spot delta to interpolate_vol_for_delta.
DELTA_GRID_MAP = {
    'C5':   0.05, 'C10': 0.10, 'C15': 0.15, 'C25': 0.25, 'C35': 0.35,
    'ATM':  0.50,
    'P35':  0.65, 'P25': 0.75, 'P15': 0.85, 'P10': 0.90, 'P5':  0.95,
}

# One-day variance weights for non-business days (Wystup §1.3.4).
# Weekends carry 15% of a normal trading day's variance.
# Holiday weight is defined here for future use once a holiday calendar is added.
_WEEKEND_WEIGHT = 0.15
_HOLIDAY_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Vol grid construction from market quotes
# ---------------------------------------------------------------------------

def build_vol_grid(atm: float,
                   rr:  Dict[int, float],
                   bf:  Dict[int, float]) -> pd.Series:
    """
    Convert ATM/RR/BF market quotes into individual call and put vols at each delta.

    Market convention:
        RR_d = vol(d-delta call) - vol(d-delta put)
        BF_d = 0.5*(vol(d-delta call) + vol(d-delta put)) - ATM

    Solving:
        call_vol = ATM + BF_d + 0.5 * RR_d
        put_vol  = ATM + BF_d - 0.5 * RR_d

    All inputs and outputs in decimal (e.g. 0.085 for 8.5% vol).
    """
    vols = {'ATM': atm}
    for d in DELTA_POINTS:
        if d in rr and d in bf and pd.notna(rr[d]) and pd.notna(bf[d]):
            vols[f'C{d}'] = atm + bf[d] + 0.5 * rr[d]
            vols[f'P{d}'] = atm + bf[d] - 0.5 * rr[d]
    return pd.Series(vols)


# ---------------------------------------------------------------------------
# SABR vol lookup — arbitrage-free, strike-space (replaces spline in the loop)
# ---------------------------------------------------------------------------

def _fit_sabr_to_grid(vol_grid: pd.Series, F: float, T: float, r_f: float):
    """
    SABR calibration step used by get_sabr_vol_at_K (vol at a strike).

    Steps
    -----
    1. Analytically invert each pillar delta to a strike:
           d1     = N^-1( dce * exp(r_f * T) )
           K_pil  = F * exp( -d1*vol*sqrt(T) + 0.5*vol^2*T )
    2. Sort ascending (QL SABRInterpolation requirement).
    3. Calibrate alpha / nu / rho with beta = 0.5 fixed.

    Returns (sabr, ks, vs, atm_vol). sabr is None on calibration failure or
    if fewer than 3 pillars are available — callers should fall back to
    atm_vol (and, for nu/rho, to "no smile signal").
    """
    available = [(DELTA_GRID_MAP[k], float(vol_grid[k]))
                 for k in vol_grid.index if k in DELTA_GRID_MAP]

    if len(available) < 3:
        atm_vol = float(vol_grid.get('ATM', available[0][1] if available else 0.09))
        return None, [], [], atm_vol

    T_safe = max(T, 1e-6)
    pairs  = []
    for dce, vol in available:
        arg   = float(np.clip(dce * np.exp(r_f * T_safe), 1e-9, 1 - 1e-9))
        d1    = _sabr_norm.ppf(arg)
        K_pil = F * np.exp(-d1 * vol * np.sqrt(T_safe) + 0.5 * vol**2 * T_safe)
        pairs.append((K_pil, vol))

    pairs.sort()
    ks = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]

    atm_idx = int(np.argmin(np.abs(np.array(ks) - F)))
    atm_vol = float(vol_grid.get('ATM', vs[atm_idx]))

    try:
        sabr = ql.SABRInterpolation(
            ks, vs, T_safe, F,
            vs[atm_idx], 0.5, 0.40, -0.30,
            False, True, False, False)   # alpha:opt, beta:fix, nu:opt, rho:opt
        sabr(F)                          # triggers calibration
        return sabr, ks, vs, atm_vol
    except Exception:
        return None, ks, vs, atm_vol


def get_sabr_vol_at_K(vol_grid: pd.Series, K: float, F: float,
                       T: float, r_f: float) -> float:
    """
    Fit SABR (beta=0.5 fixed, FX convention) to Malz pillar vols and return
    the arbitrage-free vol at a fixed strike K.

    No delta iteration needed: K is passed directly, so there is no circular
    dependency between delta and vol. Falls back to ATM vol on calibration
    failure or if fewer than 3 pillars are available.

    Parameters
    ----------
    vol_grid : Series from build_vol_grid (keys 'ATM', 'C25', 'P25', ...)
    K        : target strike (fixed at trade entry — never changes in the loop)
    F        : forward  S * exp((r_d - r_f) * T)  for this pillar's tenor
    T        : time to expiry in years for this pillar
    r_f      : foreign (base) rate, decimal
    """
    sabr, ks, vs, atm_vol = _fit_sabr_to_grid(vol_grid, F, T, r_f)
    if sabr is None:
        return atm_vol
    T_safe = max(T, 1e-6)
    # Clip extrapolation to 25% beyond outermost pillar to avoid blow-up
    K_clipped = float(np.clip(K, ks[0] * 0.75, ks[-1] * 1.25))
    return float(ql.sabrVolatility(
        K_clipped, F, T_safe, sabr.alpha(), 0.5, sabr.nu(), sabr.rho()))


# ---------------------------------------------------------------------------
# Delta interpolation — natural cubic spline (OLD PROCESS)
# ---------------------------------------------------------------------------

def interpolate_vol_for_delta(vol_grid: pd.Series,
                               option_delta: float,
                               lam: float = 0.25) -> float:
    """
    Interpolate the smile at a specific delta using a natural cubic spline.

    Fits a natural cubic spline (CubicSpline, bc_type='natural') through the
    pillar vols on the call-delta axis and evaluates at option_delta. Passes
    exactly through every pillar, removing the systematic smoothing bias of the
    prior Gaussian kernel approach at liquid strikes (25d, 10d, ATM).

    Between pillars the curve is C² smooth. The natural boundary condition
    (second derivative = 0 at both endpoints) prevents artificial curvature
    beyond the outermost wing strikes. Query points outside the pillar range
    are clamped to the nearest endpoint rather than extrapolated.

    Parameters
    ----------
    vol_grid     : Series from build_vol_grid, keys are 'ATM','C25','P25', etc.
    option_delta : call-delta-equivalent of the target strike, e.g. 0.25 for a
                   25d call, 0.50 for ATM, 0.75 for a 25d put (see DELTA_GRID_MAP).
    lam          : unused — retained for backward compatibility.
    """
    available    = {k: DELTA_GRID_MAP[k] for k in vol_grid.index if k in DELTA_GRID_MAP}
    sorted_items = sorted(available.items(), key=lambda x: x[1])
    deltas       = np.array([d for _, d in sorted_items])
    vols         = np.array([float(vol_grid[k]) for k, _ in sorted_items])

    if len(deltas) < 2:
        return float(vols[0]) if len(vols) == 1 else 0.0

    x = float(np.clip(option_delta, deltas[0], deltas[-1]))
    return float(CubicSpline(deltas, vols, bc_type='natural')(x))


# ---------------------------------------------------------------------------
# Weekend-weighted variance helpers (Wystup §1.3.4)
# ---------------------------------------------------------------------------

def _classify_period(as_of: date, t1: int, t2: int, pair: str):
    """
    Classify every calendar day in the half-open interval [t1, t2) relative to
    as_of, using the pair's JointCalendar (both legs). Returns
    (n_weekend, n_holiday, n_biz, weekend_mask, holiday_mask). For the i-th day,
    weekend_mask[i] is True on Sat/Sun, holiday_mask[i] is True on a weekday that
    either leg's market is closed. The two masks are mutually exclusive; a day is
    a business day when both are False.
    """
    cal = calendar_for_pair(pair)
    weekend_mask, holiday_mask = [], []
    for d in range(t1, t2):
        day = as_of + timedelta(days=d)
        qd  = ql.Date(day.day, day.month, day.year)
        is_wknd = day.weekday() >= 5
        weekend_mask.append(is_wknd)
        holiday_mask.append((not is_wknd) and (not cal.isBusinessDay(qd)))
    n_wkd = sum(weekend_mask)
    n_hol = sum(holiday_mask)
    n_biz = (t2 - t1) - n_wkd - n_hol
    return n_wkd, n_hol, n_biz, weekend_mask, holiday_mask


def _calibrate_biz_weight(t1: int, t2: int, n_wkd: int, n_hol: int, n_biz: int) -> float:
    """
    Solve for the business-day variance multiplier α such that the sum of all
    weighted daily variances over [t1, t2) equals exactly (t2 - t1):

        n_wkd × 0.15 + n_hol × 0.25 + n_biz × α = (t2 - t1)
        → α = [(t2 - t1) - n_wkd × 0.15 - n_hol × 0.25] / n_biz

    This preserves the total variance of the pillar forward vol exactly.
    """
    if n_biz == 0:
        return 1.0
    return ((t2 - t1) - n_wkd * _WEEKEND_WEIGHT - n_hol * _HOLIDAY_WEIGHT) / n_biz



# ---------------------------------------------------------------------------
# Calendar-arbitrage removal — Pool-Adjacent-Violators (PAVA)
# ---------------------------------------------------------------------------

def _pava_nondecreasing(values: np.ndarray) -> np.ndarray:
    """
    Least-squares projection of `values` onto the non-decreasing cone
    (isotonic regression via Pool-Adjacent-Violators, unweighted, O(n)).

    Scans left to right; whenever a point is smaller than the running block
    before it, the two blocks are merged into one at their (count-)weighted
    average, and merging repeats backward until non-decreasing. Blocks that
    never violate monotonicity are returned unchanged — only the pillars
    actually involved in a violation move.
    """
    vals, weights = [], []
    for v in values:
        vals.append(float(v))
        weights.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v1, w1 = vals[-2], weights[-2]
            v2, w2 = vals[-1], weights[-1]
            vals[-2:], weights[-2:] = [(v1 * w1 + v2 * w2) / (w1 + w2)], [w1 + w2]
    out = []
    for v, w in zip(vals, weights):
        out.extend([v] * int(round(w)))
    return np.array(out)


def _enforce_monotonic_variance(items: list) -> list:
    """
    Given sorted (days, vol) pillar pairs, project cumulative variance
    (days * vol**2) onto the non-decreasing cone via PAVA so forward variance
    between ANY two pillars is >= 0 (removes calendar-spread arbitrage in the
    quoted term structure). Pillars not part of a violation are untouched.
    """
    days = [d for d, _ in items]
    days_arr = np.array(days, dtype=float)
    vol_arr  = np.array([v for _, v in items], dtype=float)
    var_arr  = _pava_nondecreasing(vol_arr ** 2 * days_arr)

    # A 0-day pillar (e.g. an ON node whose expiry falls on `as_of` itself)
    # carries zero variance by definition -- already the global minimum, so
    # PAVA never moves it. Skip the divide there (0/0) and pass its quoted
    # vol through unchanged instead of manufacturing a NaN.
    zero_mask = days_arr <= 0
    adj_vol = np.empty_like(vol_arr)
    adj_vol[~zero_mask] = np.sqrt(var_arr[~zero_mask] / days_arr[~zero_mask])
    adj_vol[zero_mask]  = vol_arr[zero_mask]
    return list(zip(days, adj_vol.tolist()))


# ---------------------------------------------------------------------------
# ATM vol interpolation — weekend-weighted variance (Wystup §1.3.4)
# ---------------------------------------------------------------------------

def interpolate_atm_vol(vol_row: pd.Series,
                         t_remaining_days: float,
                         as_of: date,
                         pillar_days: dict = None,
                         pair: str = None) -> float:
    """
    Source: FX Options and Structured Products - P25

    Interpolate ATM vol at a target tenor using weekend-weighted variance.

    Standard variance interpolation treats every calendar day equally, which
    overstates the variance accrued over weekends (markets are closed; realized
    variance is structurally lower). This function assigns a weight of 0.15 to
    each weekend day and calibrates business-day weights upward so that the
    sum of weighted daily variances between any two pillars exactly reproduces
    the pillar vols (Wystup §1.3.4, equations 1.101–1.104).

    Forward variance between two surrounding pillars t1, t2:
        σ_f² = (σ²(t2)·t2 - σ²(t1)·t1) / (t2 - t1)          [eq 1.102]

    Accumulated variance to target date tr ∈ [t1, t2]:
        σ²(tr) = [σ²(t1)·t1 + Σ_{t=t1}^{tr-1} α(t)·σ_f²] / tr  [eq 1.104]

    Parameters
    ----------
    vol_row          : Series indexed by tenor strings ('1W', '1M', …), decimal vols.
    t_remaining_days : Calendar days to target expiry.
    as_of            : Pricing date as a date object (used to classify weekdays).
    pillar_days      : Dict mapping tenor string → actual calendar days to expiry,
                       computed from real expiry dates. Falls back to the static
                       TENOR_DAYS constants when not provided.
    pair             : Currency pair (e.g. 'USDJPY'). Selects the JointCalendar
                       (both legs) used to classify weekends and holidays.
    """
    node_days = pillar_days if pillar_days is not None else TENOR_DAYS
    items = sorted(
        (days, float(v))
        for tenor, days in node_days.items()
        for v in [vol_row.get(tenor)]
        if v is not None and pd.notna(v))

    if not items:
        raise ValueError("No valid tenors found in vol_row")

    # Remove calendar-spread arbitrage (locally inverted term structure) with
    # the minimal least-squares adjustment before interpolating. Pillars that
    # are already consistent pass through unchanged.
    items = _enforce_monotonic_variance(items)

    tr = int(round(t_remaining_days))

    if tr <= items[0][0]:
        return items[0][1]
    if tr >= items[-1][0]:
        return items[-1][1]

    # Locate the two surrounding pillars
    t1, v1, t2, v2 = None, None, None, None
    for i in range(len(items) - 1):
        if items[i][0] <= tr <= items[i + 1][0]:
            t1, v1 = items[i]
            t2, v2 = items[i + 1]
            break
    if tr == t1:
        return v1
    if tr == t2:
        return v2

    # One-day forward variance between pillars (constant within the period)
    var_t1  = v1 ** 2 * t1
    var_t2  = v2 ** 2 * t2
    fwd_var = (var_t2 - var_t1) / (t2 - t1)

    if fwd_var < -1e-10:
        # Should be unreachable: items already passed through
        # _enforce_monotonic_variance above, which guarantees non-decreasing
        # cumulative variance between every pillar pair. Left in as a
        # defensive invariant check in case that step is ever bypassed.
        msg = (
            "Negative forward variance after monotonicity correction — this "
            "should not happen; check _enforce_monotonic_variance.\n"
            f"  as_of            : {as_of}\n"
            f"  pair             : {pair}\n"
            f"  t_remaining_days : {t_remaining_days}  (rounded tr={tr})\n"
            f"  pillar t1        : {t1} days, vol={v1:.6f}, var_t1={var_t1:.8f}\n"
            f"  pillar t2        : {t2} days, vol={v2:.6f}, var_t2={var_t2:.8f}\n"
            f"  fwd_var          : {fwd_var:.8f}  (var_t2 - var_t1) / (t2 - t1)\n"
            f"  full vol_row     : {vol_row.to_dict()}"
        )
        raise ValueError(msg)
    fwd_var = max(fwd_var, 0.0)
    # Classify every day in [t1, t2) and solve for the business-day weight
    n_wkd, n_hol, n_biz, weekend_mask, holiday_mask = _classify_period(as_of, t1, t2, pair)
    alpha_biz = _calibrate_biz_weight(t1, t2, n_wkd, n_hol, n_biz)

    # Accumulate weighted variance from t1 up to (but not including) tr
    extra_var = 0.0
    for i in range(tr - t1):
        if weekend_mask[i]:
            w = _WEEKEND_WEIGHT
        elif holiday_mask[i]:
            w = _HOLIDAY_WEIGHT
        else:
            w = alpha_biz
        extra_var += w * fwd_var
    return float(np.sqrt((var_t1 + extra_var) / tr))


# ---------------------------------------------------------------------------
# Smile spread interpolation — √t weighting (Wystup §1.3.8)
# ---------------------------------------------------------------------------

def interpolate_spread_sqrt_t(spread_t1: float,
                               spread_t2: float,
                               t1: int,
                               t2: int,
                               tr: int) -> float:
    """
    Source: FX Options and Structured Products - P26-27 (Equ 1.108)

    Interpolate a vol spread (smile vol minus ATM vol) between two pillar tenors
    using square root of time weighting 

        σ̃(tr) = σ̃(t1) + (√tr − √t1) / (√t2 − √t1) × (σ̃(t2) − σ̃(t1))

    The spread is interpolated in √t space rather than variance space because
    jump and skew risk scale with √t while the vol level (variance) scales with t.
    Mixing the two interpolation spaces — variance for ATM level, √t for spread —
    is the industry-standard decomposition.

    Parameters
    ----------
    spread_t1, spread_t2 : smile vol minus ATM vol at each surrounding pillar.
    t1, t2               : pillar tenors in calendar days.
    tr                   : target tenor in calendar days.
    """
    sqrt_t1 = np.sqrt(max(t1, 1))
    sqrt_t2 = np.sqrt(t2)
    sqrt_tr = np.sqrt(tr)
    denom   = sqrt_t2 - sqrt_t1
    if abs(denom) < 1e-12:
        return spread_t1
    return spread_t1 + (sqrt_tr - sqrt_t1) / denom * (spread_t2 - spread_t1)























