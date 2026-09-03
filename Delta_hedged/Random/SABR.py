"""
SABR Vol Smile Builder — Surface Calibration
=============================================
Calibrates SABR (alpha, nu, rho; beta fixed) for one or more tenors of an FX
pair, reading every input — spot, rates, and the ATM/RR/BF vol quotes — from the
backtest's own FXVolDataset. One Bloomberg pull (FXVolDataset.build) feeds every
tenor, so passing a list of tenors is cheap.

Steps (per tenor)
-----------------
0  Tenor → calendar days (FX market convention, shared with the backtest)
1  Read ATM/RR/BF quotes for the tenor from FXVolDataset.vol_surface
2  Convert ATM/RR/BF → individual call and put pillar vols
3  Spot + rates (FXVolDataset) → GK forward F
4  Invert each pillar's call-delta-equivalent → strike K
5  Fit SABR to the (K, vol) pairs — extract alpha, nu, rho (+ fit RMSE)

Usage
-----
    df = calibrate_sabr('USDJPY', ['1M', '3M', '6M'])   # one row per tenor
    df.loc['3M', ['alpha', 'nu', 'rho']]                # params for one tenor
    df.loc['3M', 'ks'], df.loc['3M', 'vs']              # pillar strikes / vols

Note
----
Quotes are read from FXVolDataset, so tenors must be pillars it pulls
(ON, 1W, 2W, 3W, 1M, 2M, 3M, 6M, 9M, 1Y) and deltas a subset of (35, 25, 15, 10, 5).
"""

import warnings
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd
import pytz
from scipy.stats import norm
import QuantLib as ql
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime


# ------------------------------------------------------------------------------------------
# ---------------------------Using parent path for my imports-------------------------------
# ------------------------------------------------------------------------------------------
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

from data import fx_calendar
from trading_calendar import preceding_business_day, next_business_day, add_tenor
from dataset import FXVolDataset

# FX value-date rolls at 17:00 New York time; the trade date advances then.
_NY_TZ        = pytz.timezone('US/Eastern')
_FX_ROLL_HOUR = 17


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Tenor → calendar days (FX market convention, same as the backtest)
# ══════════════════════════════════════════════════════════════════════════════

def current_fx_trade_date(cal, roll_hour_et: int = _FX_ROLL_HOUR):
    """
    The live FX trade/horizon date, using NEW YORK time and the 5pm-ET value-date
    roll — not the machine clock. This avoids the off-by-one you get when the
    local calendar has ticked to the next day while New York is still on the
    prior trade date (or vice-versa).

      before 17:00 ET → today  (rolled back to a good business day)
      at/after 17:00 ET → the next business day (matches Bloomberg's roll)
    """
    now_ny = datetime.now(_NY_TZ)
    if now_ny.hour >= roll_hour_et:
        return next_business_day(now_ny.date(), cal)   # past the NY cut
    return preceding_business_day(now_ny.date(), cal)   # today / last business day


def tenor_to_calendar_days(pair: str, tenor, as_of=None):
    """
    Convert a tenor into the number of calendar days to expiry,
    return (t_days, horizon, expiry).

    as_of : optional trade/valuation date (date/str/Timestamp). If None, the live
            FX trade date is used (New York time + 5pm-ET roll). Pass it explicitly
            to pin the horizon (e.g. to match a data vintage or for reproducibility).
    """
    fxc = fx_calendar(pair)

    if as_of is None:
        horizon = current_fx_trade_date(fxc.cal_trade)
    else:
        horizon = preceding_business_day(pd.Timestamp(as_of).date(), fxc.cal_trade)

    expiry  = add_tenor(horizon, tenor, fxc)
    t_days  = (expiry - horizon).days
    return t_days, horizon, expiry


def _latest_quote(dataset: FXVolDataset, pair: str, tenor: str,
                  qtype: str, as_of: pd.Timestamp) -> float:
    """
    read single ATM/RR/BF quote from the dataset's vol surface
    """
    col = (pair, tenor, qtype)
    if col not in dataset.vol_surface.columns:
        avail = sorted({c[1] for c in dataset.vol_surface.columns if c[0] == pair})
        raise KeyError(
            f"{qtype} for {pair} {tenor} not in dataset (available tenors: {avail}). "
            f"Tenor must be a pulled pillar and delta a subset of those pulled.")
    s = dataset.vol_surface.loc[:as_of, col].dropna()
    if s.empty:
        raise ValueError(f"No {qtype} data for {pair} {tenor} as of {as_of.date()}.")
    return float(s.iloc[-1]) / 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Pure SABR fit — STEPS 2/4/5 given quotes + market data (no data access)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_sabr(atm: float, rr: dict, bf: dict, delta_pts, beta: float,
              S: float, r_d: float, r_f: float, T: float) -> dict:
    """
    Given data, run the pillar-vol → strike → SABR-fit pipeline
    return {F, alpha, nu, rho, rmse_vp, ks, vs, labels}
    """
    F = S * np.exp((r_d - r_f) * T)

    # market quotes → individual call/put pillar vols
    pillar_vols = {'ATM': atm}
    for d in delta_pts:
        pillar_vols[f'C{d}'] = atm + bf[d] + 0.5 * rr[d]
        pillar_vols[f'P{d}'] = atm + bf[d] - 0.5 * rr[d]

    # delta → strike (calls = d/100, puts = 1 - d/100, ATM = 0.50)
    delta_grid = {'ATM': 0.50}
    for d in delta_pts:
        delta_grid[f'C{d}'] = d / 100.0
        delta_grid[f'P{d}'] = 1.0 - d / 100.0
    pillar_pairs = []   # (label, K, vol)
    for label, vol in pillar_vols.items():
        dce = delta_grid[label]
        arg = float(np.clip(dce * np.exp(r_f * T), 1e-9, 1 - 1e-9))  # N^-1 needs (0,1)
        d1  = float(norm.ppf(arg))
        K   = F * np.exp(-d1 * vol * np.sqrt(T) + 0.5 * vol**2 * T)
        pillar_pairs.append((label, K, vol))
    pillar_pairs.sort(key=lambda x: x[1])
    labels = [p[0] for p in pillar_pairs]
    ks     = [p[1] for p in pillar_pairs]
    vs     = [p[2] for p in pillar_pairs]

    #  fit SABR (fixed beta) to the (K, vol) pillars
    atm_idx    = int(np.argmin(np.abs(np.array(ks) - F)))
    alpha_seed = vs[atm_idx]
    sabr = ql.SABRInterpolation(
        ks, vs, T, F,
        alpha_seed, beta, 0.40, -0.30,
        False, True, False, False)  # (optimise_alpha, fixed_beta, optimise_nu, optimise_rho)
    sabr(F)   # evaluating triggers the internal least-squares calibration
    alpha, nu, rho = sabr.alpha(), sabr.nu(), sabr.rho()
    resid   = [(ql.sabrVolatility(k, F, T, alpha, beta, nu, rho) - v) * 100
               for k, v in zip(ks, vs)]
    rmse_vp = float(np.sqrt(np.mean(np.square(resid))))
    return {'F': F, 'alpha': alpha, 'nu': nu, 'rho': rho, 'rmse_vp': rmse_vp,
            'ks': ks, 'vs': vs, 'labels': labels}


# ══════════════════════════════════════════════════════════════════════════════
# Single-tenor calibration (internal) — STEPS 1–5 for one tenor
# ══════════════════════════════════════════════════════════════════════════════

def _calibrate_one_tenor(pair: str, tenor: str, dataset: FXVolDataset,
                         delta_pts, beta: float, verbose: bool) -> dict:
    # ── STEP 0 — tenor → calendar days → year fraction ───────────────────────
    t_days, horizon, expiry = tenor_to_calendar_days(pair, tenor)
    T     = t_days / 365.0   # ACT/365 Fixed
    as_of = pd.Timestamp(horizon)

    # ── STEP 1 — read ATM / RR_d / BF_d quotes from the dataset ───────────────
    atm = _latest_quote(dataset, pair, tenor, 'ATM', as_of)
    rr  = {d: _latest_quote(dataset, pair, tenor, f'RR{d}', as_of) for d in delta_pts}
    bf  = {d: _latest_quote(dataset, pair, tenor, f'BF{d}', as_of) for d in delta_pts}

    # ── STEP 3 — spot + rates ─────────────────────────────────────────────────
    S        = dataset.get_spot(pair, as_of)
    r_d, r_f = dataset.get_rates_for_tenor(pair, as_of, t_days)

    # ── STEPS 2/4/5 — pillar vols → strikes → SABR fit (shared with history) ──
    fit = _fit_sabr(atm, rr, bf, delta_pts, beta, S, r_d, r_f, T)

    if verbose:
        print(f"  {str(tenor):>4} | {t_days:>4}d  T={T:.4f} | S={S:.5f}  F={fit['F']:.5f} | "
              f"r_d={r_d*100:>+6.3f}%  r_f={r_f*100:>+6.3f}% | "
              f"alpha={fit['alpha']:.5f}  vov={fit['nu']:.5f}  svc={fit['rho']:>+.5f} | "
              f"rmse={fit['rmse_vp']:.4f}vp")

    return {
        'tenor': tenor, 't_days': t_days, 'T': T, 'expiry': expiry,
        'S': S, 'F': fit['F'], 'r_d': r_d, 'r_f': r_f, 'atm': atm,
        'alpha': fit['alpha'], 'beta': beta, 'nu': fit['nu'], 'rho': fit['rho'],
        'rmse_vp': fit['rmse_vp'],
        'ks': fit['ks'], 'vs': fit['vs'], 'labels': fit['labels'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SABR surface calibration — input pair + tenor(s), get back a tidy DataFrame
# ══════════════════════════════════════════════════════════════════════════════

# Scalar columns (in display order); list-valued pillar columns are appended last.
_SCALAR_COLS = ['t_days', 'T', 'expiry', 'S', 'F', 'r_d', 'r_f', 'atm',
                'alpha', 'beta', 'nu', 'rho', 'rmse_vp']
_LIST_COLS   = ['ks', 'vs', 'labels']


def calibrate_sabr(pair: str,
                   tenors: Union[str, int, Sequence],
                   delta_pts=(35, 25, 10), beta: float = 0.5,
                   dataset: Optional[FXVolDataset] = None,
                   history_days: int = 15,
                   verbose: bool = False) -> pd.DataFrame:
    """
    Calibrate SABR for one or many tenors and return a DataFrame (one row/tenor).

    Spot, rates and the ATM/RR/BF vol quotes all come from FXVolDataset, so any
    pair resolves correctly (USD legs = SOFR, non-USD legs = interpolated forward
    yields) and a list of tenors only needs one underlying Bloomberg pull.

    Parameters
    ----------
    pair         : FX pair, e.g. 'USDJPY', 'EURUSD', 'EURGBP'.
    tenors       : a single tenor ('3M') or a list/tuple (['1M', '3M', '6M']).
                   Each must be a pillar the dataset pulls (see module note).
    delta_pts    : delta pillars to use (subset of (35, 25, 15, 10, 5)).
    beta         : fixed SABR backbone (0.5 = FX convention).
    dataset      : optional prebuilt FXVolDataset to reuse across calls. If None,
                   one is built for `pair` (single pull, reused across all tenors).
    history_days : history window when building the dataset (latest obs only needed).

    Returns
    -------
    DataFrame indexed by tenor with columns:
        t_days, T, expiry, S, F, r_d, r_f, atm,
        alpha, beta, nu, rho, rmse_vp,            ← calibration + fit quality
        ks, vs, labels                            ← per-tenor pillar arrays
    Vols and rates are decimals (e.g. 0.085 = 8.5%). Tenors that fail (missing
    data) are skipped with a warning rather than aborting the whole surface.
    """
    if isinstance(tenors, (str, int)):
        tenors = [tenors]

    if dataset is None:
        dataset = FXVolDataset.build(pairs=[pair], days=history_days)

    if verbose:
        print("=" * 100)
        print(f"  SABR Surface Calibration — {pair}  |  {len(tenors)} tenor(s)  "
              f"|  beta={beta} fixed  |  deltas={tuple(delta_pts)}")
        print("=" * 100)

    rows = []
    for tenor in tenors:
        try:
            rows.append(_calibrate_one_tenor(pair, tenor, dataset,
                                             delta_pts, beta, verbose))
        except (KeyError, ValueError) as e:
            warnings.warn(f"Skipping {pair} {tenor}: {e}")

    cols = _SCALAR_COLS + _LIST_COLS
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.Index([], name='tenor'))

    df = pd.DataFrame(rows).set_index('tenor')[cols]
    df.attrs['pair'] = pair
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Historical calibration - Daily 
# ══════════════════════════════════════════════════════════════════════════════

def _quotes_on_date(dataset: FXVolDataset, pair: str, tenor: str,
                    delta_pts, ts: pd.Timestamp):
    def g(qtype):
        col = (pair, tenor, qtype)
        if col not in dataset.vol_surface.columns:
            raise KeyError(f"{qtype} for {pair} {tenor} not in dataset.")
        return dataset.vol_surface.at[ts, col]
    atm = g('ATM')
    rr  = {d: g(f'RR{d}') for d in delta_pts}
    bf  = {d: g(f'BF{d}') for d in delta_pts}
    if any(pd.isna(v) for v in [atm, *rr.values(), *bf.values()]):
        return None
    return atm / 100.0, {d: rr[d] / 100.0 for d in delta_pts}, {d: bf[d] / 100.0 for d in delta_pts}


def calibrate_sabr_history(pair: str, tenors, dataset: FXVolDataset,
                           delta_pts=(35, 25, 10), beta: float = 0.5,
                           verbose: bool = False) -> pd.DataFrame:
    """
    Calibrate SABR for each tenor on EVERY historical date in the dataset
    """
    if isinstance(tenors, (str, int)):
        tenors = [tenors]
    fxc       = fx_calendar(pair)
    dates     = dataset.vol_surface.index
    out = {tenor: [] for tenor in tenors}   # tenor -> list of (date, alpha, nu, rho, rmse)
    for ts in dates:
        horizon = preceding_business_day(ts.date(), fxc.cal_trade)  # trade date as-of this row
        for tenor in tenors:
            try:
                q = _quotes_on_date(dataset, pair, tenor, delta_pts, ts)
                if q is None:
                    continue
                atm, rr, bf = q
                expiry   = add_tenor(horizon, tenor, fxc)
                t_days   = (expiry - horizon).days
                T        = t_days / 365.0
                S        = dataset.get_spot(pair, ts)
                r_d, r_f = dataset.get_rates_for_tenor(pair, ts, t_days)
                fit      = _fit_sabr(atm, rr, bf, delta_pts, beta, S, r_d, r_f, T)
                out[tenor].append((ts, fit['alpha'], fit['nu'], fit['rho'], fit['rmse_vp']))
            except (KeyError, ValueError):
                continue
    frames = {}
    for tenor, recs in out.items():
        if not recs:
            warnings.warn(f"No history calibrated for {pair} {tenor}.")
            continue
        idx = [r[0] for r in recs]
        frames[tenor] = pd.DataFrame(
            {'alpha': [r[1] for r in recs], 'nu': [r[2] for r in recs],
             'rho': [r[3] for r in recs], 'rmse_vp': [r[4] for r in recs]},
            index=pd.DatetimeIndex(idx))
    if not frames:
        return pd.DataFrame()
    hist = pd.concat(frames, axis=1)
    hist.columns.names = ['tenor', 'param']
    hist.index.name = 'date'
    hist = hist.sort_index()
    if verbose:
        print(f"  Calibrated history for {pair}: "
              f"{hist.index.min().date()} → {hist.index.max().date()}  "
              f"({len(hist)} dates)")
        for tenor in tenors:
            if tenor in frames:
                print(f"    {tenor:>4}: {frames[tenor]['nu'].notna().sum()} obs")
    return hist








# ══════════════════════════════════════════════════════════════════════════════
# Percentiles — where do TODAY's nu / rho sit within 3m / 1y / 5y history?
# ══════════════════════════════════════════════════════════════════════════════

def _window_offset(window: str) -> pd.DateOffset:
    """'3M'/'1Y'/'5Y'/'2W'/'30D' → a pandas DateOffset for the lookback length."""
    n, unit = int(window[:-1]), window[-1].upper()
    return {'D': pd.DateOffset(days=n), 'W': pd.DateOffset(weeks=n),
            'M': pd.DateOffset(months=n), 'Y': pd.DateOffset(years=n)}[unit]


def _window_calendar_days(window: str) -> int:
    """Approximate calendar days for a window string (for sizing the data pull)."""
    n, unit = int(window[:-1]), window[-1].upper()
    return n * {'D': 1, 'W': 7, 'M': 31, 'Y': 366}[unit]


def sabr_percentiles(pair: str, tenors,
                     windows=('3M', '1Y', '5Y'),
                     params=('nu', 'rho'),
                     delta_pts=(35, 25, 10), beta: float = 0.5,
                     dataset: Optional[FXVolDataset] = None,
                     history_days: Optional[int] = None,
                     verbose: bool = False) -> pd.DataFrame:
    """
    For each tenor, calibrate SABR daily over history and report where the LATEST
    nu (vol-of-vol) and rho (spot-vol corr) sit as a percentile of each lookback
    window (3M / 1Y / 5Y)
    """
    if isinstance(tenors, (str, int)):
        tenors = [tenors]
    if dataset is None:
        need = max(_window_calendar_days(w) for w in windows) + 45  # + buffer
        if history_days is not None:
            need = history_days
        dataset = FXVolDataset.build(pairs=[pair], days=need)
    hist = calibrate_sabr_history(pair, tenors, dataset, delta_pts, beta, verbose)
    if hist.empty:
        return pd.DataFrame()
    last_date  = hist.index.max()
    hist_tenors = hist.columns.get_level_values('tenor').unique()
    data = {}
    for tenor in tenors:
        if tenor not in hist_tenors:
            warnings.warn(f"No history for {pair} {tenor}; omitted from percentiles.")
            continue
        sub = hist[tenor].dropna(how='all')
        rec = {('meta', 'n_obs'): int(len(sub)),
               ('meta', 'last_date'): last_date.date()}
        for p in params:
            series = sub[p].dropna()
            cur    = float(series.iloc[-1])
            rec[(p, 'current')] = cur
            for w in windows:
                win = series[series.index >= (last_date - _window_offset(w))]
                rec[(p, f'pct_{w}')] = (float((win <= cur).mean() * 100.0)
                                        if len(win) else np.nan)
        data[tenor] = rec
    if not data:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(data, orient='index')
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=['param', 'stat'])
    out.index.name = 'tenor'
    out.attrs['pair'] = pair
    return out


# # Current snapshot across tenors
# snap = calibrate_sabr('USDJPY', ['1M', '3M', '6M'])
# print("\nCurrent SABR surface:")
# print(snap[_SCALAR_COLS])



# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# SABR:  dF = alpha F^beta dW1 ;  d(alpha) = nu*alpha dW2 ;  corr(dW1,dW2) = rho
#   nu  (vol-of-vol)     ↔ annualized stdev of daily LOG-changes in the vol state
#   rho (spot-vol corr)  ↔ correlation of spot returns with those vol log-changes
#
# We use the ATM implied vol as the observable vol state (≈ SABR alpha at beta=0.5),
# so both realized twins describe how ATM vol and spot have actually co-moved.

def _tenor_business_days(pair: str, tenor) -> int:
    """Estimation window (business days) matched to the option tenor's horizon."""
    t_days, _, _ = tenor_to_calendar_days(pair, tenor)
    return max(5, int(round(t_days * 252.0 / 365.0)))


def realized_rho_nu_history(pair: str, tenors, dataset: FXVolDataset,
                            verbose: bool = False) -> pd.DataFrame:
    """
    Daily realized (rho, nu) per tenor from ATM implied-vol dynamics:
        r      = ln(S_t / S_{t-1})            spot log-return
        dlnv   = ln(sigma_t / sigma_{t-1})    log-change in ATM implied vol
        rho    = rolling corr(r, dlnv)  over a tenor-matched window
        nu     = rolling stdev(dlnv) * sqrt(252)   (annualized vol-of-vol)

    Returns a wide DataFrame indexed by date with columns MultiIndex (tenor, param),
    param ∈ {rho, nu} — same shape as the implied side, so they align 1:1.
    """
    if isinstance(tenors, (str, int)):
        tenors = [tenors]

    spot = dataset.spot[pair].dropna()
    r    = np.log(spot / spot.shift(1))

    frames = {}
    for tenor in tenors:
        col = (pair, tenor, 'ATM')
        if col not in dataset.vol_surface.columns:
            warnings.warn(f"No ATM series for {pair} {tenor}; skipping realized.")
            continue
        atm  = (dataset.vol_surface[col] / 100.0).dropna()
        dlnv = np.log(atm / atm.shift(1))

        joined = pd.concat({'r': r, 'dlnv': dlnv}, axis=1).dropna()
        N = _tenor_business_days(pair, tenor)
        rho = joined['r'].rolling(N).corr(joined['dlnv'])
        nu  = joined['dlnv'].rolling(N).std() * np.sqrt(252.0)
        frames[tenor] = pd.DataFrame({'rho': rho, 'nu': nu}).dropna()

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, axis=1)
    hist.columns.names = ['tenor', 'param']
    hist.index.name = 'date'
    hist = hist.sort_index()

    if verbose:
        print(f"  Realized rho/nu for {pair}: "
              f"{hist.index.min().date()} → {hist.index.max().date()}  ({len(hist)} dates)")
        for tenor in tenors:
            if tenor in frames:
                print(f"    {tenor:>4}: window={_tenor_business_days(pair, tenor)}bd, "
                      f"{len(frames[tenor])} obs")

    return hist


# ══════════════════════════════════════════════════════════════════════════════
# Implied vs Realized — levels, spread, and percentiles of each vs 3m/1y/5y history
# ══════════════════════════════════════════════════════════════════════════════

def _level_and_pcts(s: pd.Series, windows, last: pd.Timestamp) -> dict:
    """{'current': latest, 'pct_<w>': % of window at-or-below latest} for each window."""
    cur = float(s.iloc[-1])
    out = {'current': cur}
    for w in windows:
        win = s[s.index >= (last - _window_offset(w))]
        out[f'pct_{w}'] = float((win <= cur).mean() * 100.0) if len(win) else np.nan
    return out


def compare_implied_realized(pair: str, tenors,
                             windows=('3M', '1Y', '5Y'),
                             delta_pts=(35, 25, 10), beta: float = 0.5,
                             dataset: Optional[FXVolDataset] = None,
                             history_days: Optional[int] = None,
                             verbose: bool = False) -> dict:
    """
    Compare SABR-implied rho/nu against their realized twins (from ATM implied-vol
    dynamics), per tenor. Reports current levels, the implied−realized spread, and
    the percentile of each (implied / realized / spread) within 3m/1y/5y history.

    Returns
    -------
    dict with:
      'rho', 'nu' : summary DataFrames indexed by tenor, columns MultiIndex
                    (kind, stat) with kind ∈ {impl, real, spread, meta} and
                    stat ∈ {current, pct_3M, pct_1Y, pct_5Y} (n_obs/last_date for meta).
      'series'    : {tenor: DataFrame of the aligned daily
                    rho_impl/rho_real/rho_spread/nu_impl/nu_real/nu_spread series}
                    — ready to plot.
    """
    if isinstance(tenors, (str, int)):
        tenors = [tenors]

    if dataset is None:
        need = max(_window_calendar_days(w) for w in windows) + 45
        if history_days is not None:
            need = history_days
        dataset = FXVolDataset.build(pairs=[pair], days=need)

    imp = calibrate_sabr_history(pair, tenors, dataset, delta_pts, beta, verbose)
    rea = realized_rho_nu_history(pair, tenors, dataset, verbose)
    if imp.empty or rea.empty:
        return {'rho': pd.DataFrame(), 'nu': pd.DataFrame(), 'series': {}}

    imp_tenors = set(imp.columns.get_level_values('tenor'))
    rea_tenors = set(rea.columns.get_level_values('tenor'))

    # ── Align implied & realized per tenor, build the merged daily series ──────
    series = {}
    for tenor in tenors:
        if tenor not in imp_tenors or tenor not in rea_tenors:
            continue
        m = pd.DataFrame({
            'rho_impl': imp[(tenor, 'rho')], 'rho_real': rea[(tenor, 'rho')],
            'nu_impl':  imp[(tenor, 'nu')],  'nu_real':  rea[(tenor, 'nu')],
        }).dropna()
        if m.empty:
            continue
        series[tenor] = m

    # ── Summary percentile tables, one per param ──────────────────────────────
    results = {}
    for p in ('rho', 'nu'):
        data = {}
        for tenor, m in series.items():
            last = m.index.max()
            rec = {}
            for kind in ('impl', 'real'):
                for stat, val in _level_and_pcts(m[f'{p}_{kind}'], windows, last).items():
                    rec[(kind, stat)] = val
            rec[('meta', 'n_obs')]     = int(len(m))
            rec[('meta', 'last_date')] = last.date()
            data[tenor] = rec
        if data:
            dfp = pd.DataFrame.from_dict(data, orient='index')
            dfp.columns = pd.MultiIndex.from_tuples(dfp.columns, names=['kind', 'stat'])
            dfp.index.name = 'tenor'
            dfp.attrs['pair'] = pair
            results[p] = dfp
        else:
            results[p] = pd.DataFrame()

    results['series'] = series
    return results






pair = 'EURCAD'
tenors = ['1W', '1M', '2M', '3M']

cmp = compare_implied_realized(pair, tenors)

print(f"\n=== {pair} rho (spot-vol corr): implied vs realized ===")
print(cmp['rho'])
print(f"\n=== {pair} nu (vol-of-vol): implied vs realized ===")
print(cmp['nu'])



# delta_pts=(35, 25, 15, 10, 5)















# imp_rho = imp.loc[:, imp.columns.get_level_values("param") == "rho"]
# real_rho = rea.loc[:, rea.columns.get_level_values("param") == "rho"]





# pair = 'USDJPY'
# tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '6M']
# delta_pts = (35, 25, 10)
# beta = 0.5
# windows=('3M', '1Y')


# need = max(_window_calendar_days(w) for w in windows) + 45
# dataset = FXVolDataset.build(pairs=[pair], days=need)


# imp = calibrate_sabr_history(pair, tenors, dataset)
# rea = realized_rho_nu_history(pair, tenors, dataset)


# imp_tenors = set(imp.columns.get_level_values('tenor'))
# rea_tenors = set(rea.columns.get_level_values('tenor'))


# series = {}
# for tenor in tenors:
#     if tenor not in imp_tenors or tenor not in rea_tenors:
#         continue
#     m = pd.DataFrame({
#         'rho_impl': imp[(tenor, 'rho')], 'rho_real': rea[(tenor, 'rho')],
#         'nu_impl':  imp[(tenor, 'nu')],  'nu_real':  rea[(tenor, 'nu')],
#     }).dropna()
#     if m.empty:
#         continue
#     series[tenor] = m



# results = {}
# for p in ('rho', 'nu'):
#     data = {}


