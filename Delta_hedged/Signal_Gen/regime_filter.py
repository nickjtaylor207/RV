"""
regime_filter.py — point-in-time vol-regime gates for short-vol entries.

The problem this solves
-----------------------
Your entry signals (always-on, IV/RV, xCCY) decide *when the carry looks good*.
None of them know *what regime vol is in*, so a "sell" can fire right as vol is
breaking higher

This module generates a **gate**: a 0/1 series aligned to your signal where

    1 = regime is OK, let the short-vol entry through
    0 = veto this entry (vol is high and/or rising — stand aside)

You combine it with any existing signal via `apply_regime_gate`, because
run_signal_backtest only cares whether the signal is truthy on a given day
(backtest_signal.py:146 -> `if not bool(sig.iloc[i]): continue`). So

    filtered = apply_regime_gate(signal_series, gate)

drops straight back into run_signal_backtest with nothing else changed.

Three layers, so you can work at whichever one you need
-------------------------------------------------------
1. `check_*(series, ...) -> bool Series`   pure, testable, plottable on its own.
2. `build_check_panel(...) -> DataFrame`   every selected check as one boolean
   column, on a common index. Built ONCE per (pair, tenor, params) and cached,
   so enumerating dozens of gate combinations costs one Bloomberg pull, not dozens.
   `gate_from_panel(panel, combine=...)` then collapses columns -> 0/1 gate.
3. `GateSpec` + `gated(signal_fn, spec)`   a named, reusable gate config that
   wraps any ComboSpec-style signal_fn. This is the layer the grid uses.

Adding a new regime filter (the whole recipe)
---------------------------------------------
    # a) write the pure check — series in, bool series out (True = OK to sell)
    def check_myfilter(iv, thresh=1.5):
        vv = iv.rolling(10).std()
        return _allow_on_warmup(vv <= thresh, vv.notna())

    # b) register an adapter so gates can refer to it as 'myfilter'
    @register_check('myfilter', defaults={'thresh': 1.5},
                    lookback=lambda p: 10)          # trailing days it needs
    def _reg_myfilter(ctx, thresh=1.5):
        return check_myfilter(ctx.iv, thresh=thresh)

That is it — `'myfilter'` is now valid in `GateSpec(checks=...)`, is picked up
automatically by `enumerate_gate_specs()`, is param-overridable via
`GateSpec(params={'myfilter': {'thresh': 2.0}})`, and is cached like the rest.
No edits anywhere else. `ctx` exposes `ctx.iv`, `ctx.iv_back`,
`ctx.iv_at(tenor)`, `ctx.rv_at(tenor)`, `ctx.rv_comp(tenors)` plus
`ctx.pair / tenor / back_tenor / days_back`, all cached, so a new check never
adds a Bloomberg round-trip it doesn't need.

Combining checks
----------------
`combine='all'` (default)  ok only if EVERY check says ok — strictest, and the
                           one that starves your sample fastest.
`combine='any'`            ok if ANY check says ok — vetoes only on unanimity.
`combine='k'`, k=2         veto once >= k checks veto. The usable middle for
                           wide combinations; k=1 is identical to 'all'.

No look-ahead
-------------
Every check uses ONLY trailing windows (rolling MA / percentile / z-score).
During each check's warm-up period (before it has enough data to positively
identify a bad regime) the default is to ALLOW: absence of evidence is not
evidence of a bad regime, and we don't want to silently veto a third of the
sample. Flip `on_missing='veto'` in apply_regime_gate if you'd rather be
conservative.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import pandas as pd

from xbbg import blp


# ═══════════════════════════════════════════════════════════════════════════
# Data pull (local + light: the simple gates must not require statsmodels)
# ═══════════════════════════════════════════════════════════════════════════

def _pull_iv(pair: str, tenors: Sequence[str], days_back: int) -> pd.DataFrame:
    """
    Batched ATM implied-vol pull for one pair across several tenors.
    Columns come back named f"{pair}_V{tenor}" (e.g. 'AUDUSD_V1W'), matching
    the convention in Implied_Realized.get_ImplVol / vol_regime.get_ImplVol.
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    tickers, col_map = [], {}
    for tenor in tenors:
        ticker = f"{pair}V{tenor} Curncy"
        tickers.append(ticker)
        col_map[ticker] = f"{pair}_V{tenor}"
    df = blp.bdh(tickers=tickers, flds="PX_LAST",
                 start_date=start_date, end_date=end_date)
    df.columns = [col_map[c[0]] for c in df.columns]
    return df.astype(float).sort_index()


def _pull_rv(pair: str, tenors: Sequence[str], days_back: int) -> pd.DataFrame:
    """
    Batched Bloomberg realized-vol pull for one pair across several tenors
    (the `H` family, e.g. 'AUDUSDH1W Curncy'). Columns come back named
    f"{pair}_H{tenor}" (e.g. 'AUDUSD_H1W'), matching vol_regime.get_RealVol.
    These are already-annualized realized vols over each trailing window, so
    they slot straight into HAR as the daily/weekly/monthly components without
    any per-day return math.
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    tickers, col_map = [], {}
    for tenor in tenors:
        ticker = f"{pair}H{tenor} Curncy"
        tickers.append(ticker)
        col_map[ticker] = f"{pair}_H{tenor}"
    df = blp.bdh(tickers=tickers, flds="PX_LAST",
                 start_date=start_date, end_date=end_date)
    df.columns = [col_map[c[0]] for c in df.columns]
    return df.astype(float).sort_index()


def _days_needed(lookback_trading: int, days_back: int,
                 holiday_buffer_pct: float = 0.15) -> int:
    """Trading-day lookback -> calendar days of history to pull (mirrors
    Implied_Realized.compute_days_needed) plus the reporting window."""
    lookback_calendar = int(lookback_trading * 7 / 5 * (1 + holiday_buffer_pct))
    return lookback_calendar + days_back


def _allow_on_warmup(ok: pd.Series, valid: pd.Series) -> pd.Series:
    """Where `valid` is False (not enough data yet), force the check to ALLOW
    (True) instead of leaking a NaN/False veto. `ok` and `valid` share an index."""
    return ok.where(valid, other=True).astype(bool)


# ═══════════════════════════════════════════════════════════════════════════
# Caches — the reason a many-combination sweep is cheap
# ═══════════════════════════════════════════════════════════════════════════
# _SERIES_CACHE : (kind, pair, tenor, hist_days)                    -> vol series
# _CHECK_CACHE  : (check, pair, tenor, back_tenor, hist_days, params) -> bool series
#
# Both are keyed on `hist_days`, which build_check_panel derives from ALL
# registered checks rather than just the selected ones, so that every subset of
# checks shares one pull and one computation. See _hist_budget.

_SERIES_CACHE: Dict[tuple, pd.Series] = {}
_CHECK_CACHE:  Dict[tuple, pd.Series] = {}


def clear_regime_cache() -> None:
    """Drop all cached pulls and check results (call after a data refresh, or
    after re-registering a check with overwrite=True)."""
    _SERIES_CACHE.clear()
    _CHECK_CACHE.clear()


def regime_cache_info() -> Dict[str, int]:
    """{'series': n_pulls_cached, 'checks': n_check_results_cached}"""
    return {'series': len(_SERIES_CACHE), 'checks': len(_CHECK_CACHE)}


def _series(kind: str, pair: str, tenors: Sequence[str],
            hist_days: int) -> Dict[str, pd.Series]:
    """{tenor: series} for kind 'V' (implied) or 'H' (realized), pulling only
    the tenors not already cached — misses go out as ONE batched bdh call."""
    assert kind in ('V', 'H'), "kind must be 'V' (implied) or 'H' (realized)"
    tenors  = list(dict.fromkeys(tenors))
    missing = [t for t in tenors if (kind, pair, t, hist_days) not in _SERIES_CACHE]
    if missing:
        puller = _pull_iv if kind == 'V' else _pull_rv
        df = puller(pair, missing, hist_days)
        for t in missing:
            _SERIES_CACHE[(kind, pair, t, hist_days)] = \
                df[f"{pair}_{kind}{t}"].astype(float).dropna()
    return {t: _SERIES_CACHE[(kind, pair, t, hist_days)] for t in tenors}


# ═══════════════════════════════════════════════════════════════════════════
# Check context — what every registered check gets handed
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CheckCtx:
    """Data access + config for one (pair, tenor) regime read. Every accessor
    is cached, so a check can ask for whatever series it needs without ever
    causing a duplicate Bloomberg round-trip."""
    pair:       str
    tenor:      str
    back_tenor: str
    days_back:  int
    hist_days:  int

    # --- implied vol ------------------------------------------------------
    def iv_at(self, tenor: str) -> pd.Series:
        return _series('V', self.pair, [tenor], self.hist_days)[tenor]

    @property
    def iv(self) -> pd.Series:
        """Front implied vol — the tenor this gate is reading."""
        return self.iv_at(self.tenor)

    @property
    def iv_back(self) -> pd.Series:
        """Back implied vol — the term-structure comparison tenor."""
        return self.iv_at(self.back_tenor)

    # --- realized vol ----------------------------------------------------
    def rv_at(self, tenor: str) -> pd.Series:
        return _series('H', self.pair, [tenor], self.hist_days)[tenor]

    def rv_comp(self, tenors: Sequence[str]) -> pd.DataFrame:
        """Realized-vol components as a frame, one batched pull, columns in the
        order given (HAR wants exactly three: short, medium, long)."""
        d = _series('H', self.pair, tenors, self.hist_days)
        return pd.concat([d[t].rename(t) for t in tenors], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# Check registry — one entry per named filter; add to it, never edit the panel
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CheckDef:
    """
    name     : the string you use in GateSpec(checks=...)
    fn       : fn(ctx, **params) -> boolean Series (True = regime OK)
    defaults : this check's tunable params and their defaults. Anything not
               listed here is rejected as a typo when passed via GateSpec.params.
    lookback : fn(params) -> trailing TRADING days of warm-up the check needs.
               Used to size the history pull; overstate rather than understate.
    """
    name:     str
    fn:       Callable[..., pd.Series]
    defaults: Dict[str, Any] = field(default_factory=dict)
    lookback: Callable[[Mapping[str, Any]], int] = lambda p: 1
    doc:      str = ''


_REGISTRY: Dict[str, CheckDef] = {}


def register_check(name: str, *, defaults: Optional[Mapping[str, Any]] = None,
                   lookback: Optional[Callable[[Mapping[str, Any]], int]] = None,
                   doc: str = '', overwrite: bool = False):
    """
    Decorator registering `fn(ctx, **params) -> bool Series` as a named check.
    See the module docstring for the two-step recipe. `overwrite=True` lets you
    re-register the same name (handy when iterating in a notebook — follow it
    with clear_regime_cache() so stale results are dropped).
    """
    def deco(fn):
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"check '{name}' already registered "
                             f"(pass overwrite=True to replace it)")
        _REGISTRY[name] = CheckDef(
            name=name, fn=fn, defaults=dict(defaults or {}),
            lookback=lookback or (lambda p: 1),
            doc=doc or (fn.__doc__ or '').strip().split('\n')[0])
        return fn
    return deco


def list_checks() -> List[str]:
    """Every registered check name, in registration order."""
    return list(_REGISTRY)


def check_defaults(name: str) -> Dict[str, Any]:
    """This check's tunable params and their default values."""
    return dict(_REGISTRY[name].defaults)


def describe_checks() -> None:
    """Print the registry — name, warm-up, params, one-line doc."""
    print(f"\n{'=' * 88}\n  REGISTERED REGIME CHECKS\n{'=' * 88}")
    for n, d in _REGISTRY.items():
        try:
            lb = f"{int(d.lookback(d.defaults))}d"
        except Exception:
            lb = '?'
        prm = ', '.join(f"{k}={v!r}" for k, v in d.defaults.items()) or '-'
        print(f"  {n:<14} warmup~{lb:<7} {prm}")
        if d.doc:
            print(f"  {'':<14} {d.doc}")
    print('=' * 88)


def _validate_checks(checks: Sequence[str]) -> Tuple[str, ...]:
    bad = [c for c in checks if c not in _REGISTRY]
    if bad:
        raise ValueError(f"unknown check(s) {bad}; valid: {list_checks()}")
    return tuple(dict.fromkeys(checks))              # dedupe, keep order


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# Individual checks — pure: series in, boolean (True = safe/OK) series out
# Each is followed by its 3-line registry adapter.
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════


def check_trend(iv: pd.Series, ma_window: int = 5,
                min_periods: Optional[int] = None) -> pd.Series:
    """
    Vol-trend veto.

    True (OK) where front IV is at or below its own trailing
    moving average — (vol is NOT in an uptrend)
        - it stops you selling into a rising tape.

        ok = iv <= rolling_mean(iv, ma_window)
    """
    mp = min_periods if min_periods is not None else max(5, ma_window // 2)
    ma = iv.rolling(ma_window, min_periods=mp).mean()
    ok = iv <= ma
    return _allow_on_warmup(ok, ma.notna())


@register_check('trend', defaults={'ma_window': 20},
                lookback=lambda p: int(p['ma_window']),
                doc='front IV at or below its own trailing MA (not trending up)')
def _reg_trend(ctx: CheckCtx, ma_window: int = 20) -> pd.Series:
    return check_trend(ctx.iv, ma_window=ma_window)


def check_momentum(iv: pd.Series, change_days: int = 5) -> pd.Series:
    """
    Momentum veto.

    True (OK) where IV has NOT risen over the last `change_days`
    (`iv.diff(change_days) <= 0`). Sharper / more reactive than check_trend;
    use one or the other, or both for a stricter gate.
    """
    chg = iv.diff(change_days)
    ok = chg <= 0
    return _allow_on_warmup(ok, chg.notna())


@register_check('momentum', defaults={'change_days': 5},
                lookback=lambda p: int(p['change_days']) + 1,
                doc='front IV has not risen over the last `change_days`')
def _reg_momentum(ctx: CheckCtx, change_days: int = 5) -> pd.Series:
    return check_momentum(ctx.iv, change_days=change_days)


# -------------------------------------------------------------------------------
# Look at shift in term structure - if extreme at all to signal holdoff in short vol entry
# -------------------------------------------------------------------------------

def check_termstructure(iv_front: pd.Series, iv_back: pd.Series,
                        buffer: float = 0.0) -> pd.Series:
    """
    Term-structure inversion veto.

    True (OK) where the vol curve is NOT
    backwardated (front <= back + buffer). Front-over-back inversion is a
    classic stress tell — the market is pricing near-term panic — and a clean,
    hard-to-overfit reason to not sell front vol.

    `buffer` (in vol points) lets you require the curve to be inverted by more
    than a token amount before vetoing, e.g. buffer=0.25.
    """
    pair = pd.concat([iv_front.rename("f"), iv_back.rename("b")], axis=1)
    ok = pair["f"] <= pair["b"] + buffer
    ok = ok.reindex(iv_front.index)
    return _allow_on_warmup(ok, pair["b"].reindex(iv_front.index).notna())


@register_check('termstructure', defaults={'buffer': 0.0},
                lookback=lambda p: 2,
                doc='vol curve not backwardated (front <= back + buffer)')
def _reg_termstructure(ctx: CheckCtx, buffer: float = 0.0) -> pd.Series:
    if ctx.back_tenor == ctx.tenor:
        raise ValueError(
            f"'termstructure' needs two different tenors, got front={ctx.tenor} "
            f"back={ctx.back_tenor}. Set GateSpec.back_tenor explicitly, or add "
            f"{ctx.tenor!r} to _NEXT_TENOR.")
    return check_termstructure(ctx.iv, ctx.iv_back, buffer=buffer)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


def check_level(iv: pd.Series, lookback: int = 252,
                cap_pct: float = 90.0) -> pd.Series:
    """
    Level-percentile cap

    True (OK) where IV's *level* percentile within its trailing `lookback` window is
    at or below `cap_pct`.

    "don't sell when vol is already in the top decile of its own range."
    """
    pct = iv.rolling(lookback, min_periods=20).rank(pct=True) * 100
    ok = pct <= cap_pct
    return _allow_on_warmup(ok, pct.notna())


@register_check('level', defaults={'lookback': 252, 'cap_pct': 90.0},
                lookback=lambda p: int(p['lookback']),
                doc='front IV level percentile at or below cap_pct')
def _reg_level(ctx: CheckCtx, lookback: int = 252,
               cap_pct: float = 90.0) -> pd.Series:
    return check_level(ctx.iv, lookback=lookback, cap_pct=cap_pct)


def check_spike(iv: pd.Series, z_window: int = 60, z_thresh: float = 2.0,
                cooloff: int = 5) -> pd.Series:
    """
    Spike / vol-of-vol cool-off. Detects a day-over-day IV jump larger than
    `z_thresh` z-scores (z computed on a trailing `z_window` of daily changes)
    and vetoes the spike day plus the next `cooloff` days. Targets vol
    clustering — the "it just popped, more to come" period.

    Point-in-time: the rolling max looks BACKWARD `cooloff` days, so a day is
    vetoed only by spikes at or before it.
    """
    ret = iv.diff()
    mu  = ret.rolling(z_window, min_periods=20).mean()
    sd  = ret.rolling(z_window, min_periods=20).std()
    z   = (ret - mu) / sd
    spike = (z > z_thresh).fillna(False)
    recent_spike = spike.rolling(cooloff + 1, min_periods=1).max().astype(bool)
    return (~recent_spike).astype(bool)


@register_check('spike', defaults={'z_window': 60, 'z_thresh': 2.0, 'cooloff': 5},
                lookback=lambda p: int(p['z_window']) + int(p['cooloff']),
                doc='no >z_thresh IV jump within the last `cooloff` days')
def _reg_spike(ctx: CheckCtx, z_window: int = 60, z_thresh: float = 2.0,
               cooloff: int = 5) -> pd.Series:
    return check_spike(ctx.iv, z_window=z_window, z_thresh=z_thresh,
                       cooloff=cooloff)


# ═══════════════════════════════════════════════════════════════════════════
# HAR realized-vol gate
# ═══════════════════════════════════════════════════════════════════════════

def _har_core(comp: pd.DataFrame, train_window: int = 252, refit_every: int = 5,
              rising_ratio: float = 1.0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Rolling HAR fit + rising-regime test on an ALREADY-PULLED component frame.

    comp : DataFrame with exactly 3 columns (short, medium, long) of annualized
           realized vol. Split out of check_har so the panel builder can feed it
           a cached `ctx.rv_comp` pull instead of re-hitting Bloomberg.

    Returns (ok, detail), UNTRIMMED — the caller picks the reporting window.
    """
    comp = comp.copy()
    comp.columns = ["s", "m", "l"]
    comp = comp.astype(float).clip(lower=1e-6)        # guard the log
    X = np.log(comp).dropna()                         # model in log-vol
    y = np.log(comp["s"]).reindex(X.index).shift(-1)  # next-period short RV (log)
    recent_rv = comp["m"].reindex(X.index)            # recent realized = medium H tenor
    fc = pd.Series(index=X.index, dtype=float)        # one-step-ahead forecast (log-vol)
    Xv = X.values
    beta = None                                       # last successfully fitted coefs
    for i in range(train_window, len(X)):
        # `refit_every` is the COEFFICIENT cadence, not the forecast cadence: the
        # betas are re-estimated every `refit_every` days and carried in between,
        # but the forecast is recomputed EVERY day off that day's own RV features.
        # So a stale gate can only come from stale coefficients, never from stale
        # market data — refit_every=5 means 5-day-old betas on today's vol, not a
        # 5-day-old forecast.
        if (i - train_window) % refit_every == 0:
            Xtr = Xv[i - train_window:i]
            ytr = y.iloc[i - train_window:i].values
            keep = ~np.isnan(ytr)                    # last row NaN from shift(-1)
            Xtr, ytr = Xtr[keep], ytr[keep]
            if len(ytr) >= train_window // 2:        # too thin -> keep the old betas
                A = np.column_stack([np.ones(len(Xtr)), Xtr])
                beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        if beta is None:                             # still warming up, no fit yet
            continue
        xi = np.concatenate([[1.0], Xv[i]])          # features known at close of t
        fc.iloc[i] = xi @ beta

    fc_vol = np.exp(fc)                              # annualized forecast vol, % (IV units)
    fc_vol.name = "har_fc_vol"
    ratio = fc_vol / recent_rv
    ok = ratio <= rising_ratio
    detail = pd.DataFrame({"har_fc_vol": fc_vol, "recent_rv": recent_rv,
                           "ratio": ratio})
    ok = _allow_on_warmup(ok, fc_vol.notna() & recent_rv.notna())
    return ok, detail


def check_har(
    pair:          str,
    tenor:         str   = "1W",
    days_back:     int   = 180,
    *,
    bbg_tenors:    Sequence[str] = ("1W", "1M", "3M"),
    train_window:  int   = 252,
    refit_every:   int   = 5,
    rising_ratio:  float = 1.0,
    hist_days:     Optional[int] = None,
    verbose:       bool  = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    HAR (Heterogeneous AutoRegressive) realized-vol gate.

    Corsi's HAR models the next period's realized vol as a linear function of
    realized vol measured over three horizons — short, medium, long:

        RV_{t+1} = c + β_s·RV_t^(short) + β_m·RV_t^(med) + β_l·RV_t^(long) + ε


    Rising-regime check (no IV needed — pure RV-regime direction)
    ---------------------------------------------------------------
    OK when  HAR_forecast_vol <= rising_ratio · recent_realized_vol.
    Veto when the model predicts realized vol expanding beyond recent levels
    (`recent_realized_vol` is the medium, `bbg_tenors[1]`, H tenor).

    `refit_every` is the COEFFICIENT cadence only — betas are re-estimated every
    `refit_every` days and carried in between, while the forecast is recomputed
    EVERY day off that day's own realized-vol reading. So the gate reacts to new
    vol data daily at any refit_every; only the betas age.

    Returns (ok, detail): `ok` is the boolean gate (True = OK); `detail` carries
    'har_fc_vol', 'recent_rv', and 'ratio' for plotting.
    """
    assert len(bbg_tenors) == 3, "bbg_tenors must be 3 tenors (short, med, long)"
    days = hist_days if hist_days is not None else _days_needed(train_window + 22, days_back)
    d    = _series('H', pair, bbg_tenors, days)
    comp = pd.concat([d[t].rename(t) for t in bbg_tenors], axis=1)
    ok, detail = _har_core(comp, train_window=train_window,
                           refit_every=refit_every, rising_ratio=rising_ratio)

    if len(detail):
        cutoff = detail.index.max() - pd.Timedelta(days=days_back)
        ok, detail = ok[ok.index >= cutoff], detail[detail.index >= cutoff]

    if verbose:
        n, blk = len(ok), int((~ok).sum())
        print(f"[check_har] {pair} {tenor} | {n} days, "
              f"{blk} vetoed ({blk / n:.0%})" if n else "[check_har] empty window")
    return ok, detail










@register_check('har',
                defaults={'bbg_tenors': ('1W', '1M', '3M'), 'train_window': 252,
                          'refit_every': 5, 'rising_ratio': 1.0},
                lookback=lambda p: int(p['train_window']) + 22,
                doc='HAR RV forecast not expanding vs recent realized vol')
def _reg_har(ctx: CheckCtx, bbg_tenors: Sequence[str] = ("1W", "1M", "3M"),
             train_window: int = 252, refit_every: int = 5,
             rising_ratio: float = 1.0) -> pd.Series:
    assert len(bbg_tenors) == 3, "bbg_tenors must be 3 tenors (short, med, long)"
    ok, _ = _har_core(ctx.rv_comp(bbg_tenors), train_window=train_window,
                      refit_every=refit_every, rising_ratio=rising_ratio)
    return ok


# Kept for backward compatibility — prefer list_checks(), which stays live as
# you register more filters.
_ALL_CHECKS = tuple(_REGISTRY)


# ═══════════════════════════════════════════════════════════════════════════
# Tenor plumbing
# ═══════════════════════════════════════════════════════════════════════════
# Which tenor the term-structure check compares the front against when the
# GateSpec doesn't say. Extend as needed.
_NEXT_TENOR = {
    '1D': '1W', '24H': '1W', '1W': '1M', '2W': '1M', '3W': '2M',
    '1M': '3M', '2M': '3M', '3M': '6M', '4M': '6M', '6M': '1Y',
    '9M': '1Y', '1Y': '2Y', '2Y': '3Y',
}


def _resolve_tenors(tenor: str, back_tenor: Optional[str]) -> Tuple[str, str]:
    if back_tenor:
        return tenor, back_tenor
    return tenor, _NEXT_TENOR.get(tenor, tenor)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — the check panel (computed once, sliced many ways)
# ═══════════════════════════════════════════════════════════════════════════

def _freeze(obj):
    """Hashable snapshot of a nested params structure, for cache keys."""
    if isinstance(obj, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple, set)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _hist_budget(days_back: int, params: Mapping[str, Mapping]) -> int:
    """
    Calendar days of history to pull. Deliberately sized off the longest warm-up
    among ALL registered checks, not just the selected ones: that keeps
    `hist_days` — and therefore every cache key — constant across every subset
    of checks, so a many-combination sweep shares ONE pull per (pair, tenor)
    instead of one per subset. Costs one slightly-longer bdh call; saves dozens.
    """
    lbs = [1]
    for name, d in _REGISTRY.items():
        p = {**d.defaults, **dict(params.get(name) or {})}
        try:
            lbs.append(int(d.lookback(p)))
        except Exception:                            # a check with an odd lookback fn
            pass
    return _days_needed(max(lbs), days_back)


def _check_series(name: str, ctx: CheckCtx,
                  params: Mapping[str, Any]) -> pd.Series:
    """One check's boolean series, memoized on (check, pair, tenors, params)."""
    d = _REGISTRY[name]
    merged = {**d.defaults, **dict(params or {})}
    unknown = [k for k in merged if k not in d.defaults]
    if unknown:
        raise ValueError(f"check '{name}' got unknown param(s) {unknown}; "
                         f"valid: {list(d.defaults)}")
    key = (name, ctx.pair, ctx.tenor, ctx.back_tenor, ctx.hist_days,
           _freeze(merged))
    if key not in _CHECK_CACHE:
        _CHECK_CACHE[key] = d.fn(ctx, **merged).astype(bool)
    return _CHECK_CACHE[key]


def build_check_panel(
    pair:        str,
    tenor:       str = '1M',
    back_tenor:  Optional[str] = None,
    days_back:   int = 180,
    checks:      Optional[Sequence[str]] = None,
    params:      Optional[Mapping[str, Mapping]] = None,
    *,
    hist_days:   Optional[int] = None,
    trim:        bool = True,
    verbose:     bool = False,
) -> pd.DataFrame:
    """
    Every selected check as one boolean column (True = regime OK), aligned on
    the front-IV trading-day index and trimmed to the last `days_back` calendar
    days.

    checks : names from list_checks(); None -> every registered check.
    params : per-check overrides, e.g.
             {'level': {'cap_pct': 85}, 'trend': {'ma_window': 10}}

    Results are cached per check, so calling this repeatedly with different
    `checks` subsets — which is exactly what a gate sweep does — is nearly free
    after the first call for a given (pair, tenor, days_back, params).
    """
    checks = _validate_checks(list_checks() if checks is None else checks)
    params = dict(params or {})
    front, back = _resolve_tenors(tenor, back_tenor)
    hd  = hist_days if hist_days is not None else _hist_budget(days_back, params)
    ctx = CheckCtx(pair=pair, tenor=front, back_tenor=back,
                   days_back=days_back, hist_days=hd)

    idx   = ctx.iv.index
    panel = pd.DataFrame(index=idx)
    for name in checks:
        s = _check_series(name, ctx, params.get(name, {}))
        # A check computed off a different index (e.g. HAR on RV dates) is
        # reindexed onto the IV index; no read -> allow, matching _allow_on_warmup.
        panel[name] = s.reindex(idx).where(lambda x: x.notna(), True).astype(bool)

    if trim and len(idx):
        cutoff = idx.max() - pd.Timedelta(days=days_back)
        panel = panel[panel.index >= cutoff]

    panel.attrs.update(pair=pair, tenor=front, back_tenor=back,
                       days_back=days_back, hist_days=hd)
    if verbose:
        n = len(panel)
        print(f"[panel] {pair} {front}/{back} | {n} days | hist {hd}d")
        for c in panel.columns:
            blk = int((~panel[c]).sum())
            print(f"    {c:<14} blocks {blk:>4} day(s)"
                  + (f" ({blk / n:.0%})" if n else ""))
    return panel


def gate_from_panel(panel: pd.DataFrame, combine: str = 'all',
                    k: Optional[int] = None) -> pd.Series:
    """
    Collapse a boolean check panel into a 0/1 gate (1 = OK to sell vol).

    combine :
        'all'  ok only if EVERY check says ok        (veto if any check vetoes)
        'any'  ok if ANY check says ok               (veto only on unanimity)
        'k'    veto once >= `k` checks veto          (k=1 is identical to 'all')

    An empty panel (no checks) is all-ones: a gate that vetoes nothing.
    """
    assert combine in ('all', 'any', 'k'), "combine must be 'all', 'any' or 'k'"
    n_cols = len(panel.columns)
    if n_cols == 0:
        return pd.Series(1, index=panel.index, dtype=int)
    if combine == 'all':
        ok = panel.all(axis=1)
    elif combine == 'any':
        ok = panel.any(axis=1)
    else:
        if k is None:
            raise ValueError("combine='k' requires k (veto once >= k checks veto)")
        if not 1 <= int(k) <= n_cols:
            raise ValueError(f"k must be in 1..{n_cols} for {n_cols} check(s), got {k}")
        ok = (~panel).sum(axis=1) < int(k)
    return ok.astype(int)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — GateSpec: a named, reusable gate configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GateSpec:
    """
    ONE fully-specified regime gate — the unit you sweep over, exactly as
    ComboSpec is the unit the grid sweeps over.

        GateSpec(('trend',))                                    # single check
        GateSpec(('trend', 'har'))                              # both must pass
        GateSpec(('trend','level','spike'), combine='k', k=2)    # 2-of-3 veto
        GateSpec(('level',), params={'level': {'cap_pct': 80}})  # retuned
        GateSpec(('trend',), tenor='1W')                        # read FRONT vol
        GateSpec(())                                            # baseline, no veto

    checks     : names from list_checks()
    combine    : 'all' | 'any' | 'k'   (see gate_from_panel)
    k          : required when combine='k'
    params     : {check_name: {param: value}} overrides
    tenor      : which IV tenor the regime is read on. None (default) = the
                 tenor being TRADED. Set '1W' to always read front vol instead,
                 regardless of the trade tenor.
    back_tenor : term-structure comparison tenor. None -> _NEXT_TENOR[tenor].
    on_missing : dates the gate doesn't cover -> 'allow' (1) or 'veto' (0).
    name       : display label (this is what shows up on the grid axis);
                 auto-derived from the config if omitted.
    """
    checks:     Tuple[str, ...] = ()
    combine:    str = 'all'
    k:          Optional[int] = None
    params:     Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tenor:      Optional[str] = None
    back_tenor: Optional[str] = None
    on_missing: str = 'allow'
    name:       Optional[str] = None

    def __post_init__(self):
        if isinstance(self.checks, str):             # GateSpec('trend') convenience
            self.checks = (self.checks,)
        self.checks = _validate_checks(self.checks)
        assert self.combine in ('all', 'any', 'k'), \
            "combine must be 'all', 'any' or 'k'"
        assert self.on_missing in ('allow', 'veto'), \
            "on_missing must be 'allow' or 'veto'"
        if self.combine == 'k':
            if self.k is None:
                raise ValueError("combine='k' requires k")
            if not 1 <= int(self.k) <= len(self.checks):
                raise ValueError(f"k must be in 1..{len(self.checks)}, got {self.k}")
        bad = [c for c in self.params if c not in _REGISTRY]
        if bad:
            raise ValueError(f"params given for unknown check(s) {bad}; "
                             f"valid: {list_checks()}")
        if self.name is None:
            self.name = self._auto_name()

    def _auto_name(self) -> str:
        if not self.checks:
            return 'none'
        body = '+'.join(self.checks)
        if self.combine == 'any':
            body = f"any:{body}"
        elif self.combine == 'k':
            body = f"{self.k}of{len(self.checks)}:{body}"
        if self.params:
            body += '*'                              # marks retuned params
        if self.tenor:
            body += f"@{self.tenor}"
        return body

    @property
    def label(self) -> str:
        return self.name or self._auto_name()

    def key(self) -> tuple:
        """Config identity — used to de-dupe enumerated sweeps."""
        return (self.checks, self.combine, self.k, _freeze(self.params),
                self.tenor, self.back_tenor, self.on_missing)


NO_GATE = GateSpec(name='none')          # ungated baseline, for side-by-sides


def build_gate(spec: GateSpec, pair: str, trade_tenor: str, days_back: int,
               *, return_detail: bool = False, verbose: bool = False
               ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
    """
    Realize a GateSpec into a 0/1 series for one (pair, trade_tenor, days_back).

    The gate reads `spec.tenor` if set, otherwise the tenor being traded.
    return_detail=True -> (gate, panel + a 'GATE' column) so you can see exactly
    which check vetoed which day.
    """
    tenor = spec.tenor or trade_tenor
    panel = build_check_panel(pair, tenor=tenor, back_tenor=spec.back_tenor,
                              days_back=days_back, checks=spec.checks,
                              params=spec.params, verbose=verbose)
    gate = gate_from_panel(panel, combine=spec.combine, k=spec.k)
    gate.name = f"{pair}_{tenor}_{spec.label}"

    if verbose:
        n, blk = len(gate), int((gate == 0).sum())
        print(f"[gate] {pair} {tenor} {spec.label} | {n} days, "
              f"{blk} vetoed ({blk / n:.0%})" if n else "[gate] empty window")
    if return_detail:
        detail = panel.copy()
        detail['GATE'] = gate
        return gate, detail
    return gate


def apply_regime_gate(signal: pd.Series, gate: pd.Series,
                      on_missing: str = "allow", warn: bool = True) -> pd.Series:
    """
    AND a 0/1 regime `gate` onto a 0/1 entry `signal`, aligned on the signal's
    index. Returns a 0/1 series with the signal's name, ready to hand straight
    to run_signal_backtest.

        filtered = apply_regime_gate(signal_series, gate)

    on_missing : how to treat signal dates the gate doesn't cover
        'allow' (default) -> 1 (don't veto where we have no regime read)
        'veto'            -> 0 (conservative: no read => stand aside)

    warn : print a warning when the gate covers < 95% of the days the signal
        actually FIRES on. Without this, a gate built over a shorter window than
        the signal silently no-ops across most of the sample and reads as "the
        regime filter did nothing" — the easiest way to fool yourself here.
    """
    assert on_missing in ("allow", "veto"), "on_missing must be 'allow' or 'veto'"
    fill = 1 if on_missing == "allow" else 0
    g = gate.reindex(signal.index)

    if warn:
        fires  = signal.astype(bool)
        n_fire = int(fires.sum())
        if n_fire:
            cov = float(g[fires].notna().mean())
            if cov < 0.95:
                print(f"[apply_regime_gate] WARNING: gate "
                      f"{getattr(gate, 'name', 'gate')} covers only {cov:.0%} of "
                      f"the {n_fire} signal-firing days; the rest default to "
                      f"'{on_missing}'. Build the gate with a days_back that "
                      f"matches the signal's window.")

    g = g.fillna(fill).astype(int)
    out = (signal.astype(int) * g).astype(int)
    out.name = getattr(signal, "name", "signal")
    return out


def gated(signal_fn: Callable, spec: GateSpec, *, verbose: bool = False,
          warn: bool = True) -> Callable:
    """
    Wrap a ComboSpec-style signal_fn (`fn(combo_spec) -> Series`) with a regime gate.

        _straddle_run(gated(_sig_ivrv, GateSpec(('trend', 'har'))), ...)

    The returned callable has the same shape, so it drops into
    ComboSpec(signal_fn=...) with nothing else changed. The gate is built for
    THAT combo's pair / tenor / days_back, so the gate window always matches the
    signal's. `.gate_spec` / `.base_signal_fn` are attached for introspection.
    """
    def _fn(s):
        sig = signal_fn(s)
        if not spec.checks:
            return sig
        gate = build_gate(spec, s.pair, s.tenor, s.days_back, verbose=verbose)
        return apply_regime_gate(sig, gate, on_missing=spec.on_missing, warn=warn)

    base = getattr(signal_fn, '__name__', 'signal')
    _fn.__name__ = f"{base}|{spec.label}"
    _fn.__doc__  = f"{base} gated by {spec.label}"
    _fn.gate_spec = spec
    _fn.base_signal_fn = signal_fn
    return _fn


# ═══════════════════════════════════════════════════════════════════════════
# Enumerating gate variants — "each on its own, then in combination"
# ═══════════════════════════════════════════════════════════════════════════

def enumerate_gate_specs(
    checks:       Optional[Sequence[str]] = None,
    sizes:        Sequence[int] = (1,),
    *,
    combine:      str = 'all',
    k:            Optional[Union[int, Callable[[int], int]]] = None,
    params:       Optional[Mapping[str, Mapping]] = None,
    tenor:        Optional[str] = None,
    back_tenor:   Optional[str] = None,
    on_missing:   str = 'allow',
    include_none: bool = True,
) -> List[GateSpec]:
    """
    Every combination of `checks` at each size in `sizes`, as GateSpecs.

        enumerate_gate_specs(sizes=(1,))              # each check on its own
        enumerate_gate_specs(sizes=(1, 2))            # singles + all pairs
        enumerate_gate_specs(['trend','level','spike'], sizes=(3,),
                             combine='k', k=2)        # 2-of-3 veto

    checks : None -> every registered check, so a filter you add later is picked
             up automatically with no edit here.
    k      : int, or a callable size -> k. With combine='k' and k=None, defaults
             to a majority: ceil(size/2), floored at 2.
    include_none : prepend the ungated baseline so every sweep is self-comparing.

    Size-1 combos are skipped for combine='k'/'any', since both degenerate to
    'all' on a single check and would just duplicate the singles.
    """
    names = _validate_checks(list_checks() if checks is None else checks)
    out: List[GateSpec] = []
    if include_none:
        out.append(GateSpec(name='none'))

    for size in sizes:
        if size < 1 or size > len(names):
            continue
        if size == 1 and combine != 'all':
            continue                                  # degenerate; see docstring
        for combo in combinations(names, size):
            kk = None
            if combine == 'k':
                kk = k(size) if callable(k) else k
                if kk is None:
                    kk = max(2, -(-size // 2))        # ceil(size/2), min 2
                kk = min(int(kk), size)
            out.append(GateSpec(checks=combo, combine=combine, k=kk,
                                params=dict(params or {}), tenor=tenor,
                                back_tenor=back_tenor, on_missing=on_missing))

    seen, uniq = set(), []                            # de-dupe by config, not name
    for g in out:
        if g.key() not in seen:
            seen.add(g.key())
            uniq.append(g)
    return uniq


def describe_gate_specs(specs: Sequence[GateSpec]) -> None:
    """Print what a sweep is about to run — one line per gate."""
    print(f"\n{'=' * 84}\n  {len(specs)} GATE VARIANT(S)\n{'=' * 84}")
    for i, g in enumerate(specs, 1):
        tn = g.tenor or '<trade tenor>'
        ck = ', '.join(g.checks) or '-'
        kk = f" k={g.k}" if g.combine == 'k' else ''
        pm = f"  params={dict(g.params)}" if g.params else ''
        print(f"  {i:>3}. {g.label:<32} combine={g.combine}{kk:<6} "
              f"tenor={tn:<14} [{ck}]{pm}")
    print('=' * 84)


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible one-shot builder
# ═══════════════════════════════════════════════════════════════════════════

def make_regime_gate(
    pair:            str,
    tenor:           str = "1W",
    back_tenor:      str = "1M",
    days_back:       int = 180,
    checks:          Sequence[str] = ("trend", "termstructure"),
    *,
    ma_window:       int   = 20,
    change_days:     int   = 5,
    ts_buffer:       float = 0.0,
    level_lookback:  int   = 252,
    level_cap_pct:   float = 90.0,
    spike_z_window:  int   = 60,
    spike_z_thresh:  float = 2.0,
    spike_cooloff:   int   = 5,
    har_bbg_tenors:  Sequence[str] = ("1W", "1M", "3M"),
    har_train_window: int  = 252,
    har_refit_every: int   = 5,
    har_rising_ratio: float = 1.0,
    combine:         str   = 'all',
    k:               Optional[int] = None,
    return_detail:   bool  = False,
    verbose:         bool  = False,
) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
    """
    Build a 0/1 regime gate for `pair` over the last `days_back` days by
    combining the selected `checks`. 1 = OK to sell vol, 0 = veto.

    This is the flat-keyword front door, kept for ad-hoc/one-off inspection. For
    sweeps prefer GateSpec + build_gate (or gated(...) straight into a
    ComboSpec), which name the config, cache it, and enumerate cleanly.

    checks : any subset of list_checks() —
        'trend'         check_trend         (front IV vs its MA)
        'momentum'      check_momentum      (front IV n-day change)
        'termstructure' check_termstructure (front `tenor` vs `back_tenor`)
        'level'         check_level         (front IV level percentile cap)
        'spike'         check_spike         (front IV jump cool-off)
        'har'           check_har           (HAR RV forecast rising vs recent RV)
      ...plus anything you have registered via @register_check.

    combine / k : see gate_from_panel. Default 'all' = veto if ANY check vetoes.

    return_detail=True -> (gate, detail) where `detail` is a DataFrame with one
    boolean column per check plus 'GATE', so you can see exactly which check
    vetoed which day (great for plotting alongside the vol series).
    """
    flat = {
        'trend':         {'ma_window': ma_window},
        'momentum':      {'change_days': change_days},
        'termstructure': {'buffer': ts_buffer},
        'level':         {'lookback': level_lookback, 'cap_pct': level_cap_pct},
        'spike':         {'z_window': spike_z_window, 'z_thresh': spike_z_thresh,
                          'cooloff': spike_cooloff},
        'har':           {'bbg_tenors': tuple(har_bbg_tenors),
                          'train_window': har_train_window,
                          'refit_every': har_refit_every,
                          'rising_ratio': har_rising_ratio},
    }
    spec = GateSpec(checks=tuple(checks), combine=combine, k=k,
                    params={c: flat[c] for c in checks if c in flat},
                    tenor=tenor, back_tenor=back_tenor)
    return build_gate(spec, pair, tenor, days_back,
                      return_detail=return_detail, verbose=verbose)


__all__ = [
    # pure checks
    'check_trend', 'check_momentum', 'check_termstructure', 'check_level',
    'check_spike', 'check_har',
    # registry — how you add more filters
    'CheckCtx', 'CheckDef', 'register_check', 'list_checks', 'check_defaults',
    'describe_checks',
    # panel / gate
    'build_check_panel', 'gate_from_panel', 'GateSpec', 'NO_GATE', 'build_gate',
    'apply_regime_gate', 'gated', 'enumerate_gate_specs', 'describe_gate_specs',
    'make_regime_gate',
    # caching / plumbing
    'clear_regime_cache', 'regime_cache_info',
]


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USES / smoke test — runs ONLY when this file is executed directly,
# never on import (importing this module must not hit Bloomberg).
# ═══════════════════════════════════════════════════════════════════════════

def _demo(pair: str = 'EURUSD', tenor: str = '1M', days_back: int = 180) -> None:
    describe_checks()

    # --- one check in isolation (pure function, registry not involved) -------
    iv_df = _pull_iv(pair, [tenor, '3M'], days_back=400)
    iv    = iv_df[f'{pair}_V{tenor}'].dropna()
    print(check_trend(iv, ma_window=20).tail(10))                     # True = OK
    print(check_termstructure(iv, iv_df[f'{pair}_V3M'].dropna()).tail(10))

    # --- the full panel: every registered check, one pull -------------------
    panel = build_check_panel(pair, tenor=tenor, days_back=days_back, verbose=True)
    print(panel.tail(15))

    # --- slice that panel into gates, three ways (all cache hits) -----------
    for spec in [GateSpec(('trend',)),
                 GateSpec(('trend', 'termstructure')),
                 GateSpec(('trend', 'level', 'spike'), combine='k', k=2)]:
        gate = build_gate(spec, pair, tenor, days_back, verbose=True)
        print(f"  {spec.label:<28} last 5: {list(gate.tail(5).values)}")

    # --- HAR on its own, with its detail frame ------------------------------
    ok, det = check_har(pair, tenor=tenor, days_back=days_back,
                        bbg_tenors=('24H', '1W', '1M'), train_window=180,
                        rising_ratio=1.0, verbose=True)
    print(det.tail(10))

    # --- what a sweep would run --------------------------------------------
    describe_gate_specs(enumerate_gate_specs(sizes=(1, 2)))
    print(f"\ncache: {regime_cache_info()}")


# if __name__ == '__main__':
#     _demo()



























































# -------------------------------------------------------------------------------------------------
# ------------------------------- HAR MODEL ANALYSIS ----------------------------------------------
# -------------------------------------------------------------------------------------------------



# ------------ SUMMARIZER STATS of HAR MODEL ------------

def run_har_config(pair, tenor, days_back, bbg_tenors, train_window, rising_ratio, refit_every, verbose=False,):
    ok, det = check_har(
        pair,
        tenor=tenor,
        days_back=days_back,
        bbg_tenors=bbg_tenors,
        train_window=train_window,
        rising_ratio=rising_ratio,
        refit_every=refit_every,
        verbose=verbose,)
    det = det.copy()
    # Only evaluate dates where HAR forecast exists
    valid = (
        det["har_fc_vol"].notna()
        & det["recent_rv"].notna()
        & det["ratio"].notna())
    d = det.loc[valid].copy()
    gate = ok.reindex(d.index)
    # Number of times the gate flips True <-> False
    gate_switches = (
        gate.astype(int)
        .diff()
        .abs()
        .sum())
    # Number of times forecast itself changes
    forecast_changes = (
        d["har_fc_vol"]
        .diff()
        .abs()
        .gt(1e-12)
        .sum())
    summary = {
        "train_window": train_window,
        "rising_ratio": rising_ratio,
        "refit_every": refit_every,
        "n_days": len(d),
        "n_veto": int((~gate).sum()),
        "veto_rate": (~gate).mean(),
        "coverage": gate.mean(),
        "gate_switches": int(gate_switches),
        "forecast_changes": int(forecast_changes),
        "avg_har_fc": d["har_fc_vol"].mean(),
        "std_har_fc": d["har_fc_vol"].std(),
        "avg_recent_rv": d["recent_rv"].mean(),
        "avg_ratio": d["ratio"].mean(),
        "std_ratio": d["ratio"].std(),
        "ratio_p10": d["ratio"].quantile(0.10),
        "ratio_p50": d["ratio"].quantile(0.50),
        "ratio_p90": d["ratio"].quantile(0.90),}
    return pd.Series(summary), ok, det



# base_summary, base_ok, base_det = run_har_config(
#     pair="EURUSD",
#     tenor="1M",
#     days_back=365,
#     bbg_tenors=("24H", "1W", "1M"),
#     train_window=180,
#     rising_ratio=1.0,
#     refit_every=1,
# )

# print(base_summary)












# pair          = 'EURUSD'
# tenor         = '1M'
# days_back     = 365
# bbg_tenors    = ('24H', '1W', '1M')
# train_window  = 180
# rising_ratio  = 1.0
# refit_every   = 5



# ok, det = check_har(pair, 
#                     tenor=tenor, 
#                     days_back=days_back,
#                     bbg_tenors=bbg_tenors, 
#                     train_window=train_window,
#                     rising_ratio=rising_ratio, 
#                     refit_every=refit_every,
#                     verbose=True)



# print(det.tail(30))








