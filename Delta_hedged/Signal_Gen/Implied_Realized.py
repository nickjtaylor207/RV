"""
Different Vol Signals:
CCY    = Some Currency Pair
Tenor  = [1W, 2W, 3W, 1M, 2M, 3M, 6M, 9M, 1Y]
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Optional
from itertools import permutations
from scipy.stats import percentileofscore

from xbbg import blp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# (currency_list, tenors, days_back) -> the pulled frame. Every signal builder
# routes through get_ImplRealVol, and a grid calls signal_fn once per cell, so an
# N-column sweep on one pair/tenor otherwise makes N identical bdh calls. Keyed on
# the same args, so a different pair/tenor/window still pulls.
_IVRV_CACHE: dict = {}


def clear_ivrv_cache() -> None:
    """Drop cached get_ImplRealVol pulls (after a data refresh)."""
    _IVRV_CACHE.clear()


def get_ImplRealVol(currency_list, tenors, days_back):
    """
    Batched implied + realized ATM vol pull, MEMOIZED on its arguments — see
    _IVRV_CACHE. Returns a COPY, so a caller mutating the frame cannot poison the
    cache for the next one.
    """
    key = (tuple(currency_list), tuple(tenors), int(days_back))
    if key in _IVRV_CACHE:
        return _IVRV_CACHE[key].copy()

    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    vol_types  = ["H", "V"]
    # Build every ticker up front and pull them in ONE batched bdh call
    # (same pattern as data.pull_fx_vol_surface) instead of a round-trip per ticker.
    tickers  = []
    col_map  = {}   # ticker -> final column name
    for tenor in tenors:
        for ccy in currency_list:
            for vt in vol_types:
                ticker = f"{ccy}{vt}{tenor} Curncy"
                tickers.append(ticker)
                col_map[ticker] = f"{ccy}_{vt}{tenor}"
    df = blp.bdh(
        tickers=tickers,
        flds="PX_LAST",
        start_date=start_date,
        end_date=end_date)
    # bdh returns a (ticker, field) MultiIndex on the columns; map the ticker
    # level back to our clean names. Missing tickers simply don't appear.
    df.columns = [col_map[c[0]] for c in df.columns]
    out = df.astype(float).sort_index()
    _IVRV_CACHE[key] = out
    return out.copy()

def get_Allvol_df(ccys, tenors, days_back):
    include_iv=True; include_hv=True
    df = get_ImplRealVol(ccys, tenors, days_back)
    out = {}   
    for ccy in ccys:
        for tenor in tenors:
            vcol = f"{ccy}_V{tenor}"
            hcol = f"{ccy}_H{tenor}"
            has_v = vcol in df.columns
            has_h = hcol in df.columns
            if has_v and has_h:
                pair = df[[vcol, hcol]]
                pair = pair.dropna() 
                spread = pair[vcol].astype(float) - pair[hcol].astype(float)
                out[("VDiff", tenor, ccy)] = spread
                if include_iv and has_v:
                    iv = df[vcol].astype(float).reindex(spread.index)
                    out[("IV", tenor, ccy)] = iv
                if include_hv and has_h:
                    hv = df[hcol].astype(float).reindex(spread.index) 
                    out[("HV", tenor, ccy)] = hv
            else:
                if include_iv and has_v:
                    iv = df[vcol].astype(float)
                    iv = iv.dropna()
                    out[("IV", tenor, ccy)] = iv
    res = pd.concat(out, axis=1).sort_index()
    res.columns = [f"{ccy}_{tenor}_{metric}" for (metric, tenor, ccy) in res.columns]
    return res

# Raw ticker vol-type letter -> display metric label.
_VOL_TYPE_LABEL = {'V': 'IV', 'H': 'RV'}


def build_vol_spread(ccys, leg_a, leg_b, days_back, label='VDiff'):
    """
    Generic two-leg vol spread: leg_a - leg_b, where each leg is a
    (vol_type, tenor) pair. vol_type is 'V' (implied) or 'H' (realized).
    The two tenors need NOT match — this is what get_Allvol_df can't express.

        build_vol_spread(ccys, ('V','1M'), ('H','1W'))  # 1M implied  - 1W realized
        build_vol_spread(ccys, ('V','1M'), ('V','1W'))  # IV term structure 1M - 1W
        build_vol_spread(ccys, ('V','1W'), ('H','1W'))  # matched VDiff (same as get_Allvol_df)

    Pulls only the tenors it needs (deduped) in one batched get_ImplRealVol call,
    aligns the two legs on their common dates, and returns a df with one column
    per ccy named
        {ccy}_{IV/RV}{ta}-{IV/RV}{tb}_{label}
    e.g. 'USDJPY_IV1M-RV1W_VDiff'. The trailing _{label} keeps it compatible
    with compute_signals and the get_vol_signal / get_cross_vol_signal squeeze,
    which only key off the suffix.
    """
    a_type, a_tenor = leg_a
    b_type, b_tenor = leg_b
    assert a_type in _VOL_TYPE_LABEL and b_type in _VOL_TYPE_LABEL, \
        "leg vol_type must be 'V' (implied) or 'H' (realized)"

    tenors = list(dict.fromkeys([a_tenor, b_tenor]))   # unique, order-preserving
    raw = get_ImplRealVol(ccys, tenors, days_back)

    out = {}
    for ccy in ccys:
        col_a = f"{ccy}_{a_type}{a_tenor}"
        col_b = f"{ccy}_{b_type}{b_tenor}"
        if col_a not in raw.columns or col_b not in raw.columns:
            continue
        pair = raw[[col_a, col_b]].dropna()   # align on common dates
        if pair.empty:
            continue
        spread = pair[col_a].astype(float) - pair[col_b].astype(float)
        name = (f"{ccy}_{_VOL_TYPE_LABEL[a_type]}{a_tenor}"
                f"-{_VOL_TYPE_LABEL[b_type]}{b_tenor}_{label}")
        out[name] = spread

    if not out:
        raise ValueError(
            f"build_vol_spread: no columns built for ccys={ccys}, "
            f"leg_a={leg_a}, leg_b={leg_b} — check ticker availability.")
    return pd.concat(out, axis=1).sort_index()


def compute_percentile_tables(df, ccys, tenors):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    today = df.index.max()
    cutoffs = {
        '3M': today - pd.DateOffset(months=3),
        '1Y': today - pd.DateOffset(years=1),
        '5Y': today - pd.DateOffset(years=5)}
    def get_percentile(series, current_val, cutoff):
        window = series[series.index >= cutoff].dropna()
        if len(window) < 2:
            return np.nan
        return round(percentileofscore(window, current_val, kind='rank'), 1)
    rows = [(ccy, tenor) for ccy in ccys for tenor in tenors]
    results = {'IV': [], 'HV': [], 'VDiff': []}
    metric_map = {'IV': 'IV', 'HV': 'RV', 'VDiff': 'VDiff'}
    for ccy, tenor in rows:
        for metric, label in metric_map.items():
            col = f"{ccy}_{tenor}_{metric}"
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            current_val = series.iloc[-1]
            row = {
                'CCY': ccy,
                'Value Type': label,
                'Current': round(current_val, 2),
                '3M_Pct': get_percentile(series, current_val, cutoffs['3M']),
                '1Y_Pct': get_percentile(series, current_val, cutoffs['1Y']),
                '5Y_Pct': get_percentile(series, current_val, cutoffs['5Y']),
                '_sort_key': (ccys.index(ccy), tenors.index(tenor)),
                '_tenor': tenor}
            results[metric].append(row)
    tables = {}
    for metric in ['IV', 'HV', 'VDiff']:
        records = sorted(results[metric], key=lambda x: x['_sort_key'])
        table = pd.DataFrame(records).drop(columns=['_sort_key'])
        # Insert tenor as second column
        table.insert(1, 'Tenor', table.pop('_tenor'))
        tables[metric] = table.reset_index(drop=True)
    return tables['IV'], tables['HV'], tables['VDiff']


def compute_percentile_timeseries(df, ccys, tenors, lookback_days=252, days_back=180):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    metric_map = {'IV': 'IV', 'HV': 'RV', 'VDiff': 'VDiff'}
    out = {}
    for ccy in ccys:
        for tenor in tenors:
            for metric, label in metric_map.items():
                col = f"{ccy}_{tenor}_{metric}"
                if col not in df.columns:
                    continue
                series = df[col].dropna()
                if series.empty:
                    continue

                def rolling_pct(window):
                    return percentileofscore(window, window[-1], kind='rank')

                pct_series = series.rolling(
                    window=lookback_days, min_periods=20
                ).apply(rolling_pct, raw=True)
                out[f"{ccy}_{tenor}_{label}"] = pct_series

    res = pd.concat(out, axis=1).sort_index()
    cutoff = res.index.max() - pd.Timedelta(days=days_back)
    return res[res.index >= cutoff]

def compute_percentile_timeseries_fast(df, ccys, tenors, lookback_days=252, days_back=180):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    metric_map = {'IV': 'IV', 'HV': 'RV', 'VDiff': 'VDiff'}
    out = {}
    for ccy in ccys:
        for tenor in tenors:
            for metric, label in metric_map.items():
                col = f"{ccy}_{tenor}_{metric}"
                if col not in df.columns:
                    continue
                series = df[col].dropna()
                if series.empty:
                    continue

                pct_series = series.rolling(
                    window=lookback_days, min_periods=20
                ).rank(pct=True) * 100

                out[f"{ccy}_{tenor}_{label}"] = pct_series

    res = pd.concat(out, axis=1).sort_index()
    cutoff = res.index.max() - pd.Timedelta(days=days_back)
    return res[res.index >= cutoff]


def rolling_percentile(df, lookback_days=252, days_back=180):
    """
    Column-driven rolling percentile (0-100): rank each column against its own
    trailing `lookback_days` window (current point included -> point-in-time, no
    look-ahead), then trim to the last `days_back` calendar days.

    Unlike compute_percentile_timeseries_fast, this does NOT reconstruct column
    names from a ccy/tenor/metric convention — it ranks whatever columns you
    hand it and passes their names through unchanged. That makes it the general
    primitive for any pre-built series df (matched VDiff, cross-tenor spreads
    from build_vol_spread, term structure, outright IV/RV, ...). Pass
    days_back=None to keep the full history.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    pct = df.rolling(window=lookback_days, min_periods=20).rank(pct=True) * 100
    pct = pct.sort_index()
    if days_back is not None and len(pct):
        cutoff = pct.index.max() - pd.Timedelta(days=days_back)
        pct = pct[pct.index >= cutoff]
    return pct


def compute_days_needed(pct_lookback, days_back, holiday_buffer_pct=0.15):
    # convert trading-day lookback window into calendar days
    lookback_calendar = int(pct_lookback * 7 / 5 * (1 + holiday_buffer_pct))
    return lookback_calendar + days_back

def compute_signals(pct_ts, buy_pct=20, sell_pct=80):
    signals = pd.DataFrame(index=pct_ts.index)
    for col in pct_ts.columns:
        sig = pd.Series(0, index=pct_ts.index, dtype='Int64')  # nullable int, keeps NaN as <NA>
        sig[pct_ts[col] > sell_pct] = -1
        sig[pct_ts[col] < buy_pct] = 1
        sig[pct_ts[col].isna()] = pd.NA
        signals[f"signal_{col}"] = sig
    return signals


def _entry_series_from_signal(raw: pd.Series, side: str) -> pd.Series:
    """
    Collapse compute_signals' raw -1/0/+1/<NA> column into a 0/1 entry Series
    for the chosen side. Shared by get_vol_signal and get_cross_vol_signal so
    the entry convention lives in exactly one place.
        'buy'  -> raw == +1  (percentile < buy_pct,  vol/spread cheap)
        'sell' -> raw == -1  (percentile > sell_pct, vol/spread rich)
        'both' -> raw != 0   (either extreme)
    """
    if side == 'buy':
        entry = raw == 1
    elif side == 'sell':
        entry = raw == -1
    else:
        entry = raw.fillna(0) != 0
    return entry.fillna(False).astype(int)


def get_cross_vol_signal(
    ccys:         List[str],
    leg_a:        Tuple[str, str],
    leg_b:        Tuple[str, str],
    pct_lookback: int   = 252,
    days_back:    int   = 180,
    buy_pct:      float = 20,
    sell_pct:     float = 80,
    side:         str   = 'sell',
    label:        str   = 'VDiff',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Cross-tenor (or same-tenor) two-leg vol-spread signal — the mismatched-tenor
    counterpart to get_vol_signal. Each leg is a (vol_type, tenor) pair with
    vol_type 'V' (implied) or 'H' (realized); the spread is leg_a - leg_b.

        # 1M implied minus 1W realized, enter when the spread is rich:
        pct, sig, signal_series = get_cross_vol_signal(
            ['USDJPY'], ('V','1M'), ('H','1W'), side='sell')

    Pipeline: compute_days_needed -> build_vol_spread -> rolling_percentile ->
    compute_signals -> squeeze. Returns (pct_metric, signal_metric,
    signal_series) with the SAME shapes/convention as get_vol_signal, so
    signal_series drops straight into run_signal_backtest(). Squeezes to a
    single Series only when exactly one ccy is passed (one column); for many
    ccys, call build_vol_spread + rolling_percentile + compute_signals directly.
    """
    assert side in ('buy', 'sell', 'both'), "side must be 'buy', 'sell', or 'both'"

    data_dayhist = compute_days_needed(pct_lookback, days_back)
    spread_df    = build_vol_spread(ccys, leg_a, leg_b, data_dayhist, label=label)
    pct_all      = rolling_percentile(spread_df, pct_lookback, days_back)
    signal_all   = compute_signals(pct_all, buy_pct, sell_pct)

    assert pct_all.shape[1] == 1, (
        f"get_cross_vol_signal squeezes into a single Series only for one ccy — "
        f"found {pct_all.shape[1]} columns: {list(pct_all.columns)}. Pass a "
        f"single ccy, or use build_vol_spread + rolling_percentile + "
        f"compute_signals directly for the multi-column case.")

    pct_metric    = pct_all.copy()
    signal_metric = signal_all.copy()
    signal_series = _entry_series_from_signal(signal_all.iloc[:, 0], side)
    signal_series.name = pct_all.columns[0]
    return pct_metric, signal_metric, signal_series


def get_vol_signal(
    ccys:          List[str],
    tenors:        List[str],
    metric:        str,
    pct_lookback:  int   = 252,
    days_back:     int   = 180,
    buy_pct:       float = 20,
    sell_pct:      float = 80,
    side:          str   = 'buy',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    One-call wrapper around compute_days_needed -> get_Allvol_df ->
    compute_percentile_timeseries_fast -> compute_signals, filtered down to a
    single metric and squeezed into a 0/1 entry-signal Series ready for
    backtest_signal.run_signal_backtest().

    metric : 'IV', 'RV', or 'VD' — exactly these three strings (the assert below
             rejects anything else). 'VD' is the IV-minus-RV spread, stored
             internally as 'VDiff' by get_Allvol_df/compute_signals; passing
             'IV-RV' or 'VDiff' raises.
    side   : which side of compute_signals' raw -1/0/+1 output counts as an
             entry trigger in the returned Series:
                 'buy'  (default) -> raw == +1  (percentile < buy_pct,  vol cheap)
                 'sell'           -> raw == -1  (percentile > sell_pct, vol rich)
                 'both'           -> raw != 0   (either extreme)
             This only decides WHEN to enter — trade direction still comes
             from whatever legs/direction you build separately for
             run_backtest_multi_leg; the signal itself carries no direction
             information into run_signal_backtest.

    Returns
    -------
    pct_metric    : percentile time series (0-100) for `metric`, one column
                    per ccy/tenor combination requested.
    signal_metric : raw compute_signals output (-1/0/+1/<NA>) for `metric`,
                    same columns.
    signal_series : 0/1 pd.Series, same shape/convention as
                    pd.Series(0, index=...); .iloc[...] = 1 — directly usable
                    as the `signal` argument to run_signal_backtest(). Only
                    produced when ccys/tenors resolve to exactly ONE column;
                    narrow both to a single value to get one, otherwise this
                    raises (one Series can't represent multiple ccy/tenor
                    combinations at once — use pct_metric/signal_metric
                    directly for the multi-column case).
    """
    metric_labels = {'IV': 'IV', 'RV': 'RV', 'VD': 'VDiff'}
    assert metric in metric_labels, f"metric must be one of {list(metric_labels)}, got {metric!r}"
    assert side in ('buy', 'sell', 'both'), "side must be 'buy', 'sell', or 'both'"
    label = metric_labels[metric]
    data_dayhist = compute_days_needed(pct_lookback, days_back)
    df_vol_data  = get_Allvol_df(ccys, tenors, data_dayhist)
    pct_all      = compute_percentile_timeseries_fast(df_vol_data, ccys, tenors, pct_lookback, days_back)
    signal_all   = compute_signals(pct_all, buy_pct, sell_pct)
    pct_cols      = [c for c in pct_all.columns if c.endswith(f"_{label}")]
    signal_cols   = [f"signal_{c}" for c in pct_cols]
    pct_metric    = pct_all[pct_cols].copy()
    signal_metric = signal_all[signal_cols].copy()
    assert len(pct_cols) == 1, (
        f"get_vol_signal only squeezes into a single Series when exactly one "
        f"ccy/tenor combination matches metric={metric!r} — found "
        f"{len(pct_cols)}: {pct_cols}. Narrow ccys/tenors to one each, or use "
        f"pct_metric/signal_metric directly for the multi-column case.")
    signal_series = _entry_series_from_signal(signal_metric.iloc[:, 0], side)
    signal_series.name = pct_cols[0]
    return pct_metric, signal_metric, signal_series


"""
EXAMPLE USES

# 1W implied − 1W realized: sell when percentile > 80
_, _, sig_1w1w = get_vol_signal(
    ['USDJPY'], 
    ['1W'], 
    metric='VD',
    sell_pct=80, 
    side='sell')

print(sig_1w1w.tail(30))   # 0/1 entry signal, name = USDJPY_1W_VDiff


# 1M implied − 1W realized: sell when percentile > 80
_, _, sig_1m1w = get_cross_vol_signal(
    ['USDJPY'], 
    ('V', '1M'), 
    ('H', '1W'),
    sell_pct=80, 
    side='sell')

print(sig_1m1w.tail(30))   # 0/1 entry signal, name = USDJPY_IV1M-RV1W_VDiff

"""


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------







# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def get_always_on_signal(days_back, pair='USDJPY', tenor='1M'):
    """
    All-ones entry signal on `pair`'s own Bloomberg business dates.

    EVERY available date fires an entry — including dates within one `tenor` of
    today, whose trades cannot reach expiry. Those are handled downstream, not
    here:
      * run_signal_backtest's `skip_nodata` guard drops an entry with no market
        data at all after inception (which would otherwise make
        run_backtest_multi_leg build an empty frame and raise
        KeyError("None of ['date'] are in the columns"));
      * a trade that does run but never reaches a terminal event comes back with
        exit_reason=None -> summary['live_trade']=True, and
        trade_metrics(settled_only=True) excludes it from the per-trade stats.
    Note the daily book is NOT settled_only, so a still-live final trade does
    contribute its partial mark-to-market to sharpe/drawdown.

    Deliberately NOT trimming a tenor off the tail: an early-exit rule
    (ExitAfterNDays / ExitAtDaysRemaining / TakeProfitStopLoss) can fully settle
    a trade entered inside that window, and dropping those dates would throw away
    real completed trades.

    `pair`/`tenor` must match what you pass to run_signal_backtest so the entry
    calendar lines up with the pair actually traded (the old hard-coded EURUSD
    dates silently mismatched a USDJPY backtest).
    """
    df = get_ImplRealVol(
        currency_list=[pair],
        tenors=[tenor],
        days_back=days_back)
    valid_dates = pd.to_datetime(df[[f"{pair}_V{tenor}"]].dropna().index)

    # Last inception date that still leaves a full tenor of data before today.
    last_entry  = pd.Timestamp(datetime.now().date())
    valid_dates = valid_dates[valid_dates <= last_entry]

    signal_series = pd.Series(
        data=1,
        index=valid_dates,
        dtype="int64",
        name="AlwaysOn")
    return signal_series


def get_date_signal(dates, pair='EURUSD', tenor='1M'):
    """
    Entry signal that fires ONLY on the given date(s), 0 everywhere else —
    the exact-date counterpart to get_always_on_signal.

    dates : a single date string, or a list of date strings, each in
            '%d%b%y' form e.g. '24Jul26' -> 24 Jul 2026. Pass one date to
            fire a single entry, or a list to fire an entry on each.

    pair, tenor : same meaning as get_always_on_signal — used to pull
            `pair`'s own Bloomberg business-date calendar so the 0
            background lines up with real trading days, the same shape
            get_always_on_signal returns. The pull's history window is
            derived automatically: it starts at the EARLIEST date in
            `dates` and runs through today, so there's no separate
            days_back to keep in sync with your input dates.

    The requested dates are unioned into that calendar even if they fall
    outside the pulled history (e.g. a near-future date with no market data
    yet) — run_signal_backtest resolves every signal date to the pair's
    preceding business day itself, so an exact calendar match here isn't
    required.

    Drops straight into run_signal_backtest() / plot_full_report() the same
    way get_always_on_signal does.
    """
    if isinstance(dates, str):
        dates = [dates]
    entry_dates = pd.to_datetime([datetime.strptime(d, '%d%b%y') for d in dates])

    days_back = max((datetime.now().date() - entry_dates.min().date()).days, 0)
    df = get_ImplRealVol(
        currency_list=[pair],
        tenors=[tenor],
        days_back=days_back)
    valid_dates = pd.to_datetime(df[[f"{pair}_V{tenor}"]].dropna().index)

    all_dates = valid_dates.union(entry_dates).sort_values()

    signal_series = pd.Series(
        data=0,
        index=all_dates,
        dtype="int64",
        name="DateSignal")
    signal_series.loc[entry_dates] = 1
    return signal_series





