import pandas as pd
import numpy as np

from pandas.plotting import table
from xbbg import blp
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from scipy.stats import percentileofscore




def get_ATM_XCCY_Spread_TimeSeries(ccy_int, tenor_int, ccys, pct_lookback, days_back):
    """
    Time series of the correlation-weighted percentile for ccy_int vs the basket,
    over the last `days_back` days, using a rolling `pct_lookback`-day window
    for both the percentile and the correlation weights.

    Single Bloomberg pull; all rolling stats computed locally.
    """
    base_ccy = sorted(set(ccys))
    end_date = datetime.now().strftime("%Y-%m-%d")
    # need pct_lookback of history BEFORE the earliest date we report on,
    # plus days_back of reporting dates, plus a small buffer
    total_lookback = days_back + pct_lookback + 15
    start_date = (datetime.now() - timedelta(days=total_lookback)).strftime("%Y-%m-%d")

    # --- 1. single data pull for all ccys ---
    data_dict = {}
    for ccy_pair in base_ccy:
        ticker_IV = f"{ccy_pair}V{tenor_int} BGN Curncy"
        try:
            data_IV = blp.bdh(tickers=ticker_IV, flds="PX_LAST",
                               start_date=start_date, end_date=end_date)
            data_IV.columns = data_IV.columns.get_level_values(1)
            data_dict[ccy_pair] = data_IV['PX_LAST']
        except Exception:
            continue

    df_all = pd.DataFrame(data_dict).dropna()
    if df_all.empty or ccy_int not in df_all.columns:
        return pd.Series(dtype=float)

    # ensure a clean, sorted, de-duplicated DatetimeIndex for time-based rolling
    df_all.index = pd.to_datetime(df_all.index)
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    df_all = df_all.sort_index()

    diffs = df_all.diff().dropna()
    others = [c for c in base_ccy if c != ccy_int and c in df_all.columns]

    # --- 2. build spreads for ccy_int vs each other ccy ---
    spreads = pd.DataFrame({other: df_all[ccy_int] - df_all[other] for other in others})

    # --- 3. rolling percentile of current spread within trailing pct_lookback window ---
    window_str = f'{pct_lookback}D'

    def _last_pct(x):
        if len(x) < 2:
            return np.nan
        return percentileofscore(x, x[-1])

    # min_periods is in OBSERVATIONS even though the window is time-based, so a
    # floor of 2 lets a percentile be computed off 2 points (which can only ever
    # return 50 or 100). Require a meaningful fraction of the window instead, so
    # the warm-up produces NaN and gets dropped in step 6 rather than feeding
    # garbage percentiles into the signal. 20 matches rolling_percentile's floor
    # in Implied_Realized.py.
    min_obs = 20

    pct_ts = spreads.rolling(window_str, min_periods=min_obs).apply(_last_pct, raw=True)

    # --- 4. rolling pairwise correlation of diffs, same window ---
    corr_ts_full = diffs.rolling(window_str, min_periods=min_obs).corr()
    # extract corr(ccy_int, other) as its own time series per other ccy
    corr_ts = pd.DataFrame({
        other: corr_ts_full.xs(ccy_int, level=1)[other] for other in others
    })

    # --- 5. combine into correlation-weighted percentile per date ---
    weights = corr_ts.clip(lower=0)
    weighted_sum = (pct_ts * weights).sum(axis=1)
    weight_total = weights.sum(axis=1)
    val_ts = weighted_sum / weight_total

    # --- 6. trim to the requested reporting window ---
    # days_back is CALENDAR days, matching compute_days_needed / rolling_percentile
    # in Implied_Realized.py and the `days_back` every signal builder is handed off
    # a ComboSpec. The old `iloc[-days_back:]` treated it as a ROW count, which at
    # days_back=1825 (5y) kept every one of the ~1500 available rows — so the
    # returned signal silently started ~(pct_lookback + 15) days EARLIER than an
    # always-on or IV/RV signal built with the same days_back, and a grid comparing
    # them was comparing different samples.
    val_ts = val_ts.dropna()
    if len(val_ts):
        cutoff = val_ts.index.max() - pd.Timedelta(days=days_back)
        val_ts = val_ts[val_ts.index >= cutoff]
    val_ts.name = f"{ccy_int}_{tenor_int}_XCCY_Pct"

    return val_ts

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



def get_xccy_spread_signal(
    ccy_int:      str,
    tenor_int:    str,
    ccys:         List[str],
    pct_lookback: int   = 365,
    days_back:    int   = 180,
    buy_pct:      float = 20,
    sell_pct:     float = 80,
    side:         str   = 'buy',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Cross-currency ATM vol spread signal — same shape/convention as
    get_vol_signal / get_cross_vol_signal, but the percentile driving the
    signal comes from get_ATM_XCCY_Spread_TimeSeries (correlation-weighted
    percentile of ccy_int's vol spread vs the rest of `ccys`) rather than an
    IV/RV or cross-tenor spread.

    Pipeline: get_ATM_XCCY_Spread_TimeSeries -> compute_signals ->
    _entry_series_from_signal. buy_pct/sell_pct are applied directly to the
    0-100 percentile already returned by get_ATM_XCCY_Spread_TimeSeries (no
    separate rolling_percentile step needed — that function already computes
    a rolling, correlation-weighted percentile internally).

    side : which side counts as an entry trigger:
        'buy'  (default) -> percentile < buy_pct  (ccy_int spread cheap vs basket)
        'sell'           -> percentile > sell_pct (ccy_int spread rich vs basket)
        'both'           -> either extreme

    Returns
    -------
    pct_metric    : single-column DataFrame of the percentile time series
    signal_metric : single-column DataFrame, raw compute_signals output
                    (-1/0/+1/<NA>)
    signal_series : 0/1 pd.Series ready for run_signal_backtest()
    """
    assert side in ('buy', 'sell', 'both'), "side must be 'buy', 'sell', or 'both'"

    pct_series = get_ATM_XCCY_Spread_TimeSeries(
        ccy_int, tenor_int, ccys, pct_lookback=pct_lookback, days_back=days_back)

    if pct_series.empty:
        raise ValueError(
            f"get_xccy_spread_signal: no data returned for ccy_int={ccy_int!r}, "
            f"tenor_int={tenor_int!r}, ccys={ccys}. Check ticker availability.")

    col_name = pct_series.name or f"{ccy_int}_{tenor_int}_XCCY_Pct"
    pct_metric = pct_series.to_frame(name=col_name)

    signal_metric = compute_signals(pct_metric, buy_pct, sell_pct)
    signal_series = _entry_series_from_signal(signal_metric.iloc[:, 0], side)
    signal_series.name = col_name

    return pct_metric, signal_metric, signal_series


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------





# ccy_int = 'USDJPY'
# tenor_int = '1M'


# ccys = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
#         'AUDUSD', 'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK']

# pct_metric, signal_metric, signal_series = get_xccy_spread_signal(
#     ccy_int, tenor_int, ccys,
#     days_back=180,
#     sell_pct=80, side='sell')

# print(signal_series.tail(20))

# print(signal_metric.tail(20))

# print(pct_metric.tail(20))


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER DIAGNOSTICS — basket / pct_lookback / sell_pct
# ═══════════════════════════════════════════════════════════════════════════════
# The three inputs above that are genuine degrees of freedom, inspected on live
# data off ONE Bloomberg pull. Read-only: nothing here is used by
# get_xccy_spread_signal or the backtest path.
#
# WHY THIS IS CHEAP — the pairwise decomposition:
#   spread_Q(t) = vol_P(t) - vol_Q(t) and corr_Q(t) depend ONLY on the target P
#   and the one member Q. Neither depends on which OTHER members are in the
#   basket. Basket choice enters exclusively through which columns get summed in
#   step 5's weighted mean. So pct/corr are computed ONCE per lookback for every
#   member, and any basket subset — leave-one-out, or an arbitrary custom basket
#   — is a re-normalised sum over columns costing nothing. Only pct_lookback
#   forces a recompute, and only the pull is I/O.
#
# xccy_components mirrors steps 2-4 of get_ATM_XCCY_Spread_TimeSeries exactly
# (time-based '{n}D' window, min_periods=20 OBSERVATIONS, percentileofscore
# kind='rank', correlation on daily vol CHANGES, weights clipped at zero), and
# xccy_blend mirrors step 5. Verify with the reconciliation snippet at the end.

# Palette: reuse reporting._VIZ when it resolves (it does when imported from
# TEST.py, whose sys.path[0] is Delta_Hedged/), else fall back to the same values
# inline. Guarded rather than hard-imported so Signal_Gen/ carries no dependency
# on the top-level reporting module and this file still runs standalone.
try:
    from reporting import _VIZ
except ImportError:                                   # standalone / direct run
    _VIZ = {'surface': '#fcfcfb', 'ink': '#0b0b0b', 'ink2': '#52514e',
            'muted': '#898781', 'grid': '#e1e0d9', 'baseline': '#c3c2b7',
            'blue': '#2a78d6', 'red': '#d03b3b'}

# pct_lookback is an ORDERED parameter, so it gets one hue light->dark, not
# categorical hues. Endpoints validated (monotone lightness, adjacent dL >= 0.06,
# light end 2.04:1 vs surface) for up to 5 steps — past that the steps stop
# separating, hence the cap.
_RAMP_ENDS = ['#8ab6f0', '#17406f']
_MAX_RAMP = 5


def _ramp(n):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list('xccy_seq', _RAMP_ENDS)
    return [cm(0.5)] if n == 1 else [cm(i / (n - 1)) for i in range(n)]


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(_VIZ['surface'])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
        ax.spines[s].set_color(_VIZ['baseline'])
    ax.grid(True, color=_VIZ['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_VIZ['ink2'], labelsize=8, length=0)
    if title:
        ax.set_title(title, fontsize=10, color=_VIZ['ink'], fontweight='bold',
                     loc='left', pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=_VIZ['ink2'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=_VIZ['ink2'])


def pull_vol_panel(basket, tenor, days, verbose=True):
    """ATM vol panel for the basket — ONE pull sized for the widest lookback.

    Same pull and same inner join (.dropna()) as step 1 of
    get_ATM_XCCY_Spread_TimeSeries, which is the basket's binding data
    constraint: one gappy member truncates every other member's history too. The
    printed row count / span is that cost, made visible.
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    cols = {}
    for p in sorted(set(basket)):
        try:
            d = blp.bdh(tickers=f'{p}V{tenor} BGN Curncy', flds='PX_LAST',
                        start_date=start_date, end_date=end_date)
            d.columns = d.columns.get_level_values(1)
            cols[p] = d['PX_LAST']
        except Exception as e:
            if verbose:
                print(f'  [skip] {p}: {type(e).__name__}: {e}')
    raw = pd.DataFrame(cols)
    panel = raw.dropna()
    panel.index = pd.to_datetime(panel.index)
    panel = panel[~panel.index.duplicated(keep='last')].sort_index()
    if verbose:
        print(f'[panel] {len(panel.columns)} pairs, {len(panel)} rows '
              f'({panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d}) '
              f'| {len(raw) - len(panel)} rows dropped to the inner join')
    return panel


def xccy_components(panel, target, pct_lookback, min_obs=20):
    """(pct, weights) for target vs EACH other column — basket-independent.

    pct     : rolling percentile rank of each pairwise spread (0-100)
    weights : rolling corr of daily vol CHANGES, clipped at 0
    """
    others = [c for c in panel.columns if c != target]
    spreads = pd.DataFrame({o: panel[target] - panel[o] for o in others})
    diffs = panel.diff().dropna()
    win = f'{pct_lookback}D'

    def _last_pct(x):
        return np.nan if len(x) < 2 else percentileofscore(x, x[-1])

    pct = spreads.rolling(win, min_periods=min_obs).apply(_last_pct, raw=True)
    cfull = diffs.rolling(win, min_periods=min_obs).corr()
    corr = pd.DataFrame({o: cfull.xs(target, level=1)[o] for o in others})
    return pct, corr.clip(lower=0)


def xccy_blend(pct, weights, members=None, days_back=None):
    """Correlation-weighted blended percentile for one basket subset (step 5).

    A date where every correlation is <= 0 yields NaN rather than a
    divide-by-zero, so a degenerate basket shows as a gap instead of an inf.
    """
    m = list(members) if members is not None else list(pct.columns)
    w = weights[m]
    tot = w.sum(axis=1)
    out = ((pct[m] * w).sum(axis=1) / tot.where(tot > 0)).dropna()
    if days_back and len(out):
        out = out[out.index >= out.index.max() - pd.Timedelta(days=days_back)]
    return out


def plot_xccy_param_scan(pair='EURUSD', tenor='1M', basket=None,
                         lookbacks=(180, 365, 730), sell_pct=70,
                         days_back=365 * 3, panel=None,
                         save_path=None, show=True, verbose=True):
    """Four-panel scan of the xCCY signal's three tunable parameters.

    A  blended percentile through time, one line per pct_lookback
    B  coverage curve: P(blend > threshold) vs threshold, against the uniform
       reference — the plot sell_pct is chosen off
    C  basket leave-one-out: change in firing rate when each member is dropped
    D  weight share each member carries, mean and range over the sample

    panel : pre-pulled vol panel (see pull_vol_panel) to re-plot without
            re-pulling. Pass the returned dict's ['panel'] back on every
            follow-up call and the Bloomberg pull happens once per session.

    Returns a dict of the underlying frames plus the Figure.
    """
    import matplotlib.pyplot as plt

    basket = basket or ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
                        'AUDUSD', 'NZDUSD', 'USDCAD']
    lookbacks = list(lookbacks)
    assert len(lookbacks) <= _MAX_RAMP, \
        f'at most {_MAX_RAMP} lookbacks — beyond that the one-hue ramp stops separating'

    if panel is None:
        panel = pull_vol_panel(basket, tenor,
                               days_back + max(lookbacks) + 15, verbose=verbose)
    assert pair in panel.columns, f'{pair} missing from the panel'
    members = [c for c in panel.columns if c != pair]

    # one components pass per lookback; every basket subset reuses it
    comps = {lb: xccy_components(panel, pair, lb) for lb in lookbacks}
    blends = {lb: xccy_blend(*comps[lb], days_back=days_back) for lb in lookbacks}
    base = lookbacks[len(lookbacks) // 2]          # middle lookback drives C/D

    thr = np.arange(50, 96, 1.0)
    cov = pd.DataFrame({lb: [(b > t).mean() for t in thr]
                        for lb, b in blends.items()}, index=thr)

    b_pct, b_w = comps[base]
    full_cov = (blends[base] > sell_pct).mean()
    loo = pd.Series({
        m: (xccy_blend(b_pct, b_w, [o for o in members if o != m],
                       days_back=days_back) > sell_pct).mean() - full_cov
        for m in members}).sort_values()

    share = b_w[members].div(
        b_w[members].sum(axis=1).where(lambda s: s > 0), axis=0)
    share = share.reindex(blends[base].index).dropna()

    colors = _ramp(len(lookbacks))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    fig.patch.set_facecolor(_VIZ['surface'])
    axA, axB, axC, axD = axes.ravel()

    # ── A. blended percentile through time ────────────────────────────────────
    for c, lb in zip(colors, lookbacks):
        b = blends[lb]
        # Reindex onto the observed calendar so UNDEFINED days (all weights
        # clipped to zero) break the line. Plotting b.index directly draws a
        # straight segment across the gap, which reads as a real excursion.
        gapped = b.reindex(panel.index[(panel.index >= b.index.min()) &
                                       (panel.index <= b.index.max())])
        axA.plot(gapped.index, gapped.values, color=c, linewidth=1.6,
                 label=f'{lb}d')
        axA.annotate(f'{lb}d', xy=(b.index[-1], b.iloc[-1]),
                     xytext=(4, 0), textcoords='offset points',
                     color=c, fontsize=8, va='center', fontweight='bold')
    axA.axhline(sell_pct, color=_VIZ['ink2'], linewidth=1.2, linestyle='--')
    axA.annotate(f'sell_pct = {sell_pct}', xy=(0.01, sell_pct),
                 xycoords=('axes fraction', 'data'), xytext=(0, 4),
                 textcoords='offset points', color=_VIZ['ink2'], fontsize=8)
    axA.set_ylim(0, 100)
    _style(axA, f'A · Blended percentile — {pair} {tenor} vs basket',
           ylabel='xCCY percentile')
    axA.legend(frameon=False, fontsize=8, ncol=len(lookbacks),
               labelcolor=_VIZ['ink2'], loc='lower left')

    # ── B. coverage curve ─────────────────────────────────────────────────────
    for c, lb in zip(colors, lookbacks):
        axB.plot(thr, cov[lb].values * 100, color=c, linewidth=2, label=f'{lb}d')
    # what coverage WOULD be if the blend were uniform on [0,100]. The gap to the
    # curves is the compression from averaging N ranks — the reason sell_pct
    # cannot be read as a firing rate.
    axB.plot(thr, 100 - thr, color=_VIZ['muted'], linewidth=1.4, linestyle=':',
             label='uniform reference')
    axB.axvline(sell_pct, color=_VIZ['ink2'], linewidth=1.2, linestyle='--')
    for c, lb in zip(colors, lookbacks):
        y = (blends[lb] > sell_pct).mean() * 100      # exact, not off the grid
        axB.plot([sell_pct], [y], 'o', color=c, markersize=8,
                 markeredgecolor=_VIZ['surface'], markeredgewidth=2, zorder=5)
        axB.annotate(f'{y:.0f}%', xy=(sell_pct, y), xytext=(6, 4),
                     textcoords='offset points', color=c, fontsize=8,
                     fontweight='bold')
    _style(axB, 'B · Coverage vs threshold — where sell_pct is chosen',
           xlabel='sell_pct threshold', ylabel='% of defined days firing')
    axB.legend(frameon=False, fontsize=8, labelcolor=_VIZ['ink2'])

    # ── C. basket leave-one-out (diverging: the sign is the point) ────────────
    pp = loo.values * 100
    cmax = max(abs(pp).max(), 1e-9)
    bar_c = [_VIZ['blue'] if v > 0 else _VIZ['red'] for v in pp]
    ypos = np.arange(len(loo))
    axC.barh(ypos, pp, color=bar_c, height=0.62)
    axC.axvline(0, color=_VIZ['baseline'], linewidth=1.2)
    axC.set_yticks(ypos)
    axC.set_yticklabels([f'drop {m}' for m in loo.index], fontsize=8)
    for y, v in zip(ypos, pp):
        axC.annotate(f'{v:+.1f}pp', xy=(v, y),
                     xytext=(5 if v >= 0 else -5, 0), textcoords='offset points',
                     ha='left' if v >= 0 else 'right', va='center',
                     color=_VIZ['ink2'], fontsize=8)
    axC.set_xlim(-cmax * 1.45, cmax * 1.45)
    _style(axC, f'C · Basket leave-one-out — firing rate vs full basket '
                f'({full_cov*100:.1f}%, {base}d)',
           xlabel='change in % of days firing (pp)')

    # ── D. who carries the weight ─────────────────────────────────────────────
    mean_w = share.mean().sort_values()
    lo_w = share.min()[mean_w.index].values * 100
    hi_w = share.max()[mean_w.index].values * 100
    ypos = np.arange(len(mean_w))
    axD.barh(ypos, mean_w.values * 100, color=_VIZ['blue'], height=0.62)
    axD.hlines(ypos, lo_w, hi_w, color=_VIZ['ink2'], linewidth=1.4, alpha=0.8)
    axD.set_yticks(ypos)
    axD.set_yticklabels(mean_w.index, fontsize=8)
    # label past the WHISKER end, not the bar end — the range runs wider than the
    # mean, so anchoring on the bar puts the text on top of the whisker.
    for y, v, hi in zip(ypos, mean_w.values * 100, hi_w):
        axD.annotate(f'{v:.1f}%', xy=(max(v, hi), y), xytext=(6, 0),
                     textcoords='offset points', va='center',
                     color=_VIZ['ink2'], fontsize=8)
    axD.set_xlim(0, hi_w.max() * 1.18)
    axD.axvline(100 / len(mean_w), color=_VIZ['baseline'], linewidth=1.2,
                linestyle='--')                       # equal-weight reference
    _style(axD, f'D · Weight share per member — mean, with range ({base}d)',
           xlabel='% of total correlation weight')

    head = (f'xCCY parameter scan  |  {pair} {tenor}  |  '
            f'{len(members)}-member basket  |  {days_back}d sample')
    fig.suptitle(head, fontsize=13, color=_VIZ['ink'], fontweight='bold',
                 x=0.01, ha='left', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if verbose:
        # `defined` is not decoration: a short lookback can leave the blend
        # UNDEFINED on days where every correlation goes <= 0 (all weights clip
        # to zero, so the weighted mean is 0/0 — get_ATM_XCCY_Spread_TimeSeries
        # has the same gap, dropped by its step 6). Coverage is a fraction of
        # DEFINED days, so an unequal count means panel B's curves do not share a
        # denominator; and a low count is itself the signal that this
        # lookback/basket pair is too decorrelated to rank.
        span = max(len(b) for b in blends.values())
        print(f'\n  lookback   fires@{sell_pct}   median pct    IQR   defined')
        for lb in lookbacks:
            b = blends[lb]
            q1, q3 = b.quantile([0.25, 0.75])
            flag = '' if len(b) == span else f'   <- {span-len(b)} undefined'
            print(f'  {lb:>6}d   {(b>sell_pct).mean()*100:8.1f}%   '
                  f'{b.median():10.1f}  {q3-q1:5.1f}   {len(b):5}{flag}')

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=_VIZ['surface'],
                    bbox_inches='tight')
    if show:
        plt.show()
    return {'panel': panel, 'blends': pd.DataFrame(blends), 'coverage': cov,
            'loo': loo, 'weight_share': share, 'fig': fig}


# Reconciliation — run once to prove the local reimplementation of steps 2-5
# matches get_ATM_XCCY_Spread_TimeSeries rather than silently diverging:
#
#   basket = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','NZDUSD','USDCAD']
#   ref  = get_xccy_spread_signal('EURUSD','1M',basket, pct_lookback=365,
#                                 days_back=365, sell_pct=70, side='sell')[0].iloc[:,0]
#   p    = pull_vol_panel(basket,'1M', 365+365+15)
#   mine = xccy_blend(*xccy_components(p,'EURUSD',365), days_back=365)
#   print((ref - mine.reindex(ref.index)).abs().max())      # expect ~0









# -------------------------------------------------------------------------------



# pair = 'EURUSD'
# tenor = '1M'

# basket = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
# pct_lookback = 365
# days_back = 365
# sell_pct = 70


# scan = plot_xccy_param_scan(pair=pair, tenor=tenor, basket=basket,
#                             lookbacks=(180, 365, 730),
#                             sell_pct=sell_pct,
#                             days_back=365 * 3)


# Re-plot at other settings with NO second Bloomberg pull — pass the panel back:
#   plot_xccy_param_scan(sell_pct=80, panel=scan['panel'])
#   plot_xccy_param_scan(lookbacks=(90, 180, 365, 730), panel=scan['panel'])
#   plot_xccy_param_scan(basket=[b for b in basket if b != 'USDJPY'],
#                        panel=scan['panel'])
# scan['coverage']  threshold x lookback firing rates
# scan['loo']       per-member leave-one-out delta
# scan['blends']    blended percentile series per lookback