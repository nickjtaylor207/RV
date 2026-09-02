from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from xbbg import blp


def pull_fx_vol_data_batch(ccy, tenors, time_hist, delta='25'):
    start_date = (datetime.today() - timedelta(days=time_hist)).strftime('%Y-%m-%d')
    end_date   = datetime.today().strftime('%Y-%m-%d')

    tickers = []
    for t in tenors:
        tickers += [
            f'{ccy}V{t} BGN Curncy',
            f'{ccy}{delta}R{t} BGN Curncy',
            f'{ccy}H{t} BGN Curncy',
        ]
    tickers.append(f'{ccy} BGN Curncy')

    raw = blp.bdh(tickers=tickers, flds='PX_LAST',
                  start_date=start_date, end_date=end_date)
    raw.columns = raw.columns.get_level_values(0)
    raw.index   = pd.to_datetime(raw.index)
    raw         = raw.ffill().dropna()

    out = {}
    for t in tenors:
        iv_col = f'{ccy}V{t} BGN Curncy'
        rr_col = f'{ccy}{delta}R{t} BGN Curncy'
        rv_col = f'{ccy}H{t} BGN Curncy'
        sp_col = f'{ccy} BGN Curncy'
        if not all(c in raw.columns for c in [iv_col, rr_col, sp_col]):
            print(f'  SKIP  {t}  — missing tickers')
            continue
        df = raw[[iv_col, rr_col, sp_col]].copy()
        if rv_col in raw.columns:
            df[rv_col] = raw[rv_col]
        df.columns = ([f'V{t}', f'{delta}r{t}', 'spot']
                      + ([f'RV{t}'] if rv_col in raw.columns else []))
        out[t] = df

    return out


class SpotVolRRAnalyzer:
    def __init__(self, df: pd.DataFrame, ccy: str, tenor: str, delta: str = '25'):
        self.ccy    = ccy
        self.tenor  = tenor
        self.delta  = delta
        self.iv_col = f'V{tenor}'
        self.rr_col = f'{delta}r{tenor}'
        self.iv     = df[self.iv_col]
        self.spot   = df['spot']
        self._build_derived()

    def _build_derived(self):
        self.spot_ret = np.log(self.spot / self.spot.shift(1)).dropna()
        self.iv_chg   = self.iv.diff().dropna()
        idx = self.spot_ret.index.intersection(self.iv_chg.index)
        self.spot_ret = self.spot_ret.loc[idx]
        self.iv_chg   = self.iv_chg.loc[idx]

    def rolling_correlation(self, windows=(20, 60, 120)):
        r          = self.spot_ret
        vol_series = self.iv_chg
        out = {f'corr_{w}d': r.rolling(w).corr(vol_series) for w in windows}
        ewm_cov = r.ewm(halflife=20).cov(vol_series)
        ewm_std = r.ewm(halflife=20).std() * vol_series.ewm(halflife=20).std()
        out['corr_ewm20'] = (ewm_cov / ewm_std).replace([np.inf, -np.inf], np.nan)
        self.corr = pd.DataFrame(out)
        return self.corr


def build_corr_multiwindow(ccy, tenors, corr_windows, time_hist=252*3, delta='25'):
    raw = pull_fx_vol_data_batch(ccy=ccy, tenors=tenors,
                                  time_hist=time_hist, delta=delta)
    ts_dict = {}
    for tenor, df in raw.items():
        az   = SpotVolRRAnalyzer(df, ccy=ccy, tenor=tenor, delta=delta)
        corr = az.rolling_correlation(windows=tuple(corr_windows))
        for w in corr_windows:
            ts_dict[(tenor, w)] = corr[f'corr_{w}d']

    df_ts = pd.DataFrame(ts_dict)
    df_ts.columns = pd.MultiIndex.from_tuples(df_ts.columns, names=['tenor', 'window'])
    df_ts = df_ts.sort_index(axis=1)

    df_current = df_ts.iloc[-1].unstack(level='window').round(3)

    return df_ts, df_current, raw


def plot_corr_multiwindow(df_ts, ccy, corr_windows):
    tenors = df_ts.columns.get_level_values('tenor').unique()
    n      = len(tenors)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = ['lightblue', 'steelblue', 'darkblue', 'navy']
    for ax, tenor in zip(axes, tenors):
        for w, col in zip(corr_windows, colors):
            ax.plot(df_ts[(tenor, w)], color=col, lw=1.5, label=f'{w}d')
        ax.axhline(0,    color='black', lw=0.8)
        ax.axhline(-0.5, color='red',   lw=0.7, ls=':', alpha=0.5)
        ax.set_title(f'{ccy}  {tenor}  |  Rolling Spot-ΔIV Correlation',
                     fontweight='bold')
        ax.set_ylabel('Correlation')
        ax.set_ylim(-0.99, 0.99)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig













ccy          = 'EURUSD'
tenors       = ['1W', '1M', '2M', '3M']
corr_windows = [20, 60, 120]

time_hist    = 252 * 3
delta        = '25'

# ── Run ───────────────────────────────────────────────────────────────────────
df_ts, df_current, raw = build_corr_multiwindow(
    ccy          = ccy,
    tenors       = tenors,
    corr_windows = corr_windows,
    time_hist    = time_hist,
    delta        = delta
)

print(df_ts)
