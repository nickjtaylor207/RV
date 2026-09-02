from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from xbbg import blp


# ------------------- Realized Spot-Vol CORR -------------------------

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
            col_name = f'{ccy}_{tenor}_SVC_{w}d'
            ts_dict[col_name] = corr[f'corr_{w}d']

    df_ts = pd.DataFrame(ts_dict).sort_index(axis=1)
    return df_ts, raw
def build_corr_multi_ccy(ccys, tenors, corr_windows, time_hist=252*3, delta='25'):
    all_series = {}
    all_raw    = {}
    for ccy in ccys:
        df_ts, raw = build_corr_multiwindow(
            ccy=ccy, tenors=tenors, corr_windows=corr_windows,
            time_hist=time_hist, delta=delta)
        all_series.update(df_ts.to_dict(orient='series'))
        all_raw[ccy] = raw
    df_combined = pd.DataFrame(all_series).sort_index(axis=1)
    return df_combined, all_raw

# ------------------- Implied Spot-Vol CORR -------------------------

def get_Data(ccys, tenors, delta, days):
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rr_data, atm_data, bf_data, spot_data = {}, {}, {}, {}
    for ccy in ccys:
        for tenor in tenors:
            ticker = f"{ccy}{delta}R{tenor} BGN Curncy"
            data = blp.bdh(tickers=[ticker], flds="PX_LAST",
                           start_date=start_date, end_date=end_date)
            if not data.empty:
                col = f"{ccy}_RR{delta}_{tenor}"
                data.columns = [col]
                data.index = pd.to_datetime(data.index)
                rr_data[col] = data
            ticker = f"{ccy}V{tenor} BGN Curncy"
            data = blp.bdh(tickers=[ticker], flds="PX_LAST",
                           start_date=start_date, end_date=end_date)
            if not data.empty:
                col = f"{ccy}_IV_{tenor}"
                data.columns = [col]
                data.index = pd.to_datetime(data.index)
                atm_data[col] = data
            ticker = f"{ccy}{delta}B{tenor} BGN Curncy"      
            data = blp.bdh(tickers=[ticker], flds="PX_LAST",
                           start_date=start_date, end_date=end_date)
            if not data.empty:
                col = f"{ccy}_BF{delta}_{tenor}"
                data.columns = [col]
                data.index = pd.to_datetime(data.index)
                bf_data[col] = data
    for ccy in ccys:
        ticker = f"{ccy} BGN Curncy"
        data = blp.bdh(tickers=[ticker], flds="PX_LAST",
                       start_date=start_date, end_date=end_date)
        if not data.empty:
            col = f"{ccy}_spot"
            data.columns = [col]
            data.index = pd.to_datetime(data.index)
            spot_data[col] = data
    all_series = {**rr_data, **atm_data, **bf_data, **spot_data}
    df_raw = pd.concat(all_series.values(), axis=1)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    return df_raw
def calc_implied_spot_vol_corr(df, ccys, tenors, delta):
    """
    SABR Model Assumptions 
    ρ ≈ (RR / σ_ATM) / VoV
        - Where VoV ≈ √2 × BF / σ_ATM²
                        - BF ≈ ½VoV²T
    """
    results = {}
    for ccy in ccys:
        for tenor in tenors:
            atm_col = f"{ccy}_IV_{tenor}"
            rr_col  = f"{ccy}_RR{delta}_{tenor}"
            bf_col  = f"{ccy}_BF{delta}_{tenor}"
            if not all(c in df.columns for c in [atm_col, rr_col, bf_col]):
                print(f"  Skipping {ccy} {tenor} — missing columns")
                continue
            atm = df[atm_col] / 100  
            rr  = df[rr_col]  / 100
            bf  = df[bf_col]  / 100
            volvol = np.sqrt(2) * bf / (atm ** 2)
            rho_sabr = (rr / atm) / volvol.replace(0, np.nan)
            key = f"{ccy}_{tenor}"
            results[f"{key}_rho_sabr"]  = rho_sabr
            results[f"{key}_RR"]  = df[rr_col]
            # results[f"{key}_volvol"]    = volvol
    return pd.DataFrame(results, index=df.index).dropna(how='all')




ccys         = ['EURUSD']   
tenors       = ['1W']
delta        = '25'

corr_windows = [20]
time_hist    = 252 * 3

full_data = get_Data(ccys, tenors, delta, time_hist)

df_real_SVC, all_raw = build_corr_multi_ccy(ccys, tenors, corr_windows, time_hist, delta)
df_implied_SVC = calc_implied_spot_vol_corr(full_data, ccys, tenors, delta)

df_net = pd.concat([df_real_SVC, df_implied_SVC], axis=1).sort_index().dropna()


print(df_net)
