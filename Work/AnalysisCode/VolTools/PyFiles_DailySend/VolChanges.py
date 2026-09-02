import pandas as pd
import numpy as np
from xbbg import blp
from datetime import datetime, timedelta


def get_Vols(ccys, I_tenors, years):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    df_vols = {}
    for ccy in ccys:
        if I_tenors:
            for tenor in I_tenors:
                ticker_IV = f"{ccy}V{tenor} BGN Curncy"
                data_IV = blp.bdh(tickers=[ticker_IV], flds="PX_LAST",
                                  start_date=start_date, end_date=end_date)
                if not data_IV.empty:
                    col_name = f"{ccy}_{tenor}"
                    data_IV.columns = [col_name]
                    data_IV.index = pd.to_datetime(data_IV.index)
                    df_vols[col_name] = data_IV
                else:
                    print(f"No data for {ticker_IV}, skipping.")
    if df_vols:
        df_vols_all = pd.concat(df_vols.values(), axis=1)
        df_vols_all.index = pd.to_datetime(df_vols_all.index)
        return df_vols_all
    else:
        print("No data retrieved for any ticker.")
        return pd.DataFrame()


def get_RiskReversals(ccys, tenors, delta, years):
    """
    Fetch risk reversal data from Bloomberg.

    Parameters
    ----------
    ccys   : list of currency pairs e.g. ['EURUSD', 'USDJPY']
    tenors : list of tenors e.g. ['1M', '3M', '6M', '1Y']
    delta  : int, the delta strike e.g. 25 for 25-delta
    years  : int or float, lookback in years
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    df_rr = {}
    for ccy in ccys:
        for tenor in tenors:
            ticker_RR = f"{ccy}{delta}R{tenor} BGN Curncy"
            data_RR = blp.bdh(tickers=[ticker_RR], flds="PX_LAST",
                              start_date=start_date, end_date=end_date)
            if not data_RR.empty:
                col_name = f"{ccy}_{delta}RR_{tenor}"
                data_RR.columns = [col_name]
                data_RR.index = pd.to_datetime(data_RR.index)
                df_rr[col_name] = data_RR
            else:
                print(f"No data for {ticker_RR}, skipping.")
    if df_rr:
        df_rr_all = pd.concat(df_rr.values(), axis=1)
        df_rr_all.index = pd.to_datetime(df_rr_all.index)
        return df_rr_all
    else:
        print("No RR data retrieved.")
        return pd.DataFrame()


def summarise_moves(df, lookbacks_years):
    """
    For a given dataframe of vol/RR time series, produce a summary of:
      - today's absolute and % change
      - percentile of today's change vs each lookback window

    Parameters
    ----------
    df              : DataFrame with DatetimeIndex, columns = instrument series
    lookbacks_years : dict mapping label -> years, e.g. {'3M': 0.25, '1Y': 1, '3Y': 3, '5Y': 5}

    Returns
    -------
    DataFrame sorted by % change
    """
    # Approx trading days per year
    TRADING_DAYS = 252

    daily_changes = df.diff()
    today_change  = daily_changes.iloc[-1]
    diff          = df.iloc[-1] - df.iloc[-2]
    pct_diff      = (diff / df.iloc[-2]) * 100

    result = pd.DataFrame({'Change': diff, '%Change': pct_diff})

    for label, yrs in lookbacks_years.items():
        n_days = int(yrs * TRADING_DAYS)
        window = daily_changes.iloc[-n_days:]
        # Fraction of days on which the change was smaller than today's
        pctile = (window < today_change).mean() * 100
        result[f'Pctile_{label}'] = pctile

    return result.sort_values('Pctile_1Y')











ccys    = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
           'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDZAR']
tenors  = ['1M', '2M', '3M']
delta   = 25
years   = 5          # fetch 5Y of history so all lookback windows are available

lookbacks = {
    '3M': 3/12,
    '1Y': 1,
    '3Y': 3,
    '5Y': 5,}


df_vol = get_Vols(ccys, tenors, years)
df_rr  = get_RiskReversals(ccys, tenors, delta, years)




print("=" * 60)
print("ATM IMPLIED VOL — Daily Move Summary")
print("=" * 60)
summary_vol = summarise_moves(df_vol, lookbacks)
print(summary_vol.round(2))

print("\n" + "=" * 60)
print(f"{delta}-Delta RISK REVERSAL — Daily Move Summary")
print("=" * 60)
summary_rr = summarise_moves(df_rr, lookbacks)
print(summary_rr.round(2))



