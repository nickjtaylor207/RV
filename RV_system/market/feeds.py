"""Bloomberg pulls. The ONLY module in the stack that talks to xbbg/pdblp."""

import warnings
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import pytz
from xbbg import blp
import pdblp

from core.conventions import (
    TENOR_DAYS, FWD_YIELD_TENORS, DELTA_POINTS, CCY_FWD_YIELD_PREFIX,
    FWD_YIELD_SUFFIX, USD_OIS_TICKER,
)






# --------------------- Spot Data Pull ---------------------
# DAILY SPOT
def pull_dailyCCY_close(ccys: List[str], days: int) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    df_ccys = {}
    for ccy in ccys:
        ticker = f'{ccy} BGN Curncy'
        df = blp.bdh(tickers=ticker, flds='PX_LAST',
                     start_date=start_date, end_date=end_date)
        df.columns = [ccy]
        df_ccys[ccy] = df
    out = pd.concat(df_ccys, axis=1)
    out.columns = out.columns.get_level_values(1)
    return out


# # INTRADAY SPOT
# def pull_intradayCCY_close(tickers: List[str], days: int, interv: int) -> pd.DataFrame:
#     df_ccys = {}
#     con = pdblp.BCon(debug=False, port=8194, timeout=5000)
#     con.start()
#     eastern = pytz.timezone("US/Eastern")
#     end_time_et   = eastern.localize(datetime.now())
#     start_time_et = end_time_et - timedelta(days=days)
#     end_time_gmt   = end_time_et.astimezone(pytz.utc)
#     start_time_gmt = start_time_et.astimezone(pytz.utc)
#     for t in tickers:
#         df = con.bdib(ticker=f"{t} Curncy", start_datetime=start_time_gmt,
#                       end_datetime=end_time_gmt, event_type="TRADE", interval=interv)
#         df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
#         df = df.reset_index()
#         df['date']       = df['time'].dt.date
#         df['time_close'] = df['time'].dt.time
#         df["datetime"]   = pd.to_datetime(df["date"].astype(str) + " " + df["time_close"].astype(str))
#         df = df.sort_values("datetime").set_index("datetime")[['close']]
#         df.columns = [t]
#         df_ccys[t] = df
#     out = pd.concat(df_ccys, axis=1)
#     out.columns = out.columns.get_level_values(1)
#     return out.dropna()

# --------------------- VOL Data Pull ---------------------
# ATM, RR, FLY - 5d, 10d, 15d, 25d, 35d
def pull_fx_vol_surface(pairs: List[str], days: int) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    all_tickers = []
    col_map     = {}
    for pair in pairs:
        for tenor in TENOR_DAYS:
            # ATM
            atm_t = f'{pair}V{tenor} BGN Curncy'
            all_tickers.append(atm_t)
            col_map[atm_t] = (pair, tenor, 'ATM')
 
            # RR and BF across delta points
            for d in DELTA_POINTS:
                rr_t = f'{pair}{d}R{tenor} BGN Curncy'
                bf_t = f'{pair}{d}B{tenor} BGN Curncy'
                all_tickers += [rr_t, bf_t]
                col_map[rr_t] = (pair, tenor, f'RR{d}')
                col_map[bf_t] = (pair, tenor, f'BF{d}')
    df = blp.bdh(tickers=all_tickers, flds='PX_LAST',
                 start_date=start_date, end_date=end_date)
    df.columns = [col_map[c[0]] for c in df.columns]
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['pair', 'tenor', 'type'])
    return df





# --------------------- Rates Data Pull ---------------------
# USD Depo Rate
def pull_sofr(days: int) -> pd.Series:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    df = blp.bdh(
        tickers    = 'SOFRRATE Index',
        flds       = 'PX_LAST',
        start_date = start_date,
        end_date   = end_date)
    df.columns = ['SOFR']
    return df['SOFR'] / 100  # decimal
#
def pull_fwd_implied_yields(ccys: List[str], days: int) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    all_tickers = []
    col_map     = {}
    for ccy in ccys:
        if ccy == 'USD':
            continue  # USD handled by SOFR
        prefix = CCY_FWD_YIELD_PREFIX.get(ccy)
        if prefix is None:
            warnings.warn(f"No forward yield prefix defined for {ccy}, skipping.")
            continue
        for tenor in FWD_YIELD_TENORS:
            # The suffix is not always the tenor string -- 1Y is quoted as I12M.
            ticker = f'{prefix}{FWD_YIELD_SUFFIX[tenor]} BGN Curncy'
            all_tickers.append(ticker)
            col_map[ticker] = (ccy, tenor)
    df = blp.bdh(
        tickers    = all_tickers,
        flds       = 'PX_LAST',
        start_date = start_date,
        end_date   = end_date)
    df.columns = [col_map[c[0]] for c in df.columns]
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['ccy', 'tenor'])
    return df / 100  # decimal, SIMPLE on each ccy's MM_BASIS


def pull_usd_ois_curve(days: int) -> pd.DataFrame:
    """
    USD OIS (fixed vs compounded SOFR) par rates across FWD_YIELD_TENORS.

    Returned in the SAME shape as pull_fwd_implied_yields -- a (ccy, tenor)
    MultiIndex with ccy='USD' -- so the two frames concatenate into one rate
    table and USD stops being a special case downstream.

    This replaces the flat SOFRRATE overnight fixing as the USD leg. Flat
    overnight is ~5bp below the 1M point and ~38bp below the 1Y point, and
    since the XXXI forward-implied yields are implied off THIS curve, pairing
    them with a flat rate makes S*exp((r_d-r_f)*T) miss the market forward.

    Values are decimal and SIMPLE on ACT/360 -- convert before use.
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    all_tickers = []
    col_map     = {}
    for tenor in FWD_YIELD_TENORS:
        code = USD_OIS_TICKER.get(tenor)
        if code is None:
            warnings.warn(f"No USD OIS ticker mapped for {tenor}, skipping.")
            continue
        ticker = f'{code} Curncy'
        all_tickers.append(ticker)
        col_map[ticker] = ('USD', tenor)
    df = blp.bdh(
        tickers    = all_tickers,
        flds       = 'PX_LAST',
        start_date = start_date,
        end_date   = end_date)
    df.columns = [col_map[c[0]] for c in df.columns]
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['ccy', 'tenor'])
    return df / 100  # decimal

