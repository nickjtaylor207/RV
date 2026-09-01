import functools
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import pytz
from xbbg import blp
import pdblp
import QuantLib as ql




TENOR_DAYS = {'ON': 1, '1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60, '3M': 91, '6M': 182, '9M': 273, '1Y': 365}
FWD_YIELD_TENORS = ['1W', '2W', '3W', '1M', '2M', '3M', '6M', '9M', '1Y']


# TENOR_DAYS = {'1W': 7, '2W': 14, '3W': 21, '1M': 30}
DELTA_POINTS = [35, 25, 15, 10, 5]

# FWD_YIELD_TENORS = ['1W', '2W', '3W', '1M']


CCY_FWD_YIELD_PREFIX = {
    'EUR': 'EURI',
    'GBP': 'GBPI',
    'JPY': 'JPYI',
    'CHF': 'CHFI',
    'AUD': 'AUDI',
    'NZD': 'NZDI',
    'CAD': 'CADI',
    'NOK': 'NOKI',
    'SEK': 'SEKI',
    'BRL': 'BCNI',
    'MXN': 'MXNI'}


# Currency -> QuantLib settlement calendar, used to classify weekends/holidays
# for the weekend-weighted vol interpolation (see vol_surface._classify_period).
# Keys mirror CCY_FWD_YIELD_PREFIX, plus USD (the implicit other leg of every
# pair quoted here). EUR maps to TARGET, the eurozone interbank calendar.
CCY_CALENDAR = {
    'USD': ql.UnitedStates(ql.UnitedStates.Settlement),
    'EUR': ql.TARGET(),
    'GBP': ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
    'JPY': ql.Japan(),
    'CHF': ql.Switzerland(),
    'AUD': ql.Australia(),
    'NZD': ql.NewZealand(),
    'CAD': ql.Canada(ql.Canada.Settlement),
    'NOK': ql.Norway(),
    'SEK': ql.Sweden(),
    'BRL': ql.Brazil(ql.Brazil.Settlement),
    'MXN': ql.Mexico(),
}


def calendar_for_pair(pair: str):
    """
    Build the JointCalendar for an FX pair from both legs. A day is a business
    day only when BOTH legs' markets are open; a holiday in either leg makes the
    joined day a holiday (QuantLib's default JoinHolidays rule). Handles USD
    pairs and crosses uniformly via pair[:3] / pair[3:].

    Raises ValueError if either leg's currency is not mapped in CCY_CALENDAR,
    so an unknown pair fails loudly rather than silently using a wrong calendar.
    """
    if not pair:
        raise ValueError("calendar_for_pair requires a currency pair, got empty/None.")
    base, quote = pair[:3].upper(), pair[3:].upper()
    try:
        return ql.JointCalendar(CCY_CALENDAR[base], CCY_CALENDAR[quote])
    except KeyError as e:
        raise ValueError(
            f"No calendar mapped for currency {e.args[0]} (pair {pair}). "
            f"Add it to CCY_CALENDAR."
        ) from e




SPOT_DAYS = {
    # Default is T+2 for all G10
    'DEFAULT': 2,
    # T+1 exceptions — any pair where either leg is CAD
    'CAD': 1,
}

# 'Special' Latin American currencies (Clark / FX Option Pricing §1.4): for these,
# even the T+2 *interim* date must skip USD holidays. BRL is explicitly NOT special.
SPECIAL_LATAM = {'MXN', 'ARS', 'CLP'}


@dataclass(frozen=True)
class FXCalendar:
    """
    Per-leg calendar bundle for one FX pair, carrying everything the book-faithful
    date engine (trading_calendar) needs to distinguish USD holidays from the
    non-USD legs' holidays.

    ccy1 / ccy2     : base / quote currencies.
    spot_days       : T+1 or T+2 settlement lag.
    special_latam   : True if either leg is a 'special' LatAm ccy (MXN/ARS/CLP).
    cal_trade       : good business day for BOTH legs (JointCalendar ccy1+ccy2) —
                      used to pick the trade/horizon date.
    cal_interim     : good business day for the NON-USD leg(s) only — used for the
                      T+2 interim hop, day/week expiries, and expiry validity
                      (USD holidays are disregarded here, per the book).
    cal_settle      : good settlement day = ccy1 + ccy2 + USD holidays all skipped —
                      used for the final spot date and delivery-date validity.
    """
    ccy1:          str
    ccy2:          str
    spot_days:     int
    special_latam: bool
    cal_trade:     object
    cal_interim:   object
    cal_settle:    object


@functools.lru_cache(maxsize=None)
def fx_calendar(pair: str) -> FXCalendar:
    """
    Build (and cache) the FXCalendar for a pair. Cached so repeated calls in tight
    loops (historical calibration, pillar-day computation) are essentially free.
    """
    if not pair:
        raise ValueError("fx_calendar requires a currency pair, got empty/None.")
    base, quote = pair[:3].upper(), pair[3:].upper()
    try:
        cal_base  = CCY_CALENDAR[base]
        cal_quote = CCY_CALENDAR[quote]
    except KeyError as e:
        raise ValueError(
            f"No calendar mapped for currency {e.args[0]} (pair {pair}). "
            f"Add it to CCY_CALENDAR."
        ) from e
    cal_usd = CCY_CALENDAR['USD']

    # Non-USD legs only (USD holidays disregarded for interim / expiry).
    non_usd = [c for c in (cal_base, cal_quote) if c is not cal_usd]
    if len(non_usd) == 2:
        cal_interim = ql.JointCalendar(non_usd[0], non_usd[1])
    elif len(non_usd) == 1:
        cal_interim = non_usd[0]
    else:                              # both legs USD (not a real pair) — weekends only
        cal_interim = ql.WeekendsOnly()

    return FXCalendar(
        ccy1=base, ccy2=quote,
        spot_days=get_spot_days(pair),
        special_latam=(base in SPECIAL_LATAM or quote in SPECIAL_LATAM),
        cal_trade=ql.JointCalendar(cal_base, cal_quote),
        cal_interim=cal_interim,
        cal_settle=ql.JointCalendar(cal_base, cal_quote, cal_usd),
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


# INTRADAY SPOT
def pull_intradayCCY_close(tickers: List[str], days: int, interv: int) -> pd.DataFrame:
    df_ccys = {}
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    eastern = pytz.timezone("US/Eastern")
    end_time_et   = eastern.localize(datetime.now())
    start_time_et = end_time_et - timedelta(days=days)
    end_time_gmt   = end_time_et.astimezone(pytz.utc)
    start_time_gmt = start_time_et.astimezone(pytz.utc)
    for t in tickers:
        df = con.bdib(ticker=f"{t} Curncy", start_datetime=start_time_gmt,
                      end_datetime=end_time_gmt, event_type="TRADE", interval=interv)
        df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        df = df.reset_index()
        df['date']       = df['time'].dt.date
        df['time_close'] = df['time'].dt.time
        df["datetime"]   = pd.to_datetime(df["date"].astype(str) + " " + df["time_close"].astype(str))
        df = df.sort_values("datetime").set_index("datetime")[['close']]
        df.columns = [t]
        df_ccys[t] = df
    out = pd.concat(df_ccys, axis=1)
    out.columns = out.columns.get_level_values(1)
    return out.dropna()

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
            ticker = f'{prefix}{tenor} BGN Curncy'
            all_tickers.append(ticker)
            col_map[ticker] = (ccy, tenor)
    df = blp.bdh(
        tickers    = all_tickers,
        flds       = 'PX_LAST',
        start_date = start_date,
        end_date   = end_date)
    df.columns = [col_map[c[0]] for c in df.columns]
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['ccy', 'tenor'])
    return df / 100  # decimal


def get_spot_days(pair: str) -> int:
    """
    Returns the number of business days to spot settlement.
    CAD crosses settle T+1, everything else T+2.
    """
    ccy_base  = pair[:3]
    ccy_quote = pair[3:]
    if 'CAD' in (ccy_base, ccy_quote):
        return SPOT_DAYS['CAD']
    return SPOT_DAYS['DEFAULT']