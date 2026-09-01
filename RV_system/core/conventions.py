"""
FX market conventions — pure, no I/O, no Bloomberg.

Tenor/delta pillars, per-currency QuantLib calendars, the FXCalendar bundle
and spot-lag rules. Split out of the old data.py so that core/ never has to
import market/ (which owns the Bloomberg dependency).
"""

import functools
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import pytz
import QuantLib as ql




TENOR_DAYS = {'ON': 1, '1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60, '3M': 91, '6M': 182, '9M': 273, '1Y': 365}

# Tenor pillars for the RATE curves. Deliberately NOT the same list as
# TENOR_DAYS (which is the vol grid):
#   * 'ON' is excluded — the curves start at 1W, and both legs clamp to their
#     1W quote below that, which keeps the rate DIFFERENTIAL exact at the short
#     end. Giving one currency an ON node the other lacks would reintroduce a
#     CIP error on sub-1W options.
#   * '3W' is excluded — USOSFR3Z exists but XXXI3W does not, for any currency.
#     It used to sit in this list and was silently swallowed by a
#     `except KeyError: continue`, along with '1Y' (see FWD_YIELD_SUFFIX).
FWD_YIELD_TENORS = ['1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y']


# TENOR_DAYS = {'1W': 7, '2W': 14, '3W': 21, '1M': 30}
DELTA_POINTS = [35, 25, 15, 10, 5]

# FWD_YIELD_TENORS = ['1W', '2W', '3W', '1M']


# Ticker suffix for the forward-implied yield of each tenor. The suffix is NOT
# always the tenor string: the 1Y point is quoted as XXXI12M, and XXXI1Y does
# not resolve on any currency.
FWD_YIELD_SUFFIX = {
    '1W': '1W', '2W': '2W', '1M': '1M', '2M': '2M',
    '3M': '3M', '6M': '6M', '9M': '9M', '1Y': '12M',
}

# USD OIS (fixed vs compounded SOFR) par swap rates, quoted in percent, ACT/360.
# These are the curve Bloomberg implies the XXXI forward-implied yields off:
# backing the USD rate out of market FX forward points plus XXXI reproduces
# these to within 0.03bp across every G10 pair and tenor tested. Pulling both
# families together is therefore what makes the FX forward arbitrage-free; a
# flat overnight SOFR paired with XXXI breaks CIP by ~5bp at 1M and ~38bp at 1Y.
#
# Naming: A-K = 1M-11M, plain digit = whole years (USOSFR1 = 1Y, USOSFR2 = 2Y),
# 1A-1K = 13M-23M, and 1Z/2Z/3Z = 1W/2W/3W.
#
# NOTE these are PAR swap rates. Out to 1Y the annual fixed leg has a single
# payment, so par == zero and no bootstrap is needed. If TENOR_DAYS is ever
# extended past 1Y, the >1Y points MUST be bootstrapped before use.
USD_OIS_TICKER = {
    '1W': 'USOSFR1Z', '2W': 'USOSFR2Z', '3W': 'USOSFR3Z',
    '1M': 'USOSFRA',  '2M': 'USOSFRB',  '3M': 'USOSFRC',
    '6M': 'USOSFRF',  '9M': 'USOSFRI',  '1Y': 'USOSFR1',
}

# Money-market day-count basis per currency. Both the OIS quotes and the
# forward-implied yields are SIMPLE rates on these bases, so they must be
# converted before being used as the continuous rates the pricer expects.
# Verified by CIP: assuming 360 everywhere leaves GBP/AUD/NZD/CAD adrift by
# 3-6bp at 3M; with this map every currency closes to under 0.15bp.
MM_BASIS = {
    'USD': 360, 'EUR': 360, 'JPY': 360, 'CHF': 360, 'NOK': 360,
    'SEK': 360, 'BRL': 360, 'MXN': 360,
    'GBP': 365, 'AUD': 365, 'NZD': 365, 'CAD': 365,
}
MM_BASIS_DEFAULT = 360


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