import re
import calendar as _pycal
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import QuantLib as ql

"""
FX option date arithmetic — faithful to Clark, "Foreign Exchange Option Pricing"
(§1.4 Spot Settlement Rules, §1.5 Expiry and Delivery Rules).

Two layers:

1. Low-level business-day primitives (is_business_day, preceding/next_business_day,
   modified_following) that take a single QuantLib calendar. Used to pick the
   trade/horizon date.

2. The market-convention expiry engine (spot_from_horizon, delivery_from_spot,
   expiry_from_delivery, add_tenor) that takes an `fxc` = data.FXCalendar carrying
   the PER-LEG calendars. This is what makes USD-holiday handling correct:

     - months/years : today -> spot(T+x) -> +n months (modified following,
                      month-end aware) -> delivery -> back to expiry.
     - days/weeks   : count calendar days from today -> roll forward to a good
                      expiry day (USD holidays disregarded).

   The crucial subtlety (Clark §1.4): for T+2 the *interim* date may fall on a USD
   holiday — only the *final* spot date must avoid USD holidays. cal_interim vs
   cal_settle on the FXCalendar encode exactly that.

Build the fxc with  data.fx_calendar(pair)  and pass it in.
"""



_TENOR_RE = re.compile(r'^(\d+)([DWMY])$')


def _parse_tenor(tenor: str):
    """
    Parse a tenor string into (n, unit). Raises ValueError if the format is invalid.
    Valid forms: ND (days), NW (weeks), NM (months), NY (years).
    """
    m = _TENOR_RE.match(tenor.strip().upper())
    if not m:
        raise ValueError(
            f"Unrecognised tenor '{tenor}'. "
            f"Use ND (days), NW (weeks), NM (months), or NY (years), "
            f"e.g. '1D', '5D', '1W', '3W', '1M', '4M', '18M', '1Y', '2Y'. "
            f"Or pass an integer number of calendar days."
        )
    return int(m.group(1)), m.group(2)


def _to_ql(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


# ---------------------------------------------------------------------------
# Core business day primitives (single-calendar)
# ---------------------------------------------------------------------------

def is_business_day(d: date, cal=None) -> bool:
    """
    True if d is a business day.
    With cal: must be Mon–Fri AND not a holiday in the given calendar.
    Without cal: Mon–Fri only (weekends excluded, holidays not checked).
    """
    if d.weekday() >= 5:
        return False
    if cal is None:
        return True
    return cal.isBusinessDay(_to_ql(d))


def preceding_business_day(d: date, cal=None) -> date:
    """Roll backward until a business day is found (no-op if already one)."""
    while not is_business_day(d, cal):
        d -= timedelta(days=1)
    return d


def next_business_day(d: date, cal=None) -> date:
    """First business day strictly after d."""
    d += timedelta(days=1)
    while not is_business_day(d, cal):
        d += timedelta(days=1)
    return d


def _increment_business_days(d: date, n: int, cal=None) -> date:
    """Advance d by exactly n business days."""
    for _ in range(n):
        d = next_business_day(d, cal)
    return d


def _decrement_business_days(d: date, n: int, cal=None) -> date:
    """Move d back by exactly n business days."""
    for _ in range(n):
        d -= timedelta(days=1)
        while not is_business_day(d, cal):
            d -= timedelta(days=1)
    return d


def modified_following(d: date, cal=None) -> date:
    """
    Roll non-business days using the Modified Following convention:
      1. If d is a business day, return it unchanged.
      2. Roll forward to the next business day.
      3. If that crosses a calendar month boundary, roll backward instead.
    """
    if is_business_day(d, cal):
        return d
    candidate = next_business_day(d, cal)
    if candidate.month != d.month:
        return preceding_business_day(d, cal)
    return candidate


# ---------------------------------------------------------------------------
# tenor_offset — used to compute raw_entry when entry_days_back is None
# ---------------------------------------------------------------------------

def tenor_offset(tenor):
    """
    Return the dateutil/timedelta offset for a tenor string or integer.
    Supports subtraction from a date (used to roll back from today to entry).
    """
    if isinstance(tenor, int):
        return timedelta(days=tenor)
    if tenor.strip().upper() == 'ON':
        return timedelta(days=1)
    n, unit = _parse_tenor(tenor)
    if unit == 'D':
        return timedelta(days=n)
    if unit == 'W':
        return timedelta(weeks=n)
    if unit == 'M':
        return relativedelta(months=n)
    return relativedelta(years=n)


# ---------------------------------------------------------------------------
# Book-faithful engine (per-leg calendars via data.FXCalendar `fxc`)
# ---------------------------------------------------------------------------

def _is_jan1(d: date) -> bool:
    return d.month == 1 and d.day == 1


def _good(qlcal, d: date) -> bool:
    """True if d is a good business day in the given QuantLib calendar (weekends excluded)."""
    return qlcal.isBusinessDay(_to_ql(d))


def _next_good(qlcal, d: date) -> date:
    """First good business day strictly after d, per the given calendar."""
    d += timedelta(days=1)
    while not _good(qlcal, d):
        d += timedelta(days=1)
    return d


def spot_from_horizon(horizon: date, fxc) -> date:
    """
    Clark §1.4 SpotFromHorizonDate. Returns the spot (value) date for `horizon`.

    T+2: Step 1 advances one good business day to the interim date, skipping only
         the NON-USD legs' holidays (USD holidays are allowed on the interim) —
         unless a 'special' LatAm leg forces USD to be skipped there too.
    Step 2 (T+1 and T+2): advance one good business day to the final spot date,
         skipping ccy1, ccy2 AND USD holidays.
    """
    d = horizon
    if fxc.spot_days >= 2:                      # Step 1 — interim
        step1_cal = fxc.cal_settle if fxc.special_latam else fxc.cal_interim
        d = _next_good(step1_cal, d)
    d = _next_good(fxc.cal_settle, d)           # Step 2 — final spot
    return d


def _is_settle_month_end(spot: date, fxc) -> bool:
    """True if `spot` is the last good settlement day of its month."""
    return _next_good(fxc.cal_settle, spot).month != spot.month


def delivery_from_spot(spot: date, n: int, unit: str, fxc) -> date:
    """
    Clark §1.5.2 DeliveryFromSpot — roll the spot date forward by n months (or 12n
    for years) with month-end sticking and modified-following, validating against
    good settlement days (ccy1 + ccy2 + USD).
    """
    months = 12 * n if unit == 'Y' else n
    total  = (spot.month - 1) + months
    year   = spot.year + total // 12
    month  = total % 12 + 1
    ndays  = _pycal.monthrange(year, month)[1]

    if _is_settle_month_end(spot, fxc):
        day = ndays                            # month-end sticks to month-end
    else:
        day = min(spot.day, ndays)

    d = date(year, month, day)

    if day == ndays:                           # month-end: modified preceding
        while not _good(fxc.cal_settle, d):
            d -= timedelta(days=1)
        return d

    # otherwise: following, but don't cross the month (modified following)
    dd = d
    while not _good(fxc.cal_settle, dd):
        dd += timedelta(days=1)
    if dd.month != month:
        dd = d
        while not _good(fxc.cal_settle, dd):
            dd -= timedelta(days=1)
    return dd


def expiry_from_delivery(delivery: date, fxc) -> date:
    """
    Clark §1.5.2 ExpiryFromDelivery — the furthest date on/before `delivery` that is
    a good expiry day (not weekend / Jan-1 / non-USD-leg holiday; USD holidays are
    allowed) and whose own spot date is not beyond the delivery date.
    """
    d = delivery
    while (not _good(fxc.cal_interim, d)) or _is_jan1(d) or spot_from_horizon(d, fxc) > delivery:
        d -= timedelta(days=1)
    return d


def _days_weeks_expiry(horizon: date, n: int, unit: str, fxc) -> date:
    """
    Clark §1.5.1 — for days/weeks, count n (or 7n) CALENDAR days from today, then
    roll forward to a good expiry day (weekend / non-USD-leg holiday; USD holidays
    disregarded).
    """
    d = horizon + timedelta(days=(7 * n if unit == 'W' else n))
    while (not _good(fxc.cal_interim, d)) or _is_jan1(d):
        d += timedelta(days=1)
    return d


def add_tenor(horizon: date, tenor, fxc) -> date:
    """
    Compute the FX option EXPIRY date for `tenor` as of `horizon` (the trade date),
    faithful to Clark §1.4–1.5.

    Parameters
    ----------
    horizon : trade/valuation date (should already be a good business day).
    tenor   : 'ON', ND, NW, NM, NY (any N), or an int = N calendar days.
    fxc     : data.FXCalendar for the pair (build via data.fx_calendar(pair)).

    Returns the expiry date. For months/years it routes today -> spot -> delivery
    -> expiry; for days/weeks it counts calendar days from today to the expiry.
    """
    if isinstance(tenor, int):
        return _days_weeks_expiry(horizon, tenor, 'D', fxc)

    t = tenor.strip().upper()
    if t == 'ON':                              # overnight: next good expiry day
        return _next_good(fxc.cal_interim, horizon)

    n, unit = _parse_tenor(t)

    if unit in ('D', 'W'):
        return _days_weeks_expiry(horizon, n, unit, fxc)

    # unit == 'M' or 'Y'
    spot     = spot_from_horizon(horizon, fxc)
    delivery = delivery_from_spot(spot, n, unit, fxc)
    return expiry_from_delivery(delivery, fxc)
