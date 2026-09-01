"""
Test core/option.py by describing trades the way you'd say them out loud.

    TRADES = ["USDJPY 1M 25d call 20mm"]

Read-only: imports core/option.py, core/greeks.py, core/calendar.py,
core/conventions.py and (in live mode) market/. Edits nothing.

WHAT IT EXERCISES, IN ORDER
---------------------------
  add_tenor / fx_calendar      -> the real expiry date for '1M'
  atm_forward_strike           -> the ATMF reference
  strike_from_delta            -> '25d' -> a strike
  FXOption.time_to_expiry      -> T
  FXOption.price_domestic      -> premium in quote ccy
  FXOption.value_base          -> V = P/S, the unit the stack accounts in
  FXOption.intrinsic_base      -> intrinsic, so you can split out time value
  FXOption.spot_delta          -> and the round-trip back to your target delta
  FXOption.call_delta_equivalent
  FXOption.base_ccy_partials   -> the six derivatives of V
  FXOption.greeks              -> GreekVector, then the trader view
  GreekVector.__add__          -> the book total across every leg

TRADE SHORTHAND
---------------
  "USDJPY 1M 25d call 20mm"        long  20mm USD, 1M 25-delta call
  "short USDJPY 1M 25d call 20mm"  same, sold
  "USDJPY 1M 25d call -20mm"       same, sold (leading minus)
  "EURUSD 3M ATM put 50mm"         struck at the ATM FORWARD
  "USDJPY 2W k=156.50 put 10mm"    explicit strike

  order matters only for tenor-vs-delta: the tenor must come first, because
  '1M' and '25d' are the same shape of token. Everything else is free.

  notional needs a suffix (20mm / 20m / 500k / 1bn) or to be a plain integer
  >= 1000. A strike must be written k=156.50 (or @156.50) so it can never be
  confused with a notional.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import re
from datetime import date

import numpy as np

from core.option import (FXOption, forward_price, atm_forward_strike,
                         strike_from_delta)
from core.greeks import GreekVector
from core.calendar import add_tenor, preceding_business_day
from core.conventions import fx_calendar


# ====================================================================== #
#  1. THE TRADES
# ====================================================================== #



# ====================================================================== #
#  2. WHERE THE MARKET COMES FROM
# ====================================================================== #
#   'live'   -> FXVolDataset.build + MarketSnapshot. Real spot, real SABR
#               smile vol at the solved strike, real forward-implied rates.
#   'manual' -> the MANUAL block below. FLAT vol (no smile), so the strike
#               solve is a single pass. Use this to control every input.
MARKET = 'live'

HISTORY_DAYS = 400          # live only; needs >= 1Y so the 1Y pillar exists

MANUAL = dict(
    today = date(2026, 8, 20),
    spot  = {'USDJPY': 158.94, 'EURUSD': 1.1673},
    vol   = {'USDJPY': 0.0750, 'EURUSD': 0.0539},    # flat, decimal
    rates = {'USD': 0.036300, 'JPY': 0.008372, 'EUR': 0.021200},   # continuous
)


# ====================================================================== #
#  parsing
# ====================================================================== #
_SUFFIX = {'k': 1e3, 'm': 1e6, 'mm': 1e6, 'b': 1e9, 'bn': 1e9}


def parse_trade(s: str) -> dict:
    out = dict(pair=None, tenor=None, target_delta=None, K=None,
               option_type=None, notional=None, direction=+1, raw=s)

    for tok in s.replace(',', ' ').split():
        tl = tok.lower()

        if tl in ('short', 'sell', 'sold'):
            out['direction'] = -1;  continue
        if tl in ('long', 'buy', 'bought'):
            out['direction'] = +1;  continue
        if tl in ('call', 'c'):
            out['option_type'] = 'call';  continue
        if tl in ('put', 'p'):
            out['option_type'] = 'put';   continue
        if tl in ('atm', 'atmf', 'atmfwd'):
            out['K'] = 'ATMF';  continue

        if out['pair'] is None and re.fullmatch(r'[a-z]{6}', tl):
            out['pair'] = tok.upper();  continue

        m = re.fullmatch(r'(?:k=|@)(\d+(?:\.\d+)?)', tl)          # k=156.50
        if m:
            out['K'] = float(m.group(1));  continue

        m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*delta', tl)          # 25delta
        if m:
            out['target_delta'] = float(m.group(1)) / 100.0;  continue

        # '1M' / '25d' collide. Tenor wins while unset -- hence "tenor first".
        if out['tenor'] is None and re.fullmatch(r'(on|\d+[dwmy])', tl):
            out['tenor'] = tok.upper();  continue

        m = re.fullmatch(r'(\d+(?:\.\d+)?)d', tl)                 # 25d
        if m:
            out['target_delta'] = float(m.group(1)) / 100.0;  continue

        m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)(mm|bn|[kmb])?', tl)
        if m:
            v = float(m.group(1))
            suf = m.group(2)
            if suf:
                v *= _SUFFIX[suf]
            elif abs(v) < 1000:
                raise ValueError(
                    f"'{tok}' in {s!r}: bare number under 1000. Write a notional "
                    f"as 20mm/500k, and a strike as k={tok}.")
            if v < 0:
                out['direction'] = -1
            out['notional'] = abs(v);  continue

        raise ValueError(f"cannot parse token '{tok}' in {s!r}")

    for f in ('pair', 'tenor', 'option_type', 'notional'):
        if out[f] is None:
            raise ValueError(f"{s!r} is missing the {f}")
    if out['target_delta'] is None and out['K'] is None:
        raise ValueError(f"{s!r} needs a delta (25d), a strike (k=...), or ATM")
    return out


def signed_delta(t: dict) -> float:
    """strike_from_delta wants a SIGNED spot delta: +0.25 call, -0.25 put."""
    return t['target_delta'] * (1 if t['option_type'] == 'call' else -1)


def hdr(s: str) -> None:
    print('\n' + '=' * 96);  print(s);  print('=' * 96)


def sub(s: str) -> None:
    print(f'\n  --- {s} ' + '-' * max(0, 86 - len(s)))


# ====================================================================== #
#  market resolution
# ====================================================================== #
class LiveMarket:
    """Thin wrapper so the printing code does not care where data came from."""

    def __init__(self, pairs):
        from market.dataset import FXVolDataset
        from market.snapshot import MarketSnapshot, business_dates
        print(f'Building dataset for {sorted(pairs)} ({HISTORY_DAYS}d)...')
        ds = FXVolDataset.build(pairs=sorted(pairs), days=HISTORY_DAYS)
        self.snaps = {p: MarketSnapshot.at(ds, p, business_dates(ds, p)[-1])
                      for p in pairs}
        self.smile = True

    def today(self, pair):        return self.snaps[pair].date
    def spot(self, pair):         return self.snaps[pair].spot
    def rates(self, pair, t):     return self.snaps[pair].rates(t)
    def atm_vol(self, pair, t):   return self.snaps[pair].atm_vol(t)
    def smile_vol(self, pair, K, t):  return self.snaps[pair].smile_vol(K, t)

    def solve(self, pair, tgt, otype, expiry):
        return self.snaps[pair].solve_strike_and_vol(tgt, otype, expiry)


class ManualMarket:
    """Flat vol -- no smile, so smile_vol(K) is the ATM vol at any strike."""

    def __init__(self, pairs):
        self.smile = False
        for p in pairs:
            for d, n in ((MANUAL['spot'], 'spot'), (MANUAL['vol'], 'vol')):
                if p not in d:
                    raise KeyError(f"MANUAL['{n}'] has no entry for {p}")
            for c in (p[:3], p[3:]):
                if c not in MANUAL['rates']:
                    raise KeyError(f"MANUAL['rates'] has no entry for {c}")

    def today(self, pair):        return MANUAL['today']
    def spot(self, pair):         return MANUAL['spot'][pair]
    def atm_vol(self, pair, t):   return MANUAL['vol'][pair]
    def smile_vol(self, pair, K, t):  return MANUAL['vol'][pair]

    def rates(self, pair, t):
        return MANUAL['rates'][pair[3:]], MANUAL['rates'][pair[:3]]   # (r_d, r_f)

    def solve(self, pair, tgt, otype, expiry):
        today = MANUAL['today']
        t_days = max((expiry - today).days, 1e-6)
        S = self.spot(pair)
        r_d, r_f = self.rates(pair, t_days)
        sig = self.atm_vol(pair, t_days)
        return strike_from_delta(pair, S, expiry, today, r_d, r_f,
                                 sig, tgt, otype), sig


# ====================================================================== #
def price_leg(t: dict, mkt) -> GreekVector:
    pair, otype = t['pair'], t['option_type']
    base, quote = pair[:3], pair[3:]
    N, dirn = t['notional'], t['direction']
    side = 'LONG' if dirn > 0 else 'SHORT'

    today = mkt.today(pair)
    S     = mkt.spot(pair)

    # ---- expiry: the real one, off the pair's own calendar ---------------
    fxc    = fx_calendar(pair)
    entry  = preceding_business_day(today, fxc.cal_trade)
    expiry = add_tenor(entry, t['tenor'], fxc)
    t_days = (expiry - today).days
    r_d, r_f = mkt.rates(pair, t_days)
    T_disp = t_days / 365.0

    hdr(f"{side} {N:,.0f} {base}   {pair} {t['tenor']} "
        f"{(str(int(t['target_delta']*100)) + 'd') if t['target_delta'] else ''}"
        f"{otype.upper()}        [{t['raw']}]")

    sub('CONTRACT  (core.calendar.add_tenor + core.conventions.fx_calendar)')
    print(f"    valuation      {today}")
    print(f"    trade date     {entry}   (preceding business day, {fxc.ccy1}+{fxc.ccy2} joint)")
    print(f"    expiry         {expiry}   <- add_tenor(entry, '{t['tenor']}', fxc)")
    print(f"    t_days         {t_days}      T = {T_disp:.8f}y  (ACT/365)")
    print(f"    spot           {S:.6f}")
    print(f"    r_d ({quote})      {r_d*100:.6f}%     r_f ({base})  {r_f*100:.6f}%")
    F = forward_price(S, r_d, r_f, T_disp)
    print(f"    forward        {F:.6f}   <- forward_price(S, r_d, r_f, T)")
    print(f"    ATMF strike    {atm_forward_strike(S, r_d, r_f, expiry, today):.6f}"
          f"   <- atm_forward_strike(...)")
    atm = mkt.atm_vol(pair, t_days)
    print(f"    ATM vol        {atm*100:.6f}%")

    # ---- strike --------------------------------------------------------
    sub('STRIKE  (core.option.strike_from_delta)')
    if t['K'] == 'ATMF':
        K = atm_forward_strike(S, r_d, r_f, expiry, today)
        sigma = mkt.smile_vol(pair, K, t_days)
        print(f"    struck at the ATM FORWARD, K = {K:.6f}")
    elif t['K'] is not None:
        K = float(t['K'])
        sigma = mkt.smile_vol(pair, K, t_days)
        print(f"    explicit strike K = {K:.6f}")
    else:
        tgt = signed_delta(t)
        # One pass at the ATM vol -- what strike_from_delta alone gives you.
        try:
            K_seed = strike_from_delta(pair, S, expiry, today, r_d, r_f,
                                       atm, tgt, otype)
        except ValueError as e:
            print(f"    strike_from_delta FAILED at the ATM vol: {e}")
            print(f"    -> the [0.85S, 1.20S] bracket does not contain this strike.")
            print(f"       Widen lo_mult/hi_mult in core/option.py for this wing.")
            raise
        sig_seed = mkt.smile_vol(pair, K_seed, t_days)
        print(f"    target delta   {tgt:+.4f}  ({otype})")
        print(f"    one pass at ATM vol {atm*100:.4f}%  ->  K = {K_seed:.6f}")
        print(f"    smile vol at that K                ->  {sig_seed*100:.6f}%")

        K, sigma = mkt.solve(pair, tgt, otype, expiry)
        if mkt.smile:
            print(f"    fixed point (snapshot.solve_strike_and_vol):")
            print(f"      K = {K:.6f}   sigma = {sigma*100:.6f}%")
            print(f"      moved {K - K_seed:+.6f} in strike, "
                  f"{(sigma - atm)*100:+.4f} vol pts vs ATM")
            print(f"    (strike_from_delta takes vol as an INPUT. The smile vol")
            print(f"     depends on the strike, so one pass is not self-consistent --")
            print(f"     market/snapshot.py owns that iteration, not core/option.py.)")
        else:
            print(f"    flat vol -> no circularity, one pass IS the answer")

    opt = FXOption(pair=pair, K=K, expiry=expiry, option_type=otype,
                   S0=S, r_d=r_d, r_f=r_f, sigma0=sigma)
    print(f"\n    {opt!r}")
    print(f"    vol used for pricing: {sigma*100:.6f}%")

    # ---- delta round trip ----------------------------------------------
    sub('DELTAS  (FXOption.spot_delta / call_delta_equivalent)')
    d_spot = opt.spot_delta(S, sigma, r_d, r_f, today)
    d_ceq  = opt.call_delta_equivalent(S, sigma, r_d, r_f, today)
    print(f"    spot_delta              {d_spot:+.8f}   (unadjusted, per 1 {base})")
    print(f"    call_delta_equivalent   {d_ceq:+.8f}   (smile-grid index)")
    if t['target_delta'] is not None:
        resid = d_spot - signed_delta(t)
        tag = 'OK' if abs(resid) < 1e-5 else ('OK*' if abs(resid) < 1e-3 else 'OFF')
        print(f"    [{tag}] round trip vs target {signed_delta(t):+.4f}: "
              f"residual {resid:+.2e}")
        if mkt.smile and abs(resid) > 1e-5:
            print(f"         * left over from the 5-iteration cap in")
            print(f"           solve_strike_and_vol (STRIKE_SOLVE_ITERS). It grows")
            print(f"           in the wings, where dsigma/dK is steepest -- exactly")
            print(f"           where market/snapshot.py's docstring says it would.")

    # ---- value ----------------------------------------------------------
    sub('VALUE  (FXOption.price_domestic / value_base / intrinsic_base)')
    P    = opt.price_domestic(S, sigma, r_d, r_f, today)
    V    = opt.value_base(S, sigma, r_d, r_f, today)
    intr = opt.intrinsic_base(S)
    pip  = 0.01 if quote == 'JPY' else 0.0001
    print(f"    price_domestic   {P:>16.10f}   {quote} per 1 {base}")
    print(f"    value_base       {V:>16.10f}   {base} per 1 {base}  (= P/S)")
    print(f"    intrinsic_base   {intr:>16.10f}   {base} per 1 {base}")
    print(f"    time value       {V - intr:>16.10f}   {base} per 1 {base}")
    print()
    print(f"    premium, {quote:<4}    {P * N:>16,.2f}")
    print(f"    premium, {base:<4}    {V * N:>16,.2f}")
    print(f"    premium, %{base}     {V * 100:>16.6f} %")
    print(f"    premium, {quote} pips {P / pip:>16.4f}")

    # ---- partials --------------------------------------------------------
    sub('BASE-CCY PARTIALS  (FXOption.base_ccy_partials) -- unsized, unsigned')
    p = opt.base_ccy_partials(S, sigma, r_d, r_f, today)
    for kk, vv in p.items():
        print(f"    {kk:<12} {vv:>18.10f}")

    # ---- greeks ----------------------------------------------------------
    g = opt.greeks(S, sigma, r_d, r_f, today, notional=N, direction=dirn)
    sub(f'GREEKS  (FXOption.greeks -> GreekVector) -- {base} P&L per standard move')
    print(f"    {'field':<16} {base + ' P&L':>16} {quote + ' P&L':>18}   move")
    for nm, mv in [('spot_1pct', '+1% spot, 1st order'),
                   ('gamma_1pct', '+1% spot, 2nd order'),
                   ('vega_1vp', '+1 vol point'),
                   ('volga_1vp', '+1 vol point, 2nd order'),
                   ('vanna_1pct_1vp', '+1% spot AND +1 vol pt'),
                   ('theta_1d', '1 calendar day')]:
        v = getattr(g, nm)
        print(f"    {nm:<16} {v:>16,.2f} {v*S:>18,.2f}   {mv}")
    print(f"    {'delta_hedge':<16} {g.delta_hedge:>16,.0f} {'-':>18}   "
          f"{base} of spot to sell to flatten")

    sub('TRADER VIEW  (.as_trader_units()) -- READ ONLY, never sum these')
    tu = g.as_trader_units()
    print(f"    delta        {tu['delta']:>16,.0f}   {base} notional")
    print(f"    vega_1vp     {tu['vega_1vp']:>16,.2f}   {base} per vol pt")
    print(f"    volga_1vp    {tu['volga_1vp']:>16,.2f}   change in vega per vol pt")
    print(f"    theta_1d     {tu['theta_1d']:>16,.2f}   {base} per day")
    print(f"    gamma_1pct   {tu['gamma_1pct']:>16,.2f}   <-- WRONG by ~1/S; use")
    print(f"    vanna_1vp    {tu['vanna_1vp']:>16,.2f}   <-- the corrected pair below")
    print(f"      corrected gamma (dDelta_ntl per +1% spot) "
          f"{2*g.gamma_1pct/0.01 + 0.01*g.delta_hedge:>16,.0f}  {base}")
    print(f"      corrected vanna (dDelta_ntl per +1 vol pt) "
          f"{g.vanna_1pct_1vp/0.01:>15,.0f}  {base}")

    # ---- sign sanity ------------------------------------------------------
    # volga is NOT simply long-if-long: volga = vega*d1*d2/sigma, and near the
    # ATM forward d1 and d2 straddle zero, so d1*d2 < 0 and a LONG ATM option
    # is SHORT volga. Only in the wings (d1, d2 the same sign) does the naive
    # expectation hold. So expect sign(volga) = direction * sign(d1*d2).
    from core.pricer import d1 as _d1, d2 as _d2
    D1 = _d1(S, K, T_disp, r_d, r_f, sigma)
    D2 = _d2(S, K, T_disp, r_d, r_f, sigma)
    volga_sign = dirn * (1 if D1 * D2 > 0 else -1)

    exp_sign = {'vega_1vp': dirn, 'gamma_1pct': dirn,
                'theta_1d': -dirn, 'volga_1vp': volga_sign}
    bad = [k for k, s in exp_sign.items() if getattr(g, k) * s <= 0]
    print(f"\n    d1 = {D1:+.6f}  d2 = {D2:+.6f}  d1*d2 = {D1*D2:+.6f}"
          f"  -> {side} option is {'LONG' if volga_sign > 0 else 'SHORT'} volga")
    print(f"    [{'OK' if not bad else 'CHECK'}] {side} option signs"
          + (f' -- unexpected: {bad}' if bad else
             ': vega, gamma, theta and volga all as expected'))
    return g


# ====================================================================== #
def main() -> int:
    trades = [parse_trade(s) for s in TRADES]
    pairs  = {t['pair'] for t in trades}
    mkt    = LiveMarket(pairs) if MARKET == 'live' else ManualMarket(pairs)

    vecs = [price_leg(t, mkt) for t in trades]

    if len(vecs) > 1:
        hdr(f'BOOK TOTAL -- {len(vecs)} legs  (GreekVector.__add__)')
        book = GreekVector.total(vecs)
        for nm in GreekVector.PNL_FIELDS:
            print(f"    {nm:<16} {getattr(book, nm):>18,.2f}")
        if np.isnan(book.delta_hedge):
            print(f"    {'delta_hedge':<16} {'blanked':>18}   "
                  f"(legs span more than one pair -- spot notionals do not add)")
        else:
            print(f"    {'delta_hedge':<16} {book.delta_hedge:>18,.0f}")
        print('\n    P&L fields add across pairs, tenors and strikes. That')
        print('    additivity is the whole reason GreekVector exists.')
    return 0




TRADES = [
    "USDJPY 1M 25d call 20mm",
    # "short USDJPY 1M 10d put 15mm",
    # "EURUSD 3M ATM put 50mm",
    # "USDJPY 2W k=156.50 put -10mm",
    # "USDJPY 1W 25d call 500k",
]





if __name__ == '__main__':
    _sys.exit(main())