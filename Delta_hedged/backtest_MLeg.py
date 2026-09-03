from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from dataset import FXVolDataset
from option import FXOption, find_strike_from_delta
from data import fx_calendar
from trading_calendar import preceding_business_day, add_tenor, tenor_offset
from exit_hedge_logic import (
    ExitContext, ExitRule, HoldToExpiry, ExitAfterNDays, TakeProfitStopLoss,
    HedgeContext, HedgeRule, DailyHedge,
)

DAYS_PER_YEAR = 365.0


# ═══════════════════════════════════════════════════════════════════════════════
# LegSpec
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegSpec:
    """
    option_type  : 'call' or 'put'
    target_delta : signed spot delta — positive for calls (+0.25), negative for puts (-0.25)
    direction    : +1 = long, -1 = short
    notional     : base currency notional for this leg
    """
    option_type:  str
    target_delta: float
    direction:    int
    notional:     float

    def __post_init__(self):
        assert self.option_type in ('call', 'put'),  "option_type must be 'call' or 'put'"
        assert self.direction in (1, -1),            "direction must be +1 or -1"
        if self.option_type == 'call':
            assert self.target_delta > 0, "call target_delta must be positive"
        else:
            assert self.target_delta < 0, "put target_delta must be negative"


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Constructors  — return List[LegSpec], nothing else
# ═══════════════════════════════════════════════════════════════════════════════

def build_straddle(notional: float, direction: int = 1) -> List[LegSpec]:
    """
    ATM straddle: call + put at delta ≈ 0.50.
    """
    assert direction in (1, -1)
    return [
        LegSpec('call', +0.50, direction, notional),
        LegSpec('put',  -0.50, direction, notional),
    ]


def build_strangle(call_delta: float, put_delta: float,
                   notional: float, direction: int = 1) -> List[LegSpec]:
    """
    OTM strangle: OTM call + OTM put.
    """
    assert direction in (1, -1)
    assert 0 < call_delta <= 0.50, "call_delta must be in (0, 0.50]"
    assert 0 < put_delta  <= 0.50, "put_delta must be in (0, 0.50]"
    return [
        LegSpec('call', +call_delta, direction, notional),
        LegSpec('put',  -put_delta,  direction, notional),
    ]


def build_risk_reversal(call_delta: float, put_delta: float,
                        notional: float, direction: int = 1) -> List[LegSpec]:
    """
    Risk reversal: long call + short put, or vice versa.
        - direction = +1 → long call / short put 
        - direction = -1 → short call / long put  
    """
    assert direction in (1, -1)
    assert 0 < call_delta <= 0.50
    assert 0 < put_delta  <= 0.50
    return [
        LegSpec('call', +call_delta, +direction, notional),
        LegSpec('put',  -put_delta,  -direction, notional),
    ]


def build_call_spread(long_delta: float, short_delta: float,
                      notional: float, direction: int = 1) -> List[LegSpec]:
    """
    Call spread: long lower-strike call (higher delta) + short higher-strike call.
        - direction = +1 → long the spread 
        - direction = -1 → short the spread 
    """
    assert direction in (1, -1)
    assert 0 < short_delta < long_delta <= 1.0, \
        "long_delta > short_delta > 0 required (e.g. long_delta=0.50, short_delta=0.25)"
    return [
        LegSpec('call', +long_delta,  +direction, notional),
        LegSpec('call', +short_delta, -direction, notional),
    ]


def build_put_spread(long_delta: float, short_delta: float,
                     notional: float, direction: int = 1) -> List[LegSpec]:
    """
    Put spread: long higher abs-delta put (higher strike) + short lower abs-delta put.
        - direction = +1 → long the spread 
        - direction = -1 → short the spread 

    """
    assert direction in (1, -1)
    assert 0 < short_delta < long_delta <= 1.0, \
        "long_delta > short_delta > 0 required (e.g. long_delta=0.25, short_delta=0.10)"
    return [
        LegSpec('put', -long_delta,  +direction, notional),
        LegSpec('put', -short_delta, -direction, notional),
    ]


def build_vega_neutral_butterfly(
    wing_delta:    float,
    wing_notional: float,
    direction:     int = 1,
    *,
    pair:              str,
    expiry:            date,
    entry_date:        date,
    dataset:           FXVolDataset,
    S0:                float,
    r_d:               float,
    r_f:               float,
    sigma_atm0:        float,
    F0:                float,
    option_tenor_days: int,
    verbose:           bool = False,
) -> List[LegSpec]:
    """
    Vega-neutral butterfly: a wing_delta strangle (wings) against an offsetting
    50-delta straddle (body), with the body's notional solved so the position's
    NET ENTRY VEGA is exactly zero.

        direction = +1 -> short the 50d straddle, long the wing_delta strangle
        direction = -1 -> long the 50d straddle, short the wing_delta strangle

    wing_notional sizes the two wing legs (put + call, equal size). The two body
    legs are sized automatically:

        body_notional = wing_notional * (vega_put_wing + vega_call_wing)
                                       / (vega_put_body + vega_call_body)

    This is closed-form, not a root-find: vega-per-unit-notional for a leg
    depends only on its bootstrapped (strike, smile vol) — never on notional
    itself (see find_strike_from_delta / FXOption.greeks_foreign) — so total
    portfolio vega is linear in the two notionals and solving for one given the
    other is just algebra.

    Unlike the other constructors, this one needs the entry market snapshot
    (S0/rates/vol/expiry + a pre-built dataset) to price each leg's vega. The
    caller resolves that snapshot ONCE and passes it in so the sizing pass and
    the real backtest share one Bloomberg pull and agree on entry_date/S0/etc.
    """
    assert 0 < wing_delta < 0.50, "wing_delta must be in (0, 0.50)"
    assert direction in (1, -1),  "direction must be +1 or -1"

    entry_dt = datetime.combine(entry_date, datetime.min.time())

    # ── Bootstrap each leg SHAPE at unit notional, read off vega/unit ───────
    # Identical fixed-point (K, sigma) loop to run_backtest_multi_leg's own
    # per-leg bootstrap — reusing the same public find_strike_from_delta /
    # dataset.get_smile_vol / FXOption, not a copy of any internal engine state.
    def leg_vega_per_unit(option_type: str, target_delta: float) -> float:
        sigma = sigma_atm0
        for _ in range(5):
            K_seed = find_strike_from_delta(
                pair=pair, S0=S0, expiry=expiry, r_d=r_d, r_f=r_f,
                sigma=sigma, target_delta=target_delta,
                option_type=option_type, notional=1.0, today=entry_date)
            s_new = dataset.get_smile_vol(pair, entry_dt, option_tenor_days,
                                           K_seed, F0, r_f)
            if abs(s_new - sigma) < 1e-8:
                sigma = s_new
                break
            sigma = s_new

        K = find_strike_from_delta(
            pair=pair, S0=S0, expiry=expiry, r_d=r_d, r_f=r_f,
            sigma=sigma, target_delta=target_delta,
            option_type=option_type, notional=1.0, today=entry_date)

        opt = FXOption(pair=pair, S0=S0, K=K, expiry=expiry, r_d=r_d, r_f=r_f,
                       sigma0=sigma, option_type=option_type, notional=1.0)
        return opt.greeks_foreign(S0, sigma, r_d, r_f, entry_date)['vega']

    v_put_wing  = leg_vega_per_unit('put',  -wing_delta)
    v_call_wing = leg_vega_per_unit('call', +wing_delta)
    v_put_body  = leg_vega_per_unit('put',  -0.50)
    v_call_body = leg_vega_per_unit('call', +0.50)

    body_notional = wing_notional * (v_put_wing + v_call_wing) / (v_put_body + v_call_body)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  VEGA-NEUTRAL BUTTERFLY SIZING  |  {pair}")
        print(f"{'='*65}")
        print(f"  Wing ({wing_delta*100:.0f}d) vega/unit:  put {v_put_wing:.6f}   call {v_call_wing:.6f}")
        print(f"  Body (50d) vega/unit:      put {v_put_body:.6f}   call {v_call_body:.6f}")
        print(f"  wing_notional: {wing_notional:,.0f}   ->   body_notional: {body_notional:,.0f}")

    return [
        LegSpec('put',  -wing_delta,  direction, wing_notional),
        LegSpec('put',  -0.50,       -direction, body_notional),
        LegSpec('call', +0.50,       -direction, body_notional),
        LegSpec('call', +wing_delta,  direction, wing_notional),
    ]


def vega_neutral_butterfly_factory(wing_delta: float, wing_notional: float,
                                    direction: int, tenor, pair: str,
                                    verbose: bool = False):
    """
    Returns factory(entry_date, dataset) -> List[LegSpec], suitable for
    run_signal_backtest's `legs_factory` param — resolves the entry market
    snapshot (S0/rates/ATM vol/expiry) fresh for EACH signal date from the
    run's shared dataset, then sizes the body via build_vega_neutral_butterfly.
    wing_delta/wing_notional/direction/tenor/pair are fixed at construction
    time; only the market snapshot varies trade to trade.

    tenor/pair here must match the tenor/pair passed to run_signal_backtest —
    the shared dataset it hands to the factory is pulled for that pair/tenor.
    """
    fxc = fx_calendar(pair)

    def factory(entry_date: date, dataset: FXVolDataset) -> List[LegSpec]:
        expiry            = add_tenor(entry_date, tenor, fxc)
        option_tenor_days = (expiry - entry_date).days
        entry_dt          = datetime.combine(entry_date, datetime.min.time())

        S0         = dataset.get_spot(pair, entry_dt)
        r_d, r_f   = dataset.get_rates_for_tenor(pair, entry_dt, option_tenor_days)
        sigma_atm0 = dataset.get_atm_vol(pair, entry_dt, option_tenor_days)
        T0 = option_tenor_days / DAYS_PER_YEAR
        F0 = S0 * np.exp((r_d - r_f) * T0)

        return build_vega_neutral_butterfly(
            wing_delta, wing_notional, direction,
            pair=pair, expiry=expiry, entry_date=entry_date, dataset=dataset,
            S0=S0, r_d=r_d, r_f=r_f, sigma_atm0=sigma_atm0, F0=F0,
            option_tenor_days=option_tenor_days, verbose=verbose)

    return factory




# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_multi_leg(
    legs:             List[LegSpec],
    pair:             str           = 'USDJPY',
    tenor                           = '1M',
    entry_days_back:  Optional[int] = None,
    history_days:     int           = 365,
    tc_fraction:      float         = 0.0001, # 1bps of transaction cost
    verbose:          bool          = True,
    exit_rule:        Optional[ExitRule]  = None,
    hedge_rule:       Optional[HedgeRule] = None,
    entry_date:       Optional[date]        = None,
    dataset:          Optional[FXVolDataset] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.Series, List[pd.Series]]:
    """
    Multi-leg delta-hedged backtest engine.
        - All legs share one expiry and one net spot hedge
        - exit_rule controls early close; defaults to HoldToExpiry() (baseline)
        - hedge_rule controls daily rehedge sizing; defaults to DailyHedge() (baseline)

    entry_date : explicit trade inception date, overriding the default
                 today/entry_days_back/tenor derivation. Pass at most one of
                 entry_date / entry_days_back. Used by signal_backtest.py to
                 run one trade per historical signal date.
    dataset    : pre-built FXVolDataset to reuse across multiple calls instead
                 of pulling from Bloomberg on every call (expensive). Used by
                 signal_backtest.py to share one pull across an entire run.
    """
    n = len(legs)
    assert n >= 1, "Provide at least one LegSpec"
    assert entry_date is None or entry_days_back is None, \
        "pass at most one of entry_date / entry_days_back"
    exit_rule  = exit_rule or HoldToExpiry()
    hedge_rule = hedge_rule or DailyHedge()

    total_notional = sum(abs(leg.notional) for leg in legs)

    # ── Date logic ──────────────────────────────────────────────────────────
    today     = datetime.now().date()
    fxc       = fx_calendar(pair)
    if entry_date is None:
        raw_entry  = (today - timedelta(days=entry_days_back)) if entry_days_back is not None \
                     else (today - tenor_offset(tenor))
        entry_date = preceding_business_day(raw_entry, fxc.cal_trade)
    else:
        entry_date = preceding_business_day(entry_date, fxc.cal_trade)
    expiry            = add_tenor(entry_date, tenor, fxc)
    option_tenor_days = (expiry - entry_date).days
    days_back         = (today - entry_date).days
    exit_rule.bind(entry_date, fxc, option_tenor_days)
    hedge_rule.bind(entry_date, fxc, option_tenor_days)
    entry_dt          = datetime.combine(entry_date, datetime.min.time())
    # ── Dataset ─────────────────────────────────────────────────────────────
    if dataset is None:
        dataset_days = max(history_days, days_back + option_tenor_days + 10)
        dataset      = FXVolDataset.build(pairs=[pair], days=dataset_days)

    S0         = dataset.get_spot(pair, entry_dt)
    r_d, r_f   = dataset.get_rates_for_tenor(pair, entry_dt, option_tenor_days)
    sigma_atm0 = dataset.get_atm_vol(pair, entry_dt, option_tenor_days)

    assert 0 < sigma_atm0 < 2.0,          f"sigma={sigma_atm0}: expected decimal"
    assert abs(r_d) < 1 and abs(r_f) < 1, f"rates {r_d},{r_f}: expected decimal"

    T0 = option_tenor_days / DAYS_PER_YEAR
    F0 = S0 * np.exp((r_d - r_f) * T0)

    # ── Per-leg initialisation: bootstrap (K, sigma) → FXOption → greeks ────
    options          = []
    entry_sigmas     = []
    entry_Ks         = []
    entry_greeks_arr = []
    entry_premiums   = []   # price_foreign * notional (unsigned long value)

    for leg in legs:
        sigma = sigma_atm0
        for _ in range(5):
            K_seed = find_strike_from_delta(
                pair=pair, S0=S0, expiry=expiry, r_d=r_d, r_f=r_f,
                sigma=sigma, target_delta=leg.target_delta,
                option_type=leg.option_type, notional=leg.notional,
                today=entry_date)
            s_new = dataset.get_smile_vol(pair, entry_dt, option_tenor_days,
                                           K_seed, F0, r_f)
            if abs(s_new - sigma) < 1e-8:
                sigma = s_new
                break
            sigma = s_new

        K = find_strike_from_delta(
            pair=pair, S0=S0, expiry=expiry, r_d=r_d, r_f=r_f,
            sigma=sigma, target_delta=leg.target_delta,
            option_type=leg.option_type, notional=leg.notional,
            today=entry_date)

        opt     = FXOption(pair=pair, S0=S0, K=K, expiry=expiry,
                           r_d=r_d, r_f=r_f, sigma0=sigma,
                           option_type=leg.option_type, notional=leg.notional)
        price   = opt.price_foreign(S0, sigma, r_d, r_f, entry_date)
        greeks  = opt.greeks_foreign(S0, sigma, r_d, r_f, entry_date)
        premium = price * leg.notional

        options.append(opt)
        entry_sigmas.append(sigma)
        entry_Ks.append(K)
        entry_greeks_arr.append(greeks)
        entry_premiums.append(premium)

    # Net portfolio delta → initial hedge (direction applied per-leg)
    net_delta_entry = sum(
        entry_greeks_arr[i]['delta'] * legs[i].notional * legs[i].direction
        for i in range(n))
    initial_hedge = -net_delta_entry

    # Net premium cashflow (positive = net payment out)
    net_premium = sum(entry_premiums[i] * legs[i].direction for i in range(n))

    # Vega-weighted (not notional-weighted): legs with more vol sensitivity should
    # dominate the "was this a good vol trade" reference level more than a deep
    # OTM leg of the same notional but negligible vega. entry_greeks_arr[i]['vega']
    # is per-unit-notional (base ccy, per 1vp), so * notional gives each leg's
    # actual vega exposure; abs() since vega is a magnitude weight, not a signed P&L.
    # Falls back to notional-weighting in the (rare) all-zero-vega edge case, so
    # this never raises a ZeroDivisionError.
    entry_vegas = [abs(entry_greeks_arr[i]['vega'] * legs[i].notional) for i in range(n)]
    total_vega  = sum(entry_vegas)
    avg_entry_sigma = (
        sum(entry_sigmas[i] * entry_vegas[i] for i in range(n)) / total_vega
        if total_vega > 1e-9 else
        sum(entry_sigmas[i] * abs(legs[i].notional) for i in range(n)) / total_notional
    )

    # ── Verbose: entry print ─────────────────────────────────────────────────
    if verbose:
        # Entry-time note only: whether EXPIRY is still ahead. Deliberately not
        # named live_trade — the summary's `live_trade` is a different (post-loop)
        # question, "did this trade reach a terminal event", which an exit rule can
        # answer yes to long before expiry. See the summary block below.
        expiry_ahead = expiry > today
        status     = f"  *** EXPIRY AHEAD — {(expiry-today).days}d remaining ***" if expiry_ahead else "  (expired)"
        tenor_str  = tenor if isinstance(tenor, str) else f"{tenor}D"

        print(f"\n{'='*65}")
        print(f"  MULTI-LEG TRADE  |  {pair}  |  {tenor_str}  |  {n} legs")
        print(f"{'='*65}")
        print(f"Entry date:  {entry_date}  ({days_back} calendar days ago)")
        print(f"Expiry:      {expiry}{status}")
        print(f"Tenor:       {option_tenor_days} calendar days")
        print(f"\n--- Entry Market Data ---")
        print(f"S0:        {S0:.5f}")
        print(f"r_d:       {r_d:.6f}  ({r_d*100:.4f}%)")
        print(f"r_f:       {r_f:.6f}  ({r_f*100:.4f}%)")
        print(f"ATM vol:   {sigma_atm0*100:.4f}%")

        for i, leg in enumerate(legs):
            side   = "LONG" if leg.direction == 1 else "SHORT"
            dlabel = f"{int(abs(leg.target_delta)*100)}d {'C' if leg.option_type=='call' else 'P'}"
            g      = entry_greeks_arr[i]
            d      = leg.direction
            print(f"\n{'─'*65}")
            print(f"  Leg {i}: {side} {dlabel}  |  Notional: {leg.notional:,.0f}")
            print(f"{'─'*65}")
            print(f"  Strike:          {entry_Ks[i]:.5f}  ({(entry_Ks[i]/F0-1)*100:+.4f}% from fwd)")
            print(f"  Smile vol:       {entry_sigmas[i]*100:.4f}%  "
                  f"(skew {(entry_sigmas[i]-sigma_atm0)*100:+.4f}vp vs ATM)")
            print(f"  Premium {'paid' if d==1 else 'recv'}:  {entry_premiums[i]:,.2f}  "
                  f"(cashflow {-d * entry_premiums[i]:+,.2f})")
            print(f"  Delta:           {d * g['delta'] * leg.notional:,.2f}")
            print(f"  Gamma (1%):      {d * g['gamma'] * leg.notional:,.2f}")
            print(f"  Vega  (1vp):     {d * g['vega']  * leg.notional:,.2f}")
            print(f"  Vanna (Δ/+1vp): {d * g['vanna'] * 0.01 * leg.notional:,.2f}")
            print(f"  Volga(vega/+1vp):{d * g['volga'] * 0.01 * 0.01 / S0 * leg.notional:,.2f}")
            print(f"  Theta (daily):   {d * g['theta'] * leg.notional / DAYS_PER_YEAR:,.2f}")

        print(f"\n{'─'*65}")
        print(f"  Portfolio at Entry")
        print(f"{'─'*65}")
        print(f"  Net premium cashflow: {-net_premium:+,.2f}  "
              f"({'paid' if net_premium > 0 else 'received'})")
        print(f"  Net delta:            {net_delta_entry:,.2f}")
        print(f"  Net gamma (1%):       {sum(entry_greeks_arr[i]['gamma']*legs[i].notional*legs[i].direction for i in range(n)):,.2f}")
        print(f"  Net vega  (1vp):      {sum(entry_greeks_arr[i]['vega'] *legs[i].notional*legs[i].direction for i in range(n)):,.2f}")
        print(f"  Net theta (daily):    {sum(entry_greeks_arr[i]['theta']*legs[i].notional/DAYS_PER_YEAR*legs[i].direction for i in range(n)):,.2f}")
        print(f"  Initial hedge:        {initial_hedge:,.2f}  (spot units, neg=short)")

    # ── Backtest loop ────────────────────────────────────────────────────────
    series_end  = min(expiry, today)
    spot_series = dataset.spot.loc[str(entry_date):str(series_end), pair].dropna()

    prev_date        = entry_date
    prev_spot        = S0
    prev_sigmas      = list(entry_sigmas)
    prev_r_d         = r_d
    prev_r_f         = r_f
    prev_option_vals = list(entry_premiums)   # unsigned (long) base-ccy value per leg
    current_hedge    = initial_hedge
    days_since_hedge = 0     # calendar days since hedge_position last changed
    cum_net_pnl      = 0.0
    records          = []

    for timestamp, S_new in spot_series.items():
        current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if current_date == entry_date:
            continue

        t_remaining = (expiry - current_date).days
        dS          = S_new - prev_spot
        dt_days     = (current_date - prev_date).days
        days_since_hedge += dt_days   # age the currently-held hedge_position

        # Start-of-period greeks per leg
        prev_greeks = [
            options[i].greeks_foreign(prev_spot, prev_sigmas[i],
                                       prev_r_d, prev_r_f, prev_date)
            for i in range(n)]

        # Shared hedge P&L and carry
        hedge_pnl   = current_hedge * dS / prev_spot
        hedge_carry = current_hedge * (prev_r_f - prev_r_d) * dt_days / DAYS_PER_YEAR

        # Per-leg theta P&L (direction applied)
        leg_theta_pnls = [
            prev_greeks[i]['theta'] * legs[i].notional / DAYS_PER_YEAR * dt_days * legs[i].direction
            for i in range(n)]
        
        net_theta_pnl = sum(leg_theta_pnls)

        # Per-leg gamma Taylor (direction applied)
        leg_gamma_taylors = [
            0.5 * legs[i].notional * dS**2 * (
                options[i].get_gamma_raw(prev_spot, prev_sigmas[i], prev_r_d, prev_r_f, prev_date) / prev_spot
                - 2 * prev_greeks[i]['delta'] / prev_spot**2
            ) * legs[i].direction
            for i in range(n)]
        net_gamma_taylor = sum(leg_gamma_taylors)

        # Per-leg gamma breakeven P&L: gamma_taylor + theta_pnl combines the
        # convexity income and its time-decay funding into one number —
        # algebraically 0.5*S^2*Gamma*((dS/S)^2 - sigma^2*dt), since BS theta
        # is exactly -0.5*S^2*Gamma*sigma^2 (leading order). See run_backtest_multi_leg
        # docstring / conversation notes for the derivation.
        leg_gamma_pnls_be = [leg_gamma_taylors[i] + leg_theta_pnls[i] for i in range(n)]

        # Aggregate display greeks (signed) — start of period
        net_delta_disp = sum(prev_greeks[i]['delta'] * legs[i].notional * legs[i].direction for i in range(n))
        net_gamma_disp = sum(prev_greeks[i]['gamma'] * legs[i].notional * legs[i].direction for i in range(n))
        net_vega_disp  = sum(prev_greeks[i]['vega']  * legs[i].notional * legs[i].direction for i in range(n))
        net_vanna_disp = sum(prev_greeks[i]['vanna'] * 0.01 * legs[i].notional * legs[i].direction for i in range(n))
        net_volga_disp = sum(prev_greeks[i]['volga'] * 0.01 * 0.01 / prev_spot * legs[i].notional * legs[i].direction for i in range(n))
        net_theta_disp = sum(prev_greeks[i]['theta'] * legs[i].notional / DAYS_PER_YEAR * legs[i].direction for i in range(n))

        # Vega-weighted, off the SAME start-of-day vega snapshot used for the
        # rest of today's Taylor attribution — both avg_sigma_prev and
        # avg_sigma_new (below) use these same weights so `dsigma` reflects a
        # pure vol-level change, not a shift in weighting between the two.
        # Falls back to notional-weighting in the (rare) all-zero-vega case.
        day_vegas      = [abs(prev_greeks[i]['vega'] * legs[i].notional) for i in range(n)]
        day_total_vega = sum(day_vegas)
        def _vega_weighted_avg(sigmas):
            if day_total_vega > 1e-9:
                return sum(sigmas[i] * day_vegas[i] for i in range(n)) / day_total_vega
            return sum(sigmas[i] * abs(legs[i].notional) for i in range(n)) / total_notional

        avg_sigma_prev = _vega_weighted_avg(prev_sigmas)

        # ──────────────────────────────────────────────────────────────────
        # ── EXPIRY SETTLEMENT ─────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────────
        if t_remaining <= 0:
            leg_vals_new  = []
            leg_opt_pnls  = []
            for i, leg in enumerate(legs):
                intrinsic = (max(S_new - options[i].K, 0.0) if leg.option_type == 'call'
                             else max(options[i].K - S_new, 0.0))
                val_new = (intrinsic / S_new) * leg.notional
                leg_vals_new.append(val_new)
                leg_opt_pnls.append((val_new - prev_option_vals[i]) * leg.direction)

            net_option_pnl    = sum(leg_opt_pnls)
            # Exact-reprice residual, kept only as a diagnostic vs. the Taylor
            # estimate (recon_resid below) — no longer what gamma_pnl reports.
            gamma_pnl_reprice = net_option_pnl + hedge_pnl - net_theta_pnl
            gamma_pnl         = net_gamma_taylor
            gamma_pnl_be      = gamma_pnl + net_theta_pnl
            net_pnl           = net_option_pnl + hedge_pnl + hedge_carry

            realised_var = (dS / prev_spot) ** 2
            implied_var  = avg_sigma_prev ** 2 * (dt_days / DAYS_PER_YEAR)

            row = {
                'date':          current_date,
                't_remaining':   0,
                'spot':          S_new,
                'dS':            dS,
                'sigma':         avg_sigma_prev,
                'dsigma':        0.0,
                'option_value':  sum(leg_vals_new[i] * legs[i].direction for i in range(n)),
                'delta':         net_delta_disp,
                'gamma_1pct':    net_gamma_disp,
                'vega_1vp':      net_vega_disp,
                'vanna_1vp':     net_vanna_disp,
                'volga_1vp':     net_volga_disp,
                'theta_daily':   net_theta_disp,
                'option_pnl':    net_option_pnl,
                'hedge_pnl':     hedge_pnl,
                'hedge_carry':   hedge_carry,
                'gamma_pnl':     gamma_pnl,
                'gamma_pnl_be':  gamma_pnl_be,
                'theta_pnl':     net_theta_pnl,
                'vega_pnl':      0.0,
                'vanna_pnl':     0.0,
                'volga_pnl':     0.0,
                'vanna_pnl_be':  0.0,
                'volga_pnl_be':  0.0,
                'nu_be':         0.0,
                'rho_be':        0.0,
                'recon_resid':   gamma_pnl_reprice - gamma_pnl,
                'hedge_drift_pnl': gamma_pnl_reprice - gamma_pnl,
                'hedge_gap':       0.0,
                'hedge_fraction':  0.0,
                'hedge_trade':     0.0,
                'hedge_position':  current_hedge,
                'days_since_hedge': days_since_hedge,
                'rehedged':        False,
                'tc':            0.0,
                'net_pnl':       net_pnl,
                'realised_var':  realised_var,
                'implied_var':   implied_var,
                'var_spread':    realised_var - implied_var,
                'exit_reason':   'expiry',
            }
            for i in range(n):
                d = legs[i].direction
                row[f'option_value_leg{i}'] = leg_vals_new[i] * d
                row[f'option_pnl_leg{i}']   = leg_opt_pnls[i]
                row[f'delta_leg{i}']         = prev_greeks[i]['delta'] * legs[i].notional * d
                row[f'gamma_1pct_leg{i}']    = prev_greeks[i]['gamma'] * legs[i].notional * d
                row[f'vega_1vp_leg{i}']      = prev_greeks[i]['vega']  * legs[i].notional * d
                row[f'theta_daily_leg{i}']   = prev_greeks[i]['theta'] * legs[i].notional / DAYS_PER_YEAR * d
                # Set on the normal-day row too — without them here, a trade that
                # ends at expiry leaves NaN in its LAST row for these two columns.
                row[f'vanna_1vp_leg{i}']     = prev_greeks[i]['vanna'] * 0.01 * legs[i].notional * d
                row[f'volga_1vp_leg{i}']     = prev_greeks[i]['volga'] * 0.01 * 0.01 / prev_spot * legs[i].notional * d
                row[f'sigma_leg{i}']         = prev_sigmas[i]
                row[f'dsigma_leg{i}']        = 0.0
                row[f'theta_pnl_leg{i}']     = leg_theta_pnls[i]
                row[f'vega_pnl_leg{i}']      = 0.0
                row[f'vanna_pnl_leg{i}']     = 0.0
                row[f'volga_pnl_leg{i}']     = 0.0
                row[f'vanna_pnl_be_leg{i}']  = 0.0
                row[f'volga_pnl_be_leg{i}']  = 0.0
                row[f'gamma_pnl_leg{i}']     = leg_gamma_taylors[i]
                row[f'gamma_pnl_be_leg{i}']  = leg_gamma_pnls_be[i]

            records.append(row)
            cum_net_pnl += net_pnl
            break

        # ──────────────────────────────────────────────────────────────────
        # ── NORMAL DAY ────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────────
        current_dt       = datetime.combine(current_date, datetime.min.time())
        r_d_new, r_f_new = dataset.get_rates_for_tenor(pair, current_dt, t_remaining)
        F_new            = S_new * np.exp((r_d_new - r_f_new) * t_remaining / DAYS_PER_YEAR)

        # nu_BE/rho_BE: closed-form (Ravagli 2024) vol-of-vol / spot-vol
        # correlation, shared across all legs (all legs share one expiry ->
        # one t_remaining, and nu_BE/rho_BE describe the smile's shape, not a
        # per-leg strike).
        nu_BE, rho_BE = dataset.get_smile_nu_rho(pair, current_dt, t_remaining)
        dt_years      = dt_days / DAYS_PER_YEAR

        sigma_news        = []
        leg_vals_new       = []
        leg_opt_pnls       = []
        leg_dsigmas        = []
        leg_vega_pnls      = []
        leg_vanna_pnls     = []
        leg_volga_pnls     = []
        leg_vanna_pnls_be  = []
        leg_volga_pnls_be  = []

        for i, leg in enumerate(legs):
            sigma_new = dataset.get_smile_vol(pair, current_dt, t_remaining,
                                               options[i].K, F_new, r_f_new)
            val_new   = options[i].price_foreign(S_new, sigma_new,
                                                  r_d_new, r_f_new, current_date) * leg.notional
            dsigma    = sigma_new - prev_sigmas[i]
            d         = leg.direction
            sigma0    = prev_sigmas[i]   # this leg's breakeven vol level, same convention as gamma's sigma_BE

            opt_pnl  = (val_new - prev_option_vals[i]) * d
            vega_pnl = prev_greeks[i]['vega']  * leg.notional * (dsigma / 0.01) * d
            vanna_pnl= prev_greeks[i]['vanna'] * dS * dsigma / prev_spot * leg.notional * d
            volga_pnl= 0.5 * prev_greeks[i]['volga'] * dsigma**2 / prev_spot * leg.notional * d

            # Breakeven versions: subtract the closed-form-implied "cost" cross-term
            # before the multiply — same substitution pattern as gamma_pnl_be.
            vanna_pnl_be = prev_greeks[i]['vanna'] * (
                dS * dsigma - prev_spot * sigma0**2 * rho_BE * nu_BE * dt_years
            ) / prev_spot * leg.notional * d
            volga_pnl_be = 0.5 * prev_greeks[i]['volga'] * (
                dsigma**2 - sigma0**2 * nu_BE**2 * dt_years
            ) / prev_spot * leg.notional * d

            sigma_news.append(sigma_new)
            leg_vals_new.append(val_new)
            leg_opt_pnls.append(opt_pnl)
            leg_dsigmas.append(dsigma)
            leg_vega_pnls.append(vega_pnl)
            leg_vanna_pnls.append(vanna_pnl)
            leg_volga_pnls.append(volga_pnl)
            leg_vanna_pnls_be.append(vanna_pnl_be)
            leg_volga_pnls_be.append(volga_pnl_be)

        net_option_pnl  = sum(leg_opt_pnls)
        net_vega_pnl    = sum(leg_vega_pnls)
        net_vanna_pnl   = sum(leg_vanna_pnls)
        net_volga_pnl   = sum(leg_volga_pnls)
        net_vanna_pnl_be = sum(leg_vanna_pnls_be)
        net_volga_pnl_be = sum(leg_volga_pnls_be)
        # Exact-reprice residual, kept only as a diagnostic vs. the Taylor
        # estimate (recon_resid below) — no longer what gamma_pnl reports.
        gamma_pnl_reprice = (net_option_pnl + hedge_pnl
                             - net_theta_pnl - net_vega_pnl - net_vanna_pnl - net_volga_pnl)
        gamma_pnl      = net_gamma_taylor
        gamma_pnl_be   = gamma_pnl + net_theta_pnl

        avg_sigma_new = _vega_weighted_avg(sigma_news)

        # ── Exit-rule check ─────────────────────────────────────────────
        exit_ctx = ExitContext(
            date=current_date, day_index=len(records) + 1,
            days_in_trade=(current_date - entry_date).days,
            t_remaining=t_remaining, spot=S_new, entry_spot=S0,
            sigma_avg=avg_sigma_new, entry_sigma_avg=avg_entry_sigma,
            running_net_pnl=cum_net_pnl + net_option_pnl + hedge_pnl + hedge_carry,
        )
        exit_now = exit_rule.check(exit_ctx)

        # Re-hedge to net portfolio delta (or fully unwind the hedge on exit)
        if exit_now:
            new_net_delta  = 0.0
            target_hedge   = 0.0
            hedge_fraction = 1.0
        else:
            new_net_delta = sum(
                options[i].get_delta_pa(S_new, sigma_news[i], r_d_new, r_f_new, current_date)
                * legs[i].notional * legs[i].direction
                for i in range(n))
            target_hedge = -new_net_delta
            hedge_ctx = HedgeContext(
                date=current_date, day_index=len(records) + 1,
                days_in_trade=(current_date - entry_date).days,
                t_remaining=t_remaining, spot=S_new, prev_spot=prev_spot, dS=dS,
                entry_spot=S0, net_delta_pretrade=net_delta_disp, net_gamma_pretrade=net_gamma_disp,
                total_notional=total_notional, current_hedge=current_hedge, target_hedge=target_hedge,
            )
            hedge_fraction = hedge_rule.decide(hedge_ctx)

        hedge_gap   = target_hedge - current_hedge   # unhedged delta mismatch, pre-trade
        hedge_trade = hedge_fraction * hedge_gap
        tc          = abs(hedge_trade) * tc_fraction
        net_pnl     = net_option_pnl + hedge_pnl + hedge_carry - tc

        rehedged    = abs(hedge_trade) > 1e-9
        new_hedge   = current_hedge + hedge_trade
        if rehedged:
            days_since_hedge = 0   # hedge_position was just reset — age starts over

        realised_var  = (dS / prev_spot) ** 2
        implied_var   = avg_sigma_prev ** 2 * (dt_days / DAYS_PER_YEAR)

        row = {
            'date':          current_date,
            't_remaining':   t_remaining,
            'spot':          S_new,
            'dS':            dS,
            'sigma':         avg_sigma_new,
            'dsigma':        avg_sigma_new - avg_sigma_prev,
            'option_value':  sum(leg_vals_new[i] * legs[i].direction for i in range(n)),
            'delta':         net_delta_disp,
            'gamma_1pct':    net_gamma_disp,
            'vega_1vp':      net_vega_disp,
            'vanna_1vp':     net_vanna_disp,
            'volga_1vp':     net_volga_disp,
            'theta_daily':   net_theta_disp,
            'option_pnl':    net_option_pnl,
            'hedge_pnl':     hedge_pnl,
            'hedge_carry':   hedge_carry,
            'gamma_pnl':     gamma_pnl,
            'gamma_pnl_be':  gamma_pnl_be,
            'theta_pnl':     net_theta_pnl,
            'vega_pnl':      net_vega_pnl,
            'vanna_pnl':     net_vanna_pnl,
            'volga_pnl':     net_volga_pnl,
            'vanna_pnl_be':  net_vanna_pnl_be,
            'volga_pnl_be':  net_volga_pnl_be,
            'nu_be':         nu_BE,
            'rho_be':        rho_BE,
            'recon_resid':   gamma_pnl_reprice - gamma_pnl,
            'hedge_drift_pnl': gamma_pnl_reprice - gamma_pnl,
            'hedge_gap':       hedge_gap,
            'hedge_fraction':  hedge_fraction,
            'hedge_trade':     hedge_trade,
            'hedge_position':  new_hedge,
            'days_since_hedge': days_since_hedge,
            'rehedged':        rehedged,
            'tc':            -tc,
            'net_pnl':       net_pnl,
            'realised_var':  realised_var,
            'implied_var':   implied_var,
            'var_spread':    realised_var - implied_var,
            'exit_reason':   exit_rule.name if exit_now else None,
        }
        for i in range(n):
            d = legs[i].direction
            row[f'option_value_leg{i}'] = leg_vals_new[i] * d
            row[f'option_pnl_leg{i}']   = leg_opt_pnls[i]
            row[f'delta_leg{i}']         = prev_greeks[i]['delta'] * legs[i].notional * d
            row[f'gamma_1pct_leg{i}']    = prev_greeks[i]['gamma'] * legs[i].notional * d
            row[f'vega_1vp_leg{i}']      = prev_greeks[i]['vega']  * legs[i].notional * d
            row[f'theta_daily_leg{i}']   = prev_greeks[i]['theta'] * legs[i].notional / DAYS_PER_YEAR * d

            row[f'vanna_1vp_leg{i}']     = prev_greeks[i]['vanna'] * 0.01 * legs[i].notional * d
            row[f'volga_1vp_leg{i}']     = prev_greeks[i]['volga'] * 0.01 * 0.01 / prev_spot * legs[i].notional * d  



            row[f'sigma_leg{i}']         = sigma_news[i]
            row[f'dsigma_leg{i}']        = leg_dsigmas[i]
            row[f'theta_pnl_leg{i}']     = leg_theta_pnls[i]
            row[f'vega_pnl_leg{i}']      = leg_vega_pnls[i]
            row[f'vanna_pnl_leg{i}']     = leg_vanna_pnls[i]
            row[f'volga_pnl_leg{i}']     = leg_volga_pnls[i]
            row[f'vanna_pnl_be_leg{i}']  = leg_vanna_pnls_be[i]
            row[f'volga_pnl_be_leg{i}']  = leg_volga_pnls_be[i]
            row[f'gamma_pnl_leg{i}']     = leg_gamma_taylors[i]
            row[f'gamma_pnl_be_leg{i}']  = leg_gamma_pnls_be[i]

        records.append(row)
        cum_net_pnl += net_pnl

        if exit_now:
            break

        # Roll state
        current_hedge    = new_hedge
        prev_date        = current_date
        prev_spot        = S_new
        prev_sigmas      = sigma_news
        prev_r_d         = r_d_new
        prev_r_f         = r_f_new
        prev_option_vals = leg_vals_new

    # ── Build output DataFrames ──────────────────────────────────────────────
    df_agg = pd.DataFrame(records).set_index('date')

    shared_cols = ['t_remaining', 'spot', 'dS']
    leg_dfs = []
    for i in range(n):
        leg_specific = [
            f'sigma_leg{i}', f'dsigma_leg{i}',
            f'option_value_leg{i}', f'delta_leg{i}',
            f'gamma_1pct_leg{i}', f'vega_1vp_leg{i}', f'theta_daily_leg{i}',
            f'option_pnl_leg{i}', f'theta_pnl_leg{i}',
            f'vega_pnl_leg{i}', f'vanna_pnl_leg{i}', f'volga_pnl_leg{i}',
            f'vanna_pnl_be_leg{i}', f'volga_pnl_be_leg{i}',
            f'gamma_pnl_leg{i}', f'gamma_pnl_be_leg{i}',
        ]
        ldf = df_agg[shared_cols + leg_specific].copy()
        ldf.columns = [c.replace(f'_leg{i}', '') for c in ldf.columns]
        leg_dfs.append(ldf)

    # ── Summaries ────────────────────────────────────────────────────────────
    total_years     = max((df_agg.index[-1] - entry_date).days / DAYS_PER_YEAR, 1e-9)
    realised_vol    = np.sqrt(df_agg['realised_var'].sum() / total_years) * 100
    total_net_pnl   = df_agg['net_pnl'].sum()
    exit_reason     = df_agg['exit_reason'].iloc[-1]
    # A trade is LIVE only if the loop ended without a terminal event — it ran out
    # of spot data mid-life (§5.6). Both terminal paths set exit_reason ('expiry',
    # or the exit rule's .name) and both fully settle the position, so its net_pnl
    # is final.
    #
    # This must NOT be `expiry > today`: an exit_rule close settles the trade
    # WEEKS before expiry, so with ExitAfterNDays/ExitAtDaysRemaining/TP-SL every
    # trade entered within one tenor of today would be mislabelled live and then
    # silently dropped by trade_metrics(settled_only=True) — always the most
    # recent trades, biasing every per-trade statistic against the end of the
    # sample. Only HoldToExpiry made the old test coincidentally correct.
    live_trade      = exit_reason is None
    days_held       = (df_agg.index[-1] - entry_date).days

    if verbose:
        print(f"\n{'='*65}")
        print(f"  TRADE SUMMARY")
        print(f"{'='*65}")
        if live_trade:
            print(f"  *** LIVE — {(expiry-today).days} days remaining ***")
            print(f"  Mark-to-market P&L: {total_net_pnl:,.2f}")
        print(f"\n  Net premium cashflow at entry: {-net_premium:+,.2f}")
        print(f"\n  --- Per-Leg P&L ---")
        for i, leg in enumerate(legs):
            side   = "LONG" if leg.direction == 1 else "SHORT"
            dlabel = f"{int(abs(leg.target_delta)*100)}d {'C' if leg.option_type=='call' else 'P'}"
            print(f"  Leg {i} ({side} {dlabel}):  "
                  f"option_pnl {df_agg[f'option_pnl_leg{i}'].sum():>12,.2f}  |  "
                  f"gamma {df_agg[f'gamma_pnl_leg{i}'].sum():>10,.2f}  |  "
                  f"gamma_be {df_agg[f'gamma_pnl_be_leg{i}'].sum():>10,.2f}  |  "
                  f"theta {df_agg[f'theta_pnl_leg{i}'].sum():>10,.2f}  |  "
                  f"vega {df_agg[f'vega_pnl_leg{i}'].sum():>10,.2f}  |  "
                  f"vanna {df_agg[f'vanna_pnl_leg{i}'].sum():>10,.2f}  |  "
                  f"volga {df_agg[f'volga_pnl_leg{i}'].sum():>10,.2f}  |  "
                  f"vanna_be {df_agg[f'vanna_pnl_be_leg{i}'].sum():>10,.2f}  |  "
                  f"volga_be {df_agg[f'volga_pnl_be_leg{i}'].sum():>10,.2f}")
        print(f"\n  --- Aggregate P&L ---")
        print(f"  Gamma P&L:          {df_agg['gamma_pnl'].sum():>12,.2f}  "
              f"(Taylor: 0.5*S^2*Gamma*dS^2, base-ccy adjusted)")
        print(f"  Theta P&L:          {df_agg['theta_pnl'].sum():>12,.2f}")
        print(f"  Gamma P&L (BE):     {df_agg['gamma_pnl_be'].sum():>12,.2f}  "
              f"(= gamma + theta, netted vs. breakeven vol)")
        print(f"  Vega P&L:           {df_agg['vega_pnl'].sum():>12,.2f}")
        print(f"  Vanna P&L:          {df_agg['vanna_pnl'].sum():>12,.2f}")
        print(f"  Volga P&L:          {df_agg['volga_pnl'].sum():>12,.2f}")
        print(f"  Vanna P&L (BE):     {df_agg['vanna_pnl_be'].sum():>12,.2f}  "
              f"(netted vs. closed-form-implied spot-vol correlation)")
        print(f"  Volga P&L (BE):     {df_agg['volga_pnl_be'].sum():>12,.2f}  "
              f"(netted vs. closed-form-implied vol-of-vol)")
        print(f"  Hedge carry:        {df_agg['hedge_carry'].sum():>12,.2f}")
        print(f"  Transaction costs:  {df_agg['tc'].sum():>12,.2f}")
        print(f"  Total net P&L:      {total_net_pnl:>12,.2f}"
              f"{'  <-- MTM (live)' if live_trade else ''}")
        print(f"\n  --- Vol Check ---")
        print(f"  Avg entry sigma:    {avg_entry_sigma*100:.4f}%  (vega-weighted smile)")
        print(f"  Realised vol:       {realised_vol:.4f}%")
        print(f"  Spread (real-impl): {realised_vol - avg_entry_sigma*100:+.4f} vp")

    summary = pd.Series({
        'pair':              pair,
        'n_legs':            n,
        'entry_date':        entry_date,
        'expiry':            expiry,
        'tenor_days':        option_tenor_days,
        'live_trade':        live_trade,
        'exit_date':         df_agg.index[-1],
        'exit_reason':       exit_reason,
        'days_held':         days_held,
        'net_premium':       net_premium,
        'gamma_pnl':         df_agg['gamma_pnl'].sum(),
        'gamma_pnl_be':      df_agg['gamma_pnl_be'].sum(),
        'theta_pnl':         df_agg['theta_pnl'].sum(),
        'vega_pnl':          df_agg['vega_pnl'].sum(),
        'vanna_pnl':         df_agg['vanna_pnl'].sum(),
        'volga_pnl':         df_agg['volga_pnl'].sum(),
        'vanna_pnl_be':      df_agg['vanna_pnl_be'].sum(),
        'volga_pnl_be':      df_agg['volga_pnl_be'].sum(),
        'hedge_carry_pnl':   df_agg['hedge_carry'].sum(),
        'transaction_costs': df_agg['tc'].sum(),
        'net_pnl':           total_net_pnl,
        'realised_vol':      realised_vol,
        'avg_entry_sigma':   avg_entry_sigma * 100,
        'vol_spread':        realised_vol - avg_entry_sigma * 100,
        'atm_entry_vol':     sigma_atm0 * 100,   # pure ATM vol at inception, not vega-weighted across legs
    })

    leg_summaries = []
    for i, leg in enumerate(legs):
        leg_summaries.append(pd.Series({
            'option_type':  leg.option_type,
            'target_delta': leg.target_delta,
            'direction':    leg.direction,
            'notional':     leg.notional,
            'strike':       entry_Ks[i],
            'sigma_entry':  entry_sigmas[i] * 100,
            'premium':      entry_premiums[i] * leg.direction,
            'option_pnl':   df_agg[f'option_pnl_leg{i}'].sum(),
            'gamma_pnl':    df_agg[f'gamma_pnl_leg{i}'].sum(),
            'gamma_pnl_be': df_agg[f'gamma_pnl_be_leg{i}'].sum(),
            'theta_pnl':    df_agg[f'theta_pnl_leg{i}'].sum(),
            'vega_pnl':     df_agg[f'vega_pnl_leg{i}'].sum(),
            'vanna_pnl':    df_agg[f'vanna_pnl_leg{i}'].sum(),
            'volga_pnl':    df_agg[f'volga_pnl_leg{i}'].sum(),
            'vanna_pnl_be': df_agg[f'vanna_pnl_be_leg{i}'].sum(),
            'volga_pnl_be': df_agg[f'volga_pnl_be_leg{i}'].sum(),
        }))

    return df_agg, leg_dfs, summary, leg_summaries


# ═══════════════════════════════════════════════════════════════════════════════
# Hedge Detail Columns — shared by both the Exposure and P&L views below.
# Portfolio-level only (the hedge is one net position across the whole book,
# never attributed to a single leg — see the P&L Views note further down).
#   hedge_gap        : target_hedge - hedge_position, pre-trade mismatch that day
#   hedge_trade       : signed notional actually traded that day (hedge_fraction * hedge_gap)
#   hedge_fraction   : fraction of hedge_gap the active HedgeRule chose to trade, in [0, 1]
#   hedge_position   : outstanding spot hedge notional AFTER that day's trade (neg = short)
#   days_since_hedge : calendar days since hedge_position last changed (0 = traded today)
#   rehedged         : True if hedge_trade was nonzero that day
# ═══════════════════════════════════════════════════════════════════════════════

_HEDGE_DETAIL_COLS = ['hedge_gap', 'hedge_trade', 'hedge_fraction',
                      'hedge_position', 'days_since_hedge', 'rehedged']

HEDGE_FORMATTERS = {
    'hedge_gap':        '{:,.2f}'.format,
    'hedge_trade':      '{:,.2f}'.format,
    'hedge_fraction':   '{:.2f}'.format,
    'hedge_position':   '{:,.2f}'.format,
    'days_since_hedge': '{:,.0f}'.format,
    'rehedged':         (lambda v: 'Y' if v else '.'),
}



# ═══════════════════════════════════════════════════════════════════════════════
# Exposure Views  — split aggregate df into per-leg + whole-position breakdowns
# ═══════════════════════════════════════════════════════════════════════════════

# Per-leg exposure columns (clean names; the suffix _legN is added/stripped automatically).
# 't_remaining', 'spot', 'dS' are shared across legs and carry no suffix in df.
_SHARED_EXPOSURE_COLS = ['t_remaining', 'spot', 'dS']
_LEG_EXPOSURE_COLS    = ['sigma', 'dsigma', 'option_value', 'delta',
                         'vega_1vp', 'gamma_1pct', 'vanna_1vp', 'volga_1vp', 'theta_daily']
_FULL_EXPOSURE_COLS   = _SHARED_EXPOSURE_COLS + [
    'option_value', 'delta', 'gamma_1pct', 'vega_1vp', 'vanna_1vp', 'volga_1vp', 'theta_daily'
    ] + _HEDGE_DETAIL_COLS

# One formatter keyed on the clean column names — reused for every leg and the full view.
EXPOSURE_FORMATTERS = {
    't_remaining':  '{:,.5f}'.format,
    'spot':         '{:,.5f}'.format,
    'dS':           '{:,.5f}'.format,
    'sigma':        '{:,.5f}'.format,
    'dsigma':       '{:,.5f}'.format,
    'option_value': '{:,.2f}'.format,
    'delta':        '{:,.2f}'.format,
    'gamma_1pct':   '{:,.2f}'.format,
    'vega_1vp':     '{:,.2f}'.format,
    'vanna_1vp':    '{:,.2f}'.format,
    'volga_1vp':    '{:,.2f}'.format,
    'theta_daily':  '{:,.2f}'.format,
    **HEDGE_FORMATTERS,
}


def build_exposure_views(df: pd.DataFrame, n_legs: Optional[int] = None
                         ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Split the aggregate backtest DataFrame into one exposure view per leg plus a
    single whole-position exposure view, all with clean (un-suffixed) columns.

    Parameters
    ----------
    df      : aggregate DataFrame returned by run_backtest_multi_leg (df_agg)
    n_legs  : number of legs. If None, auto-detected from the delta_leg* columns.

    Returns
    -------
    leg_views : list of DataFrames; leg_views[i] holds leg i's daily exposures
                (columns: t_remaining, spot, dS, sigma, dsigma, option_value,
                 delta, vega_1vp, gamma_1pct, vanna_1vp, volga_1vp, theta_daily)
    full_view : single DataFrame of the netted whole-position exposures

    """
    if n_legs is None:
        n_legs = sum(1 for c in df.columns if c.startswith('delta_leg'))

    leg_views = []
    for i in range(n_legs):
        cols = _SHARED_EXPOSURE_COLS + [f'{c}_leg{i}' for c in _LEG_EXPOSURE_COLS]
        view = df[cols].copy()
        view.columns = [c.replace(f'_leg{i}', '') for c in view.columns]
        leg_views.append(view)

    full_view = df[_FULL_EXPOSURE_COLS].copy()
    return leg_views, full_view


def _leg_header(i: int, kind: str,
                leg_summaries: Optional[List[pd.Series]] = None,
                summary: Optional[pd.Series] = None) -> str:
    """
    Build the boxed header for leg i, enriched with that leg's details
    (side, delta label, expiry, strike, notional) when leg_summaries/summary
    are supplied. Falls back to a plain 'Leg i — <kind>' header otherwise.
    """
    title = f"Leg {i} — {kind}"

    if leg_summaries is not None and i < len(leg_summaries):
        ls     = leg_summaries[i]
        side   = "LONG" if ls['direction'] == 1 else "SHORT"
        dlabel = f"{int(abs(ls['target_delta'])*100)}d {'C' if ls['option_type']=='call' else 'P'}"
        details = [f"{side} {dlabel}"]
        if summary is not None and 'expiry' in summary:
            details.append(f"Exp {summary['expiry']:%d%b-%y}")
        details.append(f"K {ls['strike']:.5f}")
        details.append(f"Notl {ls['notional']:,.0f}")
        title += "  |  " + "  |  ".join(details)

    return f"\n{'─'*max(65, len(title)+4)}\n  {title}\n{'─'*max(65, len(title)+4)}"


def print_exposures(df: pd.DataFrame, n_legs: Optional[int] = None,
                    show_full: bool = True,
                    leg_summaries: Optional[List[pd.Series]] = None,
                    summary: Optional[pd.Series] = None
                    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Pretty-print each leg's exposures and the whole-position exposures, and
    return the underlying (unformatted) DataFrames for further analysis.

    Pass leg_summaries (and summary) from run_backtest_multi_leg to show each
    leg's side / strike / expiry / notional in its header.
    """
    leg_views, full_view = build_exposure_views(df, n_legs)

    for i, view in enumerate(leg_views):
        print(_leg_header(i, "Daily Exposures", leg_summaries, summary))
        print(view.to_string(formatters=EXPOSURE_FORMATTERS))

    if show_full:
        print(f"\n{'─'*65}\n  Whole Position — Daily Exposures\n{'─'*65}")
        print(full_view.to_string(formatters=EXPOSURE_FORMATTERS))

    return leg_views, full_view


# ═══════════════════════════════════════════════════════════════════════════════
# P&L Views  — split aggregate df into per-leg + whole-position P&L attribution
# ═══════════════════════════════════════════════════════════════════════════════

# Per-leg P&L columns that the engine actually attributes to a single leg.
# NOTE: hedge_pnl / hedge_carry / net_pnl / hedge details are PORTFOLIO-level
# only (they depend on the single shared hedge) —
_LEG_PNL_COLS  = ['option_pnl', 'gamma_pnl', 'gamma_pnl_be', 'theta_pnl', 'vega_pnl',
                  'vanna_pnl', 'volga_pnl', 'vanna_pnl_be', 'volga_pnl_be']
_FULL_PNL_COLS = ['option_pnl', 'hedge_pnl', 'net_pnl', 'hedge_carry'
                  ] + _HEDGE_DETAIL_COLS + [
                  'gamma_pnl', 'gamma_pnl_be', 'theta_pnl', 'vega_pnl',
                  'vanna_pnl', 'volga_pnl', 'vanna_pnl_be', 'volga_pnl_be']

# Optional orientation columns prepended to each view (set show_context=False to drop).
_PNL_CONTEXT_COLS = ['t_remaining', 'spot', 'dS']
# Extra per-leg context: this leg's own vol level and daily vol change (legs only,
# NOT the whole-position view). Inserted after the shared spot/dS columns.
_LEG_PNL_CONTEXT_COLS = ['sigma', 'dsigma']

PNL_FORMATTERS = {
    't_remaining': '{:,.5f}'.format,
    'spot':        '{:,.5f}'.format,
    'dS':          '{:,.5f}'.format,
    'sigma':       '{:,.5f}'.format,
    'dsigma':      '{:,.5f}'.format,
    'option_pnl':  '{:,.2f}'.format,
    'hedge_pnl':   '{:,.2f}'.format,
    'net_pnl':     '{:,.2f}'.format,
    'hedge_carry': '{:,.2f}'.format,
    'gamma_pnl':   '{:,.2f}'.format,
    'gamma_pnl_be':'{:,.2f}'.format,
    'theta_pnl':   '{:,.2f}'.format,
    'vega_pnl':    '{:,.2f}'.format,
    'vanna_pnl':   '{:,.2f}'.format,
    'volga_pnl':   '{:,.2f}'.format,
    'vanna_pnl_be':'{:,.2f}'.format,
    'volga_pnl_be':'{:,.2f}'.format,
    **HEDGE_FORMATTERS,
}


def build_pnl_views(df: pd.DataFrame, n_legs: Optional[int] = None,
                    show_context: bool = True
                    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Split the aggregate backtest DataFrame into one P&L view per leg plus a
    single whole-position P&L view, all with clean (un-suffixed) columns.

    Parameters
    ----------
    df           : aggregate DataFrame returned by run_backtest_multi_leg (df_agg)
    n_legs       : number of legs. If None, auto-detected from the delta_leg* columns.
    show_context : prepend t_remaining / spot / dS for orientation.

    Returns
    -------
    leg_pnl_views : list of DataFrames; leg_pnl_views[i] holds leg i's own vol
                    (sigma, dsigma) plus its daily P&L (option_pnl, gamma_pnl,
                    theta_pnl, vega_pnl, vanna_pnl, volga_pnl).
                    hedge/carry/net are portfolio-level and NOT included here.
    full_pnl_view : whole-position daily P&L incl. the shared hedge legs
                    (option_pnl, hedge_pnl, net_pnl, hedge_carry, gamma_pnl,
                     theta_pnl, vega_pnl, vanna_pnl, volga_pnl).
    """
    if n_legs is None:
        n_legs = sum(1 for c in df.columns if c.startswith('delta_leg'))

    context = _PNL_CONTEXT_COLS if show_context else []

    leg_pnl_views = []
    for i in range(n_legs):
        cols = context + [f'{c}_leg{i}' for c in (_LEG_PNL_CONTEXT_COLS + _LEG_PNL_COLS)]
        view = df[cols].copy()
        view.columns = [c.replace(f'_leg{i}', '') for c in view.columns]
        leg_pnl_views.append(view)

    full_pnl_view = df[context + _FULL_PNL_COLS].copy()
    return leg_pnl_views, full_pnl_view


def print_pnl(df: pd.DataFrame, n_legs: Optional[int] = None,
              show_full: bool = True, show_context: bool = True,
              leg_summaries: Optional[List[pd.Series]] = None,
              summary: Optional[pd.Series] = None
              ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Pretty-print each leg's P&L and the whole-position P&L, and return the
    underlying (unformatted) DataFrames for further analysis.

    Pass leg_summaries (and summary) from run_backtest_multi_leg to show each
    leg's side / strike / expiry / notional in its header.
    """
    leg_pnl_views, full_pnl_view = build_pnl_views(df, n_legs, show_context)

    for i, view in enumerate(leg_pnl_views):
        print(_leg_header(i, "Daily P&L", leg_summaries, summary))
        print(view.to_string(formatters=PNL_FORMATTERS))

    if show_full:
        print(f"\n{'─'*65}\n  Whole Position — Daily P&L\n{'─'*65}")
        print(full_pnl_view.to_string(formatters=PNL_FORMATTERS))

    return leg_pnl_views, full_pnl_view


# ═══════════════════════════════════════════════════════════════════════════════
# Top-Level Strategy Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_straddle(
    pair:            str   = 'USDJPY',
    tenor                  = '1M',
    notional:        float = 1_000_000,
    direction:       int   = 1,
    entry_days_back: int   = None,
    history_days:    int   = 365,
    tc_fraction:     float = 0.0001,
    verbose:         bool  = True,
    exit_rule:       Optional[ExitRule]  = None,
    hedge_rule:      Optional[HedgeRule] = None,
):
    """
    Delta-hedged straddle. ATM call + ATM put, same notional per leg.
    direction = +1 → long vol (pay theta / collect gamma)
    direction = -1 → short vol (collect theta / pay gamma)
    exit_rule / hedge_rule : passed straight through to run_backtest_multi_leg;
    default to HoldToExpiry() / DailyHedge() there if not given.
    """
    legs = build_straddle(notional, direction)
    return run_backtest_multi_leg(
        legs, pair=pair, tenor=tenor, entry_days_back=entry_days_back,
        history_days=history_days, tc_fraction=tc_fraction, verbose=verbose,
        exit_rule=exit_rule, hedge_rule=hedge_rule)


def run_backtest_strangle(
    pair:            str   = 'USDJPY',
    tenor                  = '1M',
    call_delta:      float = 0.25,
    put_delta:       float = 0.25,
    notional:        float = 1_000_000,
    direction:       int   = 1,
    entry_days_back: int   = None,
    history_days:    int   = 365,
    tc_fraction:     float = 0.0001,
    verbose:         bool  = True,
    exit_rule:       Optional[ExitRule]  = None,
    hedge_rule:      Optional[HedgeRule] = None,
):
    """
    Delta-hedged strangle. OTM call + OTM put at specified deltas.
    direction = +1 → long vol   direction = -1 → short vol
    exit_rule / hedge_rule : passed straight through to run_backtest_multi_leg;
    default to HoldToExpiry() / DailyHedge() there if not given.
    """
    legs = build_strangle(call_delta, put_delta, notional, direction)
    return run_backtest_multi_leg(
        legs, pair=pair, tenor=tenor, entry_days_back=entry_days_back,
        history_days=history_days, tc_fraction=tc_fraction, verbose=verbose,
        exit_rule=exit_rule, hedge_rule=hedge_rule)


def run_backtest_risk_reversal(
    pair:            str   = 'USDJPY',
    tenor                  = '1M',
    call_delta:      float = 0.25,
    put_delta:       float = 0.25,
    notional:        float = 1_000_000,
    direction:       int   = 1,
    entry_days_back: int   = None,
    history_days:    int   = 365,
    tc_fraction:     float = 0.0001,
    verbose:         bool  = True,
    exit_rule:       Optional[ExitRule]  = None,
    hedge_rule:      Optional[HedgeRule] = None,
):
    """
    Delta-hedged risk reversal.
    direction = +1 → long call / short put  (buy topside skew)
    direction = -1 → short call / long put  (sell topside skew)
    exit_rule / hedge_rule : passed straight through to run_backtest_multi_leg;
    default to HoldToExpiry() / DailyHedge() there if not given.
    """
    legs = build_risk_reversal(call_delta, put_delta, notional, direction)
    return run_backtest_multi_leg(
        legs, pair=pair, tenor=tenor, entry_days_back=entry_days_back,
        history_days=history_days, tc_fraction=tc_fraction, verbose=verbose,
        exit_rule=exit_rule, hedge_rule=hedge_rule)


def run_backtest_call_spread(
    pair:            str   = 'USDJPY',
    tenor                  = '1M',
    long_delta:      float = 0.50,
    short_delta:     float = 0.25,
    notional:        float = 1_000_000,
    direction:       int   = 1,
    entry_days_back: int   = None,
    history_days:    int   = 365,
    tc_fraction:     float = 0.0001,
    verbose:         bool  = True,
    exit_rule:       Optional[ExitRule]  = None,
    hedge_rule:      Optional[HedgeRule] = None,
):
    """
    Delta-hedged bull call spread.
    direction = +1 → long the spread (debit: long lower-strike / short higher-strike call)
    direction = -1 → short the spread (credit)
    long_delta > short_delta > 0  (e.g. 0.50 vs 0.25)
    exit_rule / hedge_rule : passed straight through to run_backtest_multi_leg;
    default to HoldToExpiry() / DailyHedge() there if not given.
    """
    legs = build_call_spread(long_delta, short_delta, notional, direction)
    return run_backtest_multi_leg(
        legs, pair=pair, tenor=tenor, entry_days_back=entry_days_back,
        history_days=history_days, tc_fraction=tc_fraction, verbose=verbose,
        exit_rule=exit_rule, hedge_rule=hedge_rule)


def run_backtest_put_spread(
    pair:            str   = 'USDJPY',
    tenor                  = '1M',
    long_delta:      float = 0.25,
    short_delta:     float = 0.10,
    notional:        float = 1_000_000,
    direction:       int   = 1,
    entry_days_back: int   = None,
    history_days:    int   = 365,
    tc_fraction:     float = 0.0001,
    verbose:         bool  = True,
    exit_rule:       Optional[ExitRule]  = None,
    hedge_rule:      Optional[HedgeRule] = None,
):
    """
    Delta-hedged bear put spread.
    direction = +1 → long the spread (debit: long higher abs-delta / short lower abs-delta put)
    direction = -1 → short the spread (credit)
    long_delta > short_delta > 0  (e.g. 0.25 vs 0.10, positive magnitudes)
    exit_rule / hedge_rule : passed straight through to run_backtest_multi_leg;
    default to HoldToExpiry() / DailyHedge() there if not given.
    """
    legs = build_put_spread(long_delta, short_delta, notional, direction)
    return run_backtest_multi_leg(
        legs, pair=pair, tenor=tenor, entry_days_back=entry_days_back,
        history_days=history_days, tc_fraction=tc_fraction, verbose=verbose,
        exit_rule=exit_rule, hedge_rule=hedge_rule)


def run_backtest_vega_neutral_butterfly(
    pair:            str   = 'USDJPY',
    tenor                  = '1M',
    wing_delta:      float = 0.10,
    direction:       int   = 1,
    wing_notional:   float = 1_000_000,
    entry_days_back: int   = None,
    history_days:    int   = 365,
    tc_fraction:     float = 0.0001,
    verbose:         bool  = True,
    exit_rule:       Optional[ExitRule]  = None,
    hedge_rule:      Optional[HedgeRule] = None,
):
    """
    Delta-hedged vega-neutral butterfly. Solves the body (50d straddle) notional
    so the position's NET ENTRY VEGA is zero, then runs one backtest.

    direction = +1 -> short the 50d straddle, long the wing_delta strangle
    direction = -1 -> long the 50d straddle, short the wing_delta strangle

    wing_notional sizes the two wing legs; the two body legs are sized
    automatically by build_vega_neutral_butterfly (see its docstring).
    exit_rule / hedge_rule : passed straight through to run_backtest_multi_leg;
    default to HoldToExpiry() / DailyHedge() there if not given.
    """
    assert 0 < wing_delta < 0.50, "wing_delta must be in (0, 0.50)"
    assert direction in (1, -1), "direction must be +1 or -1"

    # ── Resolve the entry market snapshot ONCE (mirrors run_backtest_multi_leg
    #    §5.1-5.2) so the sizing pass and the real run share one Bloomberg pull
    #    and are guaranteed to agree on entry_date/S0/rates/vol. ─────────────
    today      = datetime.now().date()
    fxc        = fx_calendar(pair)
    raw_entry  = (today - timedelta(days=entry_days_back)) if entry_days_back is not None \
                 else (today - tenor_offset(tenor))
    entry_date = preceding_business_day(raw_entry, fxc.cal_trade)
    expiry     = add_tenor(entry_date, tenor, fxc)
    option_tenor_days = (expiry - entry_date).days
    days_back  = (today - entry_date).days
    entry_dt   = datetime.combine(entry_date, datetime.min.time())

    dataset_days = max(history_days, days_back + option_tenor_days + 10)
    dataset      = FXVolDataset.build(pairs=[pair], days=dataset_days)

    S0         = dataset.get_spot(pair, entry_dt)
    r_d, r_f   = dataset.get_rates_for_tenor(pair, entry_dt, option_tenor_days)
    sigma_atm0 = dataset.get_atm_vol(pair, entry_dt, option_tenor_days)
    T0 = option_tenor_days / DAYS_PER_YEAR
    F0 = S0 * np.exp((r_d - r_f) * T0)

    # ── Solve for vega-neutral sizing, then run the real trade reusing the
    #    same entry_date/dataset ──────────────────────────────────────────────
    legs = build_vega_neutral_butterfly(
        wing_delta, wing_notional, direction,
        pair=pair, expiry=expiry, entry_date=entry_date, dataset=dataset,
        S0=S0, r_d=r_d, r_f=r_f, sigma_atm0=sigma_atm0, F0=F0,
        option_tenor_days=option_tenor_days, verbose=verbose)

    return run_backtest_multi_leg(
        legs, pair=pair, tenor=tenor,
        entry_date=entry_date, dataset=dataset,
        tc_fraction=tc_fraction, verbose=verbose,
        exit_rule=exit_rule, hedge_rule=hedge_rule,
    )




#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------






# ── Straddle ─────────────────────────────────────────────────────────
#
# df, leg_dfs, summary, leg_summaries = run_backtest_straddle(
#     pair='USDJPY', 
#     tenor='1M',
#     notional=10_000_000, 
#     direction=-1,
#     entry_days_back=10)







# ── Strangle ─────────────────────────────────────────────────────────

# df, leg_dfs, summary, leg_summaries = run_backtest_strangle(
#     pair='USDJPY', 
#     tenor='1M',
#     call_delta=0.30, 
#     put_delta=0.30,
#     notional=75_000_000,
#     direction=-1,
#     entry_days_back=6
#     )





# ── Risk Reversal ─────────────────────────────────────────────────────
# direction=-1 reverses: short call / long put (bearish USD, sell topside skew).
#
# df, leg_dfs, summary, leg_summaries = run_backtest_risk_reversal(
#     pair='USDJPY', 
#     tenor='1M',
#     call_delta=0.25, 
#     put_delta=0.25,
#     notional=10_000_000, 
#     direction=1,
#     entry_days_back=30)


# ── Call Spread ───────────────────────────────────────────────────────

#
# df, leg_dfs, summary, leg_summaries = run_backtest_call_spread(
#     pair='USDJPY', 
#     tenor='1M',
#     long_delta=0.50, 
#     short_delta=0.25,
#     notional=10_000_000, 
#     direction=1,
#     entry_days_back=30)



# ── Put Spread ────────────────────────────────────────────────────────

# df, leg_dfs, summary, leg_summaries = run_backtest_put_spread(
#     pair='USDJPY', 
#     tenor='1M',
#     long_delta=0.25, short_delta=0.10,
#     notional=10_000_000, 
#     direction=1,
#     entry_days_back=30)



# ── Vega Weighted Fly ──────────────────────────────────────────────────

# df_agg, leg_dfs, summary, leg_summaries = run_backtest_vega_neutral_butterfly(
#     pair= 'USDJPY',
#     tenor= '1M',
#     wing_delta= 0.10,
#     direction= -1,
#     wing_notional= 10_000_000,
#     entry_days_back= 40,
#     verbose= True,
# )









# ── multi-leg ─────────────────────────────────────────────────────────

# legs = [
#     LegSpec('put', -0.10, +1, 75_000_000),
#     LegSpec('put', -0.30, -1, 75_000_000),
#     LegSpec('call', +0.30, -1, 50_000_000),   
#     LegSpec('call', +0.10, +1, 50_000_000),   
# ]

# df, leg_dfs, summary, leg_summaries = run_backtest_multi_leg(
#             legs, 
#             pair='USDJPY', 
#             tenor='1M', 
#             entry_days_back=12,
#             verbose=True)



