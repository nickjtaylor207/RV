from datetime import datetime, timedelta, date
from dataclasses import dataclass
import pandas as pd
import numpy as np

from dataset import FXVolDataset
from option import FXOption, find_atm_forward_strike, find_strike_from_delta
from data import fx_calendar
from trading_calendar import preceding_business_day, add_tenor, tenor_offset

DAYS_PER_YEAR = 365.0   # ACT/365 Fixed 






# ------------ Price Option Now ------------



def get_smile_option_greeks_now(
    pair: str = 'USDJPY',
    option_type: str = 'call',
    notional: float = 1_000_000,
    tenor = '1M',
    deltas: list = None):
    """
    Print greeks and exposures for a range of delta strikes across the smile.

    Parameters
    ----------
    pair        : FX pair, e.g. 'USDJPY'
    option_type : 'call' or 'put'
    notional    : notional in base currency units
    tenor       : tenor string ('1D','1W','2W','1M','3M',...) or integer calendar days
    deltas      : list of signed spot deltas (e.g. [0.25, 0.10] for calls,
                  [-0.25, -0.10] for puts). Defaults to standard 10d/25d/ATM.
    """

    if deltas is None:
        if option_type == 'call':
            deltas = [0.10, 0.25, 0.50]
        elif option_type == 'put':
            deltas = [-0.10, -0.25, -0.50]
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    dataset   = FXVolDataset.build(pairs=[pair], days=5)
    today     = datetime.now().date()
    today_dt  = datetime.combine(today, datetime.min.time())
    fxc       = fx_calendar(pair)

    # Resolve tenor to expiry and calendar day count (mirrors backtest logic)
    entry_date = preceding_business_day(today, fxc.cal_trade)
    expiry     = add_tenor(entry_date, tenor, fxc)
    tenor_days = (expiry - today).days

    S         = dataset.get_spot(pair, today_dt)
    r_d, r_f  = dataset.get_rates_for_tenor(pair, today_dt, tenor_days)
    sigma_atm = dataset.get_atm_vol(pair, today_dt, tenor_days)

    assert 0 < sigma_atm < 2.0,           f"sigma={sigma_atm}: expected decimal"
    assert abs(r_d) < 1 and abs(r_f) < 1, f"rates {r_d},{r_f}: expected decimal"

    tenor_label = tenor if isinstance(tenor, str) else f"{tenor}d"
    print(f"\n{'='*65}")
    print(f"  {pair}  |  {tenor_label} {option_type.upper()} SMILE  |  Notional: {notional:,.0f}")
    print(f"{'='*65}")
    print(f"\n--- Market Data ---")
    print(f"Spot:      {S:.5f}")
    print(f"r_d:       {r_d:.6f}  ({r_d*100:.4f}%)")
    print(f"r_f:       {r_f:.6f}  ({r_f*100:.4f}%)")
    print(f"ATM Vol:   {sigma_atm*100:.4f}%")
    print(f"Expiry:    {expiry}")

    results = []

    # ATM forward — used for % distance display
    F = find_atm_forward_strike(S, r_d, r_f, expiry, today)

    for target_delta in deltas:

        # Entry bootstrap: K and sigma are mutually dependent.
        # Seed with ATM vol, converges in 2-3 iterations.
        sigma = sigma_atm
        for _ in range(5):
            K_seed = find_strike_from_delta(
                pair         = pair,
                S0           = S,
                expiry       = expiry,
                r_d          = r_d,
                r_f          = r_f,
                sigma        = sigma,
                target_delta = target_delta,
                option_type  = option_type,
                notional     = notional,
                today        = today)
            s_new = dataset.get_smile_vol(pair, today_dt, tenor_days, K_seed, F, r_f)
            if abs(s_new - sigma) < 1e-8:
                sigma = s_new
                break
            sigma = s_new

        K = find_strike_from_delta(
            pair         = pair,
            S0           = S,
            expiry       = expiry,
            r_d          = r_d,
            r_f          = r_f,
            sigma        = sigma,
            target_delta = target_delta,
            option_type  = option_type,
            notional     = notional,
            today        = today)

        option = FXOption(
            pair        = pair,
            S0          = S,
            K           = K,
            expiry      = expiry,
            r_d         = r_d,
            r_f         = r_f,
            sigma0      = sigma,
            option_type = option_type,
            notional    = notional)

        price    = option.price_foreign(S, sigma, r_d, r_f, today)
        greeks   = option.greeks_foreign(S, sigma, r_d, r_f, today)
        gamma_pa = option.get_gamma_pa(S, sigma, r_d, r_f, today)

        premium      = price           * notional
        delta_exp    = greeks['delta'] * notional
        gamma_exp    = greeks['gamma'] * notional
        gamma_pa_exp = gamma_pa        * notional
        vega_exp     = greeks['vega']  * notional
        theta_exp    = greeks['theta'] * notional / DAYS_PER_YEAR

        spot_move_1pct = 0.01 * S
        spot_move_2pct = 0.02 * S
        gamma_pnl_1pct = 0.5 * gamma_pa * notional * spot_move_1pct**2 / S   # base ccy (/S), matches vega P&L
        gamma_pnl_2pct = 0.5 * gamma_pa * notional * spot_move_2pct**2 / S
        vega_pnl_1vp   = greeks['vega'] * notional * (0.01 / 0.01)

        label = f"{int(abs(target_delta)*100)}d {'C' if option_type == 'call' else 'P'}"

        print(f"\n{'─'*65}")
        print(f"  {label}  |  Delta target: {target_delta:+.2f}  |  Strike: {K:.5f}"
              f"  ({(K/F - 1)*100:+.4f}% from fwd)")
        print(f"{'─'*65}")
        print(f"  Smile Vol:             {sigma*100:.4f}%"
              f"  (vs ATM {sigma_atm*100:.4f}%,"
              f"  skew {(sigma - sigma_atm)*100:+.4f} vp)")
        print(f"\n  --- Price ---")
        print(f"  Premium (base ccy):    {premium:,.2f}  ({price*100:.4f}% of notional)")
        print(f"\n  --- Greeks (base ccy, scaled to notional) ---")
        print(f"  Delta:                 {delta_exp:,.2f}")
        print(f"  Gamma (1% move):       {gamma_exp:,.2f}")
        print(f"  Vega (1 vol pt):       {vega_exp:,.2f}")
        print(f"  Theta (daily):         {theta_exp:,.2f}")
        print(f"\n  --- Scenario P&L ---")
        print(f"  Gamma P&L spot +1%:    {gamma_pnl_1pct:,.2f}")
        print(f"  Gamma P&L spot +2%:    {gamma_pnl_2pct:,.2f}")
        print(f"  Vega P&L vol +1vp:     {vega_pnl_1vp:,.2f}")
        print(f"  Theta decay (1 day):   {theta_exp:,.2f}")

        results.append({
            'label':          label,
            'target_delta':   target_delta,
            'strike':         K,
            'sigma':          sigma,
            'skew_vs_atm':    (sigma - sigma_atm) * 100,
            'premium':        premium,
            'delta':          delta_exp,
            'gamma_1pct':     gamma_exp,
            'gamma_pa':       gamma_pa_exp,
            'vega':           vega_exp,
            'theta_daily':    theta_exp,
            'gamma_pnl_1pct': gamma_pnl_1pct,
            'gamma_pnl_2pct': gamma_pnl_2pct,
            'vega_pnl_1vp':   vega_pnl_1vp})
    print(f"\n{'='*65}")
    return pd.DataFrame(results).set_index('label')





get_smile_option_greeks_now(
    pair = 'EURUSD',
    option_type = 'put',
    notional = 100_000_000,
    tenor = '1M',
    # deltas = [25, 10]
    )



def get_smile_option_greeks_ALL(
    pair: str = 'USDJPY',
    option_type: str = 'call',
    notional: float = 1_000_000,
    tenor = '1M',
    deltas: list = None):
    """
    Print greeks and exposures for a range of delta strikes across the smile,
    including Vanna and Volga (second-order vol exposures).

    Parameters
    ----------
    pair        : FX pair, e.g. 'USDJPY'
    option_type : 'call' or 'put'
    notional    : notional in base currency units
    tenor       : tenor string ('1D','1W','2W','1M','3M',...) or integer calendar days
    deltas      : list of signed spot deltas (e.g. [0.25, 0.10] for calls,
                  [-0.25, -0.10] for puts). Defaults to standard 10d/25d/ATM.

    Vanna / Volga scaling (consistent with the base-ccy conventions used elsewhere)
    -------------------------------------------------------------------------------
    Vanna  = d(delta)/d(sigma) = d(vega)/d(spot).
             Reported as the change in DELTA (base-ccy spot-equivalent) for a
             +1 vol point move:   vanna_raw * 0.01 * notional
             (delta family — no /S, mirrors how delta is scaled.)
    Volga  = d(vega)/d(sigma).
             Reported as the change in VEGA (base ccy, per vol pt) for a
             +1 vol point move:   volga_raw * 0.01 * 0.01 / S * notional
             (vega family — /S to convert to base ccy, mirrors how vega is scaled.)
    """

    if deltas is None:
        if option_type == 'call':
            deltas = [0.10, 0.25, 0.50]
        elif option_type == 'put':
            deltas = [-0.10, -0.25, -0.50]
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    dataset   = FXVolDataset.build(pairs=[pair], days=5)
    today     = datetime.now().date()
    today_dt  = datetime.combine(today, datetime.min.time())
    fxc       = fx_calendar(pair)

    # Resolve tenor to expiry and calendar day count (mirrors backtest logic)
    entry_date = preceding_business_day(today, fxc.cal_trade)
    expiry     = add_tenor(entry_date, tenor, fxc)
    tenor_days = (expiry - today).days

    S         = dataset.get_spot(pair, today_dt)
    r_d, r_f  = dataset.get_rates_for_tenor(pair, today_dt, tenor_days)
    sigma_atm = dataset.get_atm_vol(pair, today_dt, tenor_days)

    assert 0 < sigma_atm < 2.0,           f"sigma={sigma_atm}: expected decimal"
    assert abs(r_d) < 1 and abs(r_f) < 1, f"rates {r_d},{r_f}: expected decimal"

    tenor_label = tenor if isinstance(tenor, str) else f"{tenor}d"
    print(f"\n{'='*65}")
    print(f"  {pair}  |  {tenor_label} {option_type.upper()} SMILE (ALL GREEKS)  |  Notional: {notional:,.0f}")
    print(f"{'='*65}")
    print(f"\n--- Market Data ---")
    print(f"Spot:      {S:.5f}")
    print(f"r_d:       {r_d:.6f}  ({r_d*100:.4f}%)")
    print(f"r_f:       {r_f:.6f}  ({r_f*100:.4f}%)")
    print(f"ATM Vol:   {sigma_atm*100:.4f}%")
    print(f"Expiry:    {expiry}")

    results = []
    F        = find_atm_forward_strike(S, r_d, r_f, expiry, today)

    for target_delta in deltas:

        # Entry bootstrap: K and sigma are mutually dependent.
        sigma = sigma_atm
        for _ in range(5):
            K_seed = find_strike_from_delta(
                pair         = pair,
                S0           = S,
                expiry       = expiry,
                r_d          = r_d,
                r_f          = r_f,
                sigma        = sigma,
                target_delta = target_delta,
                option_type  = option_type,
                notional     = notional,
                today        = today)
            s_new = dataset.get_smile_vol(pair, today_dt, tenor_days, K_seed, F, r_f)
            if abs(s_new - sigma) < 1e-8:
                sigma = s_new
                break
            sigma = s_new

        K = find_strike_from_delta(
            pair         = pair,
            S0           = S,
            expiry       = expiry,
            r_d          = r_d,
            r_f          = r_f,
            sigma        = sigma,
            target_delta = target_delta,
            option_type  = option_type,
            notional     = notional,
            today        = today)
        option = FXOption(
            pair        = pair,
            S0          = S,
            K           = K,
            expiry      = expiry,
            r_d         = r_d,
            r_f         = r_f,
            sigma0      = sigma,
            option_type = option_type,
            notional    = notional)
        price    = option.price_foreign(S, sigma, r_d, r_f, today)
        greeks   = option.greeks_foreign(S, sigma, r_d, r_f, today)
        gamma_pa = option.get_gamma_pa(S, sigma, r_d, r_f, today)
        vanna_pa  = greeks['vanna']   # PA: d(delta_pa)/d(sigma); consistent with PA hedge
        volga_raw = greeks['volga']   # d(vega)/d(sigma)
        premium      = price           * notional
        delta_exp    = greeks['delta'] * notional
        gamma_exp    = greeks['gamma'] * notional
        gamma_pa_exp = gamma_pa        * notional
        vega_exp     = greeks['vega']  * notional
        theta_exp    = greeks['theta'] * notional / DAYS_PER_YEAR
        # Second-order vol exposures, per +1 vol point move
        vanna_exp = vanna_pa  * 0.01 * notional              # delta change (base ccy) / +1vp
        volga_exp = volga_raw * 0.01 * 0.01 / S * notional   # vega change  (base ccy) / +1vp

        spot_move_1pct = 0.01 * S
        spot_move_2pct = 0.02 * S
        gamma_pnl_1pct = 0.5 * gamma_pa * notional * spot_move_1pct**2 / S   # base ccy (/S), matches vega P&L
        gamma_pnl_2pct = 0.5 * gamma_pa * notional * spot_move_2pct**2 / S
        vega_pnl_1vp   = greeks['vega'] * notional * (0.01 / 0.01)

        # Vanna P&L: joint +1% spot & +1 vol pt move (cross term, base ccy)
        vanna_pnl = vanna_pa  * spot_move_1pct * 0.01 / S * notional
        # Volga P&L: vol convexity on a +1 vol pt move (base ccy)
        volga_pnl = 0.5 * volga_raw * 0.01**2 / S * notional

        label = f"{int(abs(target_delta)*100)}d {'C' if option_type == 'call' else 'P'}"

        print(f"\n{'─'*65}")
        print(f"  {label}  |  Delta target: {target_delta:+.2f}  |  Strike: {K:.5f}"
              f"  ({(K/F - 1)*100:+.4f}% from fwd)")
        print(f"{'─'*65}")
        print(f"  Smile Vol:             {sigma*100:.4f}%"
              f"  (vs ATM {sigma_atm*100:.4f}%,"
              f"  skew {(sigma - sigma_atm)*100:+.4f} vp)")
        print(f"\n  --- Price ---")
        print(f"  Premium (base ccy):    {premium:,.2f}  ({price*100:.4f}% of notional)")
        print(f"\n  --- Greeks (base ccy, scaled to notional) ---")
        print(f"  Delta:                 {delta_exp:,.2f}")
        print(f"  Gamma (1% move):       {gamma_exp:,.2f}")
        print(f"  Vega (1 vol pt):       {vega_exp:,.2f}")
        print(f"  Vanna (delta/+1vp):    {vanna_exp:,.2f}   (delta change per +1 vol pt)")
        print(f"  Volga (vega/+1vp):     {volga_exp:,.2f}   (vega change per +1 vol pt)")
        print(f"  Theta (daily):         {theta_exp:,.2f}")
        print(f"\n  --- Scenario P&L ---")
        print(f"  Gamma P&L spot +1%:    {gamma_pnl_1pct:,.2f}")
        print(f"  Gamma P&L spot +2%:    {gamma_pnl_2pct:,.2f}")
        print(f"  Vega P&L vol +1vp:     {vega_pnl_1vp:,.2f}")
        print(f"  Vanna P&L +1%&+1vp:    {vanna_pnl:,.2f}   (spot & vol co-move)")
        print(f"  Volga P&L vol +1vp:    {volga_pnl:,.2f}   (vol convexity)")
        print(f"  Theta decay (1 day):   {theta_exp:,.2f}")

        results.append({
            'label':          label,
            'target_delta':   target_delta,
            'strike':         K,
            'sigma':          sigma,
            'skew_vs_atm':    (sigma - sigma_atm) * 100,
            'premium':        premium,
            'delta':          delta_exp,
            'gamma_1pct':     gamma_exp,
            'gamma_pa':       gamma_pa_exp,
            'vega':           vega_exp,
            'vanna':          vanna_exp,
            'volga':          volga_exp,
            'theta_daily':    theta_exp,
            'gamma_pnl_1pct': gamma_pnl_1pct,
            'gamma_pnl_2pct': gamma_pnl_2pct,
            'vega_pnl_1vp':   vega_pnl_1vp,
            'vanna_pnl':      vanna_pnl,
            'volga_pnl':      volga_pnl})
    print(f"\n{'='*65}")
    return pd.DataFrame(results).set_index('label')


# get_smile_option_greeks_ALL(
#     pair        = 'USDBRL',
#     option_type = 'put',
#     notional    = 20_000_000,
#     tenor       = 78,
#     deltas      = [0.37])

