"""
Garman-Kohlhagen / Black-76 pricing and greeks for FX vanillas.

THE THREE CLOCKS
----------------
An FX option runs on three different time windows, and collapsing them into a
single T is an approximation, not a convention choice:

    tau_var   today      -> expiry     how much variance accumulates
    tau_fwd   spot date  -> delivery   over which the rate differential accrues
    tau_disc  today      -> delivery   over which the payoff is discounted

They differ by the T+2 settlement lag at each end. Usually the two lags cancel
and all three are within a day of each other, but they come apart whenever a
holiday cluster stretches one lag: USDJPY 1M struck 21-Aug-2026 runs 28 / 31 /
35 days, because Tokyo's Silver Week pushes the expiry->delivery lag out to
seven calendar days.

Every function here takes `T`, which may be EITHER

    a float       -> all three windows equal (the old behaviour, unchanged), or
    an OptionTime -> the three windows stated separately.

so existing float call sites keep returning exactly what they always did.

Written in forward terms:

    F  = S * exp((r_d - r_f) * tau_fwd)
    DF = exp(-r_d * tau_disc)
    d1 = (ln(F/K) + sigma^2 * tau_var / 2) / (sigma * sqrt(tau_var))
    d2 = d1 - sigma * sqrt(tau_var)

    call = DF * (F * N(d1) - K * N(d2))

When the three windows coincide, DF*F collapses to S*exp(-r_f*T) and every
formula below reduces algebraically to the textbook spot-based GK form.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# --- The three clocks ---
@dataclass(frozen=True)
class OptionTime:
    """
    The three time windows of an FX option, in YEARS (ACT/365).

    var  : today -> expiry      (variance)
    fwd  : spot date -> delivery (forward accrual)
    disc : today -> delivery    (discounting)

    Build one with core.option.FXOption.time_basis(), which derives all three
    from the pair's settlement calendar.
    """
    var:  float
    fwd:  float
    disc: float

    @classmethod
    def flat(cls, T: float) -> 'OptionTime':
        """All three windows equal — the single-T approximation, made explicit."""
        return cls(T, T, T)


def _taus(T):
    """Normalise a float-or-OptionTime into (tau_var, tau_fwd, tau_disc)."""
    if isinstance(T, OptionTime):
        return T.var, T.fwd, T.disc
    return T, T, T


def fwd_price(S, T, r_d, r_f):
    """Outright forward to the DELIVERY date: F = S*exp((r_d-r_f)*tau_fwd)."""
    _, tf, _ = _taus(T)
    return S * np.exp((r_d - r_f) * tf)


def df_domestic(T, r_d):
    """Quote-ccy discount factor to the DELIVERY date: exp(-r_d*tau_disc)."""
    _, _, td = _taus(T)
    return np.exp(-r_d * td)


# --- Helper Functions ---
def d1(S, K, T, r_d, r_f, sigma):
    tv, tf, _ = _taus(T)
    return (np.log(S / K) + (r_d - r_f) * tf + 0.5 * sigma**2 * tv) / (sigma * np.sqrt(tv))

def d2(S, K, T, r_d, r_f, sigma):
    tv, _, _ = _taus(T)
    return d1(S, K, T, r_d, r_f, sigma) - sigma * np.sqrt(tv)


# --- Garman-Kohlhagen Pricing ---
def bs_price(S, K, T, r_d, r_f, sigma, option_type='call'):
    """Price a European FX option, per unit base notional, in QUOTE ccy."""
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    D2 = d2(S, K, T, r_d, r_f, sigma)

    if option_type == 'call':
        return DF * (F * norm.cdf(D1) - K * norm.cdf(D2))
    elif option_type == 'put':
        return DF * (K * norm.cdf(-D2) - F * norm.cdf(-D1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")




# --- Greeks ---
def delta(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Spot delta — eq (2.33).
    Standard delta for pnumccy premium pairs (most G10).
    Amount of base ccy to trade in spot hedge.

        call:   DF * (F/S) * N(d1)
        put:   -DF * (F/S) * N(-d1)

    DF*(F/S) is exp(-r_f*T) when the three clocks coincide.
    """
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    if option_type == 'call':
        return DF * (F / S) * norm.cdf(D1)
    else:
        return -DF * (F / S) * norm.cdf(-D1)


def delta_premium_adjusted(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Premium-adjusted spot delta — eq (2.34) / (2.35).
    Use when premium is paid in base (foreign) currency (pbaseccy%).
    Nets out the FX risk of the premium itself from the hedge amount.

        call:   DF * (K/S) * N(d2)
        put:   -DF * (K/S) * N(-d2)

    Identically equal to:  standard_delta - bs_price / S
    """
    DF = df_domestic(T, r_d)
    D2 = d2(S, K, T, r_d, r_f, sigma)
    if option_type == 'call':
        return  DF * (K / S) * norm.cdf(D2)
    elif option_type == 'put':
        return -DF * (K / S) * norm.cdf(-D2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r_d, r_f, sigma):
    """
    Spot gamma — eq (2.36). Same for calls and puts.
    Second derivative of option price w.r.t. spot.

        G = DF * (F/S) * phi(d1) / (S * sigma * sqrt(tau_var))

    F/S is independent of S, so differentiating the delta twice is clean.
    """
    tv, _, _ = _taus(T)
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    return DF * (F / S) * norm.pdf(D1) / (S * sigma * np.sqrt(tv))


def gamma_trader(S, K, T, r_d, r_f, sigma):
    """
    Trader's gamma — eq (2.38).
    Rescaled gamma showing delta change for a 1% move in spot.

        G_trader = G * S / 100
    """
    return gamma(S, K, T, r_d, r_f, sigma) * S / 100


def gamma_premium_adjusted(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Premium-adjusted gamma — eq (2.37).
    Gamma of the premium-adjusted delta; required when premium is pbaseccy%.

        G_pa = G - (G * 0.01*pbaseccy% - 2 * D_pa) / S
    """
    G     = gamma(S, K, T, r_d, r_f, sigma)
    D_pa  = delta_premium_adjusted(S, K, T, r_d, r_f, sigma, option_type)
    price = bs_price(S, K, T, r_d, r_f, sigma, option_type)
    pbaseccy_pct = price / S * 100           # convert to pbaseccy%
    return G - (G * 0.01 * pbaseccy_pct - 2 * D_pa) / S


def vega(S, K, T, r_d, r_f, sigma):
    """
    Vega — eq (2.39). Same for calls and puts.
    First derivative of option price w.r.t. implied volatility.

        V = DF * F * sqrt(tau_var) * phi(d1)
    """
    tv, _, _ = _taus(T)
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    return DF * F * norm.pdf(D1) * np.sqrt(tv)



# ----------------------------------------------------------------------------------------

def volga(S, K, T, r_d, r_f, sigma):
    """
    Volga (vomma) — eq (2.40). Same for calls and puts.
    Second derivative of option price w.r.t. implied volatility.

        W = Vega * d1 * d2 / sigma

    Note d1*d2 < 0 near the ATM forward, so a LONG ATM option is SHORT volga.
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    D2 = d2(S, K, T, r_d, r_f, sigma)
    return vega(S, K, T, r_d, r_f, sigma) * D1 * D2 / sigma

def vanna(S, K, T, r_d, r_f, sigma):
    """
    Vanna — eq (2.41). Same for calls and puts.
    Mixed second derivative: sensitivity of vega to spot (= of delta to vol).

        X = -DF * (F/S) * phi(d1) * d2 / sigma
    """
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    D2 = d2(S, K, T, r_d, r_f, sigma)
    return -DF * (F / S) * norm.pdf(D1) * D2 / sigma

def vanna_premium_adjusted(S, K, T, r_d, r_f, sigma):
    """
    Premium-adjusted vanna — eq (2.42).

        X_pa = X - V / S
    """
    return vanna(S, K, T, r_d, r_f, sigma) - vega(S, K, T, r_d, r_f, sigma) / S


# ----------------------------------------------------------------------------------------




def theta(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Theta. Rate of change of option price w.r.t. calendar time (per year).
    Divide by 365 for daily decay.

    All three clocks shrink as today advances, so

        theta = -(dP/dtau_var + dP/dtau_fwd + dP/dtau_disc)

    with

        dP/dtau_var  =  DF * F * phi(d1) * sigma / (2*sqrt(tau_var))
        dP/dtau_fwd  = +/- DF * F * N(+/-d1) * (r_d - r_f)
        dP/dtau_disc =  -r_d * P

    This is the SMOOTH derivative. The spot date advances in discrete
    business-day jumps, so tau_fwd is flat on some calendar days and drops by
    three across a weekend; realised decay therefore jitters around this
    number. That jitter is a convention artifact, not a greek, and belongs in
    the reconciliation residual rather than in theta.
    """
    tv, _, _ = _taus(T)
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    px = bs_price(S, K, T, r_d, r_f, sigma, option_type)

    term_var = -DF * F * norm.pdf(D1) * sigma / (2 * np.sqrt(tv))
    if option_type == 'call':
        return term_var - DF * F * norm.cdf(D1) * (r_d - r_f) + r_d * px
    else:
        return term_var + DF * F * norm.cdf(-D1) * (r_d - r_f) + r_d * px




def rho_d(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Domestic rho — eq (2.43).
    Sensitivity of option price to the domestic (quote ccy) rate.

    r_d enters twice: through the discount factor over tau_disc, and through
    the forward over tau_fwd.

        dP/dr_d = -tau_disc * P  +  dP/dF * F * tau_fwd
    """
    _, tf, td = _taus(T)
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    px = bs_price(S, K, T, r_d, r_f, sigma, option_type)
    if option_type == 'call':
        return -td * px + DF * F * norm.cdf(D1) * tf
    elif option_type == 'put':
        return -td * px - DF * F * norm.cdf(-D1) * tf
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho_f(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Foreign rho — eq (2.44).
    Sensitivity of option price to the foreign (base ccy) rate, which enters
    only through the forward.

        dP/dr_f = -dP/dF * F * tau_fwd

    Note the sign: higher foreign rates reduce call value (like a dividend).
    """
    _, tf, _ = _taus(T)
    F  = fwd_price(S, T, r_d, r_f)
    DF = df_domestic(T, r_d)
    D1 = d1(S, K, T, r_d, r_f, sigma)
    if option_type == 'call':
        return -tf * DF * F * norm.cdf(D1)
    elif option_type == 'put':
        return  tf * DF * F * norm.cdf(-D1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Convenience wrapper returning both rhos as a tuple — (rho_d, rho_f).
    """
    return rho_d(S, K, T, r_d, r_f, sigma, option_type), \
           rho_f(S, K, T, r_d, r_f, sigma, option_type)
