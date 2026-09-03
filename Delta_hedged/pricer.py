import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq



# --- Helper Functions ---
def d1(S, K, T, r_d, r_f, sigma):
    return (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r_d, r_f, sigma):
    return d1(S, K, T, r_d, r_f, sigma) - sigma * np.sqrt(T)


# --- Garman-Kohlhagen Pricing ---
def bs_price(S, K, T, r_d, r_f, sigma, option_type='call'):
    """Price a European FX option using the Garman-Kohlhagen formula."""
    D1 = d1(S, K, T, r_d, r_f, sigma)
    D2 = d2(S, K, T, r_d, r_f, sigma)

    if option_type == 'call':
        return S * np.exp(-r_f * T) * norm.cdf(D1) - K * np.exp(-r_d * T) * norm.cdf(D2)
    elif option_type == 'put':
        return K * np.exp(-r_d * T) * norm.cdf(-D2) - S * np.exp(-r_f * T) * norm.cdf(-D1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")




# --- Greeks ---
def delta(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Spot delta — eq (2.33).
    Standard delta for pnumccy premium pairs (most G10).
    Amount of base ccy to trade in spot hedge.

        call:  exp(-r_f*T) * N(d1)
        put:  -exp(-r_f*T) * N(-d1)
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    if option_type == 'call':
        return np.exp(-r_f * T) * norm.cdf(D1)
    else:
        return -np.exp(-r_f * T) * norm.cdf(-D1)


def delta_premium_adjusted(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Premium-adjusted spot delta — eq (2.34) / (2.35).
    Use when premium is paid in base (foreign) currency (pbaseccy%).
    Nets out the FX risk of the premium itself from the hedge amount.

        call:   exp(-r_d*T) * (K/S) * N(d2)
        put:   -exp(-r_d*T) * (K/S) * N(-d2)

    Identically equal to:  standard_delta - bs_price / S
    """
    D2 = d2(S, K, T, r_d, r_f, sigma)
    if option_type == 'call':
        return  np.exp(-r_d * T) * (K / S) * norm.cdf(D2)
    elif option_type == 'put':
        return -np.exp(-r_d * T) * (K / S) * norm.cdf(-D2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r_d, r_f, sigma):
    """
    Spot gamma — eq (2.36). Same for calls and puts under GK.
    Second derivative of option price w.r.t. spot.
    Measures rate of change of delta per unit move in spot.

        Γ = exp(-r_f*T) * φ(d1) / (S * σ * √T)
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    return np.exp(-r_f * T) * norm.pdf(D1) / (S * sigma * np.sqrt(T))


def gamma_trader(S, K, T, r_d, r_f, sigma):
    """
    Trader's gamma — eq (2.38).
    Rescaled gamma showing delta change for a 1% move in spot.
    More intuitive for risk management than raw gamma.

        Γ_trader = Γ * S / 100
    """
    return gamma(S, K, T, r_d, r_f, sigma) * S / 100


def gamma_premium_adjusted(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Premium-adjusted gamma — eq (2.37).
    Gamma of the premium-adjusted delta; required when premium is pbaseccy%.

        Γ_pa = Γ - (Γ * 0.01*pbaseccy% - 2 * Δ_pa) / S

    Equivalently written as:
        Γ_pa = Γ - (price/S - 2*Δ_pa) / S
               where price is expressed as pbaseccy% (i.e. bs_price/S * 100)
    """
    G     = gamma(S, K, T, r_d, r_f, sigma)
    D_pa  = delta_premium_adjusted(S, K, T, r_d, r_f, sigma, option_type)
    price = bs_price(S, K, T, r_d, r_f, sigma, option_type)
    pbaseccy_pct = price / S * 100           # convert to pbaseccy%
    return G - (G * 0.01 * pbaseccy_pct - 2 * D_pa) / S


def vega(S, K, T, r_d, r_f, sigma):
    """
    Vega — eq (2.39). Same for calls and puts under GK.
    First derivative of option price w.r.t. implied volatility.

        V = P_f * S * √T * φ(d1)
          = exp(-r_f*T) * S * √T * φ(d1)
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    return S * np.exp(-r_f * T) * norm.pdf(D1) * np.sqrt(T)



# ----------------------------------------------------------------------------------------

def volga(S, K, T, r_d, r_f, sigma):
    """
    Volga (vomma) — eq (2.40). Same for calls and puts under GK.
    Second derivative of option price w.r.t. implied volatility.
    Measures convexity of option value w.r.t. vol — analogous to gamma for vol.

        W = P_f * S * φ(d1) * d1 * d2 * √T / σ
          = Vega * d1 * d2 / σ
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    D2 = d2(S, K, T, r_d, r_f, sigma)
    return S * np.exp(-r_f * T) * norm.pdf(D1) * np.sqrt(T) * D1 * D2 / sigma

def vanna(S, K, T, r_d, r_f, sigma):
    """
    Vanna — eq (2.41). Same for calls and puts under GK.
    Mixed second derivative: sensitivity of vega to spot (= sensitivity of delta to vol).
    Key greek for skew risk — drives P&L when spot and vol move together.

        X = -P_f * φ(d1) * d2 / σ
          = -exp(-r_f*T) * φ(d1) * d2 / σ
          = -Vega * d2 / (S * σ * √T)       [equivalent form]
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    D2 = d2(S, K, T, r_d, r_f, sigma)
    return -np.exp(-r_f * T) * norm.pdf(D1) * D2 / sigma

def vanna_premium_adjusted(S, K, T, r_d, r_f, sigma):
    """
    Premium-adjusted vanna — eq (2.42).
    Modifies standard vanna when premium is pbaseccy% by subtracting vega/S.
    Accounts for the spot sensitivity of the premium itself.

        X_pa = X - V / S
    """
    return vanna(S, K, T, r_d, r_f, sigma) - vega(S, K, T, r_d, r_f, sigma) / S


# ----------------------------------------------------------------------------------------




def theta(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Theta. Rate of change of option price w.r.t. time (per year).
    Divide by 365 for daily decay.
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    D2 = d2(S, K, T, r_d, r_f, sigma)
    term1 = -S * np.exp(-r_f * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    if option_type == 'call':
        return term1 + r_f * S * np.exp(-r_f * T) * norm.cdf(D1) - r_d * K * np.exp(-r_d * T) * norm.cdf(D2)
    else:
        return term1 - r_f * S * np.exp(-r_f * T) * norm.cdf(-D1) + r_d * K * np.exp(-r_d * T) * norm.cdf(-D2)




def rho_d(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Domestic rho — eq (2.43).
    Sensitivity of option price to the domestic interest rate.

        call:  K * T * exp(-r_d*T) * N(d2)
        put:  -K * T * exp(-r_d*T) * N(-d2)
    """
    D2 = d2(S, K, T, r_d, r_f, sigma)
    if option_type == 'call':
        return  K * T * np.exp(-r_d * T) * norm.cdf(D2)
    elif option_type == 'put':
        return -K * T * np.exp(-r_d * T) * norm.cdf(-D2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho_f(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Foreign rho — eq (2.44).
    Sensitivity of option price to the foreign interest rate.
    Note the sign: higher foreign rates reduce call value (like a dividend).

        call:  -S * T * exp(-r_f*T) * N(d1)
        put:    S * T * exp(-r_f*T) * N(-d1)
    """
    D1 = d1(S, K, T, r_d, r_f, sigma)
    if option_type == 'call':
        return -S * T * np.exp(-r_f * T) * norm.cdf(D1)
    elif option_type == 'put':
        return  S * T * np.exp(-r_f * T) * norm.cdf(-D1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r_d, r_f, sigma, option_type='call'):
    """
    Convenience wrapper returning both rhos as a tuple — (rho_d, rho_f).
    """
    return rho_d(S, K, T, r_d, r_f, sigma, option_type), \
           rho_f(S, K, T, r_d, r_f, sigma, option_type)