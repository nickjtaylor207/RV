
from dataclasses import dataclass
from datetime import date
from typing import Dict
import numpy as np
from scipy.optimize import brentq 


from pricer import (bs_price, delta, delta_premium_adjusted,
                    gamma, gamma_premium_adjusted, vega, volga,
                    vanna, vanna_premium_adjusted, theta)
 
 
@dataclass
class FXOption:
    """
    Represents a single European FX vanilla option.
 
    All rates and vols passed at call time (not stored) so the object
    can be repriced freely at any point in a backtest loop.
 
    Attributes
    ----------
    pair        : currency pair string e.g. 'EURUSD'
    S0          : spot rate at inception
    K           : strike
    expiry      : fixed expiry date
    r_d         : domestic interest rate at inception (decimal)
    r_f         : foreign interest rate at inception (decimal)
    sigma0      : implied vol at inception (decimal)
    option_type : 'call' or 'put'
    notional    : notional in base currency units
    """
    pair:        str
    S0:          float
    K:           float
    expiry:      date
    r_d:         float
    r_f:         float
    sigma0:      float
    option_type: str   = 'call'
    notional:    float = 1_000_000

    def time_to_expiry(self, current_date: date) -> float:
        """Remaining time to expiry in years. Floored at a small epsilon."""
        return max((self.expiry - current_date).days / 365.0, 1e-6)

    # -------------------------------------------------------------------------------------------------

    def price(self, S, sigma, r_d, r_f, current_date):
        """Present value of the option in domestic currency (pure GK, no spot-date roll)."""
        T = self.time_to_expiry(current_date)
        return bs_price(S, self.K, T, r_d, r_f, sigma, self.option_type)

    def price_foreign(self, S, sigma, r_d, r_f, current_date):
        """Option price in foreign (base) currency units: domestic price / S."""
        return self.price(S, sigma, r_d, r_f, current_date) / S
 
    # -------------------------------------------------------------------------------------------------


    def get_delta(self, S: float, sigma: float, r_d: float, r_f: float,
                  current_date: date) -> float:
        T = self.time_to_expiry(current_date)
        return delta(S, self.K, T, r_d, r_f, sigma, self.option_type)

    def get_delta_pa(self, S: float, sigma: float, r_d: float, r_f: float,
                     current_date: date) -> float:
        """Premium-adjusted spot delta (premium paid in base/foreign ccy)."""
        T = self.time_to_expiry(current_date)
        return delta_premium_adjusted(S, self.K, T, r_d, r_f, sigma, self.option_type)

    def get_call_delta(self, S: float, sigma: float, r_d: float, r_f: float,
                       current_date: date) -> float:
        """
        Call-delta-equivalent of this strike, for smile lookups (DELTA_GRID_MAP axis).
        Calls return their spot delta directly; puts convert to the same-strike
        call delta: delta_c = delta_p + exp(-r_f*T).
        """
        d = self.get_delta(S, sigma, r_d, r_f, current_date)
        if self.option_type == 'put':
            d += np.exp(-r_f * self.time_to_expiry(current_date))
        return d
    

    # -------------------------------------------------------------------------------------------------
 
    # -------------------------------------------------------------------------------------------------
    # Gamma
    # -------------------------------------------------------------------------------------------------

    def get_gamma_raw(self, S: float, sigma: float, r_d: float, r_f: float,
                      current_date: date) -> float:
        T = self.time_to_expiry(current_date)
        return gamma(S, self.K, T, r_d, r_f, sigma)

    def get_gamma_1pct(self, S: float, sigma: float, r_d: float, r_f: float,
                       current_date: date) -> float:
        """Delta change per unit notional for a +1% spot move: gamma_raw * S * 0.01"""
        return self.get_gamma_raw(S, sigma, r_d, r_f, current_date) * S * 0.01

    def get_gamma_pa(self, S: float, sigma: float, r_d: float, r_f: float,
                     current_date: date) -> float:
        """Premium-adjusted gamma — use for Taylor diagnostic in base-ccy P&L space."""
        T = self.time_to_expiry(current_date)
        return gamma_premium_adjusted(S, self.K, T, r_d, r_f, sigma, self.option_type)

    # -------------------------------------------------------------------------------------------------
    # Vega, Volga, Vanna
    # -------------------------------------------------------------------------------------------------

    def get_vega(self, S: float, sigma: float, r_d: float, r_f: float,
                 current_date: date) -> float:
        T = self.time_to_expiry(current_date)
        return vega(S, self.K, T, r_d, r_f, sigma)

    def get_volga(self, S: float, sigma: float, r_d: float, r_f: float,
                  current_date: date) -> float:
        """Vol convexity — second derivative of price w.r.t. sigma."""
        T = self.time_to_expiry(current_date)
        return volga(S, self.K, T, r_d, r_f, sigma)

    def get_vanna(self, S: float, sigma: float, r_d: float, r_f: float,
                  current_date: date) -> float:
        """Mixed spot-vol sensitivity — drives P&L when spot and vol co-move."""
        T = self.time_to_expiry(current_date)
        return vanna(S, self.K, T, r_d, r_f, sigma)
    
    def get_vanna_pa(self, S: float, sigma: float, r_d: float, r_f: float,
                 current_date: date) -> float:
        """
        Premium-adjusted vanna — eq (2.42).
        Used by Bloomberg and most interbank systems when premium is pbaseccy%.
            X_pa = X - V / S
        """
        T = self.time_to_expiry(current_date)
        return vanna_premium_adjusted(S, self.K, T, r_d, r_f, sigma)

    # -------------------------------------------------------------------------------------------------
    # Theta
    # -------------------------------------------------------------------------------------------------

    def get_theta(self, S: float, sigma: float, r_d: float, r_f: float,
                  current_date: date) -> float:
        T = self.time_to_expiry(current_date)
        return theta(S, self.K, T, r_d, r_f, sigma, self.option_type)

    # -------------------------------------------------------------------------------------------------
    # Foreign (base) currency greek bundle
    # -------------------------------------------------------------------------------------------------

    def greeks_foreign(self, S: float, sigma: float, r_d: float,
                       r_f: float, current_date: date) -> Dict[str, float]:
        """
        All greeks in foreign (base) currency terms.

        Conversions applied:
            delta : premium-adjusted (nets out base-ccy premium FX risk)
            gamma : trader 1% scaled (raw * S * 0.01); NOT PA — kept in
                    spot-delta units for display. Use get_gamma_pa() for
                    the Taylor diagnostic in the backtest.
            vega  : scaled to 1 vol point move, converted to base ccy  (/ S)
            volga : raw (base ccy); scaled by caller if needed
            vanna : premium-adjusted (vanna_raw - vega/S); consistent with
                    the PA delta hedge — use get_vanna() for raw if needed
            theta : per year, converted to base ccy (/ S)
        """
        return {
            'delta': self.get_delta_pa(S, sigma, r_d, r_f, current_date),
            'gamma': self.get_gamma_1pct(S, sigma, r_d, r_f, current_date),
            'vega':  (self.get_vega(S, sigma, r_d, r_f, current_date) * 0.01) / S,
            'volga': self.get_volga(S, sigma, r_d, r_f, current_date),
            'vanna': self.get_vanna_pa(S, sigma, r_d, r_f, current_date),
            'theta': self.get_theta(S, sigma, r_d, r_f, current_date) / S,
        }
    



# =============================================================================
# FORWARD PRICE
# The GK forward: F = S * exp((r_d - r_f) * T)
# This is the strike that makes a straddle delta-neutral at inception —
# i.e. true ATM-forward, as opposed to ATM-spot (K = S).
# =============================================================================
 
def forward_price(S: float, r_d: float, r_f: float, T: float) -> float:
    """
    Garman-Kohlhagen forward price.
        F = S * exp((r_d - r_f) * T)
    """
    return S * np.exp((r_d - r_f) * T)


def find_atm_forward_strike(S0:     float,
                             r_d:    float,
                             r_f:    float,
                             expiry: date,
                             today:  date = None) -> float:
    """
    Return the ATM-forward strike: F = S * exp((r_d - r_f) * T).
 
    This is the theoretically correct ATM strike — the forward price
    is where a straddle is delta-neutral at inception, and is always
    preferred over ATM-spot (K = S0) for any option with meaningful
    carry (r_d != r_f).
 
    Parameters
    ----------
    S0     : spot at inception
    r_d    : domestic rate (decimal)
    r_f    : foreign rate (decimal)
    expiry : option expiry date
    today  : pricing date, defaults to today if not provided
 
    Returns
    -------
    float : the ATM-forward strike
    """
    if today is None:
        from datetime import date as _date
        today = _date.today()
    T = max((expiry - today).days / 365.25, 1e-6)
    return forward_price(S0, r_d, r_f, T)
















# =============================================================================
# STRIKE FINDER
# Solve for the strike that corresponds to a target delta under a given vol.
# Use this at trade entry to convert a delta (e.g. 25d) into a strike.
# =============================================================================
def find_strike_from_delta(pair:         str,
                            S0:           float,
                            expiry:       date,
                            r_d:          float,
                            r_f:          float,
                            sigma:        float,
                            target_delta: float,
                            option_type:  str   = 'call',
                            notional:     float = 1_000_000,
                            today:        date  = None) -> float:
    """
    Solve for the strike that gives exactly target_delta under the given vol.
 
    Parameters
    ----------
    sigma        : smile vol at the target delta, from dataset.get_smile_vol()
    target_delta : signed spot delta e.g.
                       +0.25  25d call
                       -0.25  25d put
                        0.0   ATM
    today        : pricing date — defaults to expiry if not provided
                   (only affects time_to_expiry inside the solver)
    Returns
    -------
    float : the solved strike K
    """
    if today is None:
        today = expiry
    def delta_error(K: float) -> float:
        temp = FXOption(pair=pair, S0=S0, K=K, expiry=expiry,
                        r_d=r_d, r_f=r_f, sigma0=sigma,
                        option_type=option_type, notional=notional)
        return temp.get_delta(S0, sigma, r_d, r_f, today) - target_delta
    return brentq(delta_error, S0 * 0.85, S0 * 1.20)






