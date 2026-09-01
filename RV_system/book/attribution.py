"""
Daily marking and P&L attribution, per position.

WHAT THIS DOES
--------------
Given a position, its state at the START of the period, and a snapshot at the
END, produce one record row: the exact repriced P&L, the Taylor decomposition
of it, the realised-vs-implied (`_be`) buckets, and the reconciliation residual.

Hedging is NOT here. In the old stack every trade hedged itself, so a book of
eight overlapping trades ran eight hedges that partly offset and paid
transaction cost on GROSS delta where a desk pays on NET. Here a position only
ever reports its own delta; the netting and the single hedge live in engine/.
That separation is the main reason for the rewrite.

THE ONE-PERIOD CONVENTION
-------------------------
All attribution for a period is priced off START-of-period greeks. So the
greeks written into a row describe the position going INTO that day, not coming
out of it. Same convention as the old engine; keep it, because mixing the two
is the classic way to get an attribution that looks right and drifts.

`dt_days` is the ACTUAL calendar gap between consecutive observations -- 3 over
a weekend, not 1. Theta and carry scale by it. Gamma and vol P&L over a gap are
realised as one lump move off Friday's greeks, which is the standard daily-bar
limitation: expect an outsized weekend `gamma_pnl` occasionally, and read
`recon_resid` when you see one.

THE BREAKEVEN (_be) BUCKETS ARE THE POINT OF THE STRATEGY
---------------------------------------------------------
Each `_be` bucket nets the realised move against what the market's own implied
dynamics predicted, leaving only the surprise. They were a diagnostic in the old
stack. Here they are the objective function:

    gamma_pnl_be = gamma_pnl + theta_pnl
        Realised vs implied VARIANCE, scaled by gamma. Works because BS theta is
        to leading order -0.5*S^2*Gamma*sigma^2, so adding it back cancels the
        variance the position was priced for. This is the ATM VRP.

    vanna_pnl_be : realised dS*dsigma vs the S*sigma^2*rho*nu*dt the smile implied.
        This is the SKEW premium.

    volga_pnl_be : realised dsigma^2 vs the sigma^2*nu^2*dt the smile implied.
        This is the CONVEXITY / vol-of-vol premium -- the wings.

A single day of any of these is mostly noise. The DRIFT of the cumulative is the
signal (Ravagli 2024's own framing). Rank strategies on that drift per unit of
the corresponding greek carried, not on calmar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np

from core.greeks import GreekVector, SPOT_MOVE, VOL_MOVE


@dataclass
class PositionMark:
    """
    A position's state at a point in time -- everything the next period's
    attribution needs, cached so the surface is not re-hit for yesterday.

    Held by the Book, one per open position, rolled forward each day.
    """
    on:          date
    spot:        float
    sigma:       float          # the position's OWN smile vol at its fixed strike
    r_d:         float
    r_f:         float
    t_days:      float
    value_base:  float          # SIGNED mark-to-market, notional applied
    greeks:      GreekVector
    nu:          float = 0.0    # breakeven vol-of-vol at this t_remaining
    rho:         float = 0.0    # breakeven spot/vol correlation
    notional:    float = np.nan # the size that was on risk, for auditing


def take_mark(pos, snap, with_nu_rho: bool = True) -> PositionMark:
    """
    Snapshot a position's current state. Called once per position per day, and
    the result becomes the START-of-period state for tomorrow's attribution.
    """
    st = snap.price_state(pos.option.K, pos.expiry)
    g  = pos.greeks_from_state(st, snap.date)
    v  = pos.option.value_base(st['S'], st['sigma'], st['r_d'], st['r_f'],
                               snap.date) * pos.signed_notional

    nu = rho = 0.0
    if with_nu_rho:
        try:
            nu, rho = snap.nu_rho(st['t_days'])
        except Exception:
            # get_smile_nu_rho already falls back to (0,0) on missing pillar
            # quotes, which collapses the _be buckets onto the plain ones.
            # Catching here too so one bad pillar cannot kill a whole run.
            nu, rho = 0.0, 0.0

    return PositionMark(on=snap.date, spot=st['S'], sigma=st['sigma'],
                        r_d=st['r_d'], r_f=st['r_f'], t_days=st['t_days'],
                        value_base=v, greeks=g, nu=nu, rho=rho,
                        notional=pos.notional)


# --------------------------------------------------------------------- #
# The mark
# --------------------------------------------------------------------- #
def mark_position(pos, prev: PositionMark, snap,
                  nu_rho_at_end: bool = False) -> Tuple[Dict, Optional[PositionMark]]:
    """
    Advance one position from `prev` to `snap`, producing its record row.

    nu_rho_at_end
        Which nu/rho to build the `_be` expectations from.

        False (default, and correct): nu/rho as of the START of the period.
            The `_be` buckets ask "what did the market, AT THE START, imply
            this period's dS*dsigma and dsigma^2 would be?" That expectation
            must be formed from start-of-period information, exactly like
            every other term in the attribution, all of which is priced off
            start-of-day greeks.

        True (legacy): nu/rho as of the END of the period. This is what the
            old engine did -- `get_smile_nu_rho(pair, current_dt, t_remaining)`
            at backtest_MLeg.py:630, using TODAY's date and TODAY's remaining
            tenor, then applying it to yesterday's greeks. It is a mild
            look-ahead: the expectation of a move is formed using information
            revealed by that move. It also makes the buckets jump whenever
            t_remaining crosses the midpoint between two tenor pillars, since
            get_smile_nu_rho snaps to the NEAREST pillar -- which is why the
            reconciliation differences in volga_pnl_be are large, sporadic and
            concentrated on particular dates rather than uniformly spread.

        Set True only to reconcile against the old engine.

    Returns
    -------
    (row, new_mark)
        row      : dict of everything that happened over the period
        new_mark : the end-of-period state, or None if the position expired
                   (nothing left to roll forward)

    Two branches, exactly as in the old engine:

    NATURAL EXPIRY  -- the option settles to intrinsic. All vol buckets are
        hard-zeroed (there is no vol sensitivity left to have), but gamma and
        theta ARE still computed from start-of-day greeks, because the realised
        spot move over that final period was real and gamma_pnl never depended
        on the vol greeks.

    ORDINARY DAY    -- exact reprice for the P&L, Taylor buckets off prev
        greeks for the attribution, and the gap between them recorded as
        `recon_resid`.
    """
    dt_days = float((snap.date - prev.on).days)
    if dt_days <= 0:
        raise ValueError(f"non-advancing mark for {pos}: {prev.on} -> {snap.date}")

    expired = pos.is_expired(snap.date)

    S_new  = snap.spot
    dS     = S_new - prev.spot
    u      = (dS / prev.spot) / SPOT_MOVE          # spot move in "1% units"

    g = prev.greeks                                # ALL attribution off these

    # ---- value change (the truth; everything else is a decomposition of it)
    if expired:
        v_new  = pos.settle_base(S_new)
        sigma_new = np.nan
        dsigma = 0.0
        w      = 0.0
    else:
        st = snap.price_state(pos.option.K, pos.expiry)
        sigma_new = st['sigma']
        v_new  = pos.option.value_base(st['S'], st['sigma'], st['r_d'], st['r_f'],
                                       snap.date) * pos.signed_notional
        dsigma = sigma_new - prev.sigma
        w      = dsigma / VOL_MOVE                 # vol move in "1 vol pt units"

    option_pnl = v_new - prev.value_base

    # ---- Taylor buckets. No scaling constants: the greeks are already
    #      money-per-standard-move (see core/greeks.py).
    delta_pnl = g.spot_1pct  * u
    gamma_pnl = g.gamma_1pct * u * u
    theta_pnl = g.theta_1d   * dt_days

    if expired:
        vega_pnl = volga_pnl = vanna_pnl = 0.0
        vanna_pnl_be = volga_pnl_be = 0.0
        nu_used, rho_used = prev.nu, prev.rho
    else:
        vega_pnl  = g.vega_1vp       * w
        volga_pnl = g.volga_1vp      * w * w
        vanna_pnl = g.vanna_1pct_1vp * u * w

        # --- breakeven buckets: realised move MINUS the move the smile implied
        dt_years = dt_days / 365.0
        s0 = prev.sigma
        if nu_rho_at_end:
            try:
                nu, rho = snap.nu_rho(st['t_days'])
            except Exception:
                nu, rho = 0.0, 0.0
        else:
            nu, rho = prev.nu, prev.rho
        nu_used, rho_used = nu, rho

        cross_expected = prev.spot * s0 * s0 * rho * nu * dt_years   # E[dS * dsigma]
        var_expected   = s0 * s0 * nu * nu * dt_years                # E[dsigma^2]

        # Convert the surprise back into standard units and re-apply the greek.
        vanna_pnl_be = g.vanna_1pct_1vp * (dS * dsigma - cross_expected) \
                       / ((SPOT_MOVE * prev.spot) * VOL_MOVE)
        volga_pnl_be = g.volga_1vp * (dsigma * dsigma - var_expected) \
                       / (VOL_MOVE * VOL_MOVE)

    # gamma_be is realised-vs-implied variance and holds on expiry day too
    gamma_pnl_be = gamma_pnl + theta_pnl

    # ---- reconciliation. Unlike the old engine's version this is per-position
    #      and hedge-free: it includes the first-order delta term, so it is a
    #      clean "how good was the Taylor expansion" number rather than a
    #      quantity entangled with the hedge. A large |recon_resid| means a big
    #      discrete jump (gap risk) or that the expansion broke down -- it is a
    #      diagnostic, NOT an error to drive to zero.
    taylor_total = (delta_pnl + gamma_pnl + theta_pnl
                    + vega_pnl + vanna_pnl + volga_pnl)
    recon_resid  = option_pnl - taylor_total

    row = {
        'date':    snap.date,
        'pos_id':  pos.pos_id,
        'pair':    pos.pair,
        'sleeve':  pos.sleeve,
        'tag':     pos.tag,
        'tenor_label':  pos.tenor_label,
        'target_delta': pos.target_delta,
        'option_type':  pos.option_type,
        'strike':       pos.option.K,
        'expiry':       pos.expiry,
        'direction':    pos.direction,
        'notional':     prev.notional,      # the size actually on risk

        # market context over the period
        'dt_days':   dt_days,
        'spot':      S_new,
        'prev_spot': prev.spot,
        'dS':        dS,
        'sigma':     sigma_new,
        'prev_sigma': prev.sigma,
        'dsigma':    dsigma,
        't_days':    max((pos.expiry - snap.date).days, 0),
        'nu_be':     nu_used,
        'rho_be':    rho_used,
        'expired':   expired,

        # START-of-period greeks (what the P&L below was earned on)
        **g.as_dict(),

        # P&L
        'option_pnl': option_pnl,
        'delta_pnl':  delta_pnl,
        'gamma_pnl':  gamma_pnl,
        'theta_pnl':  theta_pnl,
        'vega_pnl':   vega_pnl,
        'vanna_pnl':  vanna_pnl,
        'volga_pnl':  volga_pnl,

        # realised-vs-implied -- the premium harvest, and the objective function
        'gamma_pnl_be': gamma_pnl_be,
        'vanna_pnl_be': vanna_pnl_be,
        'volga_pnl_be': volga_pnl_be,

        'recon_resid': recon_resid,
    }

    new_mark = None if expired else take_mark(pos, snap)
    return row, new_mark


# --------------------------------------------------------------------- #
# Hedge P&L -- book level, one per pair per day
# --------------------------------------------------------------------- #
def hedge_period_pnl(hedge_notional: float,
                     prev_spot: float,
                     dS: float,
                     r_d: float,
                     r_f: float,
                     dt_days: float) -> Dict[str, float]:
    """
    P&L and carry on the spot hedge CARRIED IN from the previous close.

        hedge_pnl   = hedge_notional * dS / prev_spot
        hedge_carry = hedge_notional * (r_f - r_d) * dt_days / 365

    `hedge_notional` is the position held coming INTO the period, not the
    target computed at the end of it. Getting that wrong gives the hedge a
    one-day look-ahead and quietly flatters every result.

    Sign: negative hedge_notional = short base ccy (the usual state when the
    book is long delta from short puts).
    """
    return {
        'hedge_pnl':   hedge_notional * dS / prev_spot,
        'hedge_carry': hedge_notional * (r_f - r_d) * dt_days / 365.0,
    }


# ====================================================================== #
# TEST BLOCK -- uncomment and run:  python book/attribution.py
# Requires a live Bloomberg connection.
# ====================================================================== #
# if __name__ == '__main__':
#     import os as _os, sys as _sys
#     _sys.path.insert(0, _os.path.dirname(
#         _os.path.dirname(_os.path.abspath(__file__))))
#     import pandas as pd
#     from market.dataset import FXVolDataset
#     from market.snapshot import MarketSnapshot, business_dates
#     from book.position import LegRequest, open_position
#     from core.calendar import add_tenor
#     from core.conventions import fx_calendar
#
#     PAIR = 'USDJPY'
#     ds    = FXVolDataset.build(pairs=[PAIR], days=500)
#     dates = business_dates(ds, PAIR)
#     fxc   = fx_calendar(PAIR)
#
#     # open a short 25d strangle 60 business days ago, 1M tenor
#     d0    = dates[-70]
#     snap0 = MarketSnapshot.at(ds, PAIR, d0)
#     expiry = add_tenor(snap0.date, '1M', fxc)
#
#     legs = [open_position(LegRequest(t, -1, 10_000_000, '1M',
#                                      target_delta=dl, sleeve='wings', tag=tag),
#                           snap0, expiry)
#             for t, dl, tag in [('call', +0.25, 'wing_c'), ('put', -0.25, 'wing_p')]]
#     for p in legs:
#         print(p)
#     marks = {p.pos_id: take_mark(p, snap0) for p in legs}
#     print(f'\nentry premium received: '
#           f'{sum(p.entry_premium for p in legs):,.2f} {snap0.base}\n')
#
#     # --- walk it forward to expiry, one position-row per leg per day
#     rows = []
#     for d in dates[dates > pd.Timestamp(d0)]:
#         snap = MarketSnapshot.at(ds, PAIR, d)
#         for p in legs:
#             m = marks.get(p.pos_id)
#             if m is None:
#                 continue
#             row, new_m = mark_position(p, m, snap)
#             rows.append(row)
#             if new_m is None:
#                 marks[p.pos_id] = None
#                 print(f'  {p.tag} settled {d.date()} '
#                       f'intrinsic={p.settle_base(snap.spot):,.2f}')
#             else:
#                 marks[p.pos_id] = new_m
#         if all(v is None for v in marks.values()):
#             break
#
#     df = pd.DataFrame(rows)
#     print(f'\n{len(df)} position-days over {df.date.nunique()} dates\n')
#
#     # --- 1. THE KEY TEST: the Taylor buckets should track the exact reprice.
#     #     They will NOT match exactly -- recon_resid is the by-design gap.
#     #     What matters is that the residual is small relative to the P&L.
#     tot = df[['option_pnl', 'delta_pnl', 'gamma_pnl', 'theta_pnl',
#               'vega_pnl', 'vanna_pnl', 'volga_pnl', 'recon_resid']].sum()
#     print('summed over the whole trade, both legs:')
#     for k, v in tot.items():
#         print(f'  {k:<12} {v:>14,.2f}')
#     taylor = tot[['delta_pnl', 'gamma_pnl', 'theta_pnl',
#                   'vega_pnl', 'vanna_pnl', 'volga_pnl']].sum()
#     print(f'\n  taylor sum   {taylor:>14,.2f}   vs option_pnl '
#           f'{tot.option_pnl:>14,.2f}')
#     print(f'  residual     {tot.recon_resid:>14,.2f}   '
#           f'({abs(tot.recon_resid) / max(abs(tot.option_pnl), 1) * 100:.2f}% of reprice)')
#
#     # --- 2. The premium buckets. For a SHORT wing position you want these
#     #     positive in aggregate -- that is the premium showing up.
#     be = df.groupby('date')[['gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be']].sum().cumsum()
#     print('\ncumulative breakeven (realised MINUS implied) buckets:')
#     print(f'  gamma_be (ATM VRP)        {be.gamma_pnl_be.iloc[-1]:>12,.2f}')
#     print(f'  vanna_be (SKEW premium)   {be.vanna_pnl_be.iloc[-1]:>12,.2f}')
#     print(f'  volga_be (CONVEXITY prem) {be.volga_pnl_be.iloc[-1]:>12,.2f}')
#     print('\n  Interpretation: positive = the realised move came in BELOW what')
#     print('  the smile implied, so the premium you sold was rich and you kept it.')
#     print('  Negative = you were run over. One trade is one sample; the drift')
#     print('  across hundreds is the actual signal.')
#
#     # --- 3. Where the Taylor expansion strained -- your gap-risk detector
#     worst = df.reindex(df.recon_resid.abs().sort_values(ascending=False).index).head(3)
#     print('\nlargest reconciliation residuals (gap-risk days):')
#     print(worst[['date', 'tag', 'dt_days', 'dS', 'dsigma',
#                  'option_pnl', 'recon_resid']].to_string(index=False))
