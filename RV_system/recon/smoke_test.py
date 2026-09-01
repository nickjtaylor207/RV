"""
Offline smoke test - exercise the whole Phase-1 engine with NO Bloomberg.

WHY
---
recon/reconcile.py is the real gate, but it needs a live terminal and the old
stack. This runs the same machinery against a synthetic market so you can:

    * check the plumbing after any refactor, in about a second;
    * run it in CI, or on a laptop with no Bloomberg;
    * test edge cases (a vol spike, a spot jump, a flat market) that real data
      may not contain in the window you happen to pull.

It is NOT a substitute for reconcile.py. A synthetic surface cannot catch a
mistake in the SABR interpolation or the pillar-day arithmetic, because it
does not use them. What it does catch is every wiring error in the loop:
ordering, sign conventions, netting, expiry handling, aggregation.

THE SYNTHETIC MARKET
--------------------
    spot     : GBM with a fixed seed
    ATM vol  : mean-reverting, negatively correlated with spot returns (so
               vanna terms are non-trivial rather than noise)
    smile    : sigma(K) = ATM * (1 + skew * m + convexity * m^2),
               m = log(K/F) / (ATM * sqrt(T))     -- a plain quadratic in
               standardised log-moneyness. Crude, but it has a real skew and a
               real convexity, which is all the engine needs to be exercised.
    rates    : constant
    nu, rho  : constants, so the _be buckets have a defined reference

Because the smile is a closed form rather than a fit, `smile_vol` is exact and
reproducible, and the reconciliation between the exact reprice and the Taylor
buckets is a clean test of core/greeks.py rather than of an interpolator.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from datetime import date, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


class SyntheticDataset:
    """
    Quacks like FXVolDataset for everything MarketSnapshot touches.

    Deliberately implements the SAME method names and signatures, so if the
    real dataset's interface drifts this file breaks loudly instead of
    silently testing a stale contract.
    """

    def __init__(self,
                 pair: str = 'USDJPY',
                 n_days: int = 400,
                 s0: float = 150.0,
                 atm0: float = 0.10,
                 r_d: float = 0.008,
                 r_f: float = 0.045,
                 skew: float = -0.06,        # RR-like: negative = puts bid
                 convexity: float = 0.05,    # BF-like: positive = wings bid
                 nu: float = 0.60,
                 rho: float = -0.30,
                 vol_of_vol: float = 0.55,
                 seed: int = 7,
                 jump_on: int = None,        # index of a deliberate spot jump
                 jump_size: float = -0.04):
        self.pair, self.skew, self.convexity = pair, skew, convexity
        self._r_d, self._r_f = r_d, r_f
        self._nu, self._rho = nu, rho

        rng = np.random.default_rng(seed)

        # business-day calendar with real weekend gaps
        start = date.today() - timedelta(days=int(n_days * 1.45))
        idx = pd.bdate_range(start, periods=n_days)

        # --- spot: GBM, plus an optional discrete jump to test gap handling
        dt = 1 / 252
        shocks = rng.normal(0, atm0 * np.sqrt(dt), n_days)
        if jump_on is not None:
            shocks[jump_on] += jump_size
        spot = s0 * np.exp(np.cumsum(shocks - 0.5 * atm0 ** 2 * dt))

        # --- ATM vol: mean-reverting, negatively correlated with spot returns
        v = np.empty(n_days)
        v[0] = atm0
        for i in range(1, n_days):
            shock = vol_of_vol * v[i - 1] * np.sqrt(dt) * rng.normal()
            corr  = rho * v[i - 1] * vol_of_vol * (shocks[i] / atm0)
            v[i]  = max(0.02, v[i - 1] + 4.0 * (atm0 - v[i - 1]) * dt + shock + corr)

        self.spot = pd.DataFrame({pair: spot}, index=idx)

        # --- a vol_surface frame in the real (pair, tenor, field) layout, in
        #     PERCENT, so quote_history's /100 is exercised
        tenors = ['1W', '1M', '2M', '3M', '6M', '1Y']
        cols, data = [], []
        for t in tenors:
            cols.append((pair, t, 'ATM'));  data.append(v * 100)
            cols.append((pair, t, 'RR25')); data.append(np.full(n_days, skew * atm0 * 100))
            cols.append((pair, t, 'BF25')); data.append(np.full(n_days, convexity * atm0 * 25))
        self.vol_surface = pd.DataFrame(
            np.array(data).T, index=idx,
            columns=pd.MultiIndex.from_tuples(cols, names=['pair', 'tenor', 'type']))

        self._atm = pd.Series(v, index=idx)

    # -- the FXVolDataset interface ------------------------------------- #
    def get_spot(self, pair: str, as_of) -> float:
        return float(self.spot.loc[:pd.Timestamp(as_of), pair].dropna().iloc[-1])

    def get_rates_for_tenor(self, pair: str, as_of, t_days: float) -> Tuple[float, float]:
        return self._r_d, self._r_f

    def get_atm_vol(self, pair: str, as_of, t_days: float, pillar_days=None) -> float:
        return float(self._atm.loc[:pd.Timestamp(as_of)].iloc[-1])

    # Standardised log-moneyness is clipped before the quadratic is applied.
    # Without this the smile explodes as T -> 0 (m ~ 1/sqrt(T)) and a far strike
    # prints an absurd vol on the last few days of a trade -- an artefact of the
    # toy parameterisation, not of the engine. Real smiles flatten in the far
    # wings too, so clipping is also the more faithful behaviour.
    M_CLIP = 3.5

    def get_smile_vol(self, pair: str, as_of, t_days: float,
                      K: float, F: float, r_f: float) -> float:
        atm = self.get_atm_vol(pair, as_of, t_days)
        T   = max(t_days / 365.0, 1e-6)
        m   = np.log(K / F) / (atm * np.sqrt(T))
        m   = float(np.clip(m, -self.M_CLIP, self.M_CLIP))
        return float(max(0.005, atm * (1.0 + self.skew * m + self.convexity * m * m)))

    def get_smile_nu_rho(self, pair: str, as_of, t_days: float) -> Tuple[float, float]:
        return self._nu, self._rho


# ====================================================================== #
def main(verbose: bool = True) -> bool:
    from market.snapshot import MarketSnapshot, business_dates
    from book.position import LegRequest
    from engine.loop import EngineConfig, HoldStatic, RollingStructure, run

    PAIR = 'USDJPY'
    ds = SyntheticDataset(PAIR, n_days=400, jump_on=250, jump_size=-0.045)
    dates = business_dates(ds, PAIR)
    ok = True

    def check(name: str, cond: bool, detail: str = ''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))

    print('=' * 72)
    print('SYNTHETIC SMOKE TEST -- Phase 1 engine, no Bloomberg')
    print('=' * 72)
    print(f'{len(dates)} business dates, '
          f'{dates[0].date()} -> {dates[-1].date()}, '
          f'spot {ds.get_spot(PAIR, dates[0]):.2f} -> {ds.get_spot(PAIR, dates[-1]):.2f}')

    # ------------------------------------------------------------------ #
    print('\n--- 1. Snapshot: point-in-time discipline -------------------')
    snap = MarketSnapshot.at(ds, PAIR, dates[200])
    h = snap.spot_history(90)
    check('spot_history never runs past as_of', h.index.max() <= snap.as_of,
          f'{h.index.max().date()} <= {snap.as_of.date()}')
    vh = snap.atm_history('1M', 90)
    check('atm_history never runs past as_of', vh.index.max() <= snap.as_of)
    check('atm_history converts percent -> decimal', 0.01 < vh.iloc[-1] < 1.0,
          f'{vh.iloc[-1]:.4f}')

    # ------------------------------------------------------------------ #
    print('\n--- 2. Strike/vol fixed point --------------------------------')
    expiry = (snap.date + timedelta(days=30))
    atm = snap.atm_vol(30)
    for tgt, typ in [(+0.25, 'call'), (-0.25, 'put'), (-0.10, 'put')]:
        K, sig = snap.solve_strike_and_vol(tgt, typ, expiry)
        recheck = snap.smile_vol(K, (expiry - snap.date).days)
        print(f'      {typ[0].upper()}{abs(tgt)*100:>3.0f}d  K={K:9.4f}  '
              f'vol={sig*100:7.3f}%  vs ATM {(sig-atm)*100:+6.3f}vp')
        check(f'{typ} {tgt:+.2f} fixed point converged',
              abs(sig - recheck) < 1e-9)
    Kp, sp = snap.solve_strike_and_vol(-0.25, 'put', expiry)
    Kc, sc = snap.solve_strike_and_vol(+0.25, 'call', expiry)
    check('synthetic skew shows up (25d put vol > 25d call vol)', sp > sc,
          f'{sp*100:.3f}% vs {sc*100:.3f}%')
    check('synthetic wings are bid vs ATM', sp > atm and sc > atm - 0.02)

    # ------------------------------------------------------------------ #
    print('\n--- 3. Hold one short strangle to expiry ---------------------')
    legs = [LegRequest('call', -1, 10_000_000, '1M', target_delta=+0.25,
                       sleeve='wings', tag='wing_c'),
            LegRequest('put',  -1, 10_000_000, '1M', target_delta=-0.25,
                       sleeve='wings', tag='wing_p')]
    cfg = EngineConfig(pairs=[PAIR],
                       start=dates[100], end=dates[135],
                       spot_tc=0.0001, verbose=False)
    res = run(HoldStatic(PAIR, legs, '1M'), ds, cfg)

    print(f'      {len(res.positions)} position-days, '
          f'{len(res.trades)} legs, net P&L {res.daily.pnl.sum():,.2f}')
    prem = res.trades.entry_premium.sum()
    print(f'      premium received {prem:,.2f}')
    check('short strangle receives premium', prem > 0)
    check('legs expired inside the window',
          bool(res.positions.expired.any()))
    check('book is empty at the end', res.book.n_open() == 0)

    # THE arithmetic test: buckets vs exact reprice
    p = res.positions
    taylor = p[['delta_pnl', 'gamma_pnl', 'theta_pnl',
                'vega_pnl', 'vanna_pnl', 'volga_pnl']].sum().sum()
    exact  = p.option_pnl.sum()
    resid  = p.recon_resid.sum()
    print(f'      exact reprice {exact:>13,.2f}   '
          f'taylor {taylor:>13,.2f}   resid {resid:>11,.2f} '
          f'({abs(resid)/max(abs(exact),1)*100:.2f}%)')
    check('taylor sum + residual == exact reprice, identically',
          abs(taylor + resid - exact) < 1e-6)
    check('residual is small relative to the reprice',
          abs(resid) < 0.25 * abs(exact))

    # sign sanity on a short book
    g = p.groupby('date')[['vega_1vp', 'volga_1vp', 'theta_1d']].sum()
    check('short strangle is short vega throughout', (g.vega_1vp < 0).all())
    check('short strangle is short volga throughout', (g.volga_1vp < 0).all())
    check('short strangle collects theta throughout', (g.theta_1d > 0).all())

    # the hedge does its job: delta_pnl should be largely offset by hedge_pnl
    dp, hp = res.daily.delta_pnl.sum(), res.daily.hedge_pnl.sum()
    print(f'      delta_pnl {dp:>13,.2f}  + hedge_pnl {hp:>13,.2f}  '
          f'= {dp+hp:>12,.2f}')
    check('full daily hedge offsets most of the delta P&L',
          abs(dp + hp) < 0.5 * max(abs(dp), 1.0))

    # ------------------------------------------------------------------ #
    print('\n--- 4. Netted hedging beats per-trade hedging ----------------')
    cfg2 = EngineConfig(pairs=[PAIR], start=dates[100], end=dates[300],
                        spot_tc=0.0001, verbose=False)
    res2 = run(RollingStructure(PAIR, legs, '1M', roll_days=5), ds, cfg2)
    peak = int(res2.daily.n_open.max())
    print(f'      {len(res2.trades)} legs opened, peak concurrent {peak}, '
          f'spot TC {res2.daily.hedge_tc.sum():,.2f}')
    check('multiple vintages coexist', peak > 2, f'peak {peak} legs')
    check('multiple distinct expiries held at once',
          res2.positions.groupby('date').expiry.nunique().max() > 1)

    # gross vs net delta on the busiest day
    busiest = res2.positions.groupby('date').size().idxmax()
    day = res2.positions[res2.positions.date == busiest]
    gross, net = day.delta_hedge.abs().sum(), abs(day.delta_hedge.sum())
    print(f'      on {busiest}: gross |delta| {gross:,.0f} vs net {net:,.0f} '
          f'-> {(1-net/gross)*100:.1f}% less to trade')
    check('netting genuinely reduces the delta traded', net < gross)

    # ------------------------------------------------------------------ #
    print('\n--- 5. The premium buckets and the ranking metric ------------')
    for c in ('gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be'):
        print(f'      {c:<14} cumulative {res2.daily[c + "_cum"].iloc[-1]:>13,.2f}')
    check('breakeven buckets are populated (nu/rho wired through)',
          res2.daily.volga_pnl_be.abs().sum() > 0)

    v = res2.pnl_per_unit_greek('volga_pnl_be', 'volga_1vp')
    n_ = res2.pnl_per_unit_greek('vanna_pnl_be', 'vanna_1pct_1vp')
    print(f'      volga_be per unit |volga| carried  {v:>10.4f}')
    print(f'      vanna_be per unit |vanna| carried  {n_:>10.4f}')
    check('pnl_per_unit_greek returns a finite number',
          np.isfinite(v) and np.isfinite(n_))

    # ------------------------------------------------------------------ #
    print('\n--- 6. Sleeve attribution is a groupby -----------------------')
    bys = res2.by_sleeve('volga_pnl_be')
    print(f'      sleeves present: {list(bys.columns)}')
    check('by_sleeve returns a cumulative frame', len(bys) == len(res2.daily))

    # ------------------------------------------------------------------ #
    print('\n--- 7. Gap risk: the jump day shows up in recon_resid --------')
    worst = res2.positions.reindex(
        res2.positions.recon_resid.abs().sort_values(ascending=False).index).head(3)
    print(worst[['date', 'tag', 'dt_days', 'dS', 'dsigma',
                 'option_pnl', 'gamma_pnl', 'recon_resid']].to_string(index=False))
    print('      Large |recon_resid| = a discrete jump or a Taylor breakdown.')
    print('      For a short-wings book those ARE the days that hurt; track the')
    print('      distribution of this column as a risk metric, not just a check.')

    # ------------------------------------------------------------------ #
    print('\n--- 8. Phase 2: option transaction costs ---------------------')
    from book.costs import OptionCostModel
    from run.breakeven_study import sweep, breakeven_scale

    cm = OptionCostModel()
    print(cm.describe(PAIR))

    atm_1m = cm.spread_vp(PAIR, 30, 0.50)
    w10_1m = cm.spread_vp(PAIR, 30, 0.10)
    atm_1w = cm.spread_vp(PAIR, 7, 0.50)
    check('wings cost more than ATM', w10_1m > atm_1m,
          f'{w10_1m:.3f} vs {atm_1m:.3f} vp')
    check('short tenors cost more than long', atm_1w > atm_1m,
          f'1W {atm_1w:.3f} vs 1M {atm_1m:.3f} vp')

    legs2 = [LegRequest('call', -1, 10_000_000, '1M', target_delta=+0.25,
                        sleeve='wings', tag='wing_c'),
             LegRequest('put', -1, 10_000_000, '1M', target_delta=-0.25,
                        sleeve='wings', tag='wing_p')]
    nets, tcs = [], []
    for sc in (0.0, 1.0, 2.0):
        c = EngineConfig(pairs=[PAIR], start=dates[100], end=dates[200],
                         cost_model=OptionCostModel(scale=sc), verbose=False)
        r = run(RollingStructure(PAIR, legs2, '1M', roll_days=5), ds, c)
        nets.append(r.daily.pnl.sum())
        tcs.append(r.daily.option_tc.sum())
        print(f'      scale {sc:>3.1f}  option TC {tcs[-1]:>12,.0f}   '
              f'net P&L {nets[-1]:>13,.0f}')
    check('scale=0 charges nothing (Phase 1 reproduced)', tcs[0] == 0.0)
    check('cost rises with scale', 0 < tcs[1] < tcs[2])
    check('cost reduces net P&L one-for-one',
          abs((nets[0] - nets[1]) - tcs[1]) < 1e-6,
          f'dPnL {nets[0] - nets[1]:,.2f} vs TC {tcs[1]:,.2f}')

    c = EngineConfig(pairs=[PAIR], start=dates[100], end=dates[140],
                     cost_model=OptionCostModel(scale=1.0), verbose=False)
    r = run(HoldStatic(PAIR, legs2, '1M'), ds, c)
    reasons = set(r.costs.reason) if len(r.costs) else set()
    check('hold-to-expiry pays entry cost only, expiry settles free',
          reasons == {'open'}, f'cost reasons: {sorted(reasons)}')

    sw = sweep(PAIR, tenor='1M', wing=0.25, roll_days=5,
               start=dates[100], end=dates[300], dataset=ds,
               scales=(0.0, 1.0, 2.0, 4.0), verbose=False)
    be = breakeven_scale(sw)
    print(f'      break-even cost scale on synthetic data: {be}')
    check('breakeven_scale returns a usable value',
          np.isnan(be) or np.isinf(be) or be > 0)
    print('      (meaningless on synthetic data -- this only proves the')
    print('       machinery runs. The real number comes from Bloomberg.)')

    print('\n' + '=' * 72)
    print('SMOKE TEST ' + ('PASSED' if ok else 'FAILED'))
    print('=' * 72)
    if ok:
        print('The plumbing is sound. This does NOT validate the SABR surface,')
        print('the pillar arithmetic, or any P&L number -- only')
        print('recon/reconcile.py (vs the old engine) and')
        print('run/breakeven_study.py (vs real data) can do that.')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
