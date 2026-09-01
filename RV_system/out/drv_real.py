"""Mirror of the test.py Example-2 block, driven through run/report.py.
Does NOT import test.py (that would execute its run block twice)."""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from market.dataset import FXVolDataset
from market.snapshot import business_dates
from book.costs import OptionCostModel
from strategy.sizer import GreekTarget
from strategy.roller import GreekTargetRoller
from engine.loop import EngineConfig, run
from run.report import report, chart_risk, chart_attribution
from run.dashboard import dashboard

class Composite:
    def __init__(self, *s): self.s = s
    def on_date(self, ctx):
        for x in self.s: x.on_date(ctx)

class MinLifeExit:
    def __init__(self, pair, sleeves, min_life_days=7, wake=()):
        self.pair=pair; self.sleeves=[sleeves] if isinstance(sleeves,str) else list(sleeves)
        self.min_life_days=min_life_days; self.wake=list(wake); self.log=[]
    def on_date(self, ctx):
        if self.min_life_days is None or self.pair not in ctx.snaps: return
        snap=ctx.snaps[self.pair]; hit=[]
        for sl in self.sleeves:
            for pos in list(ctx.book.open_positions(pair=self.pair, sleeve=sl)):
                if pos.days_to_expiry(snap.date) <= self.min_life_days:
                    ctx.book.close(pos.pos_id, snap.date, reason='min_life', snap=snap)
                    hit.append(pos)
        if not hit: return
        self.log.append({'date':snap.date,'closed':len(hit),
                         'notional':sum(p.notional for p in hit),
                         'sleeves':','.join(sorted({p.sleeve for p in hit})),
                         'min_life':min(p.days_to_expiry(snap.date) for p in hit)})
        for r in self.wake: r._last=None
    def frame(self): return pd.DataFrame(self.log)
    def report(self):
        if self.min_life_days is None: return "[min_life=off] holding every leg to expiry"
        if not self.log: return f"[min_life={self.min_life_days}d] no early exits fired"
        df=self.frame()
        return (f"[min_life={self.min_life_days}d] {len(df)} exit dates, "
                f"{int(df['closed'].sum())} legs closed early, "
                f"{df['notional'].sum():,.0f} gross unwound")

# ---- exactly the test.py config -----------------------------------------
EX_PAIR, EX_TENOR, EX_DAYS, EX_WINDOW, EX_HORIZON = 'USDCAD', '1M', 400, 200, 7
EX_MENU  = [10, 'ATM']
BUDGET   = -30_000.0
ex_cm    = OptionCostModel(scale=0.0)
ex_ds    = FXVolDataset.build(pairs=[EX_PAIR], days=EX_DAYS)
ex_dates = business_dates(ex_ds, EX_PAIR)

tgt_2 = GreekTarget(by_tenor={EX_TENOR: dict(volga=BUDGET, vega=0.0, vanna=0.0)},
                    horizon_days=EX_HORIZON, units='normalised')
roll_2 = GreekTargetRoller(EX_PAIR, EX_TENOR, lambda _s: tgt_2, roll_days=5,
                           sleeve='wing_convex', mode='top_up', min_trade=1_000_000,
                           solve_kw=dict(allow_deltas=EX_MENU, cost_model=ex_cm))
exit_2 = MinLifeExit(EX_PAIR, 'wing_convex', min_life_days=7, wake=[roll_2])
cfg = EngineConfig(pairs=[EX_PAIR], start=ex_dates[-EX_WINDOW], end=ex_dates[-5],
                   cost_model=ex_cm, hedge_fraction=1.0, spot_tc=0.0001, verbose=False)
res_2 = run(Composite(exit_2, roll_2), ex_ds, cfg)

d = report(res_2, ex_ds, EX_PAIR, EX_TENOR, EX_HORIZON,
           roller=roll_2, exiter=exit_2)
chart_risk(d, BUDGET, roller=roll_2, exiter=exit_2, path='out/real_risk.png',
           title='USDCAD 1M short wing convexity, exit at 1W  --  ')
chart_attribution(res_2, path='out/real_attrib.png',
                  title='USDCAD 1M short wing convexity, exit at 1W  --  ')
dashboard(res_2, ex_ds, EX_PAIR, EX_TENOR, EX_HORIZON, budget=BUDGET,
          roller=roll_2, exiter=exit_2, units='both', d=d,
          path='out/real_dash.png', title='exit at 1W  --  ')
print('\nREAL DRIVER OK')
