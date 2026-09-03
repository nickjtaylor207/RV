from backtest_MLeg import (LegSpec, run_backtest_multi_leg, print_pnl, print_exposures,
    build_straddle, build_strangle, build_risk_reversal, build_call_spread, build_put_spread,
    build_vega_neutral_butterfly, vega_neutral_butterfly_factory)

from exit_hedge_logic import (
    HoldToExpiry, ExitAfterNDays, ExitAtDaysRemaining, TakeProfitStopLoss,
    DailyHedge, DeltaBandHedge, SpotMoveHedge, GammaScaledHedge, PartialHedge,)

from Signal_Gen.Implied_Realized import (
    get_date_signal,
    get_always_on_signal,
    get_cross_vol_signal,
    get_vol_signal,)

from Signal_Gen.xCCY_Spread import (get_xccy_spread_signal,
                                    plot_xccy_param_scan, pull_vol_panel)

from Signal_Gen.regime_filter import (
    GateSpec, NO_GATE, gated, enumerate_gate_specs, describe_gate_specs,
    list_checks, describe_checks, build_check_panel, build_gate,
    check_defaults, clear_regime_cache, regime_cache_info)

from grid_eval import (ComboSpec, run_grid, print_grid, plot_grid_heatmaps,
                       SIGNAL_TABLE_COLS, PHASE_SUMMARY_COLS,
                       summarize_phases, print_phase_summary)
from gate_sweep import (run_gate_sweep, print_gate_attribution,
                        print_gate_consistency)



import pandas as pd





# # -------------------------- Specifications for each trial --------------------------  








# def _gate_sweep(signal_fn, gate_specs, exit_rule_factory, title,
#                 days_back=None, direction=-1, show=False):
#     """
#     Straddle + one signal + one exit, scored across every gate in `gate_specs`.
#     Returns (grid, attribution). See note A above for why this is one backtest
#     per pair rather than one per gate.
#     """
#     db = gate_days_back if days_back is None else days_back
#     describe_gate_specs(gate_specs)
#     grid, attr = run_gate_sweep(
#         pairs=grid_pairs, tenors=grid_tenors,
#         signal_fn=signal_fn, gate_specs=gate_specs,
#         legs_fn=_straddle, direction=direction,
#         hedge_rule_factory=lambda: DailyHedge(),      # hedge stays DailyHedge
#         exit_rule_factory=exit_rule_factory,
#         days_back=db, notional=grid_notional,
#         to_usd=True, verbose=True)

#     print_grid(grid, sort_by='calmar', title=title)   # pair x gate, calmar-ranked
#     print_gate_attribution(attr, title=title)         # what each gate removed
#     print_gate_consistency(grid, metric='calmar', title=title)
#     if show:
#         plot_grid_heatmaps(grid, title=title, show=True)
#     return grid, attr



































# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


""" ------------------ Signal Builders ------------------ """
# Always Active
def _always_on(s):
    return get_always_on_signal(s.days_back, pair=s.pair, tenor=s.tenor)

# Implied v Realized
# Parameterized like _strangle: call it to get the fn(s) the grid wants.
# sell_pct is the SELECTIVITY dial, and sweeping it is the strongest test of a
# signal available — a real signal traces a monotone frontier (tighter threshold
# -> lower coverage -> better ret_on_prem), a spurious one peaks at one value.
def _ivrv(sell_pct=80):
    """Enter when implied (trade tenor) is rich vs 1W realized."""
    def sig(s):
        return get_cross_vol_signal(
            ccys=[s.pair], leg_a=('V', s.tenor), leg_b=('H', '1W'),
            days_back=s.days_back, sell_pct=sell_pct, side='sell')[2]
    sig.__name__ = f'ivrv{sell_pct:.0f}'
    return sig








# xCCY vol Signal
# def _xccy(sell_pct=80):
#     """Enter when the pair's ATM vol is rich vs a cross-ccy basket."""
#     basket = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
#     def sig(s):
#         return get_xccy_spread_signal(
#             ccy_int=s.pair, tenor_int=s.tenor, 
#             ccys=basket,
#             days_back=s.days_back, 
#             sell_pct=sell_pct, 
#             side='sell')[2]
#     sig.__name__ = f'xccy{sell_pct:.0f}'
#     return sig




"""     

----------------- xCCY SIGNAL -----------------     


""" 


def _xccy(sell_pct=80, pct_lookback=365, basket=None,):
    if basket is None:
        basket = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
    def sig(s):
        return get_xccy_spread_signal(
            ccy_int=s.pair, tenor_int=s.tenor, ccys=basket,
            pct_lookback=pct_lookback,
            days_back=s.days_back, sell_pct=sell_pct,
            side='sell', )[2]
    sig.__name__ = f'xccy{sell_pct:.0f}'
    return sig








"""-----------------------------------------------------------------------"""










""" ------------------ Constant Leg Structures ------------------ """
def _straddle(s):
    return build_straddle(s.notional, s.direction)

def _strangle(call_delta=0.25, put_delta=0.25):
    def legs(s):
        return build_strangle(call_delta, put_delta, s.notional, s.direction)
    return legs

def _risk_reversal(call_delta=0.25, put_delta=0.25):
    def legs(s):
        return build_risk_reversal(call_delta, put_delta, s.notional, s.direction)
    return legs

def _Strangle_TouchATM(wing_delta=0.25, wing_notional=20_000_000, body_ratio=0.25):
    def legs(s):
        body = wing_notional * body_ratio
        return [
            LegSpec('put',  -wing_delta, s.direction,  wing_notional),
            LegSpec('put',  -0.50,      -s.direction,  body),
            LegSpec('call', +0.50,      -s.direction,  body),
            LegSpec('call', +wing_delta, s.direction,  wing_notional),
        ]
    return legs












# ------------ Time Start Altering ------------
PHASE_OFFSETS = (0, 12, 102, 153, 192)
def RUN_PHASES(title, *,
               days_back_list = None,   # explicit windows; overrides offsets
               offsets        = None,   # None -> PHASE_OFFSETS, off DAYS_BACK
               cols           = None,   # None -> PHASE_SUMMARY_COLS
               show_runs      = False,  # True -> also print each phase's own grid
               detail         = False,  # True -> stat-per-row instead of mean [std]
               note           = False,  # True -> print how to read the spread
               **kw):
    assert 'days_back' not in kw, \
        "days_back is the dial RUN_PHASES sweeps — pass days_back_list=[...] " \
        "for explicit windows, or offsets=(...) to derive them off DAYS_BACK"
    if days_back_list is None:
        days_back_list = [DAYS_BACK - o for o in
                          (PHASE_OFFSETS if offsets is None else offsets)]
    days_back_list = [int(d) for d in dict.fromkeys(days_back_list)]  # dedupe
    assert len(days_back_list) >= 2, \
        "a phase sweep needs at least 2 windows — one window has no spread"

    grids = {}
    for i, db in enumerate(days_back_list, 1):
        print(f"\n[phase {i}/{len(days_back_list)}] days_back={db}", flush=True)
        grids[f'db={db}'] = RUN(f'{title} | days_back={db}', days_back=db,
                                print_table=show_runs, **kw)

    summary = summarize_phases(grids, cols=cols)
    print_phase_summary(summary, title=title, detail=detail, note=note)
    return summary, grids


# summary, phase_grids = RUN_PHASES('short 1M 25d strangle | always-on | Max 1 Trade',
#     max_concurrent=1,
#     exit_rule_factory= lambda: ExitAfterNDays('1M'),
#     )






# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------









def RUN(title, *,
        signal_fn         = _always_on,
        signals           = None,                          # None = single signal
        gates             = None,                          # None = no gate sweep
        legs_fn           = _strangle(0.25, 0.25),
        legs_factory_fn   = None,
        direction         = -1,
        exit_rule_factory = lambda: HoldToExpiry(),
        hedge_rule_factory= lambda: DailyHedge(),
        pairs             = None,
        tenors            = None,
        days_back         = None,
        notional          = 10_000_000,
        tc_fraction       = 0.0001,      #0.0001,
        max_concurrent    = None,      # None=unlimited stacking, 1=non-overlapping, N=cap. Gate sweeps ignore this (see below).
        sort_by           = None,      # None -> 'ret_on_prem' for signals, else 'calmar'
        ascending         = None,      # None -> A->Z on a key column, best-first on a metric
        cols              = None,      # None -> SIGNAL_TABLE_COLS on every route
        print_table       = True,      # False -> return the grid silently (RUN_PHASES)


        show              = False):
    """
    Every dial of a backtest in one call.
      signals=None, gates=None -> one plain grid run          (pair x tenor)
      signals={...}            -> one backtest per cell       (pair x signal)
      gates=[...]              -> ONE backtest per pair, every gate scored off it
                                  (pair x gate)
    signals and gates are separate sweeps and cannot be combined: signals go
    through run_grid (so max_concurrent is honoured), gates go through
    run_gate_sweep (which hardcodes unlimited stacking). To gate a signal on the
    signals= route, wrap it yourself: gated(_ivrv(80), GateSpec(('har',))).
    """
    P = pairs  or grid_pairs
    T = tenors or grid_tenors
    days_back = DAYS_BACK if days_back is None else days_back
    if legs_factory_fn is not None:
        legs_fn = None                      # exactly one of the two is allowed
    if ascending is None:
        ascending = sort_by in ('pair', 'tenor', 'label')
    cols = cols or SIGNAL_TABLE_COLS
    if signals is not None:
        assert gates is None, \
            "signals= and gates= are separate sweeps — run them one at a time"
        assert 'none' in signals, \
            "signals= needs an unfiltered baseline keyed 'none' (e.g. " \
            "{'none': _always_on, 'ivrv80': _ivrv(80)}). Every column is scored " \
            "as a delta against it, and without one there is nothing to compare."
        specs = [ComboSpec(pair=p, tenor=t,
                           signal_fn=fn,
                           label=lbl,               # <- becomes the column axis
                           legs_fn=legs_fn,
                           legs_factory_fn=legs_factory_fn,
                           hedge_rule_factory=hedge_rule_factory,
                           exit_rule_factory=exit_rule_factory,
                           days_back=days_back,
                           notional=notional,
                           direction=direction,
                           tc_fraction=tc_fraction,
                           max_concurrent=max_concurrent)
                 for p in P for t in T for lbl, fn in signals.items()]
        grid = run_grid(specs, to_usd=True, verbose=True)
        if print_table:
            print_grid(grid, sort_by=sort_by or 'ret_on_prem', ascending=ascending,
                       cols=cols, title=title)
            print_gate_consistency(grid, metric='ret_on_prem', title=title,
                                   label_header='signal')
        if show:
            plot_grid_heatmaps(grid, title=title, show=True)
        return grid
    if gates is None:
        specs = [ComboSpec(pair=p, tenor=t,
                           signal_fn=signal_fn,
                           legs_fn=legs_fn,
                           legs_factory_fn=legs_factory_fn,
                           hedge_rule_factory=hedge_rule_factory,
                           exit_rule_factory=exit_rule_factory,
                           days_back=days_back,
                           notional=notional,
                           direction=direction,
                           tc_fraction=tc_fraction,
                           max_concurrent=max_concurrent)
                 for p in P for t in T]
        grid = run_grid(specs, to_usd=True, verbose=True)
        if print_table:
            print_grid(grid, sort_by=sort_by or 'calmar', ascending=ascending,
                       cols=cols, title=title)
        if show:
            plot_grid_heatmaps(grid, title=title, show=True)
        return grid
    assert max_concurrent is None, \
        "max_concurrent is not supported with gates= — run_gate_sweep hardcodes " \
        "unlimited stacking because its veto-attribution shortcut requires it " \
        "(see gate_sweep.py module docstring)"
    grid, attr = run_gate_sweep(
        pairs=P, tenors=T, 
        signal_fn=signal_fn, 
        gate_specs=gates,
        legs_fn=legs_fn, 
        legs_factory_fn=legs_factory_fn,
        hedge_rule_factory=hedge_rule_factory,
        exit_rule_factory=exit_rule_factory,
        days_back=days_back, 
        notional=notional, 
        direction=direction,
        tc_fraction=tc_fraction, 
        to_usd=True,
        verbose=True)
    if print_table:
        print_grid(grid, sort_by=sort_by or 'calmar', ascending=ascending,
                   cols=cols, title=title)
        print_gate_attribution(attr, title=title)
        print_gate_consistency(grid, metric='calmar', title=title)
    if show:
        plot_grid_heatmaps(grid, title=title, show=True)
    return grid, attr








# grid_pairs  = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
# # grid_tenors = ['1W', '1M', '3M']




# grid_notional  = 10_000_000
# grid_tenors = ['2M']    


grid_pairs  = ['USDCAD']

grid_direction = -1
DAYS_BACK = (365 * 5)


RUN('trend gate on always-on vs xccy | G7 1M 50D Straddle in 20 ',
    tenors=['1M'],
    legs_fn=_Strangle_TouchATM(),
    exit_rule_factory= lambda: ExitAtDaysRemaining('1W'),
    # max_concurrent=5,

    signals={
        # always-on family — 'none' is its ungated reference
        'none':          _always_on,

    },

    sort_by='pair')














# ──────────────────────────────────────────────────────────────────────────────
# ────────────────────────── BASELINE ──────────────────────────────────────────

# g = RUN('base | short 1M 25d strangle | always-on | Hold 1W | Max 1 Trade', 
#         max_concurrent=1, 
#         exit_rule_factory= lambda: ExitAfterNDays('1M'))

# g = RUN('base | short 1M 25d strangle | always-on | Hold Until 1M | Max 1 Trade', 
#         max_concurrent=1, 
#         exit_rule_factory= lambda: ExitAtDaysRemaining('1W'))

# g = RUN('base | short 1M 25d strangle | always-on | Hold Till Expiry | Max 1 Trade', 
#         max_concurrent=1)

# -------------------------------------------------------------------------------


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────




# g = RUN('signals | short 1M 25d strangle | Hold to Expiry | Max 1 Trade',
#         tenors=['3M'],
#         legs_fn=_strangle(0.25,0.25),
#         exit_rule_factory= lambda: ExitAfterNDays('1M'),

#         max_concurrent=1,
#         signals={'none':   _always_on,
#                  'xccy70': _xccy(70),   
#                  'xccy80': _xccy(80),
#                  'xccy90': _xccy(90),
#                  },
#         sort_by='pair')






# ─────────────────────────────────────────────────────────────────────────────
# ───────────────────────── Implied v Realized ────────────────────────────────





















# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# ───────────────────────────── GATE INSPECTION ────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────


"""
1) ---------------------- Initial Inspection ----------------------
"""

# from Signal_Gen.regime_filter import (build_check_panel, regime_cache_info, clear_regime_cache)

# panel = build_check_panel('EURUSD', tenor='1M', days_back=DAYS_BACK, verbose=True)
# print(panel.mean().sort_values())        # fraction of days each check ALLOWS
# print(regime_cache_info())











# gate, detail = build_gate(
#     GateSpec(('trend',), ),
#     'EURUSD',
#     '1M',
#     DAYS_BACK,
#     return_detail=True,
#     verbose=True)

# vetoed = detail[detail['GATE'] == 0].copy()
# vetoed.index = pd.to_datetime(vetoed.index)

# print(f"{len(vetoed)} vetoed days")
# print(vetoed.index.to_period('M').value_counts().sort_index())



# for w in (5, 10, 20, 40, 60, 90):
#     pan = build_check_panel('EURUSD', tenor='1M', days_back=DAYS_BACK,
#                             checks=['trend'], params={'trend': {'ma_window': w}})
#     print(f"ma_window={w:>3}  allows {pan['trend'].mean():6.1%}")






"""
3) ---------------------- Variable Flexing ----------------------
"""

# ------------- TREND -------------
# g, a = RUN('gates | trend ma_window frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [
#                GateSpec(('trend',), params={'trend': {'ma_window': w}}, name=f'trend@{w}d')
#                for w in (5, 10, 20)],
#            cols=SIGNAL_TABLE_COLS)



# ------------- MOMENTUM -------------
# g, a = RUN('gates | momentum change_days frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [
#                GateSpec(('momentum',), params={'momentum': {'change_days': d}}, name=f'mom@{d}d')
#                for d in (1, 3, 5, 10, 20)],
#            cols=SIGNAL_TABLE_COLS)




# ------------- SPIKE -------------
# g, a = RUN('gates | spike z_thresh frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [
#                GateSpec(('spike',), params={'spike': {'z_thresh': z}}, name=f'spk z{z:.1f}')
#                for z in (1.0, 1.5, 2.0, 2.5, 3.0)],
#            cols=SIGNAL_TABLE_COLS)

# g, a = RUN('gates | spike cooloff frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [
#                GateSpec(('spike',), params={'spike': {'cooloff': c}}, name=f'spk c{c}d')
#                for c in (0, 3, 5, 10, 20)],
#            cols=SIGNAL_TABLE_COLS)



# ------------- LEVEL -------------
# g, a = RUN('gates | level lookback frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [
#                GateSpec(('level',), params={'level': {'lookback': lb}}, name=f'lvl lb{lb}')
#                for lb in (63, 126, 252, 504)],
#            cols=SIGNAL_TABLE_COLS)




# ------------- HAR -------------

# grid_pairs  = ['EURUSD']
# grid_tenors = ['2M'] 

# DAYS_BACK = (365 * 5)

# g, a = RUN('gates | har rising_ratio frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [GateSpec(('har',), params={'har': {'rising_ratio': r}}, name=f'har r{r:.2f}') for r in (0.90, 0.95, 1.00, 1.05, 1.10)],
#            cols=SIGNAL_TABLE_COLS)




# g, a = RUN('gates | har rising_ratio frontier | G7 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),

#            gates=[NO_GATE] + [
#                GateSpec(('har',), params={'har': {'rising_ratio': r}}, name=f'har r{r:.2f}') for r in (0.90, 0.95, 1.00, 1.05, 1.10)],

#            cols=SIGNAL_TABLE_COLS)










# g, a = RUN('gates | har train_window frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [
#                GateSpec(('har',), params={'har': {'train_window': tw}}, name=f'har tw{tw}')
#                for tw in (126, 252, 504)],
#            cols=SIGNAL_TABLE_COLS)

# g, a = RUN('gates | har bbg_tenors frontier | short 1M 25d strangle',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=[NO_GATE] + [
#                GateSpec(('har',), params={'har': {'bbg_tenors': bt}},
#                         name=f'har {"/".join(bt)}')
#                for bt in (('24H','1W','1M'), ('1W','1M','3M'), ('1W','1M','6M'))],
#            cols=SIGNAL_TABLE_COLS)









"""
4) ---------------------- Variable Flexing w/ max concurrent ----------------------
"""


# grid_pairs  = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
# # grid_tenors = ['1W']

# grid_notional  = 10_000_000
# grid_direction = -1

# DAYS_BACK = (365 * 5)



# RUN('trend gate on always-on vs xccy | G7 1M 50D Straddle in 20 ',
#     tenors=['1M'],
#     legs_fn=_straddle,
#     # exit_rule_factory= lambda: ExitAtDaysRemaining('2W'),
#     # max_concurrent=5,

#     signals={
#         # always-on family — 'none' is its ungated reference
#         'none':          _always_on,
#         'ao_tr5':        gated(_always_on, GateSpec(('trend',), params={'trend': {'ma_window': 5}})),
#         'ao_tr10':        gated(_always_on, GateSpec(('trend',), params={'trend': {'ma_window': 10}})),
#         'ao_tr15':        gated(_always_on, GateSpec(('trend',), params={'trend': {'ma_window': 15}})),
#         # 'ao_tr10':        gated(_always_on, GateSpec(('trend',), params={'trend': {'ma_window': 10}})),
#         # 'ao_tr15':        gated(_always_on, GateSpec(('trend',), params={'trend': {'ma_window': 15}})),
#         # 'ao_tr20':        gated(_always_on, GateSpec(('trend',), params={'trend': {'ma_window': 20}})),

#         # 'xccy80':      _xccy(80),
#         # 'xccy80_tr5':  gated(_xccy(80), GateSpec(('trend',), params={'trend': {'ma_window': 5}})),

#     },

#     sort_by='pair')




# ------------------------------------------------------------------------------



# grid_pairs  = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD']
# # grid_tenors = ['1W']


# grid_notional  = 10_000_000
# grid_direction = -1

# DAYS_BACK = (365 * 7)





# RUN('trend vs har gate on always-on | G6 1M 25d strangle | 1W left | Max 1 Trade',
#     tenors=['1M'],
#     legs_fn=_strangle(0.25, 0.25),

#     # lambda: is NOT optional — run_signal_backtest calls this once per trade for
#     # a FRESH rule. Passing the instance raises TypeError on every cell.
#     exit_rule_factory=lambda: ExitAtDaysRemaining('1W'),

#     max_concurrent=1,

#     signals={
#         # 'none' is the ungated reference every other column is deltaed against.
#         # Keys must be DISTINCT — a duplicate silently overwrites, it does not error.
#         'none':        _always_on,
#         'ao_tr5':      gated(_always_on, GateSpec(('trend',), params={'trend': {'ma_window': 5}})),
#         'ao_har':      gated(_always_on, GateSpec(('har',), params={'har': {'bbg_tenors': ('1W','1M','2M')}})),


#         'xccy80':      _xccy(80),
#         'xccy80_tr5':  gated(_xccy(80), GateSpec(('trend',), params={'trend': {'ma_window': 5}})),
#         'xccy80_har':  gated(_xccy(80), GateSpec(('har',), params={'har': {'bbg_tenors': ('1W','1M','2M')}}))



#     },

#     sort_by='pair')



# GateSpec(('har',), params={'har': {'bbg_tenors': bt}}, name=f'har {"/".join(bt)}') for bt in 
#             (('24H','1W','1M'), ('1W','1M','3M'), ('1W','1M','6M'))





















# g, a = RUN('gates | short 1M 25d strangle | Hold to Expiry | singles',
#            tenors=['1M'],
#            legs_fn=_strangle(0.25, 0.25),
#            gates=enumerate_gate_specs(sizes=(1,)),      # none + all 6
#            cols=SIGNAL_TABLE_COLS)










# for db in (365*5, 365*5 - 12, 365*5 - 102, 365*5 - 153, 365*5 - 192):
#     RUN(f'FINAL phase check | days_back={db}',
#         tenors=['1M'], legs_fn=_strangle(0.25, 0.25),
#         max_concurrent=1, days_back=db,
#         signals={'none':  _always_on,
#                  'trend': gated(_always_on, GateSpec(('trend',)))},
#         sort_by='pair')


























# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


















# # Change the signal

# g = RUN('xccy',  signal_fn=_sig_xccy)      # vol rich vs basket
# g = RUN('ivrv',  signal_fn=_sig_ivrv)      # implied rich vs 1W realized
# g = RUN('ivpct', signal_fn=_sig_iv_rich)   # own IV in top percentile





# # Change the exit

# g = RUN('exit +1W',   exit_rule_factory=lambda: ExitAfterNDays('1W'))
# g = RUN('1W left',    exit_rule_factory=lambda: ExitAtDaysRemaining('1W'))
# g = RUN('exit +10d',  exit_rule_factory=lambda: ExitAfterNDays(10))       # int = calendar days
# g = RUN('SL 150k',    exit_rule_factory=lambda: TakeProfitStopLoss(stop_loss=150_000))
# g = RUN('TP/SL',      exit_rule_factory=lambda: TakeProfitStopLoss(take_profit=100_000,
#                                                                    stop_loss=150_000))
# # The lambda: is not optional — backtest_signal.py:53-56 needs a fresh rule per trade, because stateful rules like SpotMoveHedge would otherwise leak state between trades.

# # Change the hedge

# g = RUN('2% band',  hedge_rule_factory=lambda: DeltaBandHedge(0.02))    # rehedge above 2% of notional
# g = RUN('25bp',     hedge_rule_factory=lambda: SpotMoveHedge(0.0025))   # rehedge on 25bp spot move
# g = RUN('half',     hedge_rule_factory=lambda: PartialHedge(0.5))       # always hedge 50% of the gap
# g = RUN('gamma',    hedge_rule_factory=lambda: GammaScaledHedge(
#                         [(0.0020, 1), (0.0010, 2), (0.0, 5)]))           # daily >0.20%, else 2d, else 5d




# # Change the structure

# g = RUN('25d strangle', legs_fn=_strangle(0.25, 0.25))   # note the parens — call it
# g = RUN('10d strangle', legs_fn=_strangle(0.10, 0.10))
# g = RUN('broken wing',  legs_fn=_strangle(0.25, 0.10))   # asymmetric
# g = RUN('25d RR',       legs_fn=_risk_reversal(0.25, 0.25))
# g = RUN('condor',       legs_fn=_custom)                 # your 4-leg
# g = RUN('vn fly',       legs_factory_fn=_vn_fly(0.25))   # needs entry data -> factory
# # _straddle and _custom go in bare (they are fn(s)); _strangle(...) and _risk_reversal(...) get called first because they're parameterized and return the fn(s).




# # Flip direction, resize, change costs, change universe

# g = RUN('LONG straddle', direction=+1)
# g = RUN('50m',           notional=50_000_000)
# g = RUN('5bp costs',     tc_fraction=0.0005)
# g = RUN('G3 only',       pairs=['EURUSD', 'USDJPY', 'GBPUSD'])
# g = RUN('3M tenor',      tenors=['3M'])
# g = RUN('term struct',   tenors=['1W', '1M', '3M'])
# # One trap on tenors: ExitAtDaysRemaining('1W') asserts the threshold is shorter than the trade's own tenor, so it will fail on tenors=['1W'].

# # Add gates
# # This is where the return value changes to (grid, attr):


# # one gate, against its baseline — ALWAYS include NO_GATE or you have nothing to compare
# g, a = RUN('HAR gate', gates=[NO_GATE, GateSpec(('har',))])

# # every registered check on its own
# g, a = RUN('all singles', gates=enumerate_gate_specs(sizes=(1,)))

# # singles + every 2-check AND combination
# g, a = RUN('singles+pairs', gates=enumerate_gate_specs(sizes=(1, 2)))

# # hand-picked, showing all three combine modes
# g, a = RUN('shortlist', gates=[
#     NO_GATE,
#     GateSpec(('trend',)),                                    # one check
#     GateSpec(('trend', 'har')),                              # AND: both must pass
#     GateSpec(('trend','level','spike'), combine='k', k=2),   # veto once 2 of 3 veto
#     GateSpec(('trend', 'momentum'), combine='any'),          # veto only if BOTH veto
# ])

# # retune a check's params, and read the regime off front vol instead of 1M
# g, a = RUN('tuned', gates=[
#     NO_GATE,
#     GateSpec(('level',), params={'level': {'cap_pct': 80}}, name='level@80'),
#     GateSpec(('har',),   params={'har': {'bbg_tenors': ('24H','1W','1M'),
#                                          'train_window': 180}}, name='har_short'),
#     GateSpec(('trend',), tenor='1W', name='trend@1W'),
# ])
# # Combine dials freely — that's the whole point

# # your earlier request, in one line
# g, a = RUN('short straddle | xccy | 1W left | HAR',
#            signal_fn=_sig_xccy,
#            exit_rule_factory=lambda: ExitAtDaysRemaining('1W'),
#            gates=[NO_GATE, GateSpec(('har',))])

# # 10d strangle, band-hedged, TP/SL, gated, G3 only
# g, a = RUN('strangle | band | TP-SL | gated',
#            signal_fn=_sig_ivrv,
#            legs_fn=_strangle(0.10, 0.10),
#            hedge_rule_factory=lambda: DeltaBandHedge(0.02),
#            exit_rule_factory=lambda: TakeProfitStopLoss(take_profit=80_000,
#                                                         stop_loss=120_000),
#            gates=[NO_GATE, GateSpec(('trend', 'har'))],
#            pairs=['EURUSD', 'USDJPY', 'GBPUSD'])






# # Step 4 — the one rule for choosing the route



# for name, ex in {'hte':   lambda: HoldToExpiry(),
#                  'rem1w': lambda: ExitAtDaysRemaining('1W'),
#                  'n1w':   lambda: ExitAfterNDays('1W')}.items():
#     RUN(f'exit={name}', exit_rule_factory=ex,
#         gates=[NO_GATE, GateSpec(('har',))], show=False)