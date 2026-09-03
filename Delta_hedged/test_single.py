import pandas as pd

from backtest_MLeg import (LegSpec, run_backtest_multi_leg, print_pnl, print_exposures,

    build_straddle, build_strangle, build_risk_reversal, build_call_spread, build_put_spread,
    build_vega_neutral_butterfly, vega_neutral_butterfly_factory,

    run_backtest_straddle, run_backtest_strangle, run_backtest_risk_reversal,
    run_backtest_call_spread, run_backtest_put_spread,
    run_backtest_vega_neutral_butterfly

)

from exit_hedge_logic import (
    HoldToExpiry, ExitAfterNDays, ExitAtDaysRemaining, TakeProfitStopLoss,
    DailyHedge, DeltaBandHedge, SpotMoveHedge, GammaScaledHedge, PartialHedge,
)



# ═══════════════════════════════════════════════════════════════════════════════
# Single Structure Backtest 
# ═══════════════════════════════════════════════════════════════════════════════




# pair        =  'USDJPY'
# tenor       =  '1M'
# # call_delta  =  0.25
# # put_delta   =  0.25

# notional    =  50_000_000
# direction   =  -1
# verbose     =  True
# entry_days_back=9


# legs = [
#     LegSpec('put',    -0.10, +1, 50_000_000),
#     LegSpec('put',    -0.25, -1, 50_000_000),
#     LegSpec('call',   +0.25, -1, 50_000_000),   
#     LegSpec('call',   +0.10, +1, 50_000_000),   
# ]


# # legs = build_strangle(call_delta, put_delta, notional, direction)

# exit_rule   = HoldToExpiry()
# hedge_rule  =  DailyHedge() # DeltaBandHedge(0.05) 

# df_agg, leg_dfs, summary, leg_summaries = run_backtest_multi_leg(
#         legs,
#         pair=pair,
#         tenor=tenor,
#         entry_days_back=entry_days_back,
#         verbose=verbose,
#         hedge_rule=hedge_rule,
#         exit_rule=exit_rule,
#         )

# print_pnl(df_agg, leg_summaries=leg_summaries, summary=summary)
# print_exposures(df_agg, leg_summaries=leg_summaries, summary=summary)



# ═══════════════════════════════════════════════════════════════════════════════
# Signal-driven backtest 
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys
from backtest_signal import run_signal_backtest, print_trade_log
from reporting import (build_daily_book, print_daily_book,
                       build_pnl_book, build_exposure_book, to_usd_book,
                       evaluate, print_scorecard, plot_scorecard, plot_full_report
)


from Signal_Gen.Implied_Realized import (
    get_date_signal,
    get_always_on_signal,
    get_cross_vol_signal,
    get_vol_signal,
)

from Signal_Gen.xCCY_Spread import (
    get_xccy_spread_signal,
)



# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------



# Option Params

pair = 'USDJPY'
tenor = '1M'
days_back = int(365 * 2)    

# call_delta = 0.25
# put_delta = 0.25
# notional = 10_000_000
direction = -1

hedge_rule = DailyHedge()
exit_rule = HoldToExpiry() #ExitAfterNDays('1W')  # HoldToExpiry()   #ExitAfterNDays(1)



# signal_legs = build_strangle(call_delta, put_delta, notional, direction)

signal_legs = [
    LegSpec('put',    -0.25, -1, 40_000_000), 
    # LegSpec('put',    -0.50, +1, 5_000_000), 
    # LegSpec('call',   +0.50, +1, 5_000_000),  
    LegSpec('call',   +0.25, -1, 40_000_000),   
]



legs_factory = None


signal_series = get_date_signal(['1May26', '31Jul26', '24Jan26', '1Aug26', '20Mar26', '28Jan26', '27Jan26', '11Feb26', '7May26'])
# signal_series = get_always_on_signal(days_back, pair=pair, tenor=tenor)

trade_log, trade_dfs, trade_leg_sums = run_signal_backtest(
    signal_series,
    legs=signal_legs,
    legs_factory=legs_factory,
    pair=pair,
    tenor=tenor,
    hedge_rule_factory=lambda: hedge_rule,
    exit_rule_factory=lambda: exit_rule,
    max_concurrent=1,
    history_days= days_back,
    verbose=False,
    progress=True,          # per-step tracking: pull time, loop heartbeat, summary
)

# print_trade_log(trade_log)

plot_full_report(trade_dfs=trade_dfs, trade_log=trade_log, pair=pair,
    # title=f'{pair}: Sell 1M 25D Strangle in 10 per',
    show=True)




sc = evaluate(trade_dfs, trade_log, pair=pair, to_usd=True, settled_only=True)
print_scorecard(sc)




"""--------------- Fixed-delta legs (no market data needed) ---------------"""
# signal_legs = build_strangle(call_delta, put_delta, notional, direction)
# legs_factory = None


"""------ Vega-neutral butterfly (sized from entry market snapshot) -------"""
# signal_legs = None
# legs_factory = vega_neutral_butterfly_factory(
#     wing_delta=0.10, wing_notional=30_000_000, direction=-1,
#     tenor=tenor, pair=pair)







"""-------------- Specific Date Entry/Entries-----------------"""
# signal_series = get_date_signal('24Jun26')



"""--------------------- Always on ---------------------------"""
# signal_series = get_always_on_signal(days_back)




"""-------------------Implied v Realized-----------------------"""
# _, _, signal_series = get_cross_vol_signal(
#     ccys=[pair], 
#     leg_a=('V', '1M'), 
#     leg_b=('H', '1W'),
#     days_back= days_back,
#     sell_pct=80, side='sell',
# )



"""---------------Pair v xCCY Vol basket spread----------------"""
# ccys = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
#         'AUDUSD', 'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK']
# _, _, signal_series = get_xccy_spread_signal(
#     ccy_int=pair, 
#     tenor_int=tenor, ccys=ccys,
#     days_back=days_back,
#     sell_pct=80, side='sell')



# trade_log, trade_dfs, trade_leg_sums = run_signal_backtest(
#     signal_series,
#     legs=signal_legs,
#     legs_factory=legs_factory,
#     pair=pair,
#     tenor=tenor,
#     hedge_rule_factory=lambda: hedge_rule,
#     exit_rule_factory=lambda: exit_rule,
#     max_concurrent=None,
#     history_days= days_back,
#     verbose=False,
# )


# --------------------------------------------------------



# sc = evaluate(trade_dfs, trade_log, pair=pair, name='')
# print_scorecard(sc, title='USDJPY Daily: Selling 1M 25D Strangle in 10 per')



# --------------------------------------------------------



# print(trade_dfs)


# One-screen report: scorecard metrics + all plots on a single figure to send.
# show=True opens a window; pass save_path='report.png' to write a shareable image.
# plot_full_report(trade_dfs=trade_dfs, trade_log=trade_log, pair=pair,
#     title=f'{pair}: Selling 1M 25D Strangle in 10 per',
#     show=True)













# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# print(trade_log)

# book_base = build_daily_book(trade_dfs, trade_log) 

# print_daily_book(book_base)

# pnl_df      = build_pnl_book(book=book_base)
# exposure_df = build_exposure_book(book=book_base)

# print(pnl_df)
# print(exposure_df)


# # one call: build book -> full scorecard








# ==============================================================================================
# ==============================================================================================
# ==============================================================================================










# ═══════════════════════════════════════════════════════════════════════════════
# Multi-ccy / multi-tenor grid evaluation
#   Runs the SAME strategy (signal + legs + hedge/exit rules) across every
#   (pair, tenor) combo, collects comparable metrics, and presents them as a
#   ranked table + a one-screen heatmap figure (a panel per metric).
# ═══════════════════════════════════════════════════════════════════════════════

# from grid_eval import ComboSpec, run_grid, print_grid, plot_grid_heatmaps


# # ---- Grid axes -------------------------------------------------------------
# grid_pairs  = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
# grid_tenors = ['1M']

# grid_days_back = 94
# grid_notional  = 10_000_000
# grid_direction = -1                      # -1 = short the structure (short vol)


# # ---- Strategy definition (identical for every cell of the grid) ------------
# # signal_fn / legs_fn receive THIS combo's spec, so a pair-aware signal is
# # explicit. `s` is the ComboSpec — read s.pair / s.tenor / s.days_back off it.

# def grid_signal_fn(s):
#     # Same "always on" signal used in the single-combo run above.
#     return get_always_on_signal(s.days_back)

# def grid_legs_fn(s):
#     # Fixed-structure legs (no market data needed at entry).
#     return build_straddle(notional=s.notional, direction=s.direction)


# # ---- Build one ComboSpec per (pair, tenor) ---------------------------------
# grid_specs = [
#     ComboSpec(
#         pair=p, tenor=t,
#         signal_fn=grid_signal_fn,
#         legs_fn=grid_legs_fn,
#         hedge_rule_factory=lambda: DailyHedge(),
#         exit_rule_factory=lambda: HoldToExpiry(),
#         days_back=grid_days_back,
#         notional=grid_notional,
#         direction=grid_direction,
#     )
#     for p in grid_pairs for t in grid_tenors
# ]


# # ---- Run the sweep + present ----------------------------------------------
# # run_grid never aborts on a bad combo — missing data / zero-trade cells come
# # back as NaN rows with a `status`. Money metrics are in USD (to_usd=True) so
# # they are comparable across pairs.

# grid = run_grid(grid_specs, to_usd=True, verbose=True)

# print_grid(grid, sort_by='sharpe', title='Short straddle, DailyHedge, HoldToExpiry')













# plot_grid_heatmaps(grid, title='Short straddle sweep', show=True, save_path='grid_report.png')










# Pivot any single metric into a ccy x tenor matrix on demand:
# print(grid['sharpe'].unstack('tenor'))


"""------ Pair-aware signal alternative (implied vs realized per pair) ------"""
# Swap grid_signal_fn for this to run a genuinely per-pair signal. The [2]
# picks the signal_series out of get_cross_vol_signal's 3-tuple return.
# def grid_signal_fn(s):
#     return get_cross_vol_signal(
#         ccys=[s.pair], leg_a=('V', s.tenor), leg_b=('H', '1W'),
#         days_back=s.days_back, sell_pct=80, side='sell')[2]


"""---- Vega-neutral butterfly alternative (sized from entry snapshot) -----"""
# Use legs_factory_fn instead of legs_fn when sizing needs the entry market
# snapshot. Pass exactly one of legs_fn / legs_factory_fn.
# def grid_legs_factory_fn(s):
#     return vega_neutral_butterfly_factory(
#         wing_delta=0.10, wing_notional=30_000_000, direction=s.direction,
#         tenor=s.tenor, pair=s.pair)
