"""
Time-level (book-level) reporting .
  - FLOW columns (daily P&L increments) -> 0 off-trade, summed, cumsummed into equity
  - EXPO columns (point-in-time levels) -> 0 off-trade, summed as net book greeks
"""

from typing import List, Optional
import pandas as pd











# Daily P&L increments — fill 0 off-trade, SUM across trades, then cumsum for equity.
_FLOW_COLS = ['net_pnl', 'option_pnl', 'hedge_pnl', 'hedge_carry',
              'gamma_pnl', 'theta_pnl', 'vega_pnl', 'vanna_pnl', 'volga_pnl', 'tc',
              'gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be']

# Breakeven (realized-vs-implied) flow columns that also get a stored running
# total — unlike the plain Taylor buckets above, a single day's BE value is
# mostly noise; the diagnostic signal is in its cumulative drift (Ravagli
# 2024's own cumulative-PnL framing). Suffixed `_cum` rather than reusing the
# `pnl`->`equity` pattern since these are a diagnostic lens, not real P&L.
_BE_FLOW_COLS = ['gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be']

# Point-in-time exposure levels — fill 0 off-trade, SUM across trades, do NOT cumsum.
_EXPO_COLS = ['delta', 'gamma_1pct', 'vega_1vp', 'vanna_1vp', 'volga_1vp',
              'theta_daily', 'hedge_position']




def build_daily_book(trade_dfs: List[pd.DataFrame],
                     trade_log: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    ONE book-level daily frame, all in the BASE currency (= pair[:3]). 
            Use to_usd_book() to convert to USD accounting.

    Returns a DataFrame indexed by date with:
      pnl, equity, option_pnl..tc, gamma_pnl_be, vanna_pnl_be, volga_pnl_be,
      gamma_pnl_be_cum, vanna_pnl_be_cum, volga_pnl_be_cum, net_delta,
      net_gamma_1pct, net_vega_1vp, net_vanna_1vp, net_volga_1vp,
      net_theta_daily, net_hedge, net_delta_approx, n_open, spot

    net_delta_approx: `delta` is a START-of-day greek and `hedge_position`
    is the END-of-day hedge, so their sum is only an APPROXIMATE post-hedge residual
    (one-step timing mismatch), useful as a "is the book roughly delta-neutral" check
    rather than an exact residual.

    gamma_pnl_be_cum / vanna_pnl_be_cum / volga_pnl_be_cum: running total of each
    breakeven (realized-vs-implied) P&L bucket — the drift of these over time is
    the actual diagnostic signal (a single day's BE value is mostly noise). These
    are NOT part of `pnl`/`equity` — the plain (non-BE) buckets are the real P&L;
    the `_be` buckets are a separate lens layered on top (§5.7 of BACKTEST_GUIDE.md).
    """
    if not trade_dfs:
        return pd.DataFrame()
    norm, idx = [], None
    for df in trade_dfs:
        d = df.copy()
        d.index = pd.to_datetime(d.index)          # date objects -> Timestamps
        norm.append(d)
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    flow   = pd.DataFrame(0.0, index=idx, columns=_FLOW_COLS)
    expo   = pd.DataFrame(0.0, index=idx, columns=_EXPO_COLS)
    n_open = pd.Series(0, index=idx, dtype=int)
    spot   = pd.Series(index=idx, dtype=float)
    for d in norm:
        f = d.reindex(idx)
        flow   = flow + f[_FLOW_COLS].fillna(0.0)          # additive daily P&L
        expo   = expo + f[_EXPO_COLS].fillna(0.0)          # net level exposure
        n_open = n_open + f['net_pnl'].notna().astype(int) # trades open that day
        spot   = spot.combine_first(f['spot'])             # market context
    book = pd.DataFrame(index=idx)
    book.index.name = 'date'
    book['pnl']    = flow['net_pnl']
    book['equity'] = book['pnl'].cumsum()
    # ── daily P&L attribution (base ccy) ──
    for c in ['option_pnl', 'hedge_pnl', 'hedge_carry', 'gamma_pnl', 'theta_pnl',
              'vega_pnl', 'vanna_pnl', 'volga_pnl', 'tc']:
        book[c] = flow[c]
    # ── breakeven (realized-vs-implied) P&L: daily flow + running total ──
    for c in _BE_FLOW_COLS:
        book[c]           = flow[c]
        book[f'{c}_cum']  = book[c].cumsum()
    # ── net book exposures (base-ccy greeks), point-in-time ──
    book['net_delta']        = expo['delta']
    book['net_gamma_1pct']   = expo['gamma_1pct']
    book['net_vega_1vp']     = expo['vega_1vp']
    book['net_vanna_1vp']    = expo['vanna_1vp']
    book['net_volga_1vp']    = expo['volga_1vp']
    book['net_theta_daily']  = expo['theta_daily']
    book['net_hedge']        = expo['hedge_position']
    book['net_delta_approx'] = expo['delta'] + expo['hedge_position']  # ~post-hedge residual
    book['n_open']           = n_open
    book['spot']             = spot
    return book




# ── Day-by-day analysis views — exact column subsets of build_daily_book ─────
_PNL_VIEW_COLS = ['pnl', 'equity', 'option_pnl', 'hedge_pnl', 'hedge_carry',
                  'theta_pnl', 'vega_pnl', 'gamma_pnl', 'vanna_pnl', 'volga_pnl',
                  'tc', 'gamma_pnl_be', 'gamma_pnl_be_cum', 'vanna_pnl_be',
                  'vanna_pnl_be_cum', 'volga_pnl_be', 'volga_pnl_be_cum', 'n_open']
_EXPOSURE_VIEW_COLS = ['pnl', 'equity', 'net_delta', 'net_gamma_1pct', 'net_vega_1vp',
                       'net_vanna_1vp', 'net_volga_1vp', 'net_theta_daily',
                       'net_hedge', 'net_delta_approx', 'n_open']




def build_pnl_book(trade_dfs: Optional[List[pd.DataFrame]] = None,
                   trade_log: Optional[pd.DataFrame] = None,
                   book: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Day-by-day P&L-attribution DataFrame, indexed by date, for time-series
    analysis. Columns
        pnl, equity, option_pnl, hedge_pnl, hedge_carry, theta_pnl, vega_pnl,
        gamma_pnl, vanna_pnl, volga_pnl, tc, gamma_pnl_be, gamma_pnl_be_cum,
        vanna_pnl_be, vanna_pnl_be_cum, volga_pnl_be, volga_pnl_be_cum, n_open

    The `_be` columns are the breakeven (realized-vs-implied) P&L lens layered
    on top of the plain gamma/vanna/volga buckets — see build_daily_book's
    docstring / BACKTEST_GUIDE.md §5.7. The `_cum` columns are each `_be`
    bucket's running total, the actual signal to watch (a single day's `_be`
    value is mostly noise).
    """
    if book is None:
        book = build_daily_book(trade_dfs, trade_log)
    if book.empty:
        return pd.DataFrame(columns=_PNL_VIEW_COLS)
    return book[_PNL_VIEW_COLS].copy()



def build_exposure_book(trade_dfs: Optional[List[pd.DataFrame]] = None,
                        trade_log: Optional[pd.DataFrame] = None,
                        book: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Day-by-day net-exposure DataFrame, indexed by date, for time-series
    analysis. Columns:
        pnl, equity, net_delta, net_gamma_1pct, net_vega_1vp, net_vanna_1vp,
        net_volga_1vp, net_theta_daily, net_hedge, net_delta_approx, n_open
    """
    if book is None:
        book = build_daily_book(trade_dfs, trade_log)
    if book.empty:
        return pd.DataFrame(columns=_EXPOSURE_VIEW_COLS)
    return book[_EXPOSURE_VIEW_COLS].copy()




def _usd_factor(index: pd.Index, pair: str, spot: Optional[pd.Series] = None,
                fx_usd: Optional[pd.Series] = None) -> pd.Series:
    """
    USD-per-BASE conversion factor aligned to `index`. The SINGLE definition of the
    FX convention, shared by to_usd_book (daily book) and to_usd_trade_log
    (per-trade totals) — if those two ever disagreed, the trade lens would stop
    reconciling with the book lens and the same P&L would print two ways.

      base  == USD -> 1.0 (the frame is already USD accounting)
      quote == USD -> the pair's own spot, which IS quote(USD)-per-base
      neither      -> caller must supply fx_usd (USD-per-base), ffilled onto index

    Gaps in the pair's own spot are ffilled then bfilled: a missing rate would
    otherwise silently NaN out that day's converted P&L rather than fail loudly.
    """
    base, quote = pair[:3].upper(), pair[3:].upper()
    if base == 'USD':
        return pd.Series(1.0, index=index)
    if quote == 'USD':
        if spot is None:
            raise ValueError(
                f"to-USD conversion for {pair} needs the frame's `spot` column.")
        factor = pd.Series(spot).astype(float).reindex(index).ffill().bfill()
        if factor.isna().all():
            raise ValueError(
                f"{pair}: `spot` is entirely missing — cannot convert to USD.")
        return factor
    if fx_usd is None:
        raise ValueError(
            f"{pair} has no USD leg — cannot convert to USD from the frame alone "
            f"(its spot is {quote}-per-{base}). Supply fx_usd: a Series of "
            f"USD-per-{base} rates indexed by date.")
    factor = pd.Series(fx_usd).reindex(index).ffill()
    if factor.isna().any():
        raise ValueError(
            f"fx_usd does not cover all dates for {pair}: "
            f"{int(factor.isna().sum())} date(s) missing a USD-per-{base} rate.")
    return factor


def to_usd_book(book: pd.DataFrame, pair: str,
                fx_usd: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Convert a base-ccy book (from build_daily_book) into USD accounting, so the
    same strategy can be compared across pairs on a common-currency basis.

    NOTE: this converts each day's P&L at THAT day's FX rate (daily-rate
    convention)
    """
    if book.empty:
        return book.copy()
    factor    = _usd_factor(book.index, pair, spot=book.get('spot'), fx_usd=fx_usd)
    cum_cols  = [f'{c}_cum' for c in _BE_FLOW_COLS]
    no_scale  = {'equity', 'n_open', 'spot'} | set(cum_cols)
    money_cols = [c for c in book.columns if c not in no_scale]
    usd = book.copy()
    for c in money_cols:
        usd[c] = book[c] * factor
    usd['equity'] = usd['pnl'].cumsum()          # re-cumsum from converted daily pnl
    # Same re-cumsum treatment as equity: a cumulative BE total scaled by a single
    # day's FX rate would be wrong — re-derive each from its now-converted daily flow.
    for c, cum_c in zip(_BE_FLOW_COLS, cum_cols):
        if cum_c in usd.columns:
            usd[cum_c] = usd[c].cumsum()
    return usd


# Every money field on trade_log -> the daily flow column in that trade's own
# df_agg that it is the plain sum of (backtest_MLeg.py:898-924). net_premium is
# deliberately absent: it is a single entry-date cashflow, not a flow over the
# trade's life, so it converts at one rate (see to_usd_trade_log).
_TRADE_MONEY_FROM_FLOW = {
    'net_pnl':           'net_pnl',
    'gamma_pnl':         'gamma_pnl',
    'gamma_pnl_be':      'gamma_pnl_be',
    'theta_pnl':         'theta_pnl',
    'vega_pnl':          'vega_pnl',
    'vanna_pnl':         'vanna_pnl',
    'volga_pnl':         'volga_pnl',
    'vanna_pnl_be':      'vanna_pnl_be',
    'volga_pnl_be':      'volga_pnl_be',
    'hedge_carry_pnl':   'hedge_carry',
    'transaction_costs': 'tc',
}


def to_usd_trade_log(trade_log: pd.DataFrame, trade_dfs: List[pd.DataFrame],
                     pair: str, fx_usd: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Convert a base-ccy trade_log (from run_signal_backtest) into USD accounting —
    the PER-TRADE counterpart of to_usd_book, so trade_metrics() is cross-pair
    comparable the same way book_metrics() already is.

    Each money field is RE-SUMMED from that trade's own daily flow at THAT day's
    rate, not scaled by a single rate. Two reasons it has to be done this way:
      * a trade spanning a large FX move would otherwise be converted at a rate it
        never actually earned its P&L at;
      * only the daily convention keeps the two lenses reconciled —
        sum(trade net_pnl) == book_total_pnl, because both are
        sum_days(factor(day) * flow(day)) over the same days.

    net_premium is the one exception: an entry-date cashflow, converted at the
    ENTRY-date rate. return_on_premium therefore stays a clean USD/USD ratio.

    trade_dfs must be POSITIONALLY aligned with trade_log's rows. run_signal_backtest
    appends the two in lockstep (backtest_signal.py:199-200), and gate_sweep's
    subsetting preserves it (trade_log.iloc[idx] beside [trade_dfs[i] for i in idx]).

    Unit-free fields pass through untouched: the vol-point reads (realised_vol,
    avg_entry_sigma, vol_spread, atm_entry_vol) and every ratio derived downstream.
    """
    if trade_log is None or len(trade_log) == 0:
        return trade_log
    if len(trade_dfs) != len(trade_log):
        raise ValueError(
            f"to_usd_trade_log: {len(trade_dfs)} daily frame(s) for "
            f"{len(trade_log)} trade(s) — they must be positionally aligned.")
    if pair[:3].upper() == 'USD':
        return trade_log.copy()             # base is USD: already USD accounting

    out    = trade_log.copy()
    fields = [f for f in _TRADE_MONEY_FROM_FLOW if f in out.columns]
    conv   = {f: [] for f in fields}
    prem   = []
    has_prem = 'net_premium' in out.columns

    for i, df in enumerate(trade_dfs):
        d = df.copy()
        d.index = pd.to_datetime(d.index)   # trade frames can carry date objects
        factor  = _usd_factor(d.index, pair, spot=d.get('spot'), fx_usd=fx_usd)
        for f in fields:
            flow = d[_TRADE_MONEY_FROM_FLOW[f]].astype(float).fillna(0.0)
            conv[f].append(float((flow * factor).sum()))
        if has_prem:
            entry = pd.Timestamp(out['entry_date'].iat[i])
            f0    = (float(factor.loc[entry]) if entry in factor.index
                     else float(factor.iloc[0]))
            prem.append(float(out['net_premium'].iat[i]) * f0)

    for f in fields:                        # list assignment is positional
        out[f] = conv[f]
    if has_prem:
        out['net_premium'] = prem
    return out





# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
def _extreme(s: pd.Series) -> float:
    """Signed value of largest magnitude (works for net-short OR net-long books)."""
    s = s.dropna()
    return float(s.loc[s.abs().idxmax()]) if len(s) else float('nan')



def print_daily_book(book: pd.DataFrame, tail: int = 20, ccy: Optional[str] = None) -> None:
    """
    Time-level headline you can't get from trade_log: book P&L attribution, max
    drawdown (from the equity curve), peak concurrency, and peak net vega/gamma
    (the unhedged risk that accumulates as trades stack).

    ccy : label for the money columns (e.g. 'USD', 'EUR'). Purely cosmetic — pass
          pair[:3] for a base-ccy book, or 'USD' for a to_usd_book() result.
          Defaults to 'base ccy'.
    """
    if book.empty:
        print("No daily book — no trades.")
        return
    unit   = ccy or 'base ccy'
    dd     = book['equity'] - book['equity'].cummax()   # drawdown series (<= 0)
    active = int((book['n_open'] > 0).sum())

    print(f"\n{'='*72}")
    print(f"  TIME-LEVEL BOOK  |  {book.index[0].date()} -> {book.index[-1].date()}  "
          f"|  {len(book)} days ({active} with a position)")
    print(f"{'='*72}")

    print(f"  P&L attribution ({unit}):")
    for c, lbl in [('option_pnl', 'Option'), ('hedge_pnl', 'Hedge spot'),
                   ('hedge_carry', 'Hedge carry'), ('gamma_pnl', 'Gamma'),
                   ('theta_pnl', 'Theta'), ('vega_pnl', 'Vega'), ('vanna_pnl', 'Vanna'),
                   ('volga_pnl', 'Volga'), ('tc', 'Transaction cost')]:
        print(f"    {lbl:<18}{book[c].sum():>16,.2f}")
    print(f"    {'-'*34}")
    print(f"    {'Total net P&L':<18}{book['pnl'].sum():>16,.2f}  {unit}")

    print(f"\n  Time-level risk / exposure ({unit}):")
    print(f"    Max drawdown             {dd.min():>16,.2f}")
    print(f"    Worst day                {book['pnl'].min():>16,.2f}")
    print(f"    Best day                 {book['pnl'].max():>16,.2f}")
    print(f"    Peak concurrent trades   {book['n_open'].max():>16d}")
    print(f"    Peak net vega  (1vp)     {_extreme(book['net_vega_1vp']):>16,.2f}")
    print(f"    Peak net gamma (1%)      {_extreme(book['net_gamma_1pct']):>16,.2f}")
    print(f"    Max |approx net delta|   {book['net_delta_approx'].abs().max():>16,.2f}")

    cols = ['pnl', 'equity', 'net_delta_approx', 'net_gamma_1pct', 'net_vega_1vp',
            'net_theta_daily', 'n_open', 'spot']
    print(f"\n  Last {tail} days:")
    print(book[cols].tail(tail).to_string(
        formatters={c: '{:,.2f}'.format for c in cols if c != 'n_open'}))



# --------------------------------------------------------------------------------------------
# ------------------------------ Evalualtion Metrics -----------------------------------------
# --------------------------------------------------------------------------------------------

"""

Two lenses, each -> a labelled pd.Series so comparing configs is just pd.concat:
  trade_metrics(trade_log) : per-trade lens (edge, tail, attribution, significance)
  book_metrics(book)       : time-level lens (Sharpe/Sortino/Calmar, drawdown, exposure)

  scorecard(book, log)     : both, concatenated -> the unit of comparison
  evaluate(trade_dfs, log) : one-call convenience (build book -> scorecard)

Scale-invariant RATIOS: (win_rate, profit_factor, payoff_ratio, t_stat,
return_on_premium, Sharpe/Sortino/Calmar, attribution *shares*) compare directly
across configs/pairs with no normalization. 

ABSOLUTE figures: (expectancy,worst_trade, cvar, max_drawdown, peak vega)
only compare after normalization or in USD. evaluate(to_usd=True) does that for
BOTH lenses — to_usd_book() for the daily book, to_usd_trade_log() for the
per-trade log — off one shared daily FX factor (_usd_factor), so the two stay
reconciled: sum(trade net_pnl) == book_total_pnl.
Everything degrades to NaN rather than crashing.

"""


import numpy as np

_ANN = 252   

def _safe_div(a, b) -> float:
    b = float(b) if pd.notna(b) else float('nan')
    return float(a) / b if (pd.notna(b) and b != 0.0) else float('nan')


# PER TRADE METRICS - Eval of each trade done
def trade_metrics(trade_log: pd.DataFrame, settled_only: bool = True) -> pd.Series:
    """
    settled_only : drop live (unsettled) trades — their net_pnl is only a partial
                   mark-to-market and would bias per-trade stats.
    """
    keys = ['n_trades', 'n_live_excluded', 'total_pnl', 'expectancy', 'median_pnl',
            'win_rate', 'payoff_ratio', 'profit_factor', 'pnl_skew', 'cvar_5pct',
            'worst_trade', 't_stat', 'return_on_premium', 'avg_days_held',
            'pct_hold_to_expiry', 'theta_carry_pnl', 'gamma_pnl', 'vol_pnl', 'spot_tc',
            'theta_carry_share', 'gamma_share', 'vol_share', 'avg_vol_spread',
            'real_VRP_ave', 'worst_trade_entry_date', 'worst_trade_exit_date',
            'worst_trade_gamma_pnl_be', 'worst_trade_vega_pnl',
            'worst_trade_vanna_pnl_be', 'worst_trade_volga_pnl_be']

    if trade_log is None or len(trade_log) == 0:
        return pd.Series({k: float('nan') for k in keys})

    n_all = len(trade_log)
    tl = (trade_log[~trade_log['live_trade'].astype(bool)]
          if settled_only and 'live_trade' in trade_log.columns else trade_log)
    if len(tl) == 0:
        s = pd.Series({k: float('nan') for k in keys})
        s['n_trades'], s['n_live_excluded'] = 0, n_all
        return s

    n = len(tl)
    pnl = tl['net_pnl'].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    std = pnl.std(ddof=1) if n > 1 else float('nan')
    q05 = pnl.quantile(0.05)

    def _sum(c):
        return float(tl[c].astype(float).sum()) if c in tl.columns else float('nan')

    theta_carry = _sum('theta_pnl') + _sum('hedge_carry_pnl')
    gamma       = _sum('gamma_pnl')
    vol         = _sum('vega_pnl') + _sum('vanna_pnl') + _sum('volga_pnl')
    tc          = _sum('transaction_costs')
    gross       = abs(theta_carry) + abs(gamma) + abs(vol) + abs(tc)

    # Worst single trade (by net_pnl) — entry/exit dates plus that trade's own
    # SUMMED breakeven/vega attribution over its life (already trade-level
    # totals on trade_log, no re-aggregation needed).
    worst_idx = pnl.idxmin()
    worst_row = tl.loc[worst_idx]
    def _worst(c):
        return worst_row[c] if c in tl.columns else float('nan')

    out = {
        'n_trades':          n,
        'n_live_excluded':   n_all - n,
        'total_pnl':         float(pnl.sum()),
        'expectancy':        float(pnl.mean()),
        'median_pnl':        float(pnl.median()),
        'win_rate':          float((pnl > 0).mean()),
        'payoff_ratio':      _safe_div(wins.mean(), abs(losses.mean())) if len(losses) else float('nan'),
        'profit_factor':     _safe_div(wins.sum(), abs(losses.sum())),
        'pnl_skew':          float(pnl.skew()) if n >= 3 else float('nan'),
        'cvar_5pct':         float(pnl[pnl <= q05].mean()) if (pnl <= q05).any() else float('nan'),
        'worst_trade':       float(pnl.min()),
        't_stat':            _safe_div(pnl.mean(), std / np.sqrt(n)) if n > 1 else float('nan'),
        'return_on_premium': (_safe_div(pnl.sum(), tl['net_premium'].abs().sum())
                              if 'net_premium' in tl.columns else float('nan')),
        'avg_days_held':     float(tl['days_held'].mean()) if 'days_held' in tl.columns else float('nan'),
        'pct_hold_to_expiry':(float((tl['exit_reason'] == 'expiry').mean())
                              if 'exit_reason' in tl.columns else float('nan')),
        'theta_carry_pnl':   theta_carry,
        'gamma_pnl':         gamma,
        'vol_pnl':           vol,
        'spot_tc':           tc,
        'theta_carry_share': _safe_div(theta_carry, gross),
        'gamma_share':       _safe_div(gamma, gross),
        'vol_share':         _safe_div(vol, gross),
        'avg_vol_spread':    float(tl['vol_spread'].mean()) if 'vol_spread' in tl.columns else float('nan'),
        # Realized vol vs. pure ATM vol at inception (not vega-weighted across
        # legs, unlike avg_vol_spread) — per trade, then averaged.
        'real_VRP_ave':      (float((tl['realised_vol'] - tl['atm_entry_vol']).mean())
                              if {'realised_vol', 'atm_entry_vol'} <= set(tl.columns)
                              else float('nan')),
        'worst_trade_entry_date':      _worst('entry_date'),
        'worst_trade_exit_date':       _worst('exit_date'),
        'worst_trade_gamma_pnl_be':    _worst('gamma_pnl_be'),
        'worst_trade_vega_pnl':        _worst('vega_pnl'),
        'worst_trade_vanna_pnl_be':    _worst('vanna_pnl_be'),
        'worst_trade_volga_pnl_be':    _worst('volga_pnl_be'),
    }
    return pd.Series(out)[keys]


# PER DAY METRIC -- Eval of Daily book statistics
def book_metrics(book: pd.DataFrame, active_only: bool = False) -> pd.Series:
    """
    Time-level lens from the daily book (build_daily_book)
    Risk-adjusted ratios are computed on the daily P&L series;
     
    active_only=True judges per-deployed-day instead. Drawdown always uses the full equity curve.
    """
    keys = ['n_days', 'active_days', 'span_bdays', 'coverage', 'book_total_pnl',
            'sharpe_ann', 'sortino_ann', 'max_drawdown', 'max_drawdown_days', 'calmar',
            'daily_pnl_skew', 'cvar_5pct_daily', 'var_95_daily', 'var_99_daily',
            'worst_day', 'best_day',
            'peak_concurrency', 'avg_net_vega', 'avg_net_gamma',
            'peak_net_vega', 'peak_net_gamma',
            'worst_day_date', 'worst_day_gamma_pnl_be', 'worst_day_vega_pnl',
            'worst_day_vanna_pnl_be', 'worst_day_volga_pnl_be',
            'net_gamma_pnl_BE', 'net_vanna_pnl_BE', 'net_volga_pnl_BE']

    if book is None or book.empty:
        return pd.Series({k: float('nan') for k in keys})
    daily_all = book['pnl'].astype(float)
    active    = book['n_open'] > 0
    series    = daily_all[active] if active_only else daily_all
    n         = len(series)

    mean     = series.mean()
    std      = series.std(ddof=1) if n > 1 else float('nan')
    downside = float(np.sqrt((series.clip(upper=0.0) ** 2).mean())) if n else float('nan')
    eq      = book['equity'].astype(float)
    dd      = eq - eq.cummax()
    total   = float(daily_all.sum())

    # ── Calendar span, NOT row count ────────────────────────────────────────────
    # build_daily_book indexes on the UNION of the trades' own date indices, so a
    # day with no trade open is ABSENT from the book entirely — it is not a zero
    # row. Two consequences the old code got wrong:
    #
    #   1. `len(book)` is DEPLOYED days, not calendar days. Annualizing with it
    #      inflated calmar by (calendar / deployed) for any config that doesn't
    #      trade continuously — a signal firing a third of the time got its P&L
    #      annualized over a third of the span, ~3x overstating calmar and making
    #      it non-comparable against an always-on baseline.
    #   2. `active.sum() / len(book)` is identically 1.0, because every row in the
    #      union belongs to at least one trade. `coverage` could never report
    #      anything but 100%, which is exactly the number it exists to vary.
    #
    # Both are fixed off the book's own first/last date. busday_count ignores FX
    # holidays so span_bdays runs a few percent high, making `coverage` a mild
    # UNDER-estimate — acceptable for a deployment read, and vastly better than 1.0.
    #
    # NOTE sharpe_ann / sortino_ann are deliberately left on the deployed series:
    # with no flat rows in the book they are already per-deployed-day (i.e. what
    # active_only=True would give), which is the right "quality per day in the
    # market" read. They are NOT diluted by idle days, so there is no
    # sqrt(coverage) penalty to undo.
    span_days  = max((book.index[-1] - book.index[0]).days, 0) + 1
    span_years = max(span_days / 365.0, 1e-9)
    span_bdays = max(int(np.busday_count(book.index[0].date(),
                                        book.index[-1].date())) + 1, 1)
    ann_pnl = total / span_years
    q05     = series.quantile(0.05)
    # VaR = the loss quantile itself (a P&L level, typically negative), as
    # distinct from cvar_5pct_daily which is the MEAN of the tail beyond q05.
    q01     = series.quantile(0.01)

    # Max drawdown duration: peak-to-trough (days from the equity high to the
    # point where max_drawdown bottoms out) — not recovery time back to a new high.
    trough_date      = dd.idxmin()
    peak_date        = eq.loc[:trough_date].idxmax()
    max_drawdown_days = (trough_date - peak_date).days

    # Worst single day (book-level) — date plus that day's breakeven/vega
    # attribution, for the specific columns asked for.
    def _worst_day(c):
        return book.loc[worst_day_date, c] if (n and c in book.columns) else float('nan')
    worst_day_date = series.idxmin() if n else float('nan')

    # Cumulative breakeven P&L over the whole book's life (last row of each
    # running total added to build_daily_book — see that function's docstring).
    def _be_cum_last(c):
        return float(book[c].iloc[-1]) if c in book.columns and not book.empty else float('nan')

    out = {
        'n_days':           len(book),          # rows = DEPLOYED days (see span note)
        'active_days':      int(active.sum()),
        'span_bdays':       span_bdays,         # business days first row -> last row
        'coverage':         _safe_div(int(active.sum()), span_bdays),
        'book_total_pnl':   total,
        'sharpe_ann':       _safe_div(mean, std) * np.sqrt(_ANN),
        'sortino_ann':      _safe_div(mean, downside) * np.sqrt(_ANN),
        'max_drawdown':     float(dd.min()),
        'max_drawdown_days': int(max_drawdown_days),
        'calmar':           _safe_div(ann_pnl, abs(dd.min())),
        'daily_pnl_skew':   float(series.skew()) if n >= 3 else float('nan'),
        'cvar_5pct_daily':  float(series[series <= q05].mean()) if (series <= q05).any() else float('nan'),
        'var_95_daily':     float(q05) if n else float('nan'),
        'var_99_daily':     float(q01) if n else float('nan'),
        'worst_day':        float(series.min()) if n else float('nan'),
        'best_day':         float(series.max()) if n else float('nan'),
        'peak_concurrency': int(book['n_open'].max()),
        'avg_net_vega':     float(book.loc[active, 'net_vega_1vp'].mean()) if active.any() else float('nan'),
        'avg_net_gamma':    float(book.loc[active, 'net_gamma_1pct'].mean()) if active.any() else float('nan'),
        'peak_net_vega':    _extreme(book['net_vega_1vp']),
        'peak_net_gamma':   _extreme(book['net_gamma_1pct']),
        'worst_day_date':             worst_day_date,
        'worst_day_gamma_pnl_be':     _worst_day('gamma_pnl_be'),
        'worst_day_vega_pnl':         _worst_day('vega_pnl'),
        'worst_day_vanna_pnl_be':     _worst_day('vanna_pnl_be'),
        'worst_day_volga_pnl_be':     _worst_day('volga_pnl_be'),
        'net_gamma_pnl_BE':  _be_cum_last('gamma_pnl_be_cum'),
        'net_vanna_pnl_BE':  _be_cum_last('vanna_pnl_be_cum'),
        'net_volga_pnl_BE':  _be_cum_last('volga_pnl_be_cum'),
    }
    return pd.Series(out)[keys]


# Per-day breakeven/vega attribution columns surfaced in the worst-days table,
# mapped from the display header -> daily-book column name.
_WORST_DAY_COLS = [
    ('gamma_be_pnl', 'gamma_pnl_be'),
    ('vega_pnl',     'vega_pnl'),
    ('vanna_be_pnl', 'vanna_pnl_be'),
    ('volga_be_pnl', 'volga_pnl_be'),
]

# That day's point-in-time net greek EXPOSURES (not P&L) — optionally appended.
_WORST_DAY_EXPO_COLS = [
    ('gamma_exp', 'net_gamma_1pct'),
    ('vega_exp',  'net_vega_1vp'),
    ('vanna_exp', 'net_vanna_1vp'),
    ('volga_exp', 'net_volga_1vp'),
]


def worst_days(book: pd.DataFrame, n: int = 3,
               include_exposures: bool = False) -> pd.DataFrame:
    """
    The `n` worst P&L days from the daily book, one row each (most-negative
    first), indexed 1..n. Columns: date, pnl, and that day's breakeven/vega
    attribution — the same fields as the single-worst-day scorecard row.

    include_exposures=True also appends that day's point-in-time net greek
    exposures (gamma/vega/vanna/volga), so you can see the risk carried into
    each bad day, not just how the loss decomposed.
    """
    cols = list(_WORST_DAY_COLS)
    if include_exposures:
        cols = cols + _WORST_DAY_EXPO_COLS
    headers = ['date', 'pnl'] + [h for h, _ in cols]
    if book is None or book.empty:
        return pd.DataFrame(columns=headers)
    worst = book['pnl'].astype(float).nsmallest(n)
    rows = {
        'date': worst.index.strftime('%Y-%m-%d'),
        'pnl':  worst.values,
    }
    for hdr, col in cols:
        rows[hdr] = (book.loc[worst.index, col].astype(float).values
                     if col in book.columns else float('nan'))
    df = pd.DataFrame(rows, columns=headers)
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, name='rank')
    return df


def worst_trades(trade_log: pd.DataFrame, n: int = 3,
                 settled_only: bool = True) -> pd.DataFrame:
    """
    The `n` worst trades (by net_pnl) from the trade log, one row each
    (most-negative first), indexed 1..n. Columns: entry, exit, pnl, and that
    trade's summed breakeven/vega attribution — same structure as worst_days,
    with entry/exit dates in place of a single day's date.

    settled_only mirrors trade_metrics: drop live (unsettled) trades whose
    net_pnl is only a partial mark-to-market.
    """
    headers = ['entry', 'exit', 'pnl'] + [h for h, _ in _WORST_DAY_COLS]
    if trade_log is None or len(trade_log) == 0:
        return pd.DataFrame(columns=headers)
    tl = (trade_log[~trade_log['live_trade'].astype(bool)]
          if settled_only and 'live_trade' in trade_log.columns else trade_log)
    if len(tl) == 0:
        return pd.DataFrame(columns=headers)
    worst = tl['net_pnl'].astype(float).nsmallest(n)
    wl = tl.loc[worst.index]

    def _dates(c):
        return (pd.to_datetime(wl[c]).dt.strftime('%Y-%m-%d').values
                if c in wl.columns else '')
    rows = {
        'entry': _dates('entry_date'),
        'exit':  _dates('exit_date'),
        'pnl':   worst.values,
    }
    for hdr, col in _WORST_DAY_COLS:
        rows[hdr] = wl[col].astype(float).values if col in wl.columns else float('nan')
    df = pd.DataFrame(rows, columns=headers)
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, name='rank')
    return df






def scorecard(book: pd.DataFrame, trade_log: pd.DataFrame,
              settled_only: bool = True, name: Optional[str] = None,
              n_worst_days: int = 3, n_worst_trades: int = 3) -> pd.Series:
    """
    Full evaluation vector = trade_metrics + book_metrics, concatenated into one
    labelled Series. The top-`n_worst_days` worst days and top-`n_worst_trades`
    worst trades are stashed on ``sc.attrs['worst_days']`` /
    ``sc.attrs['worst_trades']`` (DataFrames) for print_scorecard to render.
    """
    sc = pd.concat([trade_metrics(trade_log, settled_only=settled_only),
                    book_metrics(book)])
    sc.name = name
    sc.attrs['worst_days']   = worst_days(book, n=n_worst_days)
    sc.attrs['worst_trades'] = worst_trades(trade_log, n=n_worst_trades,
                                            settled_only=settled_only)
    return sc


def evaluate(trade_dfs: List[pd.DataFrame], trade_log: pd.DataFrame,
             pair: Optional[str] = None, to_usd: bool = False,
             settled_only: bool = False, name: Optional[str] = None,
             n_worst_days: int = 3, n_worst_trades: int = 3,
             fx_usd: Optional[pd.Series] = None) -> pd.Series:
    """
    One-call convenience: build the daily book from run_signal_backtest output
    (optionally converted to USD) and return its scorecard.

    to_usd=True converts BOTH lenses off the same daily factor — the daily book
    (to_usd_book) and the per-trade log (to_usd_trade_log) — so every money figure
    in the returned scorecard is USD, and the trade lens reconciles with the book
    lens instead of quietly reporting a different currency. Requires `pair`; a pair
    with no USD leg additionally requires fx_usd (USD-per-base).

    (It used to convert the book only, leaving expectancy / worst_trade / the
    per-trade attribution buckets in base ccy — a ~0.6-1.3x distortion that made
    them unsafe to rank across pairs.)
    """
    book = build_daily_book(trade_dfs, trade_log)
    if to_usd:
        if pair is None:
            raise ValueError("to_usd=True requires `pair` (to pick the FX factor).")
        book      = to_usd_book(book, pair, fx_usd=fx_usd)
        trade_log = to_usd_trade_log(trade_log, trade_dfs, pair, fx_usd=fx_usd)
    return scorecard(book, trade_log, settled_only=settled_only, name=name,
                     n_worst_days=n_worst_days, n_worst_trades=n_worst_trades)


_SCORECARD_GROUPS = [
    ('Edge',            ['n_trades', 'expectancy', 'median_pnl', 'win_rate',
                         'payoff_ratio', 'profit_factor', 'return_on_premium']),


    ('Risk-adjusted',   ['sharpe_ann', 'sortino_ann', 'calmar']),


    ('Tail / drawdown', ['max_drawdown', 'max_drawdown_days',
                         'pnl_skew', 'daily_pnl_skew',
                         'var_95_daily', 'var_99_daily',
                         'cvar_5pct_daily', 'cvar_5pct']),

    # ('Significance',    ['t_stat', 'n_trades', 'coverage', 'peak_concurrency']),


    ('Attribution',     ['net_gamma_pnl_BE', 'net_vanna_pnl_BE', 'net_volga_pnl_BE',
                         'spot_tc', 'real_VRP_ave']),


    ('Time / exposure', ['book_total_pnl', 'n_days', 'active_days', 'span_bdays',
                         'coverage', 'avg_days_held',
                         'pct_hold_to_expiry', 'avg_net_vega', 'avg_net_gamma',
                         'peak_net_vega', 'peak_net_gamma', 'n_live_excluded']),
]


# Display-name overrides for scalar scorecard rows — used only for print labels,
# the underlying keys are unchanged. Distinguishes the two 5% CVaR measures:
# daily = mean of the worst 5% of BOOK days; trade = mean of the worst 5% of TRADES.
_KEY_LABELS = {
    'n_days':     'n_days (deployed)',
    'span_bdays': 'span_bdays (first->last)',
    'cvar_5pct_daily': 'cvar_5pct (daily)',
    'cvar_5pct':       'cvar_5pct (trade)',
}


def _fmt_cell(v) -> str:
    """Format one scorecard value: dates as YYYY-MM-DD, money with commas, ratios 4dp."""
    if pd.isna(v):
        return 'nan'
    if hasattr(v, 'strftime'):   # date / Timestamp fields (e.g. worst_day_date)
        return v.strftime('%Y-%m-%d')
    if abs(v) >= 100:            # money-ish -> thousands, 2dp
        return f"{v:,.2f}"
    return f"{v:.4f}"            # ratios / shares -> 4dp


def _print_ranked_table(df: Optional[pd.DataFrame], label: str,
                        indent: str = '    ') -> None:
    """Render a worst-days/worst-trades DataFrame as an aligned table, ranked 1..n.

    Numeric columns are formatted via _fmt_cell; date/string columns pass through.
    """
    if df is None or df.empty:
        return
    disp = df.copy()
    for c in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = disp[c].map(_fmt_cell)
        else:
            disp[c] = disp[c].astype(str)
    headers = ['#'] + list(disp.columns)
    rows    = [[str(i)] + list(disp.loc[i]) for i in disp.index]
    widths  = [max(len(headers[j]), *(len(r[j]) for r in rows)) for j in range(len(headers))]
    hline = '  '.join(h.rjust(w) for h, w in zip(headers, widths))
    print(f"\n{indent}{label}:")
    print(f"{indent}  {hline}")
    for r in rows:
        print(f"{indent}  " + '  '.join(x.rjust(w) for x, w in zip(r, widths)))


def print_scorecard(sc: pd.Series, title: Optional[str] = None) -> None:
    """Grouped, readable view of a scorecard() Series."""
    print(f"\n{'='*54}")
    print(f"  SCORECARD{('  |  ' + title) if title else ''}")
    print(f"{'='*54}")
    for gname, ks in _SCORECARD_GROUPS:
        print(f"\n  {gname}:")
        for k in ks:
            if k not in sc.index:
                continue
            print(f"    {_KEY_LABELS.get(k, k):<22}{_fmt_cell(sc[k]):>18}")
        if gname == 'Tail / drawdown':
            _print_ranked_table(sc.attrs.get('worst_days'),   'Worst days')
            _print_ranked_table(sc.attrs.get('worst_trades'), 'Worst trades')












# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# Visualization — equity/risk tearsheet + P&L distribution
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# Colours come from the data-viz reference palette (light mode). Every multi-series
# set used here is a subset of the palette's first four categorical slots
# (blue/green/magenta/yellow), which the palette certifies passes CVD/contrast
# gates in both modes; single-series panels need no separation check.

try:
    import matplotlib.pyplot as _plt
    import matplotlib.dates as _mdates
    from matplotlib.ticker import FuncFormatter as _FuncFormatter
except Exception:                       # matplotlib optional — plotting off if absent
    _plt = None

_VIZ = {
    'surface':  '#fcfcfb',
    'ink':      '#0b0b0b',
    'ink2':     '#52514e',
    'muted':    '#898781',
    'grid':     '#e1e0d9',
    'baseline': '#c3c2b7',
    'blue':     '#2a78d6',   # slot 1
    'green':    '#008300',   # slot 2
    'magenta':  '#e87ba4',   # slot 3
    'yellow':   '#eda100',   # slot 4
    'red':      '#d03b3b',   # status: critical — losses / worst markers
}

# Fixed greek -> hue identity, reused across every panel so a greek reads as the
# same colour wherever it appears. Delta is deliberately muted (a residual check).
_GREEK_COLORS = {
    'delta': _VIZ['muted'], 'gamma': _VIZ['blue'],
    'vega':  _VIZ['green'], 'vanna': _VIZ['magenta'], 'volga': _VIZ['yellow'],
}


def _require_mpl():
    if _plt is None:
        raise ImportError("matplotlib is required for plotting "
                          "(pip install matplotlib).")


def _compact_num(v) -> str:
    """Abbreviate large magnitudes: 3,000,000 -> 3.0M, 500,000 -> 500k."""
    a = abs(v)
    if a >= 1e6:
        return f"{v/1e6:.1f}M"
    if a >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def _thousands(_ax, compact: bool = False) -> None:
    fmt = _compact_num if compact else (lambda v: f"{v:,.0f}")
    _ax.yaxis.set_major_formatter(_FuncFormatter(lambda v, _: fmt(v)))


def _style_axis(ax, money: bool = False, compact: bool = False) -> None:
    """Recessive chrome: no top/right spines, hairline y-grid, muted ticks.

    compact=True abbreviates y-tick numbers (k/M) — for tight/small panels.
    """
    ax.set_facecolor(_VIZ['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_VIZ['baseline'])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_VIZ['muted'], labelsize=8, length=3)
    ax.grid(axis='y', color=_VIZ['grid'], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if money:
        _thousands(ax, compact=compact)


def _ylabel(ax, text: str) -> None:
    ax.set_ylabel(text, fontsize=8.5, color=_VIZ['ink2'])


def _cvar(series: pd.Series, q: float = 0.05):
    s = series.astype(float).dropna()
    if s.empty:
        return float('nan'), float('nan')
    cut = s.quantile(q)
    tail = s[s <= cut]
    return (float(tail.mean()) if not tail.empty else float('nan')), float(s.min())


# ── Panel drawers — each takes an Axes so they compose into any layout ───────
def _draw_equity(ax, book, unit: str, n_worst: int = 3,
                 show_open_trades: bool = False,
                 open_trades_label: str = '# open trades') -> None:
    idx, eq = book.index, book['equity'].astype(float)
    ax.plot(idx, eq, color=_VIZ['blue'], linewidth=2.0, zorder=3, label='Equity')
    ax.fill_between(idx, 0, eq, color=_VIZ['blue'], alpha=0.07, zorder=1)
    ax.axhline(0, color=_VIZ['baseline'], linewidth=0.8, zorder=1)
    worst = book['pnl'].astype(float).nsmallest(n_worst)
    ax.scatter(worst.index, eq.reindex(worst.index), s=42, color=_VIZ['red'],
               zorder=4, edgecolor=_VIZ['surface'], linewidth=1.2,
               label=f'Worst {len(worst)} days')
    _ylabel(ax, f'Cumulative P&L{unit}')

    # Optional context series on a faint secondary (right) axis: how many trades
    # are open (= short, in a short-vol book) on each day. Deliberately faint so
    # it reads as backdrop and the equity curve stays dominant.
    if show_open_trades and 'n_open' in book.columns:
        ax2 = ax.twinx()
        ax2.plot(idx, book['n_open'].astype(float), color=_VIZ['muted'],
                 alpha=0.55, linewidth=1.4, drawstyle='steps-post', zorder=2,
                 label=open_trades_label)
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel(open_trades_label, fontsize=8.5, color=_VIZ['muted'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color(_VIZ['muted'])
        ax2.spines['right'].set_linewidth(0.8)
        ax2.tick_params(colors=_VIZ['muted'], labelsize=8, length=3)
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False,
                  fontsize=8.5, labelcolor=_VIZ['ink2'])
    else:
        ax.legend(loc='upper left', frameon=False, fontsize=8.5,
                  labelcolor=_VIZ['ink2'])


def _draw_drawdown(ax, book, unit: str) -> None:
    idx, eq = book.index, book['equity'].astype(float)
    dd = eq - eq.cummax()
    ax.fill_between(idx, 0, dd, color=_VIZ['red'], alpha=0.18, zorder=2)
    ax.plot(idx, dd, color=_VIZ['red'], linewidth=1.2, zorder=3)
    ax.axhline(0, color=_VIZ['baseline'], linewidth=0.8)
    _ylabel(ax, f'Drawdown{unit}')


def _draw_exposure(ax, book, col: str, color: str, label: str) -> None:
    if col in book.columns:
        ax.plot(book.index, book[col].astype(float), color=color,
                linewidth=1.6, zorder=3)
    ax.axhline(0, color=_VIZ['baseline'], linewidth=0.8, linestyle=(0, (4, 4)))
    _ylabel(ax, label)


def _draw_be_drift(ax, book, unit: str) -> None:
    be_series = [
        ('gamma_pnl_be_cum', _VIZ['blue'],    'gamma BE'),
        ('vanna_pnl_be_cum', _VIZ['magenta'], 'vanna BE'),
        ('volga_pnl_be_cum', _VIZ['yellow'],  'volga BE'),
    ]
    for col, color, lab in be_series:
        if col in book.columns:
            ax.plot(book.index, book[col].astype(float), color=color,
                    linewidth=1.6, zorder=3, label=lab)
    ax.axhline(0, color=_VIZ['baseline'], linewidth=0.8)
    _ylabel(ax, f'BE cum. P&L{unit}')
    ax.legend(loc='upper left', frameon=False, fontsize=8, ncol=3,
              labelcolor=_VIZ['ink2'])


def _draw_hist(ax, data, label: str) -> None:
    data = pd.Series(data, dtype=float).dropna()
    if data.empty:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                color=_VIZ['muted'], transform=ax.transAxes)
        return
    bins = min(40, max(8, int(len(data) ** 0.5) * 2))
    ax.hist(data, bins=bins, color=_VIZ['blue'], alpha=0.80,
            edgecolor=_VIZ['surface'], linewidth=0.5, zorder=3)
    cvar, worst = _cvar(data)
    ax.axvline(0, color=_VIZ['baseline'], linewidth=0.8, zorder=2)
    ax.axvline(cvar, color=_VIZ['red'], linewidth=1.4, linestyle=(0, (4, 3)),
               zorder=4, label=f'CVaR 5%: {cvar:,.0f}')
    ax.axvline(worst, color=_VIZ['red'], linewidth=1.4, zorder=4,
               label=f'worst: {worst:,.0f}')
    _ylabel(ax, 'count')
    ax.set_xlabel(label, fontsize=8.5, color=_VIZ['ink2'])
    ax.legend(loc='upper left', frameon=False, fontsize=8, labelcolor=_VIZ['ink2'])


def _trade_pnl(trade_log, settled_only: bool = True):
    tl = trade_log
    if tl is not None and 'live_trade' in tl.columns and settled_only:
        tl = tl[~tl['live_trade'].astype(bool)]
    return tl['net_pnl'] if (tl is not None and 'net_pnl' in tl.columns) else []


def _date_axis(ax) -> None:
    loc = _mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(_mdates.ConciseDateFormatter(loc))


def plot_equity_tearsheet(book: pd.DataFrame,
                          title: Optional[str] = None, ccy: Optional[str] = None,
                          n_worst: int = 3):
    """
    Stacked time-series tearsheet sharing one date axis:
      1. Equity curve  (+ top-`n_worst` worst-day markers)
      2. Drawdown      (underwater, shaded)
      3-5. Net exposures — delta residual, net gamma, net vega
      6. Breakeven greek drift (gamma/vanna/volga cumulative)

    Returns the matplotlib Figure (does not call plt.show()).
    """
    _require_mpl()
    if book is None or book.empty:
        raise ValueError("book is empty — nothing to plot.")
    unit = f" ({ccy})" if ccy else ""

    fig, axes = _plt.subplots(
        6, 1, figsize=(11, 12), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.5, 1, 1, 1, 2]},
    )
    fig.patch.set_facecolor(_VIZ['surface'])
    ax_eq, ax_dd, ax_delta, ax_gamma, ax_vega, ax_be = axes

    _draw_equity(ax_eq, book, unit, n_worst=n_worst)
    _draw_drawdown(ax_dd, book, unit)
    _draw_exposure(ax_delta, book, 'net_delta_approx', _GREEK_COLORS['delta'], 'Net Δ (resid)')
    _draw_exposure(ax_gamma, book, 'net_gamma_1pct',   _GREEK_COLORS['gamma'], 'Net Γ /1%')
    _draw_exposure(ax_vega,  book, 'net_vega_1vp',      _GREEK_COLORS['vega'],  'Net vega /1vol')
    _draw_be_drift(ax_be, book, unit)

    for ax in axes:
        _style_axis(ax, money=True)
    _date_axis(ax_be)   # date ticks only on the bottom panel

    head = 'Equity & risk tearsheet' + (f'  |  {title}' if title else '')
    fig.suptitle(head, fontsize=13, color=_VIZ['ink'], fontweight='bold',
                 x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_pnl_distribution(book: pd.DataFrame, trade_log: Optional[pd.DataFrame] = None,
                          title: Optional[str] = None, ccy: Optional[str] = None,
                          settled_only: bool = True):
    """
    Two histograms side by side: daily book P&L and per-trade P&L, each with its
    5% CVaR and worst-case marked. Returns the matplotlib Figure.
    """
    _require_mpl()
    unit = f" ({ccy})" if ccy else ""
    fig, (ax_d, ax_t) = _plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(_VIZ['surface'])

    _draw_hist(ax_d, book['pnl'] if (book is not None and not book.empty) else [],
               f'Daily P&L{unit}')
    _draw_hist(ax_t, _trade_pnl(trade_log, settled_only), f'Per-trade P&L{unit}')

    for ax in (ax_d, ax_t):
        _style_axis(ax)
        ax.xaxis.set_major_formatter(_FuncFormatter(lambda v, _: f"{v:,.0f}"))

    head = 'P&L distribution' + (f'  |  {title}' if title else '')
    fig.suptitle(head, fontsize=13, color=_VIZ['ink'], fontweight='bold',
                 x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_scorecard(trade_dfs: Optional[List[pd.DataFrame]] = None,
                   trade_log: Optional[pd.DataFrame] = None,
                   book: Optional[pd.DataFrame] = None,
                   pair: Optional[str] = None, title: Optional[str] = None,
                   ccy: Optional[str] = None, save_prefix: Optional[str] = None,
                   show: bool = False):
    """
    One-call visual: build (if needed) the daily book and render both the
    equity/risk tearsheet and the P&L distribution.

    Returns (fig_tearsheet, fig_distribution). Pass save_prefix to write
    "<prefix>_tearsheet.png" / "<prefix>_distribution.png"; show=True to display.
    """
    _require_mpl()
    if book is None:
        book = build_daily_book(trade_dfs, trade_log)
    if ccy is None and pair:
        ccy = pair[:3].upper()

    fig1 = plot_equity_tearsheet(book, title=title, ccy=ccy)
    fig2 = plot_pnl_distribution(book, trade_log, title=title, ccy=ccy)

    if save_prefix:
        fig1.savefig(f"{save_prefix}_tearsheet.png", dpi=150,
                     facecolor=_VIZ['surface'], bbox_inches='tight')
        fig2.savefig(f"{save_prefix}_distribution.png", dpi=150,
                     facecolor=_VIZ['surface'], bbox_inches='tight')
    if show:
        _plt.show()
    return fig1, fig2


def _render_lines(ax, items) -> None:
    """Lay out text rows evenly across the whole axis height (axis off).

    items: list of (text, fontsize, color, bold, weight). Empty text = spacer.
    Row vertical advance is weight/total_weight, so the block always fills the
    axis regardless of how many rows — no cramping, no dead space.
    """
    ax.axis('off')
    total = sum(w for *_, w in items) or 1.0
    y = 1.0
    for text, size, color, bold, w in items:
        dy = w / total
        if text:
            ax.text(0.0, y - dy * 0.5, text, transform=ax.transAxes,
                    va='center', ha='left', fontsize=size, color=color,
                    fontfamily='DejaVu Sans Mono',
                    fontweight='bold' if bold else 'normal')
        y -= dy


def _draw_metrics_panel(ax, sc: pd.Series, header: str) -> None:
    """Grouped scorecard scalars as an evenly-spaced monospace column."""
    items = [(header, 11.5, _VIZ['ink'], True, 1.7)]
    for gname, ks in _SCORECARD_GROUPS:
        present = [k for k in ks if k in sc.index]
        if not present:
            continue
        items.append(('', None, None, False, 0.5))                 # spacer
        items.append((gname.upper(), 9.0, _VIZ['ink'], True, 1.25))
        for k in present:
            lbl = _KEY_LABELS.get(k, k)
            items.append((f"{lbl:<21}{_fmt_cell(sc[k]):>13}", 8.3,
                          _VIZ['ink2'], False, 1.0))
    _render_lines(ax, items)


def _table_lines(df: pd.DataFrame, spec) -> list:
    """Aligned monospace rows for a worst-days/trades table.

    spec: list of (source_col | None-for-rank-index, header, kind) where kind is
    'str' (verbatim) or 'num' (compact k/M). Returns [header_line, *row_lines].
    """
    if df is None or df.empty:
        return ['(none)']
    headers = [h for _, h, _ in spec]
    body = []
    for i in df.index:
        cells = []
        for col, _h, kind in spec:
            if col is None:
                cells.append(str(i))
            else:
                v = df.at[i, col]
                if kind == 'num':
                    cells.append(_compact_num(float(v)) if pd.notna(v) else 'nan')
                else:
                    cells.append(str(v))
        body.append(cells)
    widths = [max(len(headers[j]), *(len(r[j]) for r in body))
              for j in range(len(spec))]
    fmt = lambda cells: ' '.join(c.rjust(widths[j]) for j, c in enumerate(cells))
    return [fmt(headers)] + [fmt(r) for r in body]


def _draw_worst_tables(ax, book: pd.DataFrame, trade_log: Optional[pd.DataFrame],
                       n: int = 3) -> None:
    """Compact worst-days (with BE attribution + greek exposures) and
    worst-trades tables, drawn as monospace text (axis off)."""
    ax.axis('off')
    wd = worst_days(book, n=n, include_exposures=True)
    wt = worst_trades(trade_log, n=n)

    day_spec = [
        (None, '#', 'str'), ('date', 'date', 'str'), ('pnl', 'pnl', 'num'),
        ('gamma_be_pnl', 'g_be', 'num'), ('vega_pnl', 'v_be', 'num'),
        ('vanna_be_pnl', 'va_be', 'num'), ('volga_be_pnl', 'vo_be', 'num'),
        ('gamma_exp', 'g_exp', 'num'), ('vega_exp', 'v_exp', 'num'),
        ('vanna_exp', 'va_exp', 'num'), ('volga_exp', 'vo_exp', 'num'),
    ]
    trade_spec = [
        (None, '#', 'str'), ('entry', 'entry', 'str'), ('exit', 'exit', 'str'),
        ('pnl', 'pnl', 'num'), ('gamma_be_pnl', 'g_be', 'num'),
        ('vega_pnl', 'v_be', 'num'), ('vanna_be_pnl', 'va_be', 'num'),
        ('volga_be_pnl', 'vo_be', 'num'),
    ]

    dlines = _table_lines(wd, day_spec)
    tlines = _table_lines(wt, trade_spec)
    items = [
        ('WORST 3 DAYS   (be = BE P&L attribution · exp = net greek exposure)',
         9.0, _VIZ['ink'], True, 1.3),
        (dlines[0], 8.3, _VIZ['muted'], True, 1.0),
    ]
    items += [(r, 8.3, _VIZ['ink2'], False, 1.0) for r in dlines[1:]]
    items += [
        ('', None, None, False, 0.7),
        ('WORST 3 TRADES   (be = BE P&L attribution)', 9.0, _VIZ['ink'], True, 1.3),
        (tlines[0], 8.3, _VIZ['muted'], True, 1.0),
    ]
    items += [(r, 8.3, _VIZ['ink2'], False, 1.0) for r in tlines[1:]]
    _render_lines(ax, items)


def plot_full_report(trade_dfs: Optional[List[pd.DataFrame]] = None,
                     trade_log: Optional[pd.DataFrame] = None,
                     book: Optional[pd.DataFrame] = None,
                     sc: Optional[pd.Series] = None,
                     pair: Optional[str] = None, title: Optional[str] = None,
                     ccy: Optional[str] = None, name: Optional[str] = None,
                     save_path: Optional[str] = None, show: bool = False):
    """
    ONE-SCREEN report combining everything for sharing:
      - top-left sidebar: full scorecard metrics (grouped)
      - bottom-left: worst-3-days (BE P&L attribution + net greek exposures)
        and worst-3-trades tables
      - equity curve (+ worst-day markers)
      - net exposures (delta residual / gamma / vega)
      - breakeven greek drift
      - daily & per-trade P&L distributions

    Returns a single matplotlib Figure. Pass save_path='report.png' to write it;
    show=True to display.

    CURRENCY: this converts nothing — it renders the `book` and `trade_log` you hand
    it, and `ccy` is only a label. The equity curve / exposures / daily histogram read
    the book, while the per-trade histogram and worst-trades table read the log, so if
    you pass a to_usd_book() book you must pass a to_usd_trade_log() log with it (or
    neither) — otherwise one page carries two currencies. `ccy=pair[:3]` is the
    base-ccy default and matches passing neither.
    """
    _require_mpl()
    if book is None:
        book = build_daily_book(trade_dfs, trade_log)
    if book is None or book.empty:
        raise ValueError("book is empty — nothing to plot.")
    if sc is None:
        sc = scorecard(book, trade_log, name=name)
    if ccy is None and pair:
        ccy = pair[:3].upper()
    unit = f" ({ccy})" if ccy else ""

    fig = _plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(_VIZ['surface'])

    # left column: metrics (top) + worst-days/trades tables (bottom)
    ax_side = fig.add_axes([0.015, 0.37, 0.28, 0.56])
    _draw_metrics_panel(ax_side, sc, 'SCORECARD')
    ax_worst = fig.add_axes([0.015, 0.035, 0.30, 0.26])
    _draw_worst_tables(ax_worst, book, trade_log, n=3)

    # right — chart grid (drawdown removed; 3 rows)
    gs = fig.add_gridspec(3, 9, left=0.355, right=0.985, top=0.90, bottom=0.07,
                          height_ratios=[2.4, 1.4, 1.5],
                          hspace=0.42, wspace=1.0)
    ax_eq = fig.add_subplot(gs[0, :])
    ax_delta = fig.add_subplot(gs[1, 0:3])
    ax_gamma = fig.add_subplot(gs[1, 3:6])
    ax_vega  = fig.add_subplot(gs[1, 6:9])
    ax_be = fig.add_subplot(gs[2, 0:3])
    ax_hd = fig.add_subplot(gs[2, 3:6])
    ax_ht = fig.add_subplot(gs[2, 6:9])

    _draw_equity(ax_eq, book, unit, show_open_trades=True,
                 open_trades_label='# open (short) trades')
    _draw_exposure(ax_delta, book, 'net_delta_approx', _GREEK_COLORS['delta'], 'Net Δ (resid)')
    _draw_exposure(ax_gamma, book, 'net_gamma_1pct',   _GREEK_COLORS['gamma'], 'Net Γ /1%')
    _draw_exposure(ax_vega,  book, 'net_vega_1vp',      _GREEK_COLORS['vega'],  'Net vega /1vol')
    _draw_be_drift(ax_be, book, unit)
    _draw_hist(ax_hd, book['pnl'], f'Daily P&L{unit}')
    _draw_hist(ax_ht, _trade_pnl(trade_log), f'Per-trade P&L{unit}')

    for ax in (ax_eq, ax_delta, ax_gamma, ax_vega, ax_be):
        _style_axis(ax, money=True, compact=True)
    for ax in (ax_hd, ax_ht):
        _style_axis(ax)
        ax.xaxis.set_major_formatter(_FuncFormatter(lambda v, _: _compact_num(v)))

    for ax in (ax_eq, ax_delta, ax_gamma, ax_vega, ax_be):
        _date_axis(ax)

    head = 'Backtest report' + (f'  |  {title}' if title else '')
    fig.suptitle(head, fontsize=15, color=_VIZ['ink'], fontweight='bold',
                 x=0.015, ha='left', y=0.965)

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=_VIZ['surface'], bbox_inches='tight')
    if show:
        _plt.show()
    return fig





