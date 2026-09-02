import pdblp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xbbg import blp
from datetime import datetime, timedelta
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings



# Data Pulling IV and HV
def get_Vols(ccys, I_tenors, H_tenors, years):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    df_vols = {}
    for ccy in ccys:
        if I_tenors:
            for tenor in I_tenors:
                ticker_IV = f"{ccy}V{tenor} BGN Curncy"
                data_IV = blp.bdh(
                    tickers=[ticker_IV],
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date)
                if not data_IV.empty:
                    col_name = f"{ccy}_IV_{tenor}"
                    data_IV.columns = [col_name]
                    df_vols[col_name] = data_IV
                else:
                    print(f"No data for {ticker_IV}, skipping.")
        if H_tenors:
            for tenor in H_tenors:
                ticker_HV = f"{ccy}H{tenor} BGN Curncy"
                data_HV = blp.bdh(
                    tickers=[ticker_HV],
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date
                )
                if not data_HV.empty:
                    col_name = f"{ccy}_HV_{tenor}"
                    data_HV.columns = [col_name]
                    df_vols[col_name] = data_HV
                else:
                    print(f"No data for {ticker_HV}, skipping.")
    if df_vols:
        df_vols_all = pd.concat(df_vols.values(), axis=1)
        return df_vols_all
    else:
        print("No data retrieved for any ticker.")
        return pd.DataFrame()
    
# IV PERCENTILES  
def add_IV_rolling_percentiles(df):
    df_with_pct = df.copy()
    iv_1m_col = [col for col in df.columns if col.endswith('_IV_1M')][0]
    ccy_pair = iv_1m_col.replace('_IV_1M', '')
    def rolling_percentile(series):
        if len(series) < 2:
            return np.nan
        return (series < series.iloc[-1]).sum() / len(series) * 100
    window_3m = 63
    pct_3m_col_name = f"{ccy_pair}_IV_1M_3mPct"
    df_with_pct[pct_3m_col_name] = df[iv_1m_col].rolling(
        window=window_3m
    ).apply(rolling_percentile, raw=False)
    window_1y = 252 * 3
    pct_1y_col_name = f"{ccy_pair}_IV_1M_3yPct"
    df_with_pct[pct_1y_col_name] = df[iv_1m_col].rolling(
        window=window_1y
    ).apply(rolling_percentile, raw=False)
    return df_with_pct


# VOL OF VOL AND PERCENTILES
def add_VoV_and_percentile(df):
    realized_window=40
    percentile_years=3
    df_with_rv = df.copy()
    iv_1m_cols = [col for col in df.columns if col.endswith('_IV_1M')]
    for iv_col in iv_1m_cols:
        ccy = iv_col.replace('_IV_1M', '')
        log_returns = np.log(df[iv_col] / df[iv_col].shift(1))
        rv_col_name = f'{ccy}_IV_1M_RV{realized_window}D'
        df_with_rv[rv_col_name] = log_returns.rolling(
            window=realized_window,
            min_periods=realized_window
        ).std() * np.sqrt(252) * 100 
        pct_window = percentile_years * 252
        pct_col_name = f'{rv_col_name}_pct'
        def rolling_percentile(series):
            if len(series) < 2:
                return np.nan
            return (series < series.iloc[-1]).sum() / len(series) * 100
        df_with_rv[pct_col_name] = df_with_rv[rv_col_name].rolling(
            window=pct_window,
            min_periods=252
        ).apply(rolling_percentile, raw=False)
    return df_with_rv


# TERM STRUCTURE SPREADS (1W-1M, 1M-3M, 3M-1Y) AND PERCENTILES
def add_vol_Termspreads_and_percentiles(df, smooth_days=5, percentile_years=2):
    df_with_spreads = df.copy()
    ccys_in_df = list(set([col.split('_IV_')[0] for col in df.columns if '_IV_' in col]))
    for ccy in ccys_in_df:
        spread_pairs = [
            ('1W', '1M', f'{ccy}_IV_1W', f'{ccy}_IV_1M'),
            ('1M', '3M', f'{ccy}_IV_1M', f'{ccy}_IV_3M'),
            ('3M', '1Y', f'{ccy}_IV_3M', f'{ccy}_IV_1Y')]
        for short_tenor, long_tenor, short_col, long_col in spread_pairs:
            if short_col in df.columns and long_col in df.columns:
                spread_name = f'{ccy}_spread_{short_tenor}_{long_tenor}'
                df_with_spreads[spread_name] = df[short_col] - df[long_col]
                smoothed_name = f'{spread_name}_smooth'
                df_with_spreads[smoothed_name] = df_with_spreads[spread_name].rolling(
                    window=smooth_days,
                    min_periods=1).median()
                window = percentile_years * 252
                pct_name = f'{smoothed_name}_pct'
                def rolling_percentile(series):
                    if len(series) < 2:
                        return np.nan
                    return (series < series.iloc[-1]).sum() / len(series) * 100
                df_with_spreads[pct_name] = df_with_spreads[smoothed_name].rolling(
                    window=window,
                    min_periods=252
                ).apply(rolling_percentile, raw=False)
    return df_with_spreads

# IMPLIED - REALIZED AND PERCENTILES
def add_iv_hv_spreads_and_percentiles(df, percentile_years=2):
    df_with_iv_hv = df.copy()
    ccys_in_df = list(set([col.split('_IV_')[0] for col in df.columns if '_IV_' in col]))
    for ccy in ccys_in_df:
        iv_hv_pairs = [
            ('1W', f'{ccy}_IV_1W', f'{ccy}_HV_1W'),
            ('1M', f'{ccy}_IV_1M', f'{ccy}_HV_1M')]
        for tenor, iv_col, hv_col in iv_hv_pairs:
            if iv_col in df.columns and hv_col in df.columns:
                spread_name = f'{ccy}_IV_HV_spread_{tenor}'
                df_with_iv_hv[spread_name] = df[iv_col] - df[hv_col]
                window = percentile_years * 252
                pct_name = f'{spread_name}_pct'
                def rolling_percentile(series):
                    if len(series) < 2:
                        return np.nan
                    return (series < series.iloc[-1]).sum() / len(series) * 100
                df_with_iv_hv[pct_name] = df_with_iv_hv[spread_name].rolling(
                    window=window,
                    min_periods=252
                ).apply(rolling_percentile, raw=False)
    return df_with_iv_hv
# DIRECTION OF IMPLIED VOL AND ITS PERCENTILES
def add_iv_momentum_and_percentiles(df, short_window=5, long_window=20, percentile_years=2):
    df_with_mom = df.copy()
    iv_1m_cols = [col for col in df.columns if col.endswith('_IV_1M')]
    for iv_col in iv_1m_cols:
        ccy = iv_col.replace('_IV_1M', '')
        short_mom_name = f'{ccy}_IV_1M_mom{short_window}D'
        df_with_mom[short_mom_name] = df[iv_col] - df[iv_col].shift(short_window)
        long_mom_name = f'{ccy}_IV_1M_mom{long_window}D'
        df_with_mom[long_mom_name] = df[iv_col] - df[iv_col].shift(long_window)
        window = percentile_years * 252
        def rolling_percentile(series):
            if len(series) < 2:
                return np.nan
            return (series < series.iloc[-1]).sum() / len(series) * 100
        short_pct_name = f'{short_mom_name}_pct'
        df_with_mom[short_pct_name] = df_with_mom[short_mom_name].rolling(
            window=window,
            min_periods=252
        ).apply(rolling_percentile, raw=False)
        long_pct_name = f'{long_mom_name}_pct'
        df_with_mom[long_pct_name] = df_with_mom[long_mom_name].rolling(
            window=window,
            min_periods=252
        ).apply(rolling_percentile, raw=False)
    return df_with_mom




def classify_regimes(df, 
                     iv_high=70, iv_low=30,
                     vov_high=70, vov_low=30,
                     mom_high=70, mom_low=30,
                     ts_inverted=30,
                     ivhv_rich=70):
    df_regimes = df.copy()
    
    # Extract key percentile columns
    iv_pct = 'EURUSD_IV_1M_pct'
    vov_pct = 'EURUSD_IV_1M_RV40D_pct'
    mom_short_pct = 'EURUSD_IV_1M_mom5D_pct'
    mom_long_pct = 'EURUSD_IV_1M_mom20D_pct'
    ts_1w1m_pct = 'EURUSD_spread_1W_1M_smooth_pct'
    ivhv_1m_pct = 'EURUSD_IV_HV_spread_1M_pct'
    
    # Primary regime classification
    conditions = [
        # Crisis/Panic: High IV + High VoV
        (df[iv_pct] >= iv_high) & (df[vov_pct] >= vov_high),
        # Elevated: High IV + Low VoV (stable high vol)
        (df[iv_pct] >= iv_high) & (df[vov_pct] < vov_high),
        # Breakout: Low IV + High VoV (vol breaking out)
        (df[iv_pct] < iv_low) & (df[vov_pct] >= vov_high),
        
        # Calm: Low IV + Low VoV
        (df[iv_pct] < iv_low) & (df[vov_pct] < vov_low),
    ]
    
    choices = ['Crisis', 'Elevated', 'Breakout', 'Calm']
    
    df_regimes['regime_primary'] = np.select(conditions, choices, default='Neutral')
    
    # Secondary characteristics (binary flags)
    df_regimes['regime_momentum'] = np.where(
        df[mom_long_pct] >= mom_high, 'Rising',
        np.where(df[mom_long_pct] <= mom_low, 'Falling', 'Stable')
    )
    
    df_regimes['regime_term_structure'] = np.where(
        df[ts_1w1m_pct] <= ts_inverted, 'Inverted',
        'Normal'
    )
    
    df_regimes['regime_iv_richness'] = np.where(
        df[ivhv_1m_pct] >= ivhv_rich, 'Rich',
        np.where(df[ivhv_1m_pct] <= (100 - ivhv_rich), 'Cheap', 'Fair'))

    df_regimes['regime_full'] = (
        df_regimes['regime_primary'] + '_' + 
        df_regimes['regime_momentum'] + '_' + 
        df_regimes['regime_term_structure'])
    
    # Numeric regime score (-3 to +3 scale)
    regime_score = np.zeros(len(df))
    # IV level contribution
    regime_score += np.where(df[iv_pct] >= iv_high, 1, 
                   np.where(df[iv_pct] <= iv_low, -1, 0))
    # VoV contribution
    regime_score += np.where(df[vov_pct] >= vov_high, 1,
                   np.where(df[vov_pct] <= vov_low, -1, 0))
    # Momentum contribution  
    regime_score += np.where(df[mom_long_pct] >= mom_high, 1,
                   np.where(df[mom_long_pct] <= mom_low, -1, 0))
    df_regimes['regime_score'] = regime_score
    return df_regimes






def plot_indicators(df, indicators, years=None, figsize=(18, 8)):
    if years is not None:
        cutoff_date = df.index[-1] - pd.Timedelta(days=years * 365)
        df_plot = df[df.index >= cutoff_date].copy()
    else:
        df_plot = df.copy()
    fig, ax1 = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(indicators)))
    percentile_indicators = [ind for ind in indicators if 'pct' in ind.lower() or 'Pct' in ind]
    absolute_indicators = [ind for ind in indicators if ind not in percentile_indicators]
    lines = []
    labels = []
    for i, indicator in enumerate(absolute_indicators):
        if indicator in df_plot.columns:
            line = ax1.plot(df_plot.index, df_plot[indicator], 
                          label=indicator, 
                          linewidth=2, 
                          color=colors[i],
                          alpha=0.8)
            lines.extend(line)
            labels.append(indicator)
        else:
            print(f"Warning: {indicator} not found in dataframe")
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Volatility / Absolute Values', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3, linestyle='--')
    if percentile_indicators:
        ax2 = ax1.twinx()
        for i, indicator in enumerate(percentile_indicators):
            if indicator in df_plot.columns:
                line = ax2.plot(df_plot.index, df_plot[indicator], 
                              label=indicator, 
                              linewidth=2, 
                              color=colors[len(absolute_indicators) + i],
                              alpha=0.8)
                lines.extend(line)
                labels.append(indicator)
            else:
                print(f"Warning: {indicator} not found in dataframe")
        ax2.set_ylabel('Percentiles (%)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y')
        ax2.set_ylim(0, 100)
        # Add horizontal reference lines for percentiles
        ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax2.axhline(y=75, color='red', linestyle=':', alpha=0.3, linewidth=1)
        ax2.axhline(y=25, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax1.legend(lines, labels, loc='best', fontsize=10, framealpha=0.9)
    plt.title('Volatility Indicators Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()











# indicators = [
#     f'{ccys[0]}_IV_1M',
#     # f'{ccys[0]}_IV_1M_3mPct',
#     # f'{ccys[0]}_IV_1M_3yPct',
#     f'{ccys[0]}_IV_1M_RV40D_pct'
#     ]

# plot_indicators(df, indicators, years=2)


















# ccys = ['EURUSD']
# I_tenors = ['1W', '1M', '3M', '1Y']
# H_tenors = ['1W', '1M']
# Data_years = 5

# IV_Percentile_LookBack = 3

# TSpread_smoothDays = 5
# TSpread_Percentile_LookBack = 2

# VoV_Lookback = 40
# VoV_Percentile_LookBack = 3

# IVHV_Percentile_LookBack = 2 

# Momentum_Short = 5
# Momentum_Long = 20
# Momentum_Percentile_LookBack = 2


# Regime_IV_High = 70
# Regime_IV_Low = 30

# Regime_VoV_High = 70
# Regime_VoV_Low = 30

# Regime_Mom_High = 70
# Regime_Mom_Low = 30

# Regime_TS_Inverted = 30
# Regime_IVHV_Rich = 70


# df = get_Vols(ccys, I_tenors, H_tenors, Data_years)
# df = add_rolling_percentiles(df, IV_Percentile_LookBack)
# df = add_vol_spreads_and_percentiles(df, TSpread_smoothDays, TSpread_Percentile_LookBack)
# df = add_iv_realized_vol_and_percentile(df, VoV_Lookback, VoV_Percentile_LookBack)
# df = add_iv_hv_spreads_and_percentiles(df, IVHV_Percentile_LookBack)
# df = add_iv_momentum_and_percentiles(df, Momentum_Short, Momentum_Long, Momentum_Percentile_LookBack)

# df = classify_regimes(df, 
#                      iv_high=Regime_IV_High, 
#                      iv_low=Regime_IV_Low,
#                      vov_high=Regime_VoV_High, 
#                      vov_low=Regime_VoV_Low,
#                      mom_high=Regime_Mom_High, 
#                      mom_low=Regime_Mom_Low,
#                      ts_inverted=Regime_TS_Inverted,
#                      ivhv_rich=Regime_IVHV_Rich)
# print(df.tail(5).T)
# indicators = [
#     'EURUSD_IV_1M',
#     'EURUSD_IV_1M_pct',
#     'EURUSD_IV_1M_RV40D_pct',
#     ]

# plot_regime_indicators(df, indicators, years_back=2)








# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------







def backtest_vov_peak_signal(df, ccy, 
                              lookback_days=5,        # Days to confirm peak
                              min_peak_pct=75,        # Minimum VoV pct to consider
                              min_prominence=10,      # How much higher than surrounding points
                              holding_period_days=5):
    # Column names
    rv_pct_col = f'{ccy}_IV_1M_RV40D_pct'
    entry_vol_col = f'{ccy}_IV_1M'
    exit_vol_col = f'{ccy}_IV_3W'
    
    # Check columns
    required_cols = [rv_pct_col, entry_vol_col, exit_vol_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df_backtest = df.copy()
    df_backtest = df_backtest.dropna(subset=[rv_pct_col, entry_vol_col, exit_vol_col])
    
    # Calculate rate of change indicators
    df_backtest['vov_change_1d'] = df_backtest[rv_pct_col].diff(1)  # Daily change
    df_backtest['vov_change_3d'] = df_backtest[rv_pct_col].diff(3)  # 3-day change
    df_backtest[f'vov_lookback_{lookback_days}d'] = df_backtest[rv_pct_col].shift(lookback_days)
    
    trades = []
    last_entry_date = None
    min_days_between_trades = 10  # Minimum days between signals to avoid clustering
    
    for i in range(lookback_days + 3, len(df_backtest)):
        entry_date = df_backtest.index[i]
        
        # Skip if too soon after last trade
        if last_entry_date and (entry_date - last_entry_date).days < min_days_between_trades:
            continue
        
        current_vov_pct = df_backtest[rv_pct_col].iloc[i]
        vov_1d_change = df_backtest['vov_change_1d'].iloc[i]
        vov_3d_change = df_backtest['vov_change_3d'].iloc[i]
        vov_lookback = df_backtest[f'vov_lookback_{lookback_days}d'].iloc[i]
        
        # Peak detection criteria (NO LOOKAHEAD):
        # 1. Current VoV is high enough
        high_enough = current_vov_pct >= min_peak_pct
        
        # 2. VoV has risen significantly from lookback period
        risen_significantly = (current_vov_pct - vov_lookback) >= min_prominence
        
        # 3. VoV is now turning down (negative 1-day change)
        turning_down = vov_1d_change < 0
        
        # 4. Additional filter: 3-day momentum was positive (confirms we were rising)
        was_rising = vov_3d_change > 0
        
        # Combine all criteria
        peak_signal = high_enough and risen_significantly and turning_down and was_rising
        
        if peak_signal:
            # Find exit date
            future_dates = df_backtest.index[df_backtest.index > entry_date]
            
            if len(future_dates) < holding_period_days:
                continue
            
            exit_date = future_dates[holding_period_days - 1]
            
            # Get trade values
            entry_rv_pct = current_vov_pct
            entry_vol = df_backtest.loc[entry_date, entry_vol_col]
            exit_vol = df_backtest.loc[exit_date, exit_vol_col]
            
            # Calculate P&L
            vol_change = exit_vol - entry_vol
            pnl = entry_vol - exit_vol
            pnl_pct = (pnl / entry_vol) * 100
            
            # Store trade
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'holding_days': (exit_date - entry_date).days,
                'entry_rv_pct': entry_rv_pct,
                'vov_from_lookback': current_vov_pct - vov_lookback,
                'vov_1d_change': vov_1d_change,
                'entry_vol': entry_vol,
                'exit_vol': exit_vol,
                'vol_change': vol_change,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            
            last_entry_date = entry_date
            print(f"Peak signal on {entry_date.strftime('%Y-%m-%d')}: "
                  f"VoV={entry_rv_pct:.1f}%, 1d chg={vov_1d_change:.1f}pp, "
                  f"{lookback_days}d rise={current_vov_pct - vov_lookback:.1f}pp")
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)
    
    return trades_df


# # Run the backtest
# trades_df = backtest_vov_peak_signal(
#     df, 
#     ccy='EURUSD', 
#     lookback_days=5,         # Look back 5 days to confirm elevation
#     min_peak_pct=80,         # Only enter if VoV >= 75%
#     min_prominence=15,       # Must be 15pp higher than 5 days ago
#     holding_period_days=5
# )
























# # Enhanced visualization with VoV percentile on bottom plot
# if len(trades_df) > 0:
#     fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
#     # Plot 1: Cumulative P&L (left axis) + Individual Trade P&L (right axis)
#     ax1 = axes[0]
#     ax2 = ax1.twinx()
    
#     # Cumulative P&L on left axis
#     trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
#     line1 = ax1.plot(trades_df['entry_date'], trades_df['cumulative_pnl'], 
#                      linewidth=2.5, color='darkblue', label='Cumulative P&L', zorder=3)
#     ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
#     ax1.fill_between(trades_df['entry_date'], 0, trades_df['cumulative_pnl'], 
#                       where=trades_df['cumulative_pnl']>=0, alpha=0.2, color='green')
#     ax1.fill_between(trades_df['entry_date'], 0, trades_df['cumulative_pnl'], 
#                       where=trades_df['cumulative_pnl']<0, alpha=0.2, color='red')
    
#     # Individual trade P&L on right axis (as bars)
#     colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
#     bar1 = ax2.bar(trades_df['entry_date'], trades_df['pnl'], 
#                    color=colors, alpha=0.5, edgecolor='black', linewidth=0.5,
#                    label='Individual Trade P&L', width=3, zorder=1)
#     ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
#     # Labels and formatting
#     ax1.set_title('Cumulative P&L & Individual Trade P&L', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('Entry Date', fontsize=11)
#     ax1.set_ylabel('Cumulative P&L (vol points)', fontsize=11, color='darkblue')
#     ax1.tick_params(axis='y', labelcolor='darkblue')
#     ax2.set_ylabel('Individual Trade P&L (vol points)', fontsize=11, color='black')
#     ax2.tick_params(axis='y', labelcolor='black')
    
#     # Combine legends
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2: Vol Levels (left axis) + VoV Percentile (right axis)
#     ax3 = axes[1]
#     ax4 = ax3.twinx()
    
#     entry_vol_col = 'EURUSD_IV_1M'
#     exit_vol_col = 'EURUSD_IV_3W'
#     rv_pct_col = 'EURUSD_IV_1M_RV40D_pct'
    
#     # Left axis: Volatility levels
#     # Plot the full time series
#     line2 = ax3.plot(df.index, df[entry_vol_col], linewidth=1.5, 
#                      color='darkblue', label='1M Vol (Entry Level)', alpha=0.7, zorder=2)
    
#     # For each trade, plot the exit vol level at the exit date
#     for _, trade in trades_df.iterrows():
#         exit_date = trade['exit_date']
#         if exit_date in df.index:
#             # Plot a single point for the 3W vol at exit
#             ax3.scatter(exit_date, trade['exit_vol'], 
#                        color='orange', s=60, zorder=5, alpha=0.8)
    
#     # Create a line connecting exit points for visualization
#     line3 = ax3.plot(trades_df['exit_date'], trades_df['exit_vol'], 
#                      linewidth=1.5, color='orange', label='3W Vol (Exit Level)', 
#                      linestyle='--', marker='o', markersize=5, alpha=0.7, zorder=3)
    
#     # Mark entry points on the 1M vol line
#     scatter1 = ax3.scatter(trades_df['entry_date'], trades_df['entry_vol'], 
#                           color='darkgreen', s=100, marker='v', zorder=6, 
#                           label='Entry (Sell 1M Vol)', edgecolors='black', linewidth=1.5)
    
#     # Connect entry and exit points for each trade
#     for _, trade in trades_df.iterrows():
#         ax3.plot([trade['entry_date'], trade['exit_date']], 
#                 [trade['entry_vol'], trade['exit_vol']], 
#                 color='gray', linestyle=':', linewidth=1, alpha=0.4, zorder=1)
    
#     # Right axis: VoV Percentile
#     line4 = ax4.plot(df.index, df[rv_pct_col], linewidth=1.5, 
#                      color='cyan', label='VoV Percentile', alpha=0.6, zorder=1)
    
#     # Mark VoV percentile at entry points
#     scatter2 = ax4.scatter(trades_df['entry_date'], trades_df['entry_rv_pct'], 
#                           color='red', s=80, marker='s', zorder=7, 
#                           label='VoV % at Entry', edgecolors='darkred', linewidth=1.5, alpha=0.8)
    
#     # Add reference lines for VoV percentile
#     ax4.axhline(y=75, color='orange', linestyle='--', alpha=0.4, linewidth=1, label='VoV 75%')
#     ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1, label='VoV 50%')
    
#     # Labels and formatting
#     ax3.set_title('Volatility Levels: Entry (1M) vs Exit (3W) with VoV Percentile', 
#                   fontsize=14, fontweight='bold')
#     ax3.set_xlabel('Date', fontsize=11)
#     ax3.set_ylabel('Implied Volatility (%)', fontsize=11, color='darkblue')
#     ax3.tick_params(axis='y', labelcolor='darkblue')
#     ax4.set_ylabel('VoV Percentile (%)', fontsize=11, color='cyan')
#     ax4.tick_params(axis='y', labelcolor='cyan')
#     ax4.set_ylim(0, 105)  # Set VoV percentile range from 0 to 100%
    
#     # Combine legends
#     lines3, labels3 = ax3.get_legend_handles_labels()
#     lines4, labels4 = ax4.get_legend_handles_labels()
#     ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=9)
#     ax3.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Display trade details with VoV percentile
#     print("\nRecent Trades:")
#     display_cols = ['entry_date', 'exit_date', 'entry_rv_pct', 'entry_vol', 
#                     'exit_vol', 'vol_change', 'pnl', 'pnl_pct']
#     print(trades_df[display_cols].tail(10).to_string(index=False))


















