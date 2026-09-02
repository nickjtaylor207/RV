from EUR_GBP_USD_Factors import get_Vols, get_Vol_Spreads, USD_GlobalFactors, EUR_Factors, GBP_Factors
from EUR_GBP_USD_Factors import  print_dataset_structure, filter_by_period, generate_monthly_periods, generate_rolling_3month_periods, generate_quarterly_periods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings


# --------------------------------------------------------------------------------------------------
# ---------- Lagged Correlation Analysis Breakdown by Time Period of Choice ------------------------ 
# --------------------------------------------------------------------------------------------------

def compute_ccf(vol_series, factor_series, max_lag=10):
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Negative lag: factor leads vol
            corr = factor_series.iloc[:lag].corr(vol_series.iloc[-lag:])
        elif lag > 0:
            # Positive lag: vol leads factor
            corr = factor_series.iloc[:-lag].corr(vol_series.iloc[lag:])
        else:
            corr = factor_series.corr(vol_series)
        correlations[lag] = corr
    return pd.Series(correlations)
def analyze_optimalLagCorr_multiFactor(df, col_interest, factors, periods, max_lag=10, 
                                 show_dates=True, verbose=False):
    df = df.copy()
    df[f'{col_interest}_Vol_Chg'] = df[col_interest].diff()
    results = []
    if verbose:
        print("\n" + "="*80)
        print("DATE MATCHING")
        print("="*80)
    for period in periods:
        try:
            df_period = filter_by_period(df, period, verbose=verbose)
        except Exception as e:
            print(f"⚠️  Error parsing period '{period}': {e}")
            continue
        if len(df_period) == 0:
            if not verbose: 
                print(f"⚠️  No data for period {period}, skipping...")
            continue
        row = {'Period': period}
        if show_dates:
            row['Start_Date'] = df_period.index.min().strftime('%Y-%m-%d')
            row['End_Date'] = df_period.index.max().strftime('%Y-%m-%d')
            row['N_Days'] = len(df_period)
        for factor in factors:
            if factor in df_period.columns:
                ccf = compute_ccf(
                    df_period[f'{col_interest}_Vol_Chg'], 
                    df_period[factor], 
                    max_lag=max_lag)
                best_lag = ccf.abs().idxmax()
                best_corr = ccf.loc[best_lag]
                row[factor] = f"{int(best_lag):+3d} ({best_corr:+.3f})"
        results.append(row)
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------------------------------
# -------------------Rolling Corr Time Series and Plot ----------------------------------------------
# ---------------------------------------------------------------------------------------------------


def rolling_ccf(vol_series, factor_series, window=63, target_lag=0):
    rolling_corr = pd.Series(index=vol_series.index, dtype=float)
    for i in range(window, len(vol_series)):
        vol_win = vol_series.iloc[i-window:i]
        if target_lag > 0:
            # Factor leads: use factor from earlier in the window
            factor_win = factor_series.iloc[i-window-target_lag:i-target_lag]
        elif target_lag < 0:
            # Vol leads: use vol from earlier
            factor_win = factor_series.iloc[i-window-target_lag:i-target_lag]
        else:
            factor_win = factor_series.iloc[i-window:i]
        if len(vol_win) == len(factor_win) and len(vol_win) > 10:
            rolling_corr.iloc[i] = vol_win.corr(factor_win)
    return rolling_corr

def plot_rolling_correlations(rolling_contemp_df, target_col, figsize=(16, 10)):
    usd_base = ['USGG2YR', 'USGG10YR', '2s10s', 'MOVE', 'SPX', 'VIX']
    eur_base = ['EESWE1', 'EESWE2', 'EUR_OIS_1Y2Y_Slope', 
                'BTP_Bund_10Y_Spread', 'GBTPGR10', 'GDBR10', 'SX5E', 'V2X']
    usd_factors = []
    eur_factors = []
    for col in rolling_contemp_df.columns:
        if any(base in col for base in usd_base):
            usd_factors.append(col)
        elif any(base in col for base in eur_base):
            eur_factors.append(col)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for col in usd_factors:
        label = col
        axes[0].plot(rolling_contemp_df.index, rolling_contemp_df[col], 
                    label=label, linewidth=2, alpha=0.8)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    axes[0].set_title(f'Rolling 63d Correlation: {target_col} vs USD/Global Factors', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Correlation', fontsize=12)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1, 1)
    for col in eur_factors:
        label = col
        axes[1].plot(rolling_contemp_df.index, rolling_contemp_df[col], 
                    label=label, linewidth=2, alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    axes[1].set_title(f'Rolling 63d Correlation: {target_col} vs EUR Factors', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Correlation', fontsize=12)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1, 1)
    plt.tight_layout()
    plt.show()


def analyze_rolling_correlations(df, col_interest, factors, window=63, target_lag=0, 
                                 plot=True, figsize=(16, 10), date_range=None):
    print("\n" + "="*80)
    print(f"ROLLING CORRELATION ANALYSIS ({window}d window, lag={target_lag})")
    print("="*80)
    print(f"Target: {col_interest}")
    print(f"Factors: {len(factors)}")
    df = df.copy()
    if date_range is not None:
        start_date, end_date = date_range
        df = df.loc[start_date:end_date]
        print(f"Date range (filtered): {df.index.min().date()} to {df.index.max().date()}")
    else:
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    df[f'{col_interest}_Vol_Chg'] = df[col_interest].diff()
    rolling_corrs = {}
    for factor in factors:
        if factor in df.columns:
            rolling_corrs[factor] = rolling_ccf(
                df[f'{col_interest}_Vol_Chg'], 
                df[factor], 
                window=window, 
                target_lag=target_lag)
        else:
            print(f"⚠️  Warning: {factor} not found in dataset, skipping...")
    rolling_df = pd.DataFrame(rolling_corrs)
    print(f"\nRolling correlations computed for {len(rolling_corrs)} factors")
    print(f"Valid observations per factor:")
    for factor in rolling_corrs.keys():
        n_valid = rolling_df[factor].notna().sum()
        print(f"  {factor:40s} {n_valid:5d} observations")
    if plot:
        plot_rolling_correlations(rolling_df, col_interest, figsize=figsize)
    return rolling_df


# ---------------------------------------------------------------------------------------------------
# ------------------ Asymetric Correlation Breakdown ------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def analyze_asymmetric_correlations_by_period(df, col_interest, factors, periods, verbose=False):
    df = df.copy()
    df[f'{col_interest}_Vol_Chg'] = df[col_interest].diff()
    results_by_period = {}
    for period in periods:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Period: {period}")
            print(f"{'='*80}")
        try:
            df_period = filter_by_period(df, period, verbose=False)
        except Exception as e:
            print(f"⚠️  Error parsing period '{period}': {e}")
            continue
        if len(df_period) == 0:
            print(f"⚠️  No data for period {period}, skipping...")
            continue
        if verbose:
            print(f"Date range: {df_period.index.min().date()} to {df_period.index.max().date()}")
            print(f"Observations: {len(df_period)}")
        asym_results = {}
        for factor in factors:
            if factor in df_period.columns:
                asym_results[factor] = asymmetric_correlation(
                    df_period[f'{col_interest}_Vol_Chg'], 
                    df_period[factor]
                )
        asym_df = pd.DataFrame(asym_results).T
        if verbose:
            print("\nAsymmetric Correlation Results:")
            print(asym_df.round(3))
        results_by_period[period] = asym_df
    return results_by_period


def asymmetric_correlation(vol_chg, factor_chg):
    valid_mask = ~(vol_chg.isna() | factor_chg.isna())
    vol_chg = vol_chg[valid_mask]
    factor_chg = factor_chg[valid_mask]
    if len(vol_chg) < 10:  # Need minimum data
        return {
            'all': np.nan,
            'factor_up': np.nan,
            'factor_down': np.nan,
            'large_up': np.nan,
            'large_down': np.nan,
            'asymmetry': np.nan}
    corr_all = vol_chg.corr(factor_chg)
    up_mask = factor_chg > 0
    down_mask = factor_chg < 0
    corr_up = vol_chg[up_mask].corr(factor_chg[up_mask]) if up_mask.sum() > 5 else np.nan
    corr_down = vol_chg[down_mask].corr(factor_chg[down_mask]) if down_mask.sum() > 5 else np.nan
    large_up = factor_chg > factor_chg.quantile(0.75)
    large_down = factor_chg < factor_chg.quantile(0.25)
    corr_large_up = vol_chg[large_up].corr(factor_chg[large_up]) if large_up.sum() > 5 else np.nan
    corr_large_down = vol_chg[large_down].corr(factor_chg[large_down]) if large_down.sum() > 5 else np.nan
    asymmetry = corr_up - corr_down if (pd.notna(corr_up) and pd.notna(corr_down)) else np.nan
    return {
        'all': corr_all,
        'factor_up': corr_up,
        'factor_down': corr_down,
        'large_up': corr_large_up,
        'large_down': corr_large_down,
        'asymmetry': asymmetry}







df = pd.read_csv('eurusd_gbpusd_dataset.csv', index_col=0, parse_dates=True)


col_interest = 'EURUSD_GBPUSD_1M'
factors = [
    'USGG2YR_DayChg', 
    'MOVE_DayChg', 
    'SPX_DayChg', 
    'EESWE1_DayChg', 
    'BTP_Bund_10Y_Spread_DayChg', 
    'SX5E_DayChg'
]


# # ==============================================================================
# # ----------- Optimal Lag Correlation ------------------------------------------
# # ==============================================================================

# Q_periods = generate_quarterly_periods('Jan23', 'Dec25')
# lag_results = analyze_optimalLagCorr_multiFactor(df, col_interest, factors, Q_periods, max_lag=3)
# print(lag_results)


# # ==============================================================================
# # ----------- Rolling Correlations ---------------------------------------------
# # ==============================================================================
corr_window = 21
corr_lag = -1
plot = True


rolling_df_21d = analyze_rolling_correlations(df, col_interest, factors, 
                                                window=corr_window, 
                                                target_lag=corr_lag, 
                                                plot=plot,
                                                date_range=('2024-01-01', None)
                                            )


# # ==============================================================================
# # ----------- Asymmetric Correlations ------------------------------------------
# # ==============================================================================

# custom_periods = ['Jan25:Jun25', 'Jul25:Dec25']
# asym_results = analyze_asymmetric_correlations_by_period(df, col_interest, factors, 
#                                                          custom_periods, 
#                                                          verbose=True)





# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


def percentile_regimes(df, vol_col, lookback=None, quantiles=[0.33, 0.67]):
    if lookback is None:
        low_thresh = df[vol_col].quantile(quantiles[0])
        high_thresh = df[vol_col].quantile(quantiles[1])
    else:
        low_thresh = df[vol_col].rolling(lookback, min_periods=lookback//2).quantile(quantiles[0])
        high_thresh = df[vol_col].rolling(lookback, min_periods=lookback//2).quantile(quantiles[1])
    regimes = pd.Series(index=df.index, dtype='object')
    regimes[df[vol_col] <= low_thresh] = 'Low'
    regimes[(df[vol_col] > low_thresh) & (df[vol_col] <= high_thresh)] = 'Medium'
    regimes[df[vol_col] > high_thresh] = 'High'
    return regimes





df = pd.read_csv('eurusd_gbpusd_dataset.csv', index_col=0, parse_dates=True)



# df['VIX_Regime'] = percentile_regimes(df, 'VIX', lookback=None)
# df['MOVE_Regime'] = percentile_regimes(df, 'MOVE', lookback=None)

# print(df.tail(10).T)







# col_interest = 'EURUSD_1M'
# factors = [
#     'USGG2YR_DayChg', 
#     'MOVE_DayChg', 
#     'SPX_DayChg', 
#     'EESWE1_DayChg', 
#     'BTP_Bund_10Y_Spread_DayChg', 
#     'SX5E_DayChg' 
# ]





































