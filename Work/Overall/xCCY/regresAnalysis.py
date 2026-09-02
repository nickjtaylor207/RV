from EUR_GBP_USD_Factors import get_Vols, get_Vol_Spreads, USD_GlobalFactors, EUR_Factors
from EUR_GBP_USD_Factors import  print_dataset_structure, filter_by_period, generate_monthly_periods, generate_rolling_3month_periods

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




# Rolling Regression
# LASSO/Ridge Variable Selection
# PCA Decomposition
# Regime Analysis
# Granger Causality



# ----------------------------------------------------------------------------------------
# -------------------------REGRESSION ANALYSIS--------------------------------------------
# ----------------------------------------------------------------------------------------



def ols_regression(df, y_col, x_cols, period_str=None, use_changes=True, 
                   verbose=True, lags=None):
    if period_str:
        filtered_df = filter_by_period(df, period_str, verbose=verbose)
        if filtered_df.empty:
            print(f"No data for period: {period_str}")
            return None, None, None, None, None
    else:
        filtered_df = df.copy()
    if use_changes:
        y_series = filtered_df[y_col].diff().dropna()
        y_col_name = f"{y_col}_Chg"
    else:
        y_series = filtered_df[y_col]
        y_col_name = y_col
    if lags is None:
        lags = {}
    X_cols_lagged = []
    X_col_names = []
    for x_col in x_cols:
        lag = lags.get(x_col, 0)  # Default to 0 if not specified
        if lag == 0:
            X_cols_lagged.append(filtered_df[x_col])
            X_col_names.append(x_col)
        else:
            X_cols_lagged.append(filtered_df[x_col].shift(lag))
            X_col_names.append(f"{x_col}_Lag{lag}")
    X_df = pd.concat(X_cols_lagged, axis=1)
    X_df.columns = X_col_names
    X_df = X_df.loc[y_series.index].dropna()
    y_series = y_series.loc[X_df.index]
    y = y_series.values
    X = X_df.values
    X_const = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
    y_pred = X_const @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, k = X_const.shape
    mse = ss_res / (n - k)
    var_beta = mse * np.linalg.inv(X_const.T @ X_const)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
    results = pd.DataFrame({
        'Coefficient': beta,
        'Std_Error': se_beta,
        'T_Stat': t_stats,
        'P_Value': p_values
    }, index=['const'] + X_col_names)
    if verbose:
        print(f"\nRegression: {y_col_name} ~ {' + '.join(X_col_names)}")
        print(f"Period: {y_series.index.min().date()} to {y_series.index.max().date()}")
        print(f"Observations: {len(y)}")
    return results, r2, residuals, y_pred, filtered_df





# df = pd.read_csv('eurusd_gbpusd_dataset.csv', index_col=0, parse_dates=True)
# y_col = 'EURUSD_1W'
# x_cols = ['MOVE_DayChg', 'VIX_DayChg', 'V2X_DayChg']

# lags_config = {
#     'MOVE_DayChg': 1,  
#     'VIX_DayChg': 1,      
#     'V2X_DayChg': 1}
# results, r2, _, _, _ = ols_regression(df, y_col, x_cols, period_str='1Y',
#     lags=lags_config)

# print(results.round(4))
# print(f"\nR-squared: {r2:.4f}")








# ---------------------------------------------------------------------------------------

def univariate_scan(df, y_col, x_cols, period_str=None, use_changes=True, lags=None):
    results_list = []
    if period_str:
        filtered_df = filter_by_period(df, period_str, verbose=False)
    else:
        filtered_df = df.copy()
    if use_changes:
        y = filtered_df[y_col].diff().dropna()
    else:
        y = filtered_df[y_col].dropna()
    if lags is None:
        lags = {}
    for x_col in x_cols:
        var_lags = lags.get(x_col, [0])
        for lag in var_lags:
            if lag == 0:
                x_lagged = filtered_df[x_col]
                factor_name = x_col
            else:
                x_lagged = filtered_df[x_col].shift(lag)
                factor_name = f"{x_col}_Lag{lag}"
            common_idx = y.index.intersection(x_lagged.dropna().index)
            y_aligned = y.loc[common_idx]
            x_aligned = x_lagged.loc[common_idx]
            if len(y_aligned) < 30:
                continue
            X = np.column_stack([np.ones(len(x_aligned)), x_aligned.values])
            beta = np.linalg.lstsq(X, y_aligned.values, rcond=None)[0]
            y_pred = X @ beta
            ss_res = np.sum((y_aligned.values - y_pred) ** 2)
            ss_tot = np.sum((y_aligned.values - y_aligned.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            corr = y_aligned.corr(x_aligned)
            n = len(y_aligned)
            mse = ss_res / (n - 2)
            se_beta = np.sqrt(mse / np.sum((x_aligned - x_aligned.mean())**2))
            t_stat = beta[1] / se_beta
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))
            results_list.append({
                'Factor': factor_name,
                'Lag': lag,
                'Correlation': corr,
                'Beta': beta[1],
                'T_Stat': t_stat,
                'P_Value': p_value,
                'R_Squared': r2})
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('R_Squared', ascending=False)
    return results_df





# df = pd.read_csv('eurusd_gbpusd_dataset.csv', index_col=0, parse_dates=True)

# y_col = 'EURUSD_1W'
# x_cols = [
#     # Equity vol
#     'VIX_DayChg', 'V2X_DayChg',
#     # US Rates
#     'MOVE_DayChg', 'USGG2YR_DayChg', 'USGG10YR_DayChg', '2s10s_DayChg',
#     # EUR specific
#     'EESWE2_DayChg', 'EUR_OIS_1Y2Y_Slope_DayChg', 'BTP_Bund_10Y_Spread_DayChg',
#     # Equity returns (negative = vol up)
#     'SPX_PctChg', 'SX5E_PctChg',
#     # Historical vol
#     'EURUSD_HVol_1w_DayChg']

# all_same_lags = {col: [2, 3, 5] for col in x_cols}

# uni_results = univariate_scan(df, y_col, x_cols, period_str='2Y:1Y', lags=all_same_lags)
# print(uni_results.round(4).to_string(index=False))





# ---------------------------------------------------------------------------------------


def standardized_regression(df, y_col, x_cols):
    y = df[y_col]
    y_std = (y - y.mean()) / y.std()
    X_std = pd.DataFrame()
    factor_stats = {}
    for col in x_cols:
        x = df[col]
        factor_stats[col] = {'mean': x.mean(), 'std': x.std()}
        X_std[col] = (x - x.mean()) / x.std()
    X_mat = X_std.values
    y_vec = y_std.values
    X_const = np.column_stack([np.ones(len(X_mat)), X_mat])
    beta = np.linalg.lstsq(X_const, y_vec, rcond=None)[0]
    y_pred_std = X_const @ beta
    ss_res = np.sum((y_vec - y_pred_std) ** 2)
    ss_tot = np.sum(y_vec ** 2)  # mean is 0
    r2 = 1 - ss_res / ss_tot
    y_std_dev = y.std()
    raw_betas = {}
    for i, col in enumerate(x_cols):
        std_beta = beta[i + 1]  # skip constant
        raw_beta = std_beta * y_std_dev / factor_stats[col]['std']
        raw_betas[col] = raw_beta
    results = pd.DataFrame({
        'Std_Beta': beta[1:],  # standardized
        'Raw_Beta': [raw_betas[c] for c in x_cols],
        'Factor_StdDev': [factor_stats[c]['std'] for c in x_cols],
    }, index=x_cols)
    results['Abs_Importance'] = results['Std_Beta'].abs()
    results = results.sort_values('Abs_Importance', ascending=False)
    return results, r2, y_std_dev

# df = pd.read_csv('eurusd_gbpusd_dataset.csv', index_col=0, parse_dates=True)

# y_col = 'EURUSD_1M'
# x_cols = ['VIX', 'MOVE', 'BTP_Bund_10Y_Spread', 'V2X']

# results_std, r2_std, vol_std = standardized_regression(df, y_col, x_cols)

# print("\nStandardized Regression Results:")
# print(results_std.round(4))
# print(f"\nR-squared: {r2_std:.4f}")
# print(f"Vol std dev: {vol_std:.2f}")

# print("\nInterpretation:")
# print("  - Std_Beta: directly comparable across factors")
# print("  - Largest |Std_Beta| = most important factor")
# print("  - Raw_Beta: coefficient in original units")
# print(f"  - VIX Std_Beta of {results_std.loc['VIX', 'Std_Beta']:.2f} means:")
# print(f"    1 std dev VIX move → {results_std.loc['VIX', 'Std_Beta']:.2f} std dev vol move")







def regime_conditional_regression(df, y_col, x_cols, regime_col, 
                                   thresholds, period_str=None, use_changes=True):
    if period_str:
        filtered_df = filter_by_period(df, period_str, verbose=False)
    else:
        filtered_df = df.copy()
    if filtered_df.empty:
        return pd.DataFrame()
    bins = [-np.inf] + thresholds + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        if i == 0:
            labels.append(f'< {thresholds[0]}')
        elif i == len(bins) - 2:
            labels.append(f'> {thresholds[-1]}')
        else:
            labels.append(f'{thresholds[i-1]} - {thresholds[i]}')
    filtered_df = filtered_df.copy()
    filtered_df['_regime'] = pd.cut(
        filtered_df[regime_col], 
        bins=bins, 
        labels=labels)
    results_list = []
    for regime in labels:
        regime_df = filtered_df[filtered_df['_regime'] == regime]
        if len(regime_df) < 50:  # Need enough observations
            continue
        if use_changes:
            y = regime_df[y_col].diff().dropna()
        else:
            y = regime_df[y_col]
        X_df = regime_df.loc[y.index, x_cols].dropna()
        y = y.loc[X_df.index]
        if len(y) < 30:
            continue
        X_mat = np.column_stack([np.ones(len(X_df)), X_df.values])
        beta = np.linalg.lstsq(X_mat, y.values, rcond=None)[0]
        y_pred = X_mat @ beta
        ss_res = np.sum((y.values - y_pred) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        n, k = X_mat.shape
        mse = ss_res / (n - k)
        try:
            var_beta = mse * np.linalg.inv(X_mat.T @ X_mat)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
        except:
            t_stats = np.full(k, np.nan)
            p_values = np.full(k, np.nan)
        result_row = {
            'Regime': regime,
            'N_Obs': len(y),
            'R_Squared': r2}
        for i, col in enumerate(x_cols):
            result_row[f'{col}_Beta'] = beta[i + 1]
            result_row[f'{col}_TStat'] = t_stats[i + 1]
            p = p_values[i + 1]
            result_row[f'{col}_Sig'] = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        
        results_list.append(result_row)
    
    return pd.DataFrame(results_list).set_index('Regime')


def print_regime_analysis(results_df, regime_col, x_cols):
    print("\n" + "="*80)
    print(f"REGIME CONDITIONAL ANALYSIS (by {regime_col})")
    print("="*80)
    print(f"""
    This shows how factor betas differ depending on the level of {regime_col}.
    
    Key questions to answer:
    - Do factors matter MORE in high-vol regimes?
    - Do factors matter LESS when markets are calm?
    - Are there non-linear effects (beta changes with regime)?
    """)
    
    print("\n" + "-"*80)
    print("OBSERVATIONS AND MODEL FIT BY REGIME")
    print("-"*80)
    print(results_df[['N_Obs', 'R_Squared']].round(4).to_string())
    
    print("\n" + "-"*80)
    print("BETA COEFFICIENTS BY REGIME")
    print("-"*80)
    beta_cols = [f'{col}_Beta' for col in x_cols]
    betas = results_df[beta_cols].copy()
    betas.columns = x_cols
    print(betas.round(4).to_string())
    
    print("\n" + "-"*80)
    print("SIGNIFICANCE BY REGIME")
    print("-"*80)
    sig_cols = [f'{col}_Sig' for col in x_cols]
    sig_df = results_df[sig_cols].copy()
    sig_df.columns = x_cols
    print(sig_df.to_string())
    
    print("\n" + "-"*80)
    print("INTERPRETATION")
    print("-"*80)
    for col in x_cols:
        betas_by_regime = results_df[f'{col}_Beta']
        low_regime_beta = betas_by_regime.iloc[0]  # First regime (lowest)
        high_regime_beta = betas_by_regime.iloc[-1]  # Last regime (highest)
        
        print(f"\n  {col}:")
        print(f"    - Low {regime_col} beta: {low_regime_beta:.4f}")
        print(f"    - High {regime_col} beta: {high_regime_beta:.4f}")
        
        if abs(high_regime_beta) > abs(low_regime_beta) * 1.5:
            print(f"    -> Effect AMPLIFIES in high {regime_col} regimes")
        elif abs(low_regime_beta) > abs(high_regime_beta) * 1.5:
            print(f"    -> Effect DIMINISHES in high {regime_col} regimes")
        else:
            print(f"    -> Effect is relatively STABLE across regimes")









df = pd.read_csv('eurusd_gbpusd_dataset.csv', index_col=0, parse_dates=True)






# -------------------------------------------------------------------
y_col = 'EURUSD_3M'
x_cols = ['MOVE_DayChg', 'BTP_Bund_10Y_Spread_DayChg', 'SX5E_PctChg']
regime_col = 'VIX'
thresholds = [15, 20, 25]  # Low, Medium, Elevated, High
results = regime_conditional_regression(
    df=df,
    y_col=y_col,
    x_cols=x_cols,
    regime_col=regime_col,
    thresholds=thresholds,
    period_str='2Y',
    use_changes=True)
print_regime_analysis(results, regime_col, x_cols)




# -------------------------------------------------------------------



# Test VIX regimes
print("\n" + "="*80)
print("REGIME TEST 1: VIX LEVELS")
print("="*80)
results_vix = regime_conditional_regression(
    df, 'EURUSD_3M', 
    ['MOVE_DayChg', 'V2X_DayChg'],
    regime_col='VIX',
    thresholds=[15, 20, 25],
    period_str='5Y'
)
print_regime_analysis(results_vix, 'VIX', ['MOVE_DayChg', 'V2X_DayChg'])




# Test MOVE regimes
print("\n" + "="*80)
print("REGIME TEST 2: MOVE LEVELS")
print("="*80)
results_move = regime_conditional_regression(
    df, 'EURUSD_3M',
    ['VIX_DayChg', 'V2X_DayChg'],
    regime_col='MOVE',
    thresholds=[80, 100, 120],  # MOVE typically higher values
    period_str='5Y'
)
print_regime_analysis(results_move, 'MOVE', ['VIX_DayChg', 'V2X_DayChg'])





# Test the vol itself as regime
print("\n" + "="*80)
print("REGIME TEST 3: EURUSD VOL LEVELS")
print("="*80)

results_vol = regime_conditional_regression(
    df, 'EURUSD_3M',
    ['VIX_DayChg', 'MOVE_DayChg', 'BTP_Bund_10Y_Spread_DayChg'],
    regime_col='EURUSD_3M',  # Use the vol itself as regime variable
    thresholds=[6, 8, 10],
    period_str='5Y'
)
print_regime_analysis(results_vol, 'EURUSD_3M', 
                     ['VIX_DayChg', 'MOVE_DayChg', 'BTP_Bund_10Y_Spread_DayChg'])





# Rolling Regression
# LASSO/Ridge Variable Selection
# PCA Decomposition
# Regime Analysis
# Granger Causality
