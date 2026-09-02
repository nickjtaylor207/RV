import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import table
from PIL import Image

from xbbg import blp
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from itertools import permutations

from itertools import product
from scipy.stats import percentileofscore, gaussian_kde, norm
from scipy.stats import zscore
import math
from itertools import combinations



# ----------------------------------------------------------------------------------------------
# ---------------------Current Vol Level Percentile/ZScore Perspective  ------------------------
# ----------------------------------------------------------------------------------------------

"""DF -- Daily Close to Close Vols - Multiple CCYs, Multiple Tenors"""
def getDailyVols_ccysTenors_years(ccy_list, tenor_list, years):
    start_date = (datetime.today() - timedelta(days=(years * 365))).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    combined_data = pd.DataFrame()
    for ccy, tenor in product(ccy_list, tenor_list):
        ticker_IV = f"{ccy}V{tenor} BGN Curncy"
        field = "PX_LAST"
        data_IV = blp.bdh(
            tickers=ticker_IV,
            flds=field,
            start_date=start_date,
            end_date=end_date)
        data_IV.columns = [f'{ccy} Vol {tenor}']
        combined_data = pd.concat([combined_data, data_IV], axis=1)
    return combined_data



print(365 * 5)


# -------------- PERCENTILE CALCULATIONS -------------- 
"""df -- Multiple Period Percentile Calulations - all columns in df input """
def calculate_multi_period_percentiles(df):
    lookback_TradingDays = {'3M_%': 90, '6M_%': 180, '1Y_%': 365, '2Y_%': 730}
    results = {}
    for period_name, days in lookback_TradingDays.items():
        period_percentiles = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < days:
                lookback_series = series
            else:
                lookback_series = series.iloc[-days:]
            current_value = series.iloc[-1]
            percentile = percentileofscore(lookback_series, current_value, kind='rank')
            period_percentiles[col] = percentile
        results[period_name] = period_percentiles
    df_results = pd.DataFrame(results)
    return df_results.round(2)

"""plot -- Single column distribution w/ market percentile """
def plot_vol_distribution_percentile(df, column_name=None, figsize=(10, 6)):
    if column_name is None:
        column_name = df.columns[0]
    data = df[column_name].dropna()
    current_value = data.iloc[-1]
    percentile = percentileofscore(data, current_value, kind='rank')
    fig, ax = plt.subplots(figsize=figsize)
    n_bins = min(30, len(data) // 10)  # Adaptive number of bins
    counts, bins, patches = ax.hist(data, bins=n_bins, alpha=0.7, 
                                   color='lightblue', edgecolor='black', 
                                   density=True, label='Historical Data')
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_values = kde(x_range)
        ax.plot(x_range, kde_values, 'b-', linewidth=2, label='Density Curve')
    except:
        pass  
    ax.axvline(current_value, color='red', linestyle='--', linewidth=2, 
               label=f'Current: {current_value:.2f} ({percentile:.1f}th percentile)')
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    ax.set_xlabel('Volatility (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of {column_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    stats_text = f'''Statistics:
                    Mean: {data.mean():.2f}
                    Std: {data.std():.2f}
                    Min: {data.min():.2f}
                    Max: {data.max():.2f}
                    Current: {current_value:.2f}
                    Percentile: {percentile:.1f}%'''
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()

"""plot -- Multiple LookBack periods for one vol reference"""
def plot_multi_period_distributions(df, column_name, figsize=(15, 10)):

    lookback_days = {'3M_%': 90, '1Y_%': 365, '3Y_%': 365*3, "5Y_%": 365*5}
    n_periods = len(lookback_days)
    n_cols = 4
    n_rows = math.ceil(n_periods / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten() if n_periods > 1 else [axes]
    full_data = df[column_name].dropna()
    current_value = full_data.iloc[-1]
    for idx, (period_name, days) in enumerate(lookback_days.items()):
        ax = axes_flat[idx]
        if len(full_data) < days:
            lookback_data = full_data
            data_note = f"(All {len(full_data)} days)"
        else:
            lookback_data = full_data.iloc[-days:]
            data_note = f"({days} days)"
        percentile = percentileofscore(lookback_data, current_value, kind='rank')
        n_bins = min(20, len(lookback_data) // 5)  # Fewer bins for smaller subplots
        counts, bins, patches = ax.hist(lookback_data, bins=n_bins, alpha=0.7, 
                                       color='lightblue', edgecolor='black', 
                                       density=True)
        try:
            from scipy.stats import gaussian_kde
            if len(lookback_data) > 5:  # Need minimum data for KDE
                kde = gaussian_kde(lookback_data)
                x_range = np.linspace(lookback_data.min(), lookback_data.max(), 100)
                kde_values = kde(x_range)
                ax.plot(x_range, kde_values, 'b-', linewidth=1.5)
        except:
            pass
        ax.axvline(current_value, color='red', linestyle='--', linewidth=2)
        median_val = lookback_data.median()
        ax.axvline(median_val, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        period_clean = period_name.replace('_%', '')  # Remove % suffix for display
        ax.set_title(f'{period_clean} Lookback {data_note}\nCurrent: {percentile:.1f}th percentile', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Volatility (%)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(True, alpha=0.3)
        if percentile > 90:
            line_color = 'darkred'
            percentile_text = f'{percentile:.1f}% (Extreme High)'
        elif percentile > 75:
            line_color = 'red'
            percentile_text = f'{percentile:.1f}% (High)'
        elif percentile < 10:
            line_color = 'darkgreen'
            percentile_text = f'{percentile:.1f}% (Extreme Low)'
        elif percentile < 25:
            line_color = 'green'
            percentile_text = f'{percentile:.1f}% (Low)'
        else:
            line_color = 'orange'
            percentile_text = f'{percentile:.1f}% (Normal)'
        for line in ax.lines:
            if line.get_linestyle() == '--':
                line.set_color(line_color)
        ax.text(0.02, 0.98, percentile_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        stats_text = f'μ: {lookback_data.mean():.2f}\nσ: {lookback_data.std():.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    for idx in range(n_periods, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


# -------------- Z-Score CALCULATIONS -------------- 
"""df -- Multiple Period Z-Score Calulations - all columns in df input """
def calculate_multi_period_zscores(df):
    lookback_TradingDays = {'3M_ZSCORE': 90, '6M_ZSCORE': 180, '1Y_ZSCORE': 365, '2Y_ZSCORE': 730}
    results = {}
    for period_name, days in lookback_TradingDays.items():
        period_zscores = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < days:
                lookback_series = series
            else:
                lookback_series = series.iloc[-days:]
            current_value = series.iloc[-1]
            mean_val = lookback_series.mean()
            std_val = lookback_series.std()
            if std_val != 0:
                z_score = (current_value - mean_val) / std_val
            else:
                z_score = 0
            period_zscores[col] = z_score 
        results[period_name] = period_zscores
    df_results = pd.DataFrame(results).T
    return df_results.round(2)

"""plot -- Single column distribution w/ market Z-Score """
def plot_vol_distribution_zscore(df, column_name=None, figsize=(10, 6)):
    if column_name is None:
        column_name = df.columns[0]
    data = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if data.empty:
        raise ValueError("No numeric data found for plotting.")
    current_value = data.iloc[-1]
    mu = data.mean()
    sigma = data.std(ddof=1)
    z = (current_value - mu) / sigma
    pct = percentileofscore(data, current_value, kind="rank")
    fig, ax = plt.subplots(figsize=figsize)
    n_bins = min(30, max(10, len(data)//10))
    ax.hist(data, bins=n_bins, density=True,
            alpha=0.6, edgecolor="black", color="lightblue",
            label="Historical Data")
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 400)
    ax.plot(x_range, kde(x_range), lw=2, color="darkblue", label="Density Curve")
    ax.axvline(mu, color="black", lw=1, linestyle="--", label=f"Mean μ = {mu:.2f}")
    ax.text(mu, ax.get_ylim()[1]*0.9, "Mean", color="black", fontsize=9,
            ha="center", va="bottom", rotation=90, fontweight="bold")
    for k in [1, 2, 3]:
        for sign, lbl in [(-1, f"–{k}σ"), (1, f"+{k}σ")]:
            x_val = mu + sign * k * sigma
            ax.axvline(x_val, color="green", lw=1, linestyle=":")
            ax.text(x_val, ax.get_ylim()[1]*0.85, lbl, color="gray", fontsize=8,
                    ha="center", va="bottom", rotation=90)
    ax.axvline(current_value, color="red", lw=2, linestyle="--",
               label=f"Z = {z:.2f} ({current_value:.2f}v)")
    ax.set_title(f"Distribution of {column_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()



ccy_list = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 
    'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN',
    'USDBRL', 'USDCNH']
tenor_list = ['1W']
years = 5 # (Max Lookback Reference) + 1


df = getDailyVols_ccysTenors_years(ccy_list, tenor_list, years)

print(calculate_multi_period_percentiles(df))

print("-"*30)

print(calculate_multi_period_percentiles(df))


lookback_TradingDays = {'3M_%': 65, '6M_%': 130, '1Y_%': 252, '2Y_%': 504}
plot_multi_period_distributions(df, 'EURUSD Vol 1M', lookback_TradingDays)


# Get the data
df = getDailyVols_ccysTenors_years(ccy_list, tenor_list, years)
df_percentiles_multi = calculate_multi_period_percentiles(df)
print(df_percentiles_multi)










# ----------------------------------------------------------------------------------------------
# -----------------Vol Spread Refference Code (CCY1 v CCY2) - Multiple iterations  -------------
# ----------------------------------------------------------------------------------------------

"""df - Vol Spreads for each ccy (no duping)"""
def getVolSpreads_noRepeats(ccy_list, tenor_list, years):
    df = getDailyVols_ccysTenors_years(ccy_list, tenor_list, years)
    spread_df = pd.DataFrame(index=df.index)
    for tenor in tenor_list:
        # Generate all pair combinations of currencies
        for ccy1, ccy2 in combinations(ccy_list, 2):
            col1 = f"{ccy1} Vol {tenor}"
            col2 = f"{ccy2} Vol {tenor}"
            spread_col = f"{ccy1}-{ccy2}_{tenor}"
            spread_df[spread_col] = df[col1] - df[col2]
    return spread_df


"""df - Vol Spread Percentiles for each spread column from above output"""
def calulate_spread_MultiPercentiles(spreads_df: pd.DataFrame, min_points=5) -> pd.DataFrame:
    windows_days = (("3M_%", 90),("1Y_%", 365),("3Y_%", 365*3),("5Y_%", 365*5))
    latest_date = spreads_df.dropna(how="all").index.max()
    out = {win_label: {} for win_label, _ in windows_days}
    out["Current"] = {}
    for col in spreads_df.columns:
        series = pd.to_numeric(spreads_df[col], errors="coerce").dropna()
        if series.empty:
            for win_label, _ in windows_days:
                out[win_label][col] = None
            out["Current"][col] = None
            continue
        latest_ts = series.index.max()
        current_spread = series.loc[latest_ts]
        out["Current"][col] = round(float(current_spread), 4)
        for win_label, days in windows_days:
            start_dt = latest_date - timedelta(days=days)
            window_series = series.loc[(series.index >= start_dt) & (series.index <= latest_ts)]
            if len(window_series) >= min_points:
                pct = percentileofscore(window_series, current_spread, kind="rank")
                out[win_label][col] = round(float(pct), 2)
            else:
                out[win_label][col] = None
    result = pd.DataFrame(out).T
    return result


"""df - Vol Spread Z-Score for each spread column from above output"""
def calculate_spread_MultiZScores(spreads_df: pd.DataFrame, min_points=5) -> pd.DataFrame:
    windows_days = (("3M_z", 90),("1Y_z", 365),("3Y_z", 365*3),("5Y_z", 365*5))
    latest_date = spreads_df.dropna(how="all").index.max()
    out = {win_label: {} for win_label, _ in windows_days}
    out["Current"] = {}
    for col in spreads_df.columns:
        series = pd.to_numeric(spreads_df[col], errors="coerce").dropna()
        if series.empty:
            for win_label, _ in windows_days:
                out[win_label][col] = None
            out["Current"][col] = None
            continue
        latest_ts = series.index.max()
        current_spread = series.loc[latest_ts]
        out["Current"][col] = round(float(current_spread), 4)
        for win_label, days in windows_days:
            start_dt = latest_date - timedelta(days=days)
            window_series = series.loc[(series.index >= start_dt) & (series.index <= latest_ts)]
            if len(window_series) >= min_points:
                mean_val = window_series.mean()
                std_val = window_series.std(ddof=1)
                if std_val > 0:
                    z = (current_spread - mean_val) / std_val
                    out[win_label][col] = round(float(z), 2)
                else:
                    out[win_label][col] = None
            else:
                out[win_label][col] = None
    result = pd.DataFrame(out).T
    return result




# ccy_list = ['EURUSD', 'USDJPY']
# tenor_list = ['1M']
# years = 5

# df = getVolSpreads_noRepeats(ccy_list, tenor_list, years)

# df_percentiles = calulate_spread_MultiPercentiles(df)
# print(df_percentiles)

# df_ZScore = calculate_spread_MultiZScores(df)
# print(df_ZScore)















