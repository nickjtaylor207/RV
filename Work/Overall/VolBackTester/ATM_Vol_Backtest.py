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


# ==================== DATA PULLING FUNCTIONS ====================

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
                    # Ensure datetime index
                    data_IV.index = pd.to_datetime(data_IV.index)
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
                    # Ensure datetime index
                    data_HV.index = pd.to_datetime(data_HV.index)
                    df_vols[col_name] = data_HV
                else:
                    print(f"No data for {ticker_HV}, skipping.")
    if df_vols:
        df_vols_all = pd.concat(df_vols.values(), axis=1)
        # Ensure the final index is DatetimeIndex
        df_vols_all.index = pd.to_datetime(df_vols_all.index)
        return df_vols_all
    else:
        print("No data retrieved for any ticker.")
        return pd.DataFrame()


def get_Vol_VRP(ccys, years):
    def calculate_fwd_vol_comparisons(df, ccys):
        tenor_days = {
            '1W': 5,
            '2W': 10,
            '3W': 15,
            '1M': 21}
        results = {}
        for ccy in ccys:  
            for tenor in ['1W', '2W', '3W', '1M']:
                iv_col = f'{ccy}_IV_{tenor}'
                hv_col = f'{ccy}_HV_{tenor}'
                if iv_col not in df.columns or hv_col not in df.columns:
                    continue
                days_forward = tenor_days[tenor]
                future_rv = df[hv_col].shift(-days_forward)
                spread = df[iv_col] - future_rv
                results[f'{ccy}_{tenor}_VRP'] = spread
        return pd.DataFrame(results, index=df.index)
    I_tenors = ['1W', '2W', '3W', '1M']
    H_tenors = ['1W', '2W', '3W', '1M']
    df = get_Vols(ccys, I_tenors, H_tenors, years)
    df = calculate_fwd_vol_comparisons(df, ccys)
    return df


def get_Vol_MTM(ccys, years):
    I_tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '6M', '9M', '1Y']
    H_tenors = [] 
    df = get_Vols(ccys, I_tenors, H_tenors, years)
    mtm_pairs = [
        ('1Y', '9M', 63, '1Y9M'),
        ('1Y', '6M', 126, '1Y6M'),
        ('1Y', '3M', 189, '1Y3M'), 
        ('6M', '3M', 63, '6M3M'),
        ('6M', '2M', 84, '6M2M'),   
        ('6M', '1M', 105, '6M1M'),  
        ('3M', '2M', 21, '3M2M'),
        ('3M', '1M', 42, '3M1M'),  
        ('1M', '3W', 5, '1M3W'),
        ('1M', '2W', 10, '1M2W'),   
        ('1M', '1W', 15, '1M1W'),  
        ('3W', '2W', 5, '3W2W'),
        ('3W', '1W', 10, '3W1W'),  
        ('2W', '1W', 5, '2W1W'),]
    results = {}
    for ccy in ccys:
        for long_tenor, short_tenor, days_forward, label in mtm_pairs:
            long_col = f'{ccy}_IV_{long_tenor}'
            short_col = f'{ccy}_IV_{short_tenor}'
            if long_col not in df.columns or short_col not in df.columns:
                print(f"Warning: Missing data for {ccy} {label}, skipping.")
                continue
            initial_long_vol = df[long_col]
            future_short_vol = df[short_col].shift(-days_forward)
            mtm_pnl = initial_long_vol - future_short_vol
            results[f'{ccy}_{label}_MTM'] = mtm_pnl
    return pd.DataFrame(results, index=df.index)


def get_combined_analysis(ccys, years):
    df_vrp = get_Vol_VRP(ccys, years)
    df_mtm = get_Vol_MTM(ccys, years)
    df_combined = pd.concat([df_vrp, df_mtm], axis=1)
    return df_combined



# --------------- FRESH DATA PULLING ---------------

# ccys = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN', 'USDBRL']
# years = 10
# df_full = get_combined_analysis(ccys, years)


# df_full.to_csv('ATM_VRP_MTM.csv')















# ==================== FILTERING HELPER FUNCTIONS ====================

def _parse_offset(reference_date, offset_str):
    match = re.match(r'(\d+)([YMWD])', offset_str.upper())
    if not match:
        raise ValueError(f"Invalid offset format: {offset_str}. Use format like '1Y', '6M', '3W', '5D'")
    value = int(match.group(1))
    unit = match.group(2)
    reference_date = pd.Timestamp(reference_date)
    if unit == 'Y':
        result = reference_date - pd.DateOffset(years=value)
    elif unit == 'M':
        result = reference_date - pd.DateOffset(months=value)
    elif unit == 'W':
        result = reference_date - pd.DateOffset(weeks=value)
    elif unit == 'D':
        result = reference_date - pd.DateOffset(days=value)
    else:
        result = reference_date
    return pd.Timestamp(result)


def _parse_years_from_offset(offset_str):
    match = re.match(r'(\d+)([YMWD])', offset_str.upper())
    if not match:
        return 1
    value = int(match.group(1))
    unit = match.group(2)
    if unit == 'Y':
        return value
    elif unit == 'M':
        return value / 12
    elif unit == 'W':
        return value / 52
    elif unit == 'D':
        return value / 365
    else:
        return 1


def filter_by_date_range(df, date_range):
    end_date = pd.Timestamp(df.index.max())
    if ':' in date_range:
        start_offset, end_offset = date_range.split(':')
        start_date = _parse_offset(end_date, start_offset)
        end_date_filter = _parse_offset(end_date, end_offset)
    elif date_range == 'YTD':
        start_date = pd.Timestamp(f'{end_date.year}-01-01')
        end_date_filter = end_date
    elif date_range == 'MTD':
        start_date = pd.Timestamp(f'{end_date.year}-{end_date.month:02d}-01')
        end_date_filter = end_date
    else:
        start_date = _parse_offset(end_date, date_range)
        end_date_filter = end_date
    mask = (df.index >= start_date) & (df.index <= end_date_filter)
    result = df.loc[mask]
    if result.empty:
        print(f"Warning: No data found between {start_date} and {end_date_filter}")
        print(f"Available data range: {df.index.min()} to {df.index.max()}")
    return result


def filter_by_currency(df, ccy):
    if isinstance(ccy, str):
        ccy = [ccy]
    ccy_cols = []
    for c in ccy:
        cols = [col for col in df.columns if col.startswith(f'{c}_')]
        if not cols:
            print(f"Warning: No columns found for currency: {c}")
        ccy_cols.extend(cols)
    if not ccy_cols:
        raise ValueError(f"No columns found for any of the currencies: {ccy}")
    return df[ccy_cols]


def filter_by_metric(df, metric_type):
    if isinstance(metric_type, str):
        metric_type = [metric_type]
    metric_cols = []
    for m in metric_type:
        cols = [col for col in df.columns if f'_{m}' in col]
        if not cols:
            print(f"Warning: No columns found for metric type: {m}")
        metric_cols.extend(cols)
    if not metric_cols:
        raise ValueError(f"No columns found for any of the metric types: {metric_type}")
    return df[metric_cols]


def filter_by_tenor(df, tenor, metric_type=None):
    if isinstance(tenor, str):
        tenor = [tenor]
    if metric_type and isinstance(metric_type, str):
        metric_type = [metric_type]
    tenor_cols = []
    for t in tenor:
        if metric_type:
            for m in metric_type:
                cols = [col for col in df.columns if f'_{t}_{m}' in col]
                tenor_cols.extend(cols)
        else:
            cols = [col for col in df.columns if f'_{t}_' in col]
            tenor_cols.extend(cols)
    if not tenor_cols:
        raise ValueError(f"No columns found for tenors: {tenor}")
    return df[tenor_cols]


def get_data(df, ccy=None, metric_type=None, tenor=None, date_range=None):
    result = df.copy()
    if ccy:
        if isinstance(ccy, str):
            ccy = [ccy]
        ccy_cols = []
        for c in ccy:
            ccy_cols.extend([col for col in result.columns if col.startswith(f'{c}_')])
        result = result[ccy_cols]
    if metric_type:
        if isinstance(metric_type, str):
            metric_type = [metric_type]
        metric_cols = []
        for m in metric_type:
            metric_cols.extend([col for col in result.columns if f'_{m}' in col])
        result = result[metric_cols]
    if tenor:
        if isinstance(tenor, str):
            tenor = [tenor]
        tenor_cols = []
        for t in tenor:
            tenor_cols.extend([col for col in result.columns if f'_{t}_' in col])
        result = result[tenor_cols]
    if date_range:
        result = filter_by_date_range(result, date_range)
    if result.empty:
        print("Warning: No data matches the specified filters")
    return result





# ==================== DATE RANGE ENABLED VOL FUNCTIONS ====================

def get_Vols_with_date_range(ccys, I_tenors, H_tenors, date_range):
    if isinstance(ccys, str):
        ccys = [ccys]
    end_date_ref = datetime.now()
    if ':' in date_range:
        start_offset = date_range.split(':')[0]
        years_needed = _parse_years_from_offset(start_offset)
    elif date_range in ['YTD', 'MTD']:
        years_needed = 1
    else:
        years_needed = _parse_years_from_offset(date_range)
    years_to_pull = int(np.ceil(years_needed * 1.2))  # 20% buffer
    df_full = get_Vols(ccys, I_tenors, H_tenors, years_to_pull)
    df_filtered = filter_by_date_range(df_full, date_range)
    return df_filtered


def get_complete_vol_analysis(ccys, I_tenors, metric_type, date_range, H_tenors=None):
    df_full = pd.read_csv('ATM_VRP_MTM.csv', index_col=0, parse_dates=True)
    
    df_result = get_data(
        df_full,
        ccy=ccys,
        metric_type=metric_type,
        date_range=date_range)
    if 'MTM' in (metric_type if isinstance(metric_type, list) else [metric_type]):
        ccy_list = [ccys] if isinstance(ccys, str) else ccys
        tenor_list = I_tenors if isinstance(I_tenors, list) else [I_tenors]
        allowed_mtm_cols = []
        for ccy in ccy_list:
            for tenor in tenor_list:
                pattern_cols = [col for col in df_result.columns
                               if col.startswith(f'{ccy}_{tenor}') and col.endswith('_MTM')]
                allowed_mtm_cols.extend(pattern_cols)
        vrp_cols = [col for col in df_result.columns if '_VRP' in col]
        df_result = df_result[vrp_cols + allowed_mtm_cols]
    
    return df_result




# ==================== PLOTTING FUNCTION ====================
def plot_timeseries(df, columns, figsize=(14, 6), title=None, ylabel='Value', 
                    show_mean=False, show_stats=False, grid=True):
    if isinstance(columns, str):
        columns = [columns]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
    for idx, col in enumerate(columns):
        data = df[col].dropna()
        if len(data) == 0:
            print(f"Warning: {col} has no data, skipping.")
            continue
        mean_val = data.mean()
        std_val = data.std()
        if show_stats:
            label = f'{col} (μ={mean_val:.2f}, σ={std_val:.2f})'
        else:
            label = col
        ax.plot(data.index, data.values, label=label, color=colors[idx], 
                linewidth=1.5, alpha=0.8)
        if show_mean:
            ax.axhline(y=mean_val, color=colors[idx], linestyle='--', 
                      linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title(' & '.join(columns), fontsize=14, fontweight='bold', pad=20)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_timeseries_dual_axis(df, left_columns, right_columns, figsize=(14, 6), 
                               title=None, left_ylabel='Vol Points', right_ylabel='Percentile',
                               show_mean=True, show_stats=False, grid=True, align_zero_fifty=True):
    if isinstance(left_columns, str):
        left_columns = [left_columns]
    if isinstance(right_columns, str):
        right_columns = [right_columns]
    missing = [col for col in left_columns + right_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    colors_left = plt.cm.Blues(np.linspace(0.4, 0.9, len(left_columns)))
    colors_right = plt.cm.Oranges(np.linspace(0.4, 0.9, len(right_columns)))
    for idx, col in enumerate(left_columns):
        data = df[col].dropna()
        if len(data) == 0:
            print(f"Warning: {col} has no data, skipping.")
            continue
        mean_val = data.mean()
        std_val = data.std()
        if show_stats:
            label = f'{col} (μ={mean_val:.2f}, σ={std_val:.2f})'
        else:
            label = col
        ax1.plot(data.index, data.values, label=label, color=colors_left[idx], 
                linewidth=2, alpha=0.8)
        if show_mean:
            ax1.axhline(y=mean_val, color=colors_left[idx], linestyle='--', 
                       linewidth=1, alpha=0.5)
    for idx, col in enumerate(right_columns):
        data = df[col].dropna()
        if len(data) == 0:
            print(f"Warning: {col} has no data, skipping.")
            continue
        mean_val = data.mean()
        std_val = data.std()
        if show_stats:
            label = f'{col} (μ={mean_val:.2f}, σ={std_val:.2f})'
        else:
            label = col
        ax2.plot(data.index, data.values, label=label, color=colors_right[idx], 
                linewidth=2, alpha=0.8, linestyle='--')
        if show_mean:
            ax2.axhline(y=mean_val, color=colors_right[idx], linestyle=':', 
                       linewidth=1, alpha=0.5)
    if align_zero_fifty:
        left_data = pd.concat([df[col].dropna() for col in left_columns])
        right_data = pd.concat([df[col].dropna() for col in right_columns])
        left_min, left_max = left_data.min(), left_data.max()
        right_min, right_max = right_data.min(), right_data.max()
        left_abs_max = max(abs(left_min), abs(left_max))
        left_lim = left_abs_max * 1.1  # Add 10% padding
        right_range = max(50 - right_min, right_max - 50)
        right_lower = 50 - right_range * 1.1
        right_upper = 50 + right_range * 1.1
        ax1.set_ylim(-left_lim, left_lim)
        ax2.set_ylim(right_lower, right_upper)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
    ax2.axhline(y=50, color='darkred', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel(left_ylabel, fontsize=12, fontweight='bold', color='steelblue')
    ax2.set_ylabel(right_ylabel, fontsize=12, fontweight='bold', color='orangered')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='orangered')
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    if grid:
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax1, ax2


def plot_timeseries_smart(df, columns, figsize=(14, 6), title=None, 
                          show_mean=False, show_stats=False, grid=True, align_zero_fifty=True):
    if isinstance(columns, str):
        columns = [columns]
    percentile_keywords = ['Pct', '_pct', 'percentile']
    left_columns = []
    right_columns = []
    for col in columns:
        is_percentile = any(keyword in col for keyword in percentile_keywords)
        if is_percentile:
            right_columns.append(col)
        else:
            left_columns.append(col)
    if not right_columns:
        return plot_timeseries(df, columns, figsize, title, 'Value', 
                              show_mean, show_stats, grid)
    if not left_columns:
        return plot_timeseries(df, columns, figsize, title, 'Percentile', 
                              show_mean, show_stats, grid)
    return plot_timeseries_dual_axis(
        df, 
        left_columns=left_columns,
        right_columns=right_columns,
        figsize=figsize,
        title=title,
        left_ylabel='Vol Points',
        right_ylabel='Percentile',
        show_mean=show_mean,
        show_stats=show_stats,
        grid=grid,
        align_zero_fifty=align_zero_fifty
    )





# ==================== ANALYSIS FUNCTIONS ====================

def add_IV_rolling_percentiles(df, tenor='1M', lookback_3m=63, lookback_3y=756):
    df_with_pct = df.copy()
    iv_cols = [col for col in df.columns if col.endswith(f'_IV_{tenor}')]
    if not iv_cols:
        print(f"Warning: No IV columns found for tenor {tenor}")
        return df_with_pct
    def rolling_percentile(series):
        if len(series) < 2:
            return np.nan
        return (series < series.iloc[-1]).sum() / len(series) * 100
    for iv_col in iv_cols:
        ccy_pair = iv_col.replace(f'_IV_{tenor}', '')
        pct_3m_col_name = f"{ccy_pair}_IV_{tenor}_3mPct"
        df_with_pct[pct_3m_col_name] = df[iv_col].rolling(
            window=lookback_3m
        ).apply(rolling_percentile, raw=False)
        pct_3y_col_name = f"{ccy_pair}_IV_{tenor}_3yPct"
        df_with_pct[pct_3y_col_name] = df[iv_col].rolling(
            window=lookback_3y
        ).apply(rolling_percentile, raw=False)
    return df_with_pct

def add_VoV_and_percentile(df, tenor='1M', realized_window=40):
    df_with_rv = df.copy()
    iv_cols = [col for col in df.columns if col.endswith(f'_IV_{tenor}')]
    if not iv_cols:
        print(f"Warning: No IV columns found for tenor {tenor}")
        return df_with_rv
    def rolling_percentile(series):
        if len(series) < 2:
            return np.nan
        return (series < series.iloc[-1]).sum() / len(series) * 100
    for iv_col in iv_cols:
        ccy = iv_col.replace(f'_IV_{tenor}', '')
        log_returns = np.log(df[iv_col] / df[iv_col].shift(1))
        rv_col_name = f'{ccy}_IV_{tenor}_RV{realized_window}D'
        df_with_rv[rv_col_name] = log_returns.rolling(
            window=realized_window,
            min_periods=realized_window
        ).std() * np.sqrt(252) * 100
        pct_3m_col = f'{ccy}_IV_{tenor}_VoV_3mPct'
        df_with_rv[pct_3m_col] = df_with_rv[rv_col_name].rolling(
            window=63,
            min_periods=63
        ).apply(rolling_percentile, raw=False)
        pct_3y_col = f'{ccy}_IV_{tenor}_VoV_3yPct'
        df_with_rv[pct_3y_col] = df_with_rv[rv_col_name].rolling(
            window=756,
            min_periods=252
        ).apply(rolling_percentile, raw=False)
    return df_with_rv
def add_iv_hv_spreads_and_percentiles(df):
    df_with_iv_hv = df.copy()
    ccys_in_df = list(set([col.split('_IV_')[0] for col in df.columns if '_IV_' in col]))
    def rolling_percentile(series):
        if len(series) < 2:
            return np.nan
        return (series < series.iloc[-1]).sum() / len(series) * 100
    for ccy in ccys_in_df:
        iv_hv_pairs = [
            ('1W', f'{ccy}_IV_1W', f'{ccy}_HV_1W'),
            ('1M', f'{ccy}_IV_1M', f'{ccy}_HV_1M')]
        for tenor, iv_col, hv_col in iv_hv_pairs:
            if iv_col in df.columns and hv_col in df.columns:
                spread_name = f'{ccy}_IV_HV_spread_{tenor}'
                df_with_iv_hv[spread_name] = df[iv_col] - df[hv_col]
                
                df_with_iv_hv[f'{spread_name}_3mPct'] = df_with_iv_hv[spread_name].rolling(
                    window=63, min_periods=63
                ).apply(rolling_percentile, raw=False)
                
                df_with_iv_hv[f'{spread_name}_3yPct'] = df_with_iv_hv[spread_name].rolling(
                    window=756, min_periods=252
                ).apply(rolling_percentile, raw=False)
    return df_with_iv_hv


def add_vol_Termspreads_and_percentiles(df, smooth_days=5):
    df_with_spreads = df.copy()
    ccys_in_df = list(set([col.split('_IV_')[0] for col in df.columns if '_IV_' in col]))
    def rolling_percentile(series):
        if len(series) < 2:
            return np.nan
        return (series < series.iloc[-1]).sum() / len(series) * 100
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
                    window=smooth_days, min_periods=1).median()
                df_with_spreads[f'{smoothed_name}_3mPct'] = df_with_spreads[smoothed_name].rolling(
                    window=63, min_periods=63
                ).apply(rolling_percentile, raw=False)
                df_with_spreads[f'{smoothed_name}_3yPct'] = df_with_spreads[smoothed_name].rolling(
                    window=756, min_periods=252
                ).apply(rolling_percentile, raw=False)
    return df_with_spreads

# -------------------------------------------------------------------------------------------



ccys = ['USDJPY']
metric_type = ['VRP', 'MTM']
date_range = '3M'
I_tenors = ['1W', '2W', '1M']

df_complete = get_complete_vol_analysis(
    ccys=ccys,
    I_tenors=I_tenors,
    metric_type=metric_type,
    date_range=date_range)



print(df_complete.tail(20).T)







columns = [
    f'{ccys[0]}_1W_VRP',

    f'{ccys[0]}_2W_VRP',



    # f'{ccys[0]}_1M3W_MTM',
    # f'{ccys[0]}_1M2W_MTM',

]



plot_timeseries_smart(df_complete, columns, 
                      title=f'{ccys[0]}')