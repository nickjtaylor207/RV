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
from sklearn.linear_model import LinearRegression
import time



"""
Data Pull --- One Large Pull

    -   Input:  List of CCYs and Tenors of Interest
    -  Output:  Dict by [Tenor][CCY]
"""
def get_all_volatility_data_bulk(currency_list, tenors, start_date, end_date):
    all_tickers = []
    ticker_map = {}  # Map ticker back to (currency, tenor)
    for tenor in tenors:
        for ccy_pair in currency_list:
            ticker_IV = f"{ccy_pair}V{tenor} BGN Curncy"
            all_tickers.append(ticker_IV)
            ticker_map[ticker_IV] = (ccy_pair, tenor)
    try:
        bulk_data = blp.bdh(
            tickers=all_tickers,
            flds="PX_LAST",
            start_date=start_date,
            end_date=end_date)
        organized_data = {}
        for tenor in tenors:
            organized_data[tenor] = {}
        for ticker in all_tickers:
            if ticker in bulk_data.columns.get_level_values(0):
                ccy_pair, tenor = ticker_map[ticker]
                try:
                    series_data = bulk_data[ticker]['PX_LAST']
                    organized_data[tenor][ccy_pair] = series_data
                except:
                    continue        
        return organized_data
    except Exception as e:
        print(f"Bulk data fetch failed: {e}")
        return {}



def load_cointegration_scores(excel_path, currencies, tenors):
    cointegration_scores = {}
    try:
        xls = pd.ExcelFile(excel_path)
        for tenor in tenors:
            if tenor in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=tenor, index_col=0)
                cointegration_scores[tenor] = {}
                for pair_str in df.index:
                    if 'cointegration_score' in df.columns:
                        score = df.loc[pair_str, 'cointegration_score']
                        parts = pair_str.split(' / ')
                        if len(parts) == 2:
                            ccy1 = parts[0].split('V')[0].replace(' Curncy', '').strip()
                            ccy2 = parts[1].split('V')[0].replace(' Curncy', '').strip()
                            if ccy1 in currencies and ccy2 in currencies:
                                cointegration_scores[tenor][(ccy1, ccy2)] = score
                                cointegration_scores[tenor][(ccy2, ccy1)] = score  # Symmetric
                print(f"Loaded {len(cointegration_scores[tenor])} pair scores for tenor {tenor}")
            else:
                print(f"Warning: Tenor {tenor} not found in Excel file")
                cointegration_scores[tenor] = {}
    except Exception as e:
        print(f"Error loading cointegration scores: {e}")
        # Return empty structure
        for tenor in tenors:
            cointegration_scores[tenor] = {}
    return cointegration_scores



# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

"""
Historical Refference Calculations

    -   Input: Full Vol Data, Tenor List, Lookback Period for Testing
    -  Output [Dict]: 
            -     results_by_tenor[Tenor] - Each Spread iteration with Historical Measure and Corr Between CCY in Spread
            - currency_avg_metrics[Tenor] - Spread Consolidated Statistics
            -     correlation_data[Tenor] - Correlation For Each pair's vol for given tenor   
"""
def xCCY_AllPairs_MultiTenor_Analysis(bulk_vol_data, tenor_list, date_range):
    results_by_date = {}  # {analysis_ts: {'results_by_tenor':..., 'currency_avg_metrics':..., 'correlation_data':...}}
    for analysis_date in date_range:
        analysis_ts = pd.to_datetime(analysis_date)

        results_by_tenor = {}
        currency_avg_metrics = {}
        correlation_data = {}

        for tenor in tenor_list:
            if tenor not in bulk_vol_data:
                continue
            tenor_data = bulk_vol_data[tenor]
            if not tenor_data:
                continue
            data_dict = {}
            for ccy_pair, series in tenor_data.items():
                if series is None or series.empty:
                    continue
                if not isinstance(series.index, pd.DatetimeIndex):
                    series = series.copy()
                    series.index = pd.to_datetime(series.index)
                filtered_series = series.loc[:analysis_ts]
                if not filtered_series.empty:
                    data_dict[ccy_pair] = filtered_series
            if not data_dict:
                continue
            df_all = pd.DataFrame(data_dict).dropna()
            if df_all.empty:
                continue
            correlation_matrix = df_all.corr()
            correlation_data[tenor] = correlation_matrix
            latest_date = df_all.index[-1]
            currency_pairs = list(permutations(list(data_dict.keys()), 2))
            pair_results = []
            currency_volatilities = {ccy_pair: [] for ccy_pair in data_dict.keys()}
            for ccy_pair1, ccy_pair2 in currency_pairs:
                if ccy_pair1 not in df_all.columns or ccy_pair2 not in df_all.columns:
                    continue
                current_spread = df_all.loc[latest_date, ccy_pair1] - df_all.loc[latest_date, ccy_pair2]
                historical_spreads = df_all[ccy_pair1] - df_all[ccy_pair2]
                three_months_ago = pd.Timestamp(latest_date) - pd.Timedelta(days=90)
                one_year_ago     = pd.Timestamp(latest_date) - pd.Timedelta(days=365)
                three_years_ago  = pd.Timestamp(latest_date) - pd.Timedelta(days=365*3)
                five_years_ago   = pd.Timestamp(latest_date) - pd.Timedelta(days=365*5)
                past_3_months_spreads = historical_spreads.loc[historical_spreads.index >= three_months_ago]
                past_year_spreads     = historical_spreads.loc[historical_spreads.index >= one_year_ago]
                past_3_years_spreads  = historical_spreads.loc[historical_spreads.index >= three_years_ago]
                past_5_years_spreads  = historical_spreads.loc[historical_spreads.index >= five_years_ago]
                percentile_3_months = percentileofscore(past_3_months_spreads, current_spread)
                percentile_1_year   = percentileofscore(past_year_spreads, current_spread)
                percentile_3_years  = percentileofscore(past_3_years_spreads, current_spread) if len(past_3_years_spreads) > 1 else None
                percentile_5_years  = percentileofscore(past_5_years_spreads, current_spread) if len(past_5_years_spreads) > 1 else None
                spread_volatility_3m = past_3_months_spreads.std()
                spread_volatility_1y = past_year_spreads.std()
                spread_volatility_3y = past_3_years_spreads.std() if len(past_3_years_spreads) > 1 else None
                spread_volatility_5y = past_5_years_spreads.std() if len(past_5_years_spreads) > 1 else None
                mean_spread_3m = past_3_months_spreads.mean()
                mean_spread_1y = past_year_spreads.mean()
                mean_spread_3y = past_3_years_spreads.mean() if len(past_3_years_spreads) > 1 else None
                mean_spread_5y = past_5_years_spreads.mean() if len(past_5_years_spreads) > 1 else None
                relative_position_3m = abs(current_spread - mean_spread_3m) / spread_volatility_3m if (spread_volatility_3m and spread_volatility_3m > 0) else 0
                relative_position_1y = abs(current_spread - mean_spread_1y) / spread_volatility_1y if (spread_volatility_1y and spread_volatility_1y > 0) else 0
                relative_position_3y = (abs(current_spread - mean_spread_3y) / spread_volatility_3y if (spread_volatility_3y and spread_volatility_3y > 0 and mean_spread_3y is not None) else None)
                relative_position_5y = (abs(current_spread - mean_spread_5y) / spread_volatility_5y if (spread_volatility_5y and spread_volatility_5y > 0 and mean_spread_5y is not None) else None)
                correlation = correlation_matrix.loc[ccy_pair1, ccy_pair2] if (
                    ccy_pair1 in correlation_matrix.index and ccy_pair2 in correlation_matrix.columns
                ) else None
                pair_result = {
                    'Tenor': tenor,
                    'Pair': f"{ccy_pair1}-{ccy_pair2}",
                    'CCY1': ccy_pair1,
                    'CCY2': ccy_pair2,
                    'Correlation': round(correlation, 4) if correlation is not None else None,
                    'Current_Spread': round(current_spread, 4),
                    'Spread_STD_3M': round(spread_volatility_3m, 4) if spread_volatility_3m is not None else None,
                    'Spread_STD_1Y': round(spread_volatility_1y, 4) if spread_volatility_1y is not None else None,
                    'Spread_STD_3Y': round(spread_volatility_3y, 4) if spread_volatility_3y is not None else None,
                    'Spread_STD_5Y': round(spread_volatility_5y, 4) if spread_volatility_5y is not None else None,
                    'Spread_ZScore_3M': round(relative_position_3m, 4) if relative_position_3m is not None else None,
                    'Spread_ZScore_1Y': round(relative_position_1y, 4) if relative_position_1y is not None else None,
                    'Spread_ZScore_3Y': round(relative_position_3y, 4) if relative_position_3y is not None else None,
                    'Spread_ZScore_5Y': round(relative_position_5y, 4) if relative_position_5y is not None else None,
                    'Percentile_3M': round(percentile_3_months, 3) if percentile_3_months is not None else None,
                    'Percentile_1Y': round(percentile_1_year, 3) if percentile_1_year is not None else None,
                    'Percentile_3Y': round(percentile_3_years, 3) if percentile_3_years is not None else None,
                    'Percentile_5Y': round(percentile_5_years, 3) if percentile_5_years is not None else None}
                pair_results.append(pair_result)
                # --------------------- Average STD of Vol Spreads over history ---------------------
                if (spread_volatility_1y is not None) and (relative_position_1y is not None) and (percentile_1_year is not None):
                    currency_volatilities[ccy_pair1].append({
                        'Spread_MEAN_3m': mean_spread_3m,
                        'Spread_MEAN_1y': mean_spread_1y,
                        'Spread_MEAN_3y': mean_spread_3y,
                        'Spread_MEAN_5y': mean_spread_5y,       
                        'Spread_STD_3m': spread_volatility_3m,
                        'Spread_STD_1y': spread_volatility_1y,
                        'Spread_STD_3y': spread_volatility_3y,
                        'Spread_STD_5y': spread_volatility_5y,
                        'pair': f"{ccy_pair1}-{ccy_pair2}"})
            results_by_tenor[tenor] = pd.DataFrame(pair_results)
            tenor_currency_averages = []
            for ccy_pair in data_dict.keys():
                if currency_volatilities[ccy_pair]:
                    avg_spread_mean_3m = np.mean([x['Spread_MEAN_3m'] for x in currency_volatilities[ccy_pair]])
                    avg_spread_mean_1y = np.mean([x['Spread_MEAN_1y'] for x in currency_volatilities[ccy_pair]])
                    avg_spread_mean_3y = np.mean([x['Spread_MEAN_3y'] for x in currency_volatilities[ccy_pair]])
                    avg_spread_mean_5y = np.mean([x['Spread_MEAN_5y'] for x in currency_volatilities[ccy_pair]])        
                    avg_spread_std_3m = np.mean([x['Spread_STD_3m'] for x in currency_volatilities[ccy_pair]])
                    avg_spread_std_1y = np.mean([x['Spread_STD_1y'] for x in currency_volatilities[ccy_pair]])
                    avg_spread_std_3y = np.mean([x['Spread_STD_3y'] for x in currency_volatilities[ccy_pair]])
                    avg_spread_std_5y = np.mean([x['Spread_STD_5y'] for x in currency_volatilities[ccy_pair]])
                    num_pairs = len(currency_volatilities[ccy_pair])
                    tenor_currency_averages.append({
                        'Currency': ccy_pair,
                        'Tenor': tenor,
                        'Ave_histSpread_MEAN_3m': round(avg_spread_mean_3m, 4),
                        'Ave_histSpread_MEAN_1y': round(avg_spread_mean_1y, 4),
                        'Ave_histSpread_MEAN_3y': round(avg_spread_mean_3y, 4),
                        'Ave_histSpread_MEAN_5y': round(avg_spread_mean_5y, 4),
                        'Avg_histSpread_STD_3m': round(avg_spread_std_3m, 4),
                        'Avg_histSpread_STD_1y': round(avg_spread_std_1y, 4),
                        'Avg_histSpread_STD_3y': round(avg_spread_std_3y, 4),
                        'Avg_histSpread_STD_5y': round(avg_spread_std_5y, 4),
                        'Num_Pairs': num_pairs})
            currency_avg_metrics[tenor] = (
                pd.DataFrame(tenor_currency_averages).sort_values('Avg_histSpread_STD_5y')
                if tenor_currency_averages else pd.DataFrame(columns=['Currency','Tenor',
                                                            'Ave_histSpread_MEAN_3m', 'Ave_histSpread_MEAN_1y', 'Ave_histSpread_MEAN_3y', 'Ave_histSpread_MEAN_5y',
                                                            'Avg_histSpread_STD_3m','Avg_histSpread_STD_1y','Avg_histSpread_STD_3y', 'Avg_histSpread_STD_5y','Num_Pairs']))
        results_by_date[analysis_ts] = {'results_by_tenor': results_by_tenor,'currency_avg_metrics': currency_avg_metrics, 'correlation_data': correlation_data}
    return results_by_date


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

""" Bucketing CCys Into Tenors - Averaging out each spread percentile conbination for respective CCY"""

# --------- Equal weighting ---------
def find_ave_allCCYallLookback_equalWeights(results_by_tenor, tenors):
    tenor_results = {}  
    for tenor in tenors:
        if tenor in results_by_tenor:
            df = results_by_tenor[tenor]
            all_currency_pairs = set(df['CCY1'].tolist() + df['CCY2'].tolist())
            currency_percentile_summary = []
            for ccy_pair in all_currency_pairs:  
                ccy_pairs = df[df['CCY1'] == ccy_pair] 
                if len(ccy_pairs) > 0:
                    avg_percentile_3m = ccy_pairs['Percentile_3M'].mean()
                    avg_percentile_1y = ccy_pairs['Percentile_1Y'].mean()
                    percentile_3y_values = ccy_pairs['Percentile_3Y'].dropna()
                    percentile_5y_values = ccy_pairs['Percentile_5Y'].dropna()
                    avg_percentile_3y = percentile_3y_values.mean() if len(percentile_3y_values) > 0 else None
                    avg_percentile_5y = percentile_5y_values.mean() if len(percentile_5y_values) > 0 else None
                    currency_percentile_summary.append({
                        'Currency_Pair': ccy_pair, 
                        'Tenor': tenor,
                        'Avg_EquWeight_3M%': round(avg_percentile_3m, 3),
                        'Avg_EquWeight_1Y%': round(avg_percentile_1y, 3),
                        'Avg_EquWeight_3Y%': round(avg_percentile_3y, 3) if avg_percentile_3y is not None else None,
                        'Avg_EquWeight_5Y%': round(avg_percentile_5y, 3) if avg_percentile_5y is not None else None,
                        'Num_Pairs': len(ccy_pairs)})
            percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_EquWeight_5Y%')
            tenor_results[tenor] = percentile_df
    return tenor_results


# --------- Correlation Weighting ---------
def find_ave_allCCYallLookback_correlationWeights(results_by_tenor, tenors):
        tenor_results = {}  
        for tenor in tenors:
            if tenor in results_by_tenor:
                df = results_by_tenor[tenor]
                all_currency_pairs = set(df['CCY1'].tolist() + df['CCY2'].tolist())
                currency_percentile_summary = []
                for ccy_pair in all_currency_pairs:  
                    ccy_pairs = df[df['CCY1'] == ccy_pair].copy()  
                    if len(ccy_pairs) > 0:
                        ccy_pairs_valid = ccy_pairs.dropna(subset=['Correlation'])
                        if len(ccy_pairs_valid) == 0:
                            weights = np.ones(len(ccy_pairs)) / len(ccy_pairs)
                            ccy_pairs_for_calc = ccy_pairs
                        else:
                            correlation_weights = ccy_pairs_valid['Correlation'].abs()  
                            weights = correlation_weights / correlation_weights.sum()  
                            ccy_pairs_for_calc = ccy_pairs_valid
                        weighted_avg_percentile_3m = np.average(ccy_pairs_for_calc['Percentile_3M'], weights=weights)
                        weighted_avg_percentile_1y = np.average(ccy_pairs_for_calc['Percentile_1Y'], weights=weights)
                        percentile_3y_valid = ccy_pairs_for_calc['Percentile_3Y'].dropna()
                        percentile_5y_valid = ccy_pairs_for_calc['Percentile_5Y'].dropna()
                        weighted_avg_percentile_3y = None
                        weighted_avg_percentile_5y = None
                        if len(percentile_3y_valid) > 0:
                            valid_3y_indices = ccy_pairs_for_calc['Percentile_3Y'].dropna().index
                            weights_3y = weights[ccy_pairs_for_calc.index.isin(valid_3y_indices)]
                            weights_3y = weights_3y / weights_3y.sum()  
                            weighted_avg_percentile_3y = np.average(percentile_3y_valid, weights=weights_3y)
                        if len(percentile_5y_valid) > 0:
                            # Get weights for valid 5Y data
                            valid_5y_indices = ccy_pairs_for_calc['Percentile_5Y'].dropna().index
                            weights_5y = weights[ccy_pairs_for_calc.index.isin(valid_5y_indices)]
                            weights_5y = weights_5y / weights_5y.sum() 
                            weighted_avg_percentile_5y = np.average(percentile_5y_valid, weights=weights_5y)
                        avg_correlation = ccy_pairs_valid['Correlation'].mean() if len(ccy_pairs_valid) > 0 else None
                        currency_percentile_summary.append({
                            'Currency_Pair': ccy_pair,  
                            'Tenor': tenor,
                            'Avg_CorrWeight_3M%': round(weighted_avg_percentile_3m, 3),
                            'Avg_CorrWeight_1Y%': round(weighted_avg_percentile_1y, 3),
                            'Avg_CorrWeight_3Y%': round(weighted_avg_percentile_3y, 3) if weighted_avg_percentile_3y is not None else None,
                            'Avg_CorrWeight_5Y%': round(weighted_avg_percentile_5y, 3) if weighted_avg_percentile_5y is not None else None,
                            'Num_Pairs': len(ccy_pairs)})
                percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_CorrWeight_5Y%', ascending=False)
                tenor_results[tenor] = percentile_df
        return tenor_results


def find_ave_allCCYallLookback_cointegrationWeights(results_by_tenor, tenors, cointegration_scores):
    tenor_results = {}
    for tenor in tenors:
        if tenor in results_by_tenor and tenor in cointegration_scores:
            df = results_by_tenor[tenor]
            tenor_coint_scores = cointegration_scores[tenor]
            all_currency_pairs = set(df['CCY1'].tolist() + df['CCY2'].tolist())
            currency_percentile_summary = []
            for ccy_pair in all_currency_pairs:
                ccy_pairs = df[df['CCY1'] == ccy_pair].copy()
                if len(ccy_pairs) > 0:
                    weights_list = []
                    for _, row in ccy_pairs.iterrows():
                        ccy2 = row['CCY2']
                        pair_key = (ccy_pair, ccy2)
                        if pair_key in tenor_coint_scores:
                            weight = tenor_coint_scores[pair_key]
                        else:
                            weight = 1.0
                        weights_list.append(weight)
                    weights = np.array(weights_list)
                    weights = np.maximum(weights, 0.01)
                    weights = weights / weights.sum()
                    weighted_avg_percentile_3m = np.average(ccy_pairs['Percentile_3M'], weights=weights)
                    weighted_avg_percentile_1y = np.average(ccy_pairs['Percentile_1Y'], weights=weights)
                    percentile_3y_valid = ccy_pairs['Percentile_3Y'].dropna()
                    percentile_5y_valid = ccy_pairs['Percentile_5Y'].dropna()
                    weighted_avg_percentile_3y = None
                    weighted_avg_percentile_5y = None
                    if len(percentile_3y_valid) > 0:
                        valid_3y_indices = ccy_pairs['Percentile_3Y'].notna()
                        weights_3y = weights[valid_3y_indices]
                        weights_3y = weights_3y / weights_3y.sum()
                        weighted_avg_percentile_3y = np.average(percentile_3y_valid, weights=weights_3y)
                    if len(percentile_5y_valid) > 0:
                        valid_5y_indices = ccy_pairs['Percentile_5Y'].notna()
                        weights_5y = weights[valid_5y_indices]
                        weights_5y = weights_5y / weights_5y.sum()
                        weighted_avg_percentile_5y = np.average(percentile_5y_valid, weights=weights_5y)
                    avg_coint_score = np.mean(weights_list)
                    currency_percentile_summary.append({
                        'Currency_Pair': ccy_pair,
                        'Tenor': tenor,
                        'Avg_CointWeight_3M%': round(weighted_avg_percentile_3m, 3),
                        'Avg_CointWeight_1Y%': round(weighted_avg_percentile_1y, 3),
                        'Avg_CointWeight_3Y%': round(weighted_avg_percentile_3y, 3) if weighted_avg_percentile_3y is not None else None,
                        'Avg_CointWeight_5Y%': round(weighted_avg_percentile_5y, 3) if weighted_avg_percentile_5y is not None else None,
                        'Avg_Coint_Score': round(avg_coint_score, 2),
                        'Num_Pairs': len(ccy_pairs)})
            percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_CointWeight_5Y%', ascending=False)
            tenor_results[tenor] = percentile_df
    return tenor_results





# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

"""
Generating Histroical Results 
"""


"""
Using Average 

Inputs:
    - days_backtest: Amount of days in past you want to get results for 
    - frequency: How often you want the percentile calculations over the backtesting days (Typically weekly)
"""
def ATM_PercentileEquWeightAve_HistoricalResults(currency_list, tenors, days_backtested, frequency):
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    if frequency == 'weekly': 
        date_range = pd.date_range(start=start_date, end=end_date, freq=pd.offsets.Week(weekday=datetime.now().weekday()))
    elif frequency == 'monthly': 
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    else:  
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    bulk_start_date = (datetime.now() - timedelta(days=days_backtested + 365*5)).strftime("%Y-%m-%d")
    bulk_vol_data = get_all_volatility_data_bulk(currency_list, tenors, bulk_start_date, end_date)
    historical_results = {}
    for tenor in tenors:
        historical_results[tenor] = {}
        for currency_pair in currency_list:
            historical_results[tenor][currency_pair] = []
    for i, current_date in enumerate(date_range):
        try:
            daily_results = xCCY_AllPairs_MultiTenor_Analysis(
                bulk_vol_data, tenors, [current_date]) 
            if current_date in daily_results:
                single_date_results = daily_results[current_date]['results_by_tenor']
                daily_percentiles = find_ave_allCCYallLookback_equalWeights(
                    single_date_results, tenors)
            else:
                print(f"No results found for {current_date}")
                continue
            for tenor in tenors:
                if tenor in daily_percentiles:
                    df = daily_percentiles[tenor]
                    for _, row in df.iterrows():
                        historical_results[tenor][row['Currency_Pair']].append({
                            'date': current_date,
                            'Avg_EquWeight_3M%': row['Avg_EquWeight_3M%'],
                            'Avg_EquWeight_1Y%': row['Avg_EquWeight_1Y%'],
                            'Avg_EquWeight_3Y%': row['Avg_EquWeight_3Y%'],
                            'Avg_EquWeight_5Y%': row['Avg_EquWeight_5Y%'],
                            'Num_Pairs': row['Num_Pairs']})
        except Exception as e:
            print(f"Error processing {current_date}: {e}")
            continue
    formatted_results = {}
    for tenor in tenors:
        formatted_results[tenor] = {}
        for currency_pair in currency_list:
            if historical_results[tenor][currency_pair]:
                df = pd.DataFrame(historical_results[tenor][currency_pair])
                df.set_index('date', inplace=True)
                formatted_results[tenor][currency_pair] = df
    return formatted_results





def ATM_PercentileCorrelationWeightAve_HistoricalResults(currency_list, tenors, days_backtested, frequency):
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    if frequency == 'weekly': 
        date_range = pd.date_range(start=start_date, end=end_date, freq=pd.offsets.Week(weekday=datetime.now().weekday()))
    elif frequency == 'monthly': 
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    else:  
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    bulk_start_date = (datetime.now() - timedelta(days=days_backtested + 365*5)).strftime("%Y-%m-%d")
    bulk_vol_data = get_all_volatility_data_bulk(currency_list, tenors, bulk_start_date, end_date)
    historical_results = {}
    for tenor in tenors:
        historical_results[tenor] = {}
        for currency_pair in currency_list:
            historical_results[tenor][currency_pair] = []
    for i, current_date in enumerate(date_range):
        try:
            daily_results = xCCY_AllPairs_MultiTenor_Analysis(
                bulk_vol_data, tenors, [current_date]) 
            if current_date in daily_results:
                single_date_results = daily_results[current_date]['results_by_tenor']
                # Use correlation weighting instead of equal weighting
                daily_percentiles = find_ave_allCCYallLookback_correlationWeights(
                    single_date_results, tenors)
            else:
                print(f"No results found for {current_date}")
                continue
            for tenor in tenors:
                if tenor in daily_percentiles:
                    df = daily_percentiles[tenor]
                    for _, row in df.iterrows():
                        historical_results[tenor][row['Currency_Pair']].append({
                            'date': current_date,
                            'Avg_CorrWeight_3M%': row['Avg_CorrWeight_3M%'],
                            'Avg_CorrWeight_1Y%': row['Avg_CorrWeight_1Y%'],
                            'Avg_CorrWeight_3Y%': row['Avg_CorrWeight_3Y%'],
                            'Avg_CorrWeight_5Y%': row['Avg_CorrWeight_5Y%'],
                            'Num_Pairs': row['Num_Pairs']})                
        except Exception as e:
            print(f"Error processing {current_date}: {e}")
            continue
    formatted_results = {}
    for tenor in tenors:
        formatted_results[tenor] = {}
        for currency_pair in currency_list:
            if historical_results[tenor][currency_pair]:
                df = pd.DataFrame(historical_results[tenor][currency_pair])
                df.set_index('date', inplace=True)
                formatted_results[tenor][currency_pair] = df
    return formatted_results




def ATM_PercentileCointegrationWeightAve_HistoricalResults(currency_list, tenors, days_backtested, frequency, 
                                                           cointegration_excel_path):
    cointegration_scores = load_cointegration_scores(cointegration_excel_path, currency_list, tenors)
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    if frequency == 'weekly':
        date_range = pd.date_range(start=start_date, end=end_date, 
                                 freq=pd.offsets.Week(weekday=datetime.now().weekday()))
    elif frequency == 'monthly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    else:  # daily
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    bulk_start_date = (datetime.now() - timedelta(days=days_backtested + 365*5)).strftime("%Y-%m-%d")
    bulk_vol_data = get_all_volatility_data_bulk(currency_list, tenors, bulk_start_date, end_date)
    historical_results = {}
    for tenor in tenors:
        historical_results[tenor] = {}
        for currency_pair in currency_list:
            historical_results[tenor][currency_pair] = []
    for i, current_date in enumerate(date_range):
        try:
            daily_results = xCCY_AllPairs_MultiTenor_Analysis(
                bulk_vol_data, tenors, [current_date])
            if current_date in daily_results:
                single_date_results = daily_results[current_date]['results_by_tenor']
                daily_percentiles = find_ave_allCCYallLookback_cointegrationWeights(
                    single_date_results, tenors, cointegration_scores)
            else:
                print(f"No results found for {current_date}")
                continue
            for tenor in tenors:
                if tenor in daily_percentiles:
                    df = daily_percentiles[tenor]
                    for _, row in df.iterrows():
                        historical_results[tenor][row['Currency_Pair']].append({
                            'date': current_date,
                            'Avg_CointWeight_3M%': row['Avg_CointWeight_3M%'],
                            'Avg_CointWeight_1Y%': row['Avg_CointWeight_1Y%'],
                            'Avg_CointWeight_3Y%': row['Avg_CointWeight_3Y%'],
                            'Avg_CointWeight_5Y%': row['Avg_CointWeight_5Y%'],
                            'Avg_Coint_Score': row.get('Avg_Coint_Score', np.nan),
                            'Num_Pairs': row['Num_Pairs']})
        except Exception as e:
            print(f"Error processing {current_date}: {e}")
            continue
    formatted_results = {}
    for tenor in tenors:
        formatted_results[tenor] = {}
        for currency_pair in currency_list:
            if historical_results[tenor][currency_pair]:
                df = pd.DataFrame(historical_results[tenor][currency_pair])
                df.set_index('date', inplace=True)
                formatted_results[tenor][currency_pair] = df
    return formatted_results









# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


"""Get Daily Fixing Historical Vol """
# Historical Realized Vol for 
def get_realized_volatility_dataBloomberg(currency_list, tenors, days_backtested):
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    realized_vol_data = {}
    for tenor in tenors:
        tenor_dict = {}
        for ccy_pair in currency_list:
            ticker_HV = f"{ccy_pair}H{tenor} CMPN Curncy"
            try:
                data_HV = blp.bdh(
                    tickers=ticker_HV,
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date
                )
                if not data_HV.empty:
                    data_HV.columns = data_HV.columns.get_level_values(1)
                    tenor_dict[ccy_pair] = data_HV['PX_LAST']
            except Exception as e:
                print(f"Error getting data for {ccy_pair} {tenor}: {e}")
                continue
        realized_vol_data[tenor] = tenor_dict
    return realized_vol_data



def get_implied_volatility_dataBloomberg(currency_list, tenors, days_backtested):
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    realized_vol_data = {}
    for tenor in tenors:
        tenor_dict = {}
        for ccy_pair in currency_list:
            ticker_HV = f"{ccy_pair}V{tenor} Curncy"
            try:
                data_HV = blp.bdh(
                    tickers=ticker_HV,
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date)
                if not data_HV.empty:
                    data_HV.columns = data_HV.columns.get_level_values(1)
                    tenor_dict[ccy_pair] = data_HV['PX_LAST']
            except Exception as e:
                print(f"Error getting data for {ccy_pair} {tenor}: {e}")
                continue
        realized_vol_data[tenor] = tenor_dict
    return realized_vol_data



# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

""" Utility Functions """
def tenor_to_business_days(tenor: str) -> int:
    tenor_map = {
        'ON': 1,
        '1W': 7,   
        '2W': 14,
        '3W': 21, 
        '1M': 30,  
        '2M': 60, 
        '3M': 90}
    return tenor_map.get(tenor, 30) 

def find_closest_business_day(target_date: pd.Timestamp, available_dates: pd.Index, max_days: int = 5) -> pd.Timestamp:
    if target_date in available_dates:
        return target_date
    available_dates = pd.to_datetime(available_dates)
    target_date = pd.to_datetime(target_date)
    for days_offset in range(1, max_days + 1):
        for direction in [1, -1]:  # Try both forward and backward
            candidate_date = target_date + pd.Timedelta(days=direction * days_offset)
            if candidate_date in available_dates:
                return candidate_date
    return None


def calculate_max_drawdown_stats(trades_df):
    cheap_trades = trades_df[trades_df['signal_type'] == 'cheap']
    expensive_trades = trades_df[trades_df['signal_type'] == 'expensive']
    cheap_max_drawdown = cheap_trades['vol_diff'].min() if len(cheap_trades) > 0 else np.nan
    cheap_max_drawdown_pct = cheap_trades['vol_diff_pct'].min() if len(cheap_trades) > 0 else np.nan
    expensive_max_drawdown = expensive_trades['vol_diff'].max() if len(expensive_trades) > 0 else np.nan
    expensive_max_drawdown_pct = expensive_trades['vol_diff_pct'].max() if len(expensive_trades) > 0 else np.nan
    cheap_losses = cheap_trades['vol_diff'][cheap_trades['vol_diff'] < 0] if len(cheap_trades) > 0 else pd.Series(dtype=float)
    expensive_losses = expensive_trades['vol_diff'][expensive_trades['vol_diff'] > 0] if len(expensive_trades) > 0 else pd.Series(dtype=float)
    cheap_losses_pct = cheap_trades['vol_diff_pct'][cheap_trades['vol_diff_pct'] < 0] if len(cheap_trades) > 0 else pd.Series(dtype=float)
    expensive_losses_pct = expensive_trades['vol_diff_pct'][expensive_trades['vol_diff_pct'] > 0] if len(expensive_trades) > 0 else pd.Series(dtype=float)
    all_losses = []
    all_losses_pct = []
    if len(cheap_losses) > 0:
        all_losses.extend(cheap_losses.tolist())
    if len(expensive_losses) > 0:
        all_losses.extend(expensive_losses.tolist())
    if len(cheap_losses_pct) > 0:
        all_losses_pct.extend(cheap_losses_pct.tolist())
    if len(expensive_losses_pct) > 0:
        all_losses_pct.extend(expensive_losses_pct.tolist())
    if len(all_losses) > 0:
        overall_max_drawdown = max(all_losses, key=abs)
        overall_max_drawdown_pct = max(all_losses_pct, key=abs)
    else:
        overall_max_drawdown = np.nan
        overall_max_drawdown_pct = np.nan
    return {
        'cheap_max_drawdown': cheap_max_drawdown,
        'cheap_max_drawdown_pct': cheap_max_drawdown_pct,
        'expensive_max_drawdown': expensive_max_drawdown,
        'expensive_max_drawdown_pct': expensive_max_drawdown_pct,
        'overall_max_drawdown': overall_max_drawdown,
        'overall_max_drawdown_pct': overall_max_drawdown_pct}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def backtest_ImpliedRealizedDiff_EqualWeightAve_Results(currency_list, gamma_tenors, days_backtested, percentile_threshold_low, percentile_threshold_high, frequency):
    historical_percentiles = ATM_PercentileEquWeightAve_HistoricalResults(currency_list, gamma_tenors, days_backtested, frequency)
    print("Got data")
    implied_vol_data = get_implied_volatility_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    realized_vol_data = get_realized_volatility_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    backtest_results = {}
    all_trades = []
    for tenor in gamma_tenors:
        tenor_trades = []
        for currency_pair in currency_list:
            percentile_df = historical_percentiles[tenor][currency_pair]
            implied_vol_series = implied_vol_data[tenor][currency_pair]
            realized_vol_series = realized_vol_data[tenor][currency_pair]
            percentile_df.index = pd.to_datetime(percentile_df.index)
            implied_vol_series.index = pd.to_datetime(implied_vol_series.index)
            realized_vol_series.index = pd.to_datetime(realized_vol_series.index)
            tenor_days = tenor_to_business_days(tenor)
            signals_processed = 0
            trades_made = 0
            for signal_date in percentile_df.index:
                signals_processed += 1
                try:
                    signal_row = percentile_df.loc[signal_date]
                    percentile_3m = signal_row['Avg_EquWeight_3M%']
                    signal_percentile = percentile_3m
                    if pd.isna(signal_percentile):
                        continue
                except Exception as e:
                    print(f"    Error getting percentile for {signal_date}: {e}")
                    continue
                signal_type = None
                if signal_percentile <= percentile_threshold_low:
                    signal_type = 'cheap'  # Expect RV > IV
                elif signal_percentile >= percentile_threshold_high:
                    signal_type = 'expensive'  # Expect RV < IV
                else:
                    continue  # No signal
                # ----------- Get implied vol at signal date -----------
                try:
                    if signal_date in implied_vol_series.index:
                        signal_implied_vol = implied_vol_series.loc[signal_date]
                    else:
                        closest_iv_date = find_closest_business_day(signal_date, implied_vol_series.index)
                        if closest_iv_date is None:
                            continue
                        signal_implied_vol = implied_vol_series.loc[closest_iv_date]
                    if pd.isna(signal_implied_vol) or signal_implied_vol <= 0:
                        continue
                except Exception as e:
                    print(f"    Error getting implied vol for {signal_date}: {e}")
                    continue
                future_date = signal_date + pd.Timedelta(days=tenor_days) # Future Date for Realized Vol
                
                # ----------- Get realized vol at future date -----------
                try:
                    if future_date in realized_vol_series.index:
                        future_realized_vol = realized_vol_series.loc[future_date]
                        actual_future_date = future_date
                    else:
                        closest_rv_date = find_closest_business_day(future_date, realized_vol_series.index)
                        if closest_rv_date is None:
                            continue
                        future_realized_vol = realized_vol_series.loc[closest_rv_date]
                        actual_future_date = closest_rv_date
                    if pd.isna(future_realized_vol) or future_realized_vol <= 0:
                        continue
                except Exception as e:
                    print(f"    Error getting realized vol for {future_date}: {e}")
                    continue
                # Calculate trade metrics
                vol_diff = future_realized_vol - signal_implied_vol
                vol_diff_pct = vol_diff / signal_implied_vol
                if signal_type == 'cheap':# Expected RV > IV (bought volatility)
                    trade_success = future_realized_vol > signal_implied_vol
                    expected_direction = 'RV > IV'
                else:  # expensive
                    # Expected RV < IV (sold volatility)
                    trade_success = future_realized_vol < signal_implied_vol
                    expected_direction = 'RV < IV'
                trade_record = {
                    'signal_date': signal_date,
                    'future_date': actual_future_date,
                    'currency_pair': currency_pair,
                    'tenor': tenor,
                    'signal_type': signal_type,
                    'signal_percentile': signal_percentile,
                    'implied_vol': signal_implied_vol,
                    'realized_vol': future_realized_vol,
                    'vol_diff': vol_diff,
                    'vol_diff_pct': vol_diff_pct,
                    'trade_success': trade_success,
                    'expected_direction': expected_direction,
                    'days_held': (actual_future_date - signal_date).days}
                tenor_trades.append(trade_record)
                all_trades.append(trade_record)
                trades_made += 1
        if tenor_trades:
            tenor_df = pd.DataFrame(tenor_trades)
            total_trades = len(tenor_df)
            hit_rate = tenor_df['trade_success'].mean()
            cheap_trades = tenor_df[tenor_df['signal_type'] == 'cheap']
            expensive_trades = tenor_df[tenor_df['signal_type'] == 'expensive']
            cheap_hit_rate = cheap_trades['trade_success'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_hit_rate = expensive_trades['trade_success'].mean() if len(expensive_trades) > 0 else np.nan
            cheap_avg_vol_diff = cheap_trades['vol_diff'].mean() if len(cheap_trades) > 0 else np.nan
            cheap_avg_vol_diff_pct = cheap_trades['vol_diff_pct'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_avg_vol_diff = expensive_trades['vol_diff'].mean() if len(expensive_trades) > 0 else np.nan
            expensive_avg_vol_diff_pct = expensive_trades['vol_diff_pct'].mean() if len(expensive_trades) > 0 else np.nan
            avg_vol_diff = tenor_df['vol_diff'].mean()
            avg_vol_diff_pct = tenor_df['vol_diff_pct'].mean()
            drawdown_stats = calculate_max_drawdown_stats(tenor_df)
            backtest_results[tenor] = {
                'data': tenor_df,
                'total_trades': total_trades,
                'overall_hit_rate': hit_rate,
                'cheap_trades': len(cheap_trades),
                'expensive_trades': len(expensive_trades),
                'cheap_hit_rate': cheap_hit_rate,
                'expensive_hit_rate': expensive_hit_rate,
                'avg_vol_diff': avg_vol_diff,
                'avg_vol_diff_pct': avg_vol_diff_pct,
                'cheap_avg_vol_diff': cheap_avg_vol_diff,
                'cheap_avg_vol_diff_pct': cheap_avg_vol_diff_pct,
                'expensive_avg_vol_diff': expensive_avg_vol_diff,
                'expensive_avg_vol_diff_pct': expensive_avg_vol_diff_pct,
                # Add max drawdown statistics
                'cheap_max_drawdown': drawdown_stats['cheap_max_drawdown'],
                'cheap_max_drawdown_pct': drawdown_stats['cheap_max_drawdown_pct'],
                'expensive_max_drawdown': drawdown_stats['expensive_max_drawdown'],
                'expensive_max_drawdown_pct': drawdown_stats['expensive_max_drawdown_pct'],
                'overall_max_drawdown': drawdown_stats['overall_max_drawdown'],
                'overall_max_drawdown_pct': drawdown_stats['overall_max_drawdown_pct']}
    def tidy_backtest(backtest_results: dict):
        rows = []
        for tenor, d in backtest_results.items():
            rows.append({
                "tenor": tenor,
                "total_trades": int(d["total_trades"]),
                "overall_hit_rate": float(d["overall_hit_rate"]),
                "cheap_trades": int(d["cheap_trades"]),
                "expensive_trades": int(d["expensive_trades"]),
                "cheap_hit_rate": float(d["cheap_hit_rate"]) if not pd.isna(d["cheap_hit_rate"]) else np.nan,
                "expensive_hit_rate": float(d["expensive_hit_rate"]) if not pd.isna(d["expensive_hit_rate"]) else np.nan,
                "avg_vol_diff": float(d["avg_vol_diff"]),
                "avg_vol_diff_pct": float(d["avg_vol_diff_pct"]),
                "cheap_avg_vol_diff": float(d["cheap_avg_vol_diff"]) if not pd.isna(d["cheap_avg_vol_diff"]) else np.nan,
                "cheap_avg_vol_diff_pct": float(d["cheap_avg_vol_diff_pct"]) if not pd.isna(d["cheap_avg_vol_diff_pct"]) else np.nan,
                "expensive_avg_vol_diff": float(d["expensive_avg_vol_diff"]) if not pd.isna(d["expensive_avg_vol_diff"]) else np.nan,
                "expensive_avg_vol_diff_pct": float(d["expensive_avg_vol_diff_pct"]) if not pd.isna(d["expensive_avg_vol_diff_pct"]) else np.nan,
                # Add max drawdown columns
                "cheap_max_drawdown": float(d["cheap_max_drawdown"]) if not pd.isna(d["cheap_max_drawdown"]) else np.nan,
                "cheap_max_drawdown_pct": float(d["cheap_max_drawdown_pct"]) if not pd.isna(d["cheap_max_drawdown_pct"]) else np.nan,
                "expensive_max_drawdown": float(d["expensive_max_drawdown"]) if not pd.isna(d["expensive_max_drawdown"]) else np.nan,
                "expensive_max_drawdown_pct": float(d["expensive_max_drawdown_pct"]) if not pd.isna(d["expensive_max_drawdown_pct"]) else np.nan,
                "overall_max_drawdown": float(d["overall_max_drawdown"]) if not pd.isna(d["overall_max_drawdown"]) else np.nan,
                "overall_max_drawdown_pct": float(d["overall_max_drawdown_pct"]) if not pd.isna(d["overall_max_drawdown_pct"]) else np.nan})
        summary_df = pd.DataFrame(rows).set_index("tenor").sort_index()
        trades_df = pd.concat(
            {tenor: d["data"] for tenor, d in backtest_results.items()},
            names=["tenor"])         
        preferred_cols = [
            "tenor","signal_date","future_date","currency_pair","signal_type",
            "signal_percentile","implied_vol","realized_vol","vol_diff","vol_diff_pct",
            "trade_success","expected_direction","days_held"]
        trades_df = trades_df[[c for c in preferred_cols if c in trades_df.columns]]
        return summary_df, trades_df
    summary_df, trades_df = tidy_backtest(backtest_results)
    return summary_df.T, trades_df




def backtest_ImpliedRealizedDiff_CorrWeightAve_Results(currency_list, gamma_tenors, days_backtested, percentile_threshold_low, percentile_threshold_high, frequency):
    historical_percentiles = ATM_PercentileCorrelationWeightAve_HistoricalResults(currency_list, gamma_tenors, days_backtested, frequency)
    implied_vol_data = get_implied_volatility_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    realized_vol_data = get_realized_volatility_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    backtest_results = {}
    all_trades = []
    for tenor in gamma_tenors:
        tenor_trades = []
        for currency_pair in currency_list:
            percentile_df = historical_percentiles[tenor][currency_pair]
            implied_vol_series = implied_vol_data[tenor][currency_pair]
            realized_vol_series = realized_vol_data[tenor][currency_pair]
            percentile_df.index = pd.to_datetime(percentile_df.index)
            implied_vol_series.index = pd.to_datetime(implied_vol_series.index)
            realized_vol_series.index = pd.to_datetime(realized_vol_series.index)
            tenor_days = tenor_to_business_days(tenor)
            signals_processed = 0
            trades_made = 0
            for signal_date in percentile_df.index:
                signals_processed += 1
                try:
                    signal_row = percentile_df.loc[signal_date]
                    percentile_3m = signal_row['Avg_CorrWeight_3M%']
                    signal_percentile = percentile_3m
                    if pd.isna(signal_percentile):
                        continue
                except Exception as e:
                    print(f"    Error getting percentile for {signal_date}: {e}")
                    continue
                signal_type = None
                if signal_percentile <= percentile_threshold_low:
                    signal_type = 'cheap'  # Expect RV > IV
                elif signal_percentile >= percentile_threshold_high:
                    signal_type = 'expensive'  # Expect RV < IV
                else:
                    continue  # No signal
                # ----------- Get implied vol at signal date -----------
                try:
                    if signal_date in implied_vol_series.index:
                        signal_implied_vol = implied_vol_series.loc[signal_date]
                    else:
                        closest_iv_date = find_closest_business_day(signal_date, implied_vol_series.index)
                        if closest_iv_date is None:
                            continue
                        signal_implied_vol = implied_vol_series.loc[closest_iv_date]
                    if pd.isna(signal_implied_vol) or signal_implied_vol <= 0:
                        continue
                except Exception as e:
                    print(f"    Error getting implied vol for {signal_date}: {e}")
                    continue
                future_date = signal_date + pd.Timedelta(days=tenor_days) # Future Date for Realized Vol
                # ----------- Get realized vol at future date -----------
                try:
                    if future_date in realized_vol_series.index:
                        future_realized_vol = realized_vol_series.loc[future_date]
                        actual_future_date = future_date
                    else:
                        closest_rv_date = find_closest_business_day(future_date, realized_vol_series.index)
                        if closest_rv_date is None:
                            continue
                        future_realized_vol = realized_vol_series.loc[closest_rv_date]
                        actual_future_date = closest_rv_date
                    if pd.isna(future_realized_vol) or future_realized_vol <= 0:
                        continue
                except Exception as e:
                    print(f"    Error getting realized vol for {future_date}: {e}")
                    continue
                vol_diff = future_realized_vol - signal_implied_vol
                vol_diff_pct = vol_diff / signal_implied_vol
                if signal_type == 'cheap':# Expected RV > IV (bought volatility)
                    trade_success = future_realized_vol > signal_implied_vol
                    expected_direction = 'RV > IV'
                else:  # expensive
                    trade_success = future_realized_vol < signal_implied_vol
                    expected_direction = 'RV < IV'
                trade_record = {
                    'signal_date': signal_date,
                    'future_date': actual_future_date,
                    'currency_pair': currency_pair,
                    'tenor': tenor,
                    'signal_type': signal_type,
                    'signal_percentile': signal_percentile,
                    'implied_vol': signal_implied_vol,
                    'realized_vol': future_realized_vol,
                    'vol_diff': vol_diff,
                    'vol_diff_pct': vol_diff_pct,
                    'trade_success': trade_success,
                    'expected_direction': expected_direction,
                    'days_held': (actual_future_date - signal_date).days}
                tenor_trades.append(trade_record)
                all_trades.append(trade_record)
                trades_made += 1
        if tenor_trades:
            tenor_df = pd.DataFrame(tenor_trades)
            total_trades = len(tenor_df)
            hit_rate = tenor_df['trade_success'].mean()
            cheap_trades = tenor_df[tenor_df['signal_type'] == 'cheap']
            expensive_trades = tenor_df[tenor_df['signal_type'] == 'expensive']
            cheap_hit_rate = cheap_trades['trade_success'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_hit_rate = expensive_trades['trade_success'].mean() if len(expensive_trades) > 0 else np.na
            cheap_avg_vol_diff = cheap_trades['vol_diff'].mean() if len(cheap_trades) > 0 else np.nan
            cheap_avg_vol_diff_pct = cheap_trades['vol_diff_pct'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_avg_vol_diff = expensive_trades['vol_diff'].mean() if len(expensive_trades) > 0 else np.nan
            expensive_avg_vol_diff_pct = expensive_trades['vol_diff_pct'].mean() if len(expensive_trades) > 0 else np.nan
            avg_vol_diff = tenor_df['vol_diff'].mean()
            avg_vol_diff_pct = tenor_df['vol_diff_pct'].mean()
            drawdown_stats = calculate_max_drawdown_stats(tenor_df)
            backtest_results[tenor] = {
                'data': tenor_df,
                'total_trades': total_trades,
                'overall_hit_rate': hit_rate,
                'cheap_trades': len(cheap_trades),
                'expensive_trades': len(expensive_trades),
                'cheap_hit_rate': cheap_hit_rate,
                'expensive_hit_rate': expensive_hit_rate,
                'avg_vol_diff': avg_vol_diff,
                'avg_vol_diff_pct': avg_vol_diff_pct,
                'cheap_avg_vol_diff': cheap_avg_vol_diff,
                'cheap_avg_vol_diff_pct': cheap_avg_vol_diff_pct,
                'expensive_avg_vol_diff': expensive_avg_vol_diff,
                'expensive_avg_vol_diff_pct': expensive_avg_vol_diff_pct,
                'cheap_max_drawdown': drawdown_stats['cheap_max_drawdown'],
                'cheap_max_drawdown_pct': drawdown_stats['cheap_max_drawdown_pct'],
                'expensive_max_drawdown': drawdown_stats['expensive_max_drawdown'],
                'expensive_max_drawdown_pct': drawdown_stats['expensive_max_drawdown_pct'],
                'overall_max_drawdown': drawdown_stats['overall_max_drawdown'],
                'overall_max_drawdown_pct': drawdown_stats['overall_max_drawdown_pct']}
    def tidy_backtest(backtest_results: dict):
        rows = []
        for tenor, d in backtest_results.items():
            rows.append({
                "tenor": tenor,
                "total_trades": int(d["total_trades"]),
                "overall_hit_rate": float(d["overall_hit_rate"]),
                "cheap_trades": int(d["cheap_trades"]),
                "expensive_trades": int(d["expensive_trades"]),
                "cheap_hit_rate": float(d["cheap_hit_rate"]) if not pd.isna(d["cheap_hit_rate"]) else np.nan,
                "expensive_hit_rate": float(d["expensive_hit_rate"]) if not pd.isna(d["expensive_hit_rate"]) else np.nan,
                "avg_vol_diff": float(d["avg_vol_diff"]),
                "avg_vol_diff_pct": float(d["avg_vol_diff_pct"]),
                "cheap_avg_vol_diff": float(d["cheap_avg_vol_diff"]) if not pd.isna(d["cheap_avg_vol_diff"]) else np.nan,
                "cheap_avg_vol_diff_pct": float(d["cheap_avg_vol_diff_pct"]) if not pd.isna(d["cheap_avg_vol_diff_pct"]) else np.nan,
                "expensive_avg_vol_diff": float(d["expensive_avg_vol_diff"]) if not pd.isna(d["expensive_avg_vol_diff"]) else np.nan,
                "expensive_avg_vol_diff_pct": float(d["expensive_avg_vol_diff_pct"]) if not pd.isna(d["expensive_avg_vol_diff_pct"]) else np.nan,
                # Add max drawdown columns
                "cheap_max_drawdown": float(d["cheap_max_drawdown"]) if not pd.isna(d["cheap_max_drawdown"]) else np.nan,
                "cheap_max_drawdown_pct": float(d["cheap_max_drawdown_pct"]) if not pd.isna(d["cheap_max_drawdown_pct"]) else np.nan,
                "expensive_max_drawdown": float(d["expensive_max_drawdown"]) if not pd.isna(d["expensive_max_drawdown"]) else np.nan,
                "expensive_max_drawdown_pct": float(d["expensive_max_drawdown_pct"]) if not pd.isna(d["expensive_max_drawdown_pct"]) else np.nan,
                "overall_max_drawdown": float(d["overall_max_drawdown"]) if not pd.isna(d["overall_max_drawdown"]) else np.nan,
                "overall_max_drawdown_pct": float(d["overall_max_drawdown_pct"]) if not pd.isna(d["overall_max_drawdown_pct"]) else np.nan,})
        summary_df = pd.DataFrame(rows).set_index("tenor").sort_index()
        trades_df = pd.concat(
            {tenor: d["data"] for tenor, d in backtest_results.items()},
            names=["tenor"])
        preferred_cols = [
            "tenor","signal_date","future_date","currency_pair","signal_type",
            "signal_percentile","implied_vol","realized_vol","vol_diff","vol_diff_pct",
            "trade_success","expected_direction","days_held"]
        trades_df = trades_df[[c for c in preferred_cols if c in trades_df.columns]]
        return summary_df, trades_df
    summary_df, trades_df = tidy_backtest(backtest_results)
    return summary_df.T, trades_df






def backtest_ImpliedRealizedDiff_CointWeightAve_Results(currency_list, gamma_tenors, days_backtested, 
                                                        percentile_threshold_low, percentile_threshold_high, 
                                                        frequency, cointegration_excel_path):
    print("Loading cointegration weights and calculating historical percentiles...")
    historical_percentiles = ATM_PercentileCointegrationWeightAve_HistoricalResults(
        currency_list, gamma_tenors, days_backtested, frequency, cointegration_excel_path)
    print("Getting implied and realized volatility data...")
    implied_vol_data = get_implied_volatility_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    realized_vol_data = get_realized_volatility_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    backtest_results = {}
    all_trades = []
    for tenor in gamma_tenors:
        print(f"Processing tenor: {tenor}")
        tenor_trades = []
        for currency_pair in currency_list:
            percentile_df = historical_percentiles[tenor][currency_pair]
            implied_vol_series = implied_vol_data[tenor][currency_pair]
            realized_vol_series = realized_vol_data[tenor][currency_pair]
            percentile_df.index = pd.to_datetime(percentile_df.index)
            implied_vol_series.index = pd.to_datetime(implied_vol_series.index)
            realized_vol_series.index = pd.to_datetime(realized_vol_series.index)
            tenor_days = tenor_to_business_days(tenor)
            for signal_date in percentile_df.index:
                try:
                    signal_row = percentile_df.loc[signal_date]
                    signal_percentile = signal_row['Avg_CointWeight_3M%']
                    if pd.isna(signal_percentile):
                        continue
                    if signal_percentile <= percentile_threshold_low:
                        signal_type = 'cheap'  # Expect RV > IV
                    elif signal_percentile >= percentile_threshold_high:
                        signal_type = 'expensive'  # Expect RV < IV
                    else:
                        continue  # No signal
                    if signal_date in implied_vol_series.index:
                        signal_implied_vol = implied_vol_series.loc[signal_date]
                    else:
                        closest_iv_date = find_closest_business_day(signal_date, implied_vol_series.index)
                        if closest_iv_date is None:
                            continue
                        signal_implied_vol = implied_vol_series.loc[closest_iv_date]
                    if pd.isna(signal_implied_vol) or signal_implied_vol <= 0:
                        continue
                    future_date = signal_date + pd.Timedelta(days=tenor_days)
                    if future_date in realized_vol_series.index:
                        future_realized_vol = realized_vol_series.loc[future_date]
                        actual_future_date = future_date
                    else:
                        closest_rv_date = find_closest_business_day(future_date, realized_vol_series.index)
                        if closest_rv_date is None:
                            continue
                        future_realized_vol = realized_vol_series.loc[closest_rv_date]
                        actual_future_date = closest_rv_date
                    if pd.isna(future_realized_vol) or future_realized_vol <= 0:
                        continue
                    vol_diff = future_realized_vol - signal_implied_vol
                    vol_diff_pct = vol_diff / signal_implied_vol
                    if signal_type == 'cheap':
                        trade_success = future_realized_vol > signal_implied_vol
                        expected_direction = 'RV > IV'
                    else:  # expensive
                        trade_success = future_realized_vol < signal_implied_vol
                        expected_direction = 'RV < IV'
                    trade_record = {
                        'signal_date': signal_date,
                        'future_date': actual_future_date,
                        'currency_pair': currency_pair,
                        'tenor': tenor,
                        'signal_type': signal_type,
                        'signal_percentile': signal_percentile,
                        'coint_score': signal_row.get('Avg_Coint_Score', np.nan),
                        'implied_vol': signal_implied_vol,
                        'realized_vol': future_realized_vol,
                        'vol_diff': vol_diff,
                        'vol_diff_pct': vol_diff_pct,
                        'trade_success': trade_success,
                        'expected_direction': expected_direction,
                        'days_held': (actual_future_date - signal_date).days}
                    tenor_trades.append(trade_record)
                    all_trades.append(trade_record)
                except Exception as e:
                    continue
        if tenor_trades:
            tenor_df = pd.DataFrame(tenor_trades)
            total_trades = len(tenor_df)
            hit_rate = tenor_df['trade_success'].mean()
            cheap_trades = tenor_df[tenor_df['signal_type'] == 'cheap']
            expensive_trades = tenor_df[tenor_df['signal_type'] == 'expensive']
            cheap_hit_rate = cheap_trades['trade_success'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_hit_rate = expensive_trades['trade_success'].mean() if len(expensive_trades) > 0 else np.nan
            cheap_avg_vol_diff = cheap_trades['vol_diff'].mean() if len(cheap_trades) > 0 else np.nan
            cheap_avg_vol_diff_pct = cheap_trades['vol_diff_pct'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_avg_vol_diff = expensive_trades['vol_diff'].mean() if len(expensive_trades) > 0 else np.nan
            expensive_avg_vol_diff_pct = expensive_trades['vol_diff_pct'].mean() if len(expensive_trades) > 0 else np.nan
            avg_vol_diff = tenor_df['vol_diff'].mean()
            avg_vol_diff_pct = tenor_df['vol_diff_pct'].mean()
            drawdown_stats = calculate_max_drawdown_stats(tenor_df)
            backtest_results[tenor] = {
                'data': tenor_df,
                'total_trades': total_trades,
                'overall_hit_rate': hit_rate,
                'cheap_trades': len(cheap_trades),
                'expensive_trades': len(expensive_trades),
                'cheap_hit_rate': cheap_hit_rate,
                'expensive_hit_rate': expensive_hit_rate,
                'avg_vol_diff': avg_vol_diff,
                'avg_vol_diff_pct': avg_vol_diff_pct,
                'cheap_avg_vol_diff': cheap_avg_vol_diff,
                'cheap_avg_vol_diff_pct': cheap_avg_vol_diff_pct,
                'expensive_avg_vol_diff': expensive_avg_vol_diff,
                'expensive_avg_vol_diff_pct': expensive_avg_vol_diff_pct,
                'cheap_max_drawdown': drawdown_stats['cheap_max_drawdown'],
                'cheap_max_drawdown_pct': drawdown_stats['cheap_max_drawdown_pct'],
                'expensive_max_drawdown': drawdown_stats['expensive_max_drawdown'],
                'expensive_max_drawdown_pct': drawdown_stats['expensive_max_drawdown_pct'],
                'overall_max_drawdown': drawdown_stats['overall_max_drawdown'],
                'overall_max_drawdown_pct': drawdown_stats['overall_max_drawdown_pct']}
    def tidy_backtest(backtest_results: dict):
        rows = []
        for tenor, d in backtest_results.items():
            rows.append({
                "tenor": tenor,
                "total_trades": int(d["total_trades"]),
                "overall_hit_rate": float(d["overall_hit_rate"]),
                "cheap_trades": int(d["cheap_trades"]),
                "expensive_trades": int(d["expensive_trades"]),
                "cheap_hit_rate": float(d["cheap_hit_rate"]) if not pd.isna(d["cheap_hit_rate"]) else np.nan,
                "expensive_hit_rate": float(d["expensive_hit_rate"]) if not pd.isna(d["expensive_hit_rate"]) else np.nan,
                "avg_vol_diff": float(d["avg_vol_diff"]),
                "avg_vol_diff_pct": float(d["avg_vol_diff_pct"]),
                "cheap_avg_vol_diff": float(d["cheap_avg_vol_diff"]) if not pd.isna(d["cheap_avg_vol_diff"]) else np.nan,
                "cheap_avg_vol_diff_pct": float(d["cheap_avg_vol_diff_pct"]) if not pd.isna(d["cheap_avg_vol_diff_pct"]) else np.nan,
                "expensive_avg_vol_diff": float(d["expensive_avg_vol_diff"]) if not pd.isna(d["expensive_avg_vol_diff"]) else np.nan,
                "expensive_avg_vol_diff_pct": float(d["expensive_avg_vol_diff_pct"]) if not pd.isna(d["expensive_avg_vol_diff_pct"]) else np.nan,
                "cheap_max_drawdown": float(d["cheap_max_drawdown"]) if not pd.isna(d["cheap_max_drawdown"]) else np.nan,
                "cheap_max_drawdown_pct": float(d["cheap_max_drawdown_pct"]) if not pd.isna(d["cheap_max_drawdown_pct"]) else np.nan,
                "expensive_max_drawdown": float(d["expensive_max_drawdown"]) if not pd.isna(d["expensive_max_drawdown"]) else np.nan,
                "expensive_max_drawdown_pct": float(d["expensive_max_drawdown_pct"]) if not pd.isna(d["expensive_max_drawdown_pct"]) else np.nan,
                "overall_max_drawdown": float(d["overall_max_drawdown"]) if not pd.isna(d["overall_max_drawdown"]) else np.nan,
                "overall_max_drawdown_pct": float(d["overall_max_drawdown_pct"]) if not pd.isna(d["overall_max_drawdown_pct"]) else np.nan,})
        summary_df = pd.DataFrame(rows).set_index("tenor").sort_index()
        trades_df = pd.concat(
            {tenor: d["data"] for tenor, d in backtest_results.items()},
            names=["tenor"])
        preferred_cols = [
            "tenor", "signal_date", "future_date", "currency_pair", "signal_type",
            "signal_percentile", "coint_score", "implied_vol", "realized_vol", 
            "vol_diff", "vol_diff_pct", "trade_success", "expected_direction", "days_held"]
        trades_df = trades_df[[c for c in preferred_cols if c in trades_df.columns]]
        return summary_df, trades_df
    summary_df, trades_df = tidy_backtest(backtest_results)
    return summary_df.T, trades_df






