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
Data Pull --- One Large Pull for Risk Reversal

    -   Input:  List of CCYs and Tenors of Interest
    -  Output:  Dict by [Tenor][CCY] for both RR and ATM data
"""
def get_all_RR_data_bulk(currency_list, tenors, start_date, end_date):
    all_tickers = []
    ticker_map = {}
    for tenor in tenors:
        for ccy_pair in currency_list:
            ticker_RR = f"{ccy_pair}25R{tenor} BGN Curncy"
            all_tickers.append(ticker_RR)
            ticker_map[ticker_RR] = (ccy_pair, tenor)
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
        print(f"Bulk RR data fetch failed: {e}")
        return {}
    

def get_all_ATM_data_bulk(currency_list, tenors, start_date, end_date):
    all_tickers = []
    ticker_map = {}
    for tenor in tenors:
        for ccy_pair in currency_list:
            ticker_ATM = f"{ccy_pair}V{tenor} BGN Curncy"
            all_tickers.append(ticker_ATM)
            ticker_map[ticker_ATM] = (ccy_pair, tenor)
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
        print(f"Bulk ATM data fetch failed: {e}")
        return {}

# ==================================================================================
# RISK REVERSAL HISTORICAL ANALYSIS
# ==================================================================================

"""
Historical Reference Calculations for Risk Reversals (Vol-Adjusted)

    -   Input: Full RR and ATM Data, Tenor List, Date Range for Testing
    -  Output [Dict]: 
            -     results_by_tenor[Tenor] - Each Spread iteration with Historical Measures
            - currency_avg_metrics[Tenor] - Spread Consolidated Statistics
"""
def xCCY_AllPairs_MultiTenor_RR_Analysis(bulk_rr_data, bulk_atm_data, tenor_list, date_range):
    results_by_date = {}
    
    for analysis_date in date_range:
        analysis_ts = pd.to_datetime(analysis_date)
        
        results_by_tenor = {}
        currency_avg_metrics = {}
        
        for tenor in tenor_list:
            if tenor not in bulk_rr_data or tenor not in bulk_atm_data:
                continue
            
            tenor_rr_data = bulk_rr_data[tenor]
            tenor_atm_data = bulk_atm_data[tenor]
            
            if not tenor_rr_data or not tenor_atm_data:
                continue
            
            rr_dict = {}
            atm_dict = {}
            
            for ccy_pair, rr_series in tenor_rr_data.items():
                if rr_series is None or rr_series.empty:
                    continue
                
                if ccy_pair not in tenor_atm_data:
                    continue
                
                atm_series = tenor_atm_data[ccy_pair]
                if atm_series is None or atm_series.empty:
                    continue
                
                if not isinstance(rr_series.index, pd.DatetimeIndex):
                    rr_series = rr_series.copy()
                    rr_series.index = pd.to_datetime(rr_series.index)
                
                if not isinstance(atm_series.index, pd.DatetimeIndex):
                    atm_series = atm_series.copy()
                    atm_series.index = pd.to_datetime(atm_series.index)
                
                filtered_rr = rr_series.loc[:analysis_ts]
                filtered_atm = atm_series.loc[:analysis_ts]
                
                if not filtered_rr.empty and not filtered_atm.empty:
                    rr_dict[ccy_pair] = filtered_rr
                    atm_dict[ccy_pair] = filtered_atm
            
            if not rr_dict or not atm_dict:
                continue
            
            df_rr = pd.DataFrame(rr_dict).dropna()
            df_atm = pd.DataFrame(atm_dict).dropna()
            
            if df_rr.empty or df_atm.empty:
                continue
            
            # Align RR and ATM data
            df_aligned = pd.concat([df_rr, df_atm], axis=1, keys=['RR', 'ATM']).dropna()
            
            if df_aligned.empty:
                continue
            
            df_rr = df_aligned['RR']
            df_atm = df_aligned['ATM']
            
            # Vol-adjust RR: divide by ATM and multiply by 100
            df_rr_adj = df_rr.div(df_atm) * 100
            
            latest_date = df_rr_adj.index[-1]
            currency_pairs = list(permutations(list(rr_dict.keys()), 2))
            pair_results = []
            currency_volatilities = {ccy_pair: [] for ccy_pair in rr_dict.keys()}
            
            for ccy_pair1, ccy_pair2 in currency_pairs:
                if ccy_pair1 not in df_rr_adj.columns or ccy_pair2 not in df_rr_adj.columns:
                    continue
                
                current_spread = df_rr_adj.loc[latest_date, ccy_pair1] - df_rr_adj.loc[latest_date, ccy_pair2]
                historical_spreads = df_rr_adj[ccy_pair1] - df_rr_adj[ccy_pair2]
                
                three_months_ago = pd.Timestamp(latest_date) - pd.Timedelta(days=90)
                one_year_ago = pd.Timestamp(latest_date) - pd.Timedelta(days=365)
                three_years_ago = pd.Timestamp(latest_date) - pd.Timedelta(days=365*3)
                five_years_ago = pd.Timestamp(latest_date) - pd.Timedelta(days=365*5)
                
                past_3_months_spreads = historical_spreads.loc[historical_spreads.index >= three_months_ago]
                past_year_spreads = historical_spreads.loc[historical_spreads.index >= one_year_ago]
                past_3_years_spreads = historical_spreads.loc[historical_spreads.index >= three_years_ago]
                past_5_years_spreads = historical_spreads.loc[historical_spreads.index >= five_years_ago]
                
                percentile_3_months = percentileofscore(past_3_months_spreads, current_spread)
                percentile_1_year = percentileofscore(past_year_spreads, current_spread)
                percentile_3_years = percentileofscore(past_3_years_spreads, current_spread) if len(past_3_years_spreads) > 1 else None
                percentile_5_years = percentileofscore(past_5_years_spreads, current_spread) if len(past_5_years_spreads) > 1 else None
                
                spread_volatility_3m = past_3_months_spreads.std()
                spread_volatility_1y = past_year_spreads.std()
                spread_volatility_3y = past_3_years_spreads.std() if len(past_3_years_spreads) > 1 else None
                spread_volatility_5y = past_5_years_spreads.std() if len(past_5_years_spreads) > 1 else None
                
                mean_spread_3m = past_3_months_spreads.mean()
                mean_spread_1y = past_year_spreads.mean()
                mean_spread_3y = past_3_years_spreads.mean() if len(past_3_years_spreads) > 1 else None
                mean_spread_5y = past_5_years_spreads.mean() if len(past_5_years_spreads) > 1 else None
                
                # Z-scores (SIGNED for directional information)
                z_score_3m = (current_spread - mean_spread_3m) / spread_volatility_3m if (spread_volatility_3m and spread_volatility_3m > 0) else 0
                z_score_1y = (current_spread - mean_spread_1y) / spread_volatility_1y if (spread_volatility_1y and spread_volatility_1y > 0) else 0
                z_score_3y = ((current_spread - mean_spread_3y) / spread_volatility_3y if (spread_volatility_3y and spread_volatility_3y > 0 and mean_spread_3y is not None) else None)
                z_score_5y = ((current_spread - mean_spread_5y) / spread_volatility_5y if (spread_volatility_5y and spread_volatility_5y > 0 and mean_spread_5y is not None) else None)
                
                pair_result = {
                    'Tenor': tenor,
                    'Pair': f"{ccy_pair1}-{ccy_pair2}",
                    'CCY1': ccy_pair1,
                    'CCY2': ccy_pair2,
                    'Current_Spread': round(current_spread, 4),
                    'Spread_STD_3M': round(spread_volatility_3m, 4) if spread_volatility_3m is not None else None,
                    'Spread_STD_1Y': round(spread_volatility_1y, 4) if spread_volatility_1y is not None else None,
                    'Spread_STD_3Y': round(spread_volatility_3y, 4) if spread_volatility_3y is not None else None,
                    'Spread_STD_5Y': round(spread_volatility_5y, 4) if spread_volatility_5y is not None else None,
                    'Spread_ZScore_3M': round(z_score_3m, 4) if z_score_3m is not None else None,
                    'Spread_ZScore_1Y': round(z_score_1y, 4) if z_score_1y is not None else None,
                    'Spread_ZScore_3Y': round(z_score_3y, 4) if z_score_3y is not None else None,
                    'Spread_ZScore_5Y': round(z_score_5y, 4) if z_score_5y is not None else None,
                    'Percentile_3M': round(percentile_3_months, 3) if percentile_3_months is not None else None,
                    'Percentile_1Y': round(percentile_1_year, 3) if percentile_1_year is not None else None,
                    'Percentile_3Y': round(percentile_3_years, 3) if percentile_3_years is not None else None,
                    'Percentile_5Y': round(percentile_5_years, 3) if percentile_5_years is not None else None
                }
                
                pair_results.append(pair_result)
                
                # Store for currency averages
                if (spread_volatility_1y is not None) and (z_score_1y is not None) and (percentile_1_year is not None):
                    currency_volatilities[ccy_pair1].append({
                        'Spread_MEAN_3m': mean_spread_3m,
                        'Spread_MEAN_1y': mean_spread_1y,
                        'Spread_MEAN_3y': mean_spread_3y,
                        'Spread_MEAN_5y': mean_spread_5y,
                        'Spread_STD_3m': spread_volatility_3m,
                        'Spread_STD_1y': spread_volatility_1y,
                        'Spread_STD_3y': spread_volatility_3y,
                        'Spread_STD_5y': spread_volatility_5y,
                        'pair': f"{ccy_pair1}-{ccy_pair2}"
                    })
            
            results_by_tenor[tenor] = pd.DataFrame(pair_results)
            
            # Calculate currency averages
            tenor_currency_averages = []
            for ccy_pair in rr_dict.keys():
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
                        'Num_Pairs': num_pairs
                    })
            
            currency_avg_metrics[tenor] = (
                pd.DataFrame(tenor_currency_averages).sort_values('Avg_histSpread_STD_5y')
                if tenor_currency_averages else pd.DataFrame()
            )
        
        results_by_date[analysis_ts] = {
            'results_by_tenor': results_by_tenor,
            'currency_avg_metrics': currency_avg_metrics
        }
    
    return results_by_date



# ==================================================================================
# BUCKETING CCYs - EQUAL WEIGHTING FOR RISK REVERSALS
# ==================================================================================

def find_ave_allCCYallLookback_equalWeights_RR(results_by_tenor, tenors):
    """Same as ATM version - works for RR data"""
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
                        'Num_Pairs': len(ccy_pairs)
                    })
            
            percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_EquWeight_5Y%')
            tenor_results[tenor] = percentile_df
    
    return tenor_results



# ==================================================================================
# HISTORICAL RESULTS GENERATION FOR RISK REVERSALS
# ==================================================================================

def RR_PercentileEquWeightAve_HistoricalResults(currency_list, tenors, days_backtested, frequency):
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    if frequency == 'weekly':
        date_range = pd.date_range(start=start_date, end=end_date, freq=pd.offsets.Week(weekday=datetime.now().weekday()))
    elif frequency == 'monthly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    bulk_start_date = (datetime.now() - timedelta(days=days_backtested + 365*5)).strftime("%Y-%m-%d")
    
    print("Fetching bulk RR data...")
    bulk_rr_data = get_all_RR_data_bulk(currency_list, tenors, bulk_start_date, end_date)
    
    print("Fetching bulk ATM data for vol-adjustment...")
    bulk_atm_data = get_all_ATM_data_bulk(currency_list, tenors, bulk_start_date, end_date)
    
    historical_results = {}
    for tenor in tenors:
        historical_results[tenor] = {}
        for currency_pair in currency_list:
            historical_results[tenor][currency_pair] = []
    
    for i, current_date in enumerate(date_range):
        if i % 10 == 0:
            print(f"Processing date {i+1}/{len(date_range)}: {current_date}")
        
        try:
            daily_results = xCCY_AllPairs_MultiTenor_RR_Analysis(
                bulk_rr_data, bulk_atm_data, tenors, [current_date])
            
            if current_date in daily_results:
                single_date_results = daily_results[current_date]['results_by_tenor']
                daily_percentiles = find_ave_allCCYallLookback_equalWeights_RR(
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
                            'Num_Pairs': row['Num_Pairs']
                        })
        
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


# ==================================================================================
# DATA FETCHING FOR BACKTESTING
# ==================================================================================

def get_realized_RR_dataBloomberg(currency_list, tenors, days_backtested):
    """
    Get historical realized risk reversal data using Q tickers
    Ticker format: {CCY}Q{TENOR} BGN Curncy (e.g., EURUSDQ1M BGN Curncy)
    """
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    realized_rr_data = {}
    
    for tenor in tenors:
        tenor_dict = {}
        for ccy_pair in currency_list:
            ticker_RR = f"{ccy_pair}Q{tenor} BGN Curncy"
            
            try:
                data_RR = blp.bdh(
                    tickers=ticker_RR,
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not data_RR.empty:
                    data_RR.columns = data_RR.columns.get_level_values(1)
                    tenor_dict[ccy_pair] = data_RR['PX_LAST']
            
            except Exception as e:
                print(f"Error getting realized RR data for {ccy_pair} {tenor}: {e}")
                continue
        
        realized_rr_data[tenor] = tenor_dict
    
    return realized_rr_data


def get_implied_RR_dataBloomberg(currency_list, tenors, days_backtested):
    """Get historical implied risk reversal data"""
    start_date = (datetime.now() - timedelta(days=days_backtested)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    implied_rr_data = {}
    
    for tenor in tenors:
        tenor_dict = {}
        for ccy_pair in currency_list:
            ticker_RR = f"{ccy_pair}25R{tenor} BGN Curncy"
            
            try:
                data_RR = blp.bdh(
                    tickers=ticker_RR,
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not data_RR.empty:
                    data_RR.columns = data_RR.columns.get_level_values(1)
                    tenor_dict[ccy_pair] = data_RR['PX_LAST']
            
            except Exception as e:
                print(f"Error getting implied RR data for {ccy_pair} {tenor}: {e}")
                continue
        
        implied_rr_data[tenor] = tenor_dict
    
    return implied_rr_data



# ==================================================================================
# UTILITY FUNCTIONS (Same as ATM)
# ==================================================================================

def tenor_to_business_days(tenor: str) -> int:
    tenor_map = {
        'ON': 1,
        '1W': 7,
        '2W': 14,
        '3W': 21,
        '1M': 30,
        '2M': 60,
        '3M': 90
    }
    return tenor_map.get(tenor, 30)


def find_closest_business_day(target_date: pd.Timestamp, available_dates: pd.Index, max_days: int = 5) -> pd.Timestamp:
    if target_date in available_dates:
        return target_date
    
    available_dates = pd.to_datetime(available_dates)
    target_date = pd.to_datetime(target_date)
    
    for days_offset in range(1, max_days + 1):
        for direction in [1, -1]:
            candidate_date = target_date + pd.Timedelta(days=direction * days_offset)
            if candidate_date in available_dates:
                return candidate_date
    
    return None


def calculate_max_drawdown_stats_RR(trades_df):
    """Calculate drawdown stats for RR trades (uses rr_diff instead of vol_diff)"""
    cheap_trades = trades_df[trades_df['signal_type'] == 'cheap']
    expensive_trades = trades_df[trades_df['signal_type'] == 'expensive']
    cheap_max_drawdown = cheap_trades['rr_diff'].min() if len(cheap_trades) > 0 else np.nan
    cheap_max_drawdown_pct = cheap_trades['rr_diff_pct'].min() if len(cheap_trades) > 0 else np.nan
    expensive_max_drawdown = expensive_trades['rr_diff'].max() if len(expensive_trades) > 0 else np.nan
    expensive_max_drawdown_pct = expensive_trades['rr_diff_pct'].max() if len(expensive_trades) > 0 else np.nan
    cheap_losses = cheap_trades['rr_diff'][cheap_trades['rr_diff'] < 0] if len(cheap_trades) > 0 else pd.Series(dtype=float)
    expensive_losses = expensive_trades['rr_diff'][expensive_trades['rr_diff'] > 0] if len(expensive_trades) > 0 else pd.Series(dtype=float)
    cheap_losses_pct = cheap_trades['rr_diff_pct'][cheap_trades['rr_diff_pct'] < 0] if len(cheap_trades) > 0 else pd.Series(dtype=float)
    expensive_losses_pct = expensive_trades['rr_diff_pct'][expensive_trades['rr_diff_pct'] > 0] if len(expensive_trades) > 0 else pd.Series(dtype=float)
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


# ==================================================================================
# BACKTESTING FUNCTIONS FOR RISK REVERSALS
# ==================================================================================

def backtest_RR_ImpliedRealizedDiff_EqualWeightAve_Results(currency_list, gamma_tenors, days_backtested, 
                                                           percentile_threshold_low, percentile_threshold_high, 
                                                           frequency):
    print("Calculating historical RR percentiles...")
    historical_percentiles = RR_PercentileEquWeightAve_HistoricalResults(
        currency_list, gamma_tenors, days_backtested, frequency)
    print("Getting implied RR data...")
    implied_rr_data = get_implied_RR_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    print("Getting realized RR data...")
    realized_rr_data = get_realized_RR_dataBloomberg(currency_list, gamma_tenors, days_backtested)
    backtest_results = {}
    all_trades = []
    for tenor in gamma_tenors:
        print(f"Processing tenor: {tenor}")
        tenor_trades = []
        for currency_pair in currency_list:
            percentile_df = historical_percentiles[tenor][currency_pair]
            implied_rr_series = implied_rr_data[tenor][currency_pair]
            realized_rr_series = realized_rr_data[tenor][currency_pair]
            percentile_df.index = pd.to_datetime(percentile_df.index)
            implied_rr_series.index = pd.to_datetime(implied_rr_series.index)
            realized_rr_series.index = pd.to_datetime(realized_rr_series.index)
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
                    signal_type = 'cheap'  # Expect realized RR > implied RR
                elif signal_percentile >= percentile_threshold_high:
                    signal_type = 'expensive'  # Expect realized RR < implied RR
                else:
                    continue  # No signal
                try:
                    if signal_date in implied_rr_series.index:
                        signal_implied_rr = implied_rr_series.loc[signal_date]
                    else:
                        closest_iv_date = find_closest_business_day(signal_date, implied_rr_series.index)
                        if closest_iv_date is None:
                            continue
                        signal_implied_rr = implied_rr_series.loc[closest_iv_date]
                    if pd.isna(signal_implied_rr):
                        continue
                except Exception as e:
                    print(f"    Error getting implied RR for {signal_date}: {e}")
                    continue
                future_date = signal_date + pd.Timedelta(days=tenor_days)
                try:
                    if future_date in realized_rr_series.index:
                        future_realized_rr = realized_rr_series.loc[future_date]
                        actual_future_date = future_date
                    else:
                        closest_rv_date = find_closest_business_day(future_date, realized_rr_series.index)
                        if closest_rv_date is None:
                            continue
                        future_realized_rr = realized_rr_series.loc[closest_rv_date]
                        actual_future_date = closest_rv_date
                    if pd.isna(future_realized_rr):
                        continue
                except Exception as e:
                    print(f"    Error getting realized RR for {future_date}: {e}")
                    continue
                rr_diff = future_realized_rr - signal_implied_rr
                rr_diff_pct = rr_diff / abs(signal_implied_rr) if signal_implied_rr != 0 else 0
                if signal_type == 'cheap':
                    # Expected realized RR > implied RR (bought RR)
                    trade_success = future_realized_rr > signal_implied_rr
                    expected_direction = 'Realized > Implied'
                else:  # expensive
                    # Expected realized RR < implied RR (sold RR)
                    trade_success = future_realized_rr < signal_implied_rr
                    expected_direction = 'Realized < Implied'
                trade_record = {
                    'signal_date': signal_date,
                    'future_date': actual_future_date,
                    'currency_pair': currency_pair,
                    'tenor': tenor,
                    'signal_type': signal_type,
                    'signal_percentile': signal_percentile,
                    'implied_rr': signal_implied_rr,
                    'realized_rr': future_realized_rr,
                    'rr_diff': rr_diff,
                    'rr_diff_pct': rr_diff_pct,
                    'trade_success': trade_success,
                    'expected_direction': expected_direction,
                    'days_held': (actual_future_date - signal_date).days}
                tenor_trades.append(trade_record)
                all_trades.append(trade_record)
                trades_made += 1
            if trades_made > 0:
                print(f"  {currency_pair}: {trades_made} trades from {signals_processed} signals")
        if tenor_trades:
            tenor_df = pd.DataFrame(tenor_trades)
            total_trades = len(tenor_df)
            hit_rate = tenor_df['trade_success'].mean()
            cheap_trades = tenor_df[tenor_df['signal_type'] == 'cheap']
            expensive_trades = tenor_df[tenor_df['signal_type'] == 'expensive']
            cheap_hit_rate = cheap_trades['trade_success'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_hit_rate = expensive_trades['trade_success'].mean() if len(expensive_trades) > 0 else np.nan
            cheap_avg_rr_diff = cheap_trades['rr_diff'].mean() if len(cheap_trades) > 0 else np.nan
            cheap_avg_rr_diff_pct = cheap_trades['rr_diff_pct'].mean() if len(cheap_trades) > 0 else np.nan
            expensive_avg_rr_diff = expensive_trades['rr_diff'].mean() if len(expensive_trades) > 0 else np.nan
            expensive_avg_rr_diff_pct = expensive_trades['rr_diff_pct'].mean() if len(expensive_trades) > 0 else np.nan
            avg_rr_diff = tenor_df['rr_diff'].mean()
            avg_rr_diff_pct = tenor_df['rr_diff_pct'].mean()
            drawdown_stats = calculate_max_drawdown_stats_RR(tenor_df)
            backtest_results[tenor] = {
                'data': tenor_df,
                'total_trades': total_trades,
                'overall_hit_rate': hit_rate,
                'cheap_trades': len(cheap_trades),
                'expensive_trades': len(expensive_trades),
                'cheap_hit_rate': cheap_hit_rate,
                'expensive_hit_rate': expensive_hit_rate,
                'avg_rr_diff': avg_rr_diff,
                'avg_rr_diff_pct': avg_rr_diff_pct,
                'cheap_avg_rr_diff': cheap_avg_rr_diff,
                'cheap_avg_rr_diff_pct': cheap_avg_rr_diff_pct,
                'expensive_avg_rr_diff': expensive_avg_rr_diff,
                'expensive_avg_rr_diff_pct': expensive_avg_rr_diff_pct,
                'cheap_max_drawdown': drawdown_stats['cheap_max_drawdown'],
                'cheap_max_drawdown_pct': drawdown_stats['cheap_max_drawdown_pct'],
                'expensive_max_drawdown': drawdown_stats['expensive_max_drawdown'],
                'expensive_max_drawdown_pct': drawdown_stats['expensive_max_drawdown_pct'],
                'overall_max_drawdown': drawdown_stats['overall_max_drawdown'],
                'overall_max_drawdown_pct': drawdown_stats['overall_max_drawdown_pct']}
            print(f"Tenor {tenor}: {total_trades} total trades, {hit_rate:.2%} hit rate")
    
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
                "avg_rr_diff": float(d["avg_rr_diff"]),
                "avg_rr_diff_pct": float(d["avg_rr_diff_pct"]),
                "cheap_avg_rr_diff": float(d["cheap_avg_rr_diff"]) if not pd.isna(d["cheap_avg_rr_diff"]) else np.nan,
                "cheap_avg_rr_diff_pct": float(d["cheap_avg_rr_diff_pct"]) if not pd.isna(d["cheap_avg_rr_diff_pct"]) else np.nan,
                "expensive_avg_rr_diff": float(d["expensive_avg_rr_diff"]) if not pd.isna(d["expensive_avg_rr_diff"]) else np.nan,
                "expensive_avg_rr_diff_pct": float(d["expensive_avg_rr_diff_pct"]) if not pd.isna(d["expensive_avg_rr_diff_pct"]) else np.nan,
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
            "signal_percentile", "implied_rr", "realized_rr", "rr_diff", "rr_diff_pct",
            "trade_success", "expected_direction", "days_held"]
        trades_df = trades_df[[c for c in preferred_cols if c in trades_df.columns]]
        return summary_df, trades_df
    summary_df, trades_df = tidy_backtest(backtest_results)
    return summary_df.T, trades_df.sort_index


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

currency_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD']
tenors = ['1W', '1M', '3M']

# Run backtest
summary, trades = backtest_RR_ImpliedRealizedDiff_EqualWeightAve_Results(
    currency_list=currency_list,
    gamma_tenors=tenors,
    days_backtested=360,  # 2 years
    percentile_threshold_low=20,
    percentile_threshold_high=80,
    frequency='weekly'
)


print("\n=== BACKTEST SUMMARY ===")
print(summary)


print(trades)




















