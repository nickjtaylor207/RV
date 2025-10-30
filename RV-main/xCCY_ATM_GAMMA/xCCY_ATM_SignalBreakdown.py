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
import time
from sklearn.linear_model import LinearRegression


def calculate_beta_weights_instantaneous(df_all: pd.DataFrame, base_ccy: str, other_ccys: List[str], 
                                       lookback_days: int = 252) -> Dict[str, float]:
    if base_ccy not in df_all.columns:
        return {}
    base_series = df_all[base_ccy].dropna()
    if len(base_series) < 30:  # Need minimum data points
        return {}
    base_series = base_series.tail(lookback_days)
    base_returns = base_series.pct_change().dropna()
    if len(base_returns) < 20:  # Need minimum returns for regression
        return {}
    betas = {}
    for other_ccy in other_ccys:
        if other_ccy not in df_all.columns or other_ccy == base_ccy:
            continue
        other_series = df_all[other_ccy].dropna()
        if len(other_series) < 30:
            continue
        aligned_data = pd.concat([base_series, other_series], axis=1, keys=[base_ccy, other_ccy]).dropna()
        if len(aligned_data) < 30:
            continue
        aligned_data = aligned_data.tail(lookback_days)
        base_ret = aligned_data[base_ccy].pct_change().dropna()
        other_ret = aligned_data[other_ccy].pct_change().dropna()
        if len(base_ret) < 20 or len(other_ret) < 20:
            continue
        try:
            X = base_ret.values.reshape(-1, 1)
            y = other_ret.values
            reg = LinearRegression().fit(X, y)
            beta = abs(reg.coef_[0])  # Use absolute value
            r_squared = reg.score(X, y)
            adjusted_beta = beta * np.sqrt(max(0, r_squared)) 
            betas[other_ccy] = adjusted_beta
        except Exception as e:
            print(f"Error calculating beta for {other_ccy}: {e}")
            continue
    if not betas:
        return {}
    total_beta = sum(betas.values())
    if total_beta == 0:
        weights = {ccy: 1/len(betas) for ccy in betas.keys()}
    else:
        weights = {ccy: beta/total_beta for ccy, beta in betas.items()}
    return weights




def find_ave_allCCYallLookback_betaWeights_instantaneous(results_by_tenor: Dict, df_all_by_tenor: Dict, 
                                                       tenors: List[str], lookback_days: int = 252) -> Dict:
    tenor_results = {}
    for tenor in tenors:
        if tenor not in results_by_tenor or tenor not in df_all_by_tenor:
            continue
        df = results_by_tenor[tenor]
        df_all = df_all_by_tenor[tenor]  # This is the DataFrame with all currency vol data
        all_currency_pairs = set(df['CCY1'].tolist() + df['CCY2'].tolist())
        currency_percentile_summary = []
        for base_ccy in all_currency_pairs:  # base_ccy is like 'EURUSD'
            ccy_pairs = df[df['CCY1'] == base_ccy].copy()
            if len(ccy_pairs) == 0:
                continue
            other_ccys = ccy_pairs['CCY2'].tolist()
            weights = calculate_beta_weights_instantaneous(
                df_all, base_ccy, other_ccys, lookback_days)
            if not weights:
                weights = {ccy: 1/len(other_ccys) for ccy in other_ccys}
            weighted_percentile_3m = 0
            weighted_percentile_1y = 0
            weighted_percentile_3y = 0
            weighted_percentile_5y = 0
            total_weight_3y = 0
            total_weight_5y = 0
            for _, row in ccy_pairs.iterrows():
                other_ccy = row['CCY2']
                weight = weights.get(other_ccy, 0)
                if weight > 0:
                    weighted_percentile_3m += row['Percentile_3M'] * weight
                    weighted_percentile_1y += row['Percentile_1Y'] * weight
                    if pd.notna(row['Percentile_3Y']):
                        weighted_percentile_3y += row['Percentile_3Y'] * weight
                        total_weight_3y += weight
                    if pd.notna(row['Percentile_5Y']):
                        weighted_percentile_5y += row['Percentile_5Y'] * weight
                        total_weight_5y += weight
            if total_weight_3y > 0:
                weighted_percentile_3y = weighted_percentile_3y / total_weight_3y
            else:
                weighted_percentile_3y = None
            if total_weight_5y > 0:
                weighted_percentile_5y = weighted_percentile_5y / total_weight_5y
            else:
                weighted_percentile_5y = None
            currency_percentile_summary.append({
                'Currency_Pair': base_ccy,
                'Tenor': tenor,
                'Avg_BetaWeight_3M%': round(weighted_percentile_3m, 3),
                'Avg_BetaWeight_1Y%': round(weighted_percentile_1y, 3),
                'Avg_BetaWeight_3Y%': round(weighted_percentile_3y, 3) if weighted_percentile_3y is not None else None,
                'Avg_BetaWeight_5Y%': round(weighted_percentile_5y, 3) if weighted_percentile_5y is not None else None,
                'Num_Pairs': len(ccy_pairs),
                })    # 'Beta_Weights': {k: round(v, 4) for k, v in weights.items()}
            # print({k: round(v, 4) for k, v in weights.items()})                               ------------------------------
        percentile_df = pd.DataFrame(currency_percentile_summary)
        if not percentile_df.empty:
            percentile_df = percentile_df.sort_values('Avg_BetaWeight_5Y%')
        tenor_results[tenor] = percentile_df
    return tenor_results






def ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list, tenors, lookback_days=252):
    def xCCY_AllPairs_MultiTenor_Analysis(currency_list: List[str], tenor_list: List[str]):
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365*6)).strftime("%Y-%m-%d")
        base_ccy = sorted(set(currency_list))  # These are currency pairs like EURUSD, GBPUSD, etc.
        results_by_tenor: Dict[str, pd.DataFrame] = {}
        currency_avg_metrics: Dict[str, pd.DataFrame] = {}
        correlation_data: Dict[str, pd.DataFrame] = {}  # Store correlation data for each tenor
        df_all_by_tenor: Dict[str, pd.DataFrame] = {}  # Store df_all for each tenor for beta calculations
        for tenor in tenor_list:
            data_dict: Dict[str, pd.Series] = {}
            for ccy_pair in base_ccy:  # ccy_pair is like 'EURUSD'
                ticker_IV = f"{ccy_pair}V{tenor} BGN Curncy"
                try:
                    data_IV = blp.bdh(
                        tickers=ticker_IV,
                        flds="PX_LAST",
                        start_date=start_date,
                        end_date=end_date)
                    data_IV.columns = data_IV.columns.get_level_values(1)
                    data_dict[ccy_pair] = data_IV['PX_LAST']
                except Exception as e:
                    continue
            if not data_dict:
                continue
            df_all = pd.DataFrame(data_dict).dropna()
            if df_all.empty:
                continue
            df_all_by_tenor[tenor] = df_all
            # correlation_matrix = df_all.corr()
            correlation_matrix = df_all.diff().dropna().corr()
            correlation_data[tenor] = correlation_matrix
            latest_date = df_all.index[-1]
            currency_pairs = list(permutations(base_ccy, 2))  # All combinations of currency pairs
            pair_results = []
            currency_volatilities: Dict[str, List[Dict[str, float]]] = {ccy_pair: [] for ccy_pair in base_ccy}
            for ccy_pair1, ccy_pair2 in currency_pairs:  # e.g., ('EURUSD', 'USDJPY')
                if ccy_pair1 not in df_all.columns or ccy_pair2 not in df_all.columns:
                    continue
                current_spread = df_all.loc[latest_date, ccy_pair1] - df_all.loc[latest_date, ccy_pair2]
                historical_spreads = df_all[ccy_pair1] - df_all[ccy_pair2]
                three_months_ago = latest_date - timedelta(days=90)
                one_year_ago    = latest_date - timedelta(days=365)
                three_years_ago = latest_date - timedelta(days=365*3)
                five_years_ago  = latest_date - timedelta(days=365*5)
                past_3_months_spreads = historical_spreads.loc[historical_spreads.index >= three_months_ago]
                past_year_spreads     = historical_spreads.loc[historical_spreads.index >= one_year_ago]
                past_3_years_spreads  = historical_spreads.loc[historical_spreads.index >= three_years_ago]
                past_5_years_spreads  = historical_spreads.loc[historical_spreads.index >= five_years_ago]
                if len(past_5_years_spreads) < 2:
                    continue
                percentile_3_months = percentileofscore(past_3_months_spreads, current_spread)
                percentile_1_year   = percentileofscore(past_year_spreads,     current_spread)
                percentile_3_years  = percentileofscore(past_3_years_spreads,  current_spread) if len(past_3_years_spreads) > 1 else None
                percentile_5_years  = percentileofscore(past_5_years_spreads,  current_spread) if len(past_5_years_spreads) > 1 else None
                spread_volatility_3m = past_3_months_spreads.std()
                spread_volatility_1y = past_year_spreads.std()
                spread_volatility_3y = past_3_years_spreads.std() if len(past_3_years_spreads) > 1 else None
                spread_volatility_5y = past_5_years_spreads.std() if len(past_5_years_spreads) > 1 else None
                mean_spread_3m = past_3_months_spreads.mean()
                mean_spread_1y = past_year_spreads.mean()
                mean_spread_3y = past_3_years_spreads.mean() if len(past_3_years_spreads) > 1 else None
                mean_spread_5y = past_5_years_spreads.mean() if len(past_5_years_spreads) > 1 else None
                relative_position_3m = abs(current_spread - mean_spread_3m) / spread_volatility_3m if spread_volatility_3m and spread_volatility_3m > 0 else 0
                relative_position_1y = abs(current_spread - mean_spread_1y) / spread_volatility_1y if spread_volatility_1y and spread_volatility_1y > 0 else 0
                relative_position_3y = (abs(current_spread - mean_spread_3y) / spread_volatility_3y
                                        if (spread_volatility_3y and spread_volatility_3y > 0 and mean_spread_3y is not None) else None)
                relative_position_5y = (abs(current_spread - mean_spread_5y) / spread_volatility_5y
                                        if (spread_volatility_5y and spread_volatility_5y > 0 and mean_spread_5y is not None) else None)
                correlation = correlation_matrix.loc[ccy_pair1, ccy_pair2] if (ccy_pair1 in correlation_matrix.index and ccy_pair2 in correlation_matrix.columns) else None
                pair_result = {
                    'Tenor': tenor,
                    'Pair': f"{ccy_pair1}-{ccy_pair2}",
                    'CCY1': ccy_pair1,  # Now this is a currency pair like 'EURUSD'
                    'CCY2': ccy_pair2,  # Now this is a currency pair like 'USDJPY'
                    'Correlation': round(correlation, 4) if correlation is not None else None,
                    'Current_Spread': round(current_spread, 4),
                    'Spread_Mean_3M': round(mean_spread_3m, 4) if mean_spread_3m is not None else None,
                    'Spread_Mean_1Y': round(mean_spread_1y, 4) if mean_spread_1y is not None else None,
                    'Spread_Mean_3Y': round(mean_spread_3y, 4) if mean_spread_3y is not None else None,
                    'Spread_Mean_5Y': round(mean_spread_5y, 4) if mean_spread_5y is not None else None,
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
                if spread_volatility_1y is not None and relative_position_1y is not None:
                    currency_volatilities[ccy_pair1].append({
                        'Spread_STD_1y': spread_volatility_1y,
                        'Spread_ZScore_1y': relative_position_1y,
                        'pair': f"{ccy_pair1}-{ccy_pair2}"})
            results_by_tenor[tenor] = pd.DataFrame(pair_results)
            tenor_currency_averages = []
            for ccy_pair in base_ccy:
                if currency_volatilities[ccy_pair]:
                    avg_spread_vol = np.mean([x['Spread_STD_1y'] for x in currency_volatilities[ccy_pair]])
                    avg_relative_pos = np.mean([x['Spread_ZScore_1y'] for x in currency_volatilities[ccy_pair]])
                    num_pairs = len(currency_volatilities[ccy_pair])
                    tenor_currency_averages.append({
                        'Currency': ccy_pair,  # Now this is a currency pair like 'EURUSD'
                        'Tenor': tenor,
                        'Avg_Spread_STD_1Y': round(avg_spread_vol, 4),
                        'Avg_Spread_ZScore_1Y': round(avg_relative_pos, 4),
                        'Num_Pairs': num_pairs})
            currency_avg_metrics[tenor] = pd.DataFrame(tenor_currency_averages).sort_values('Avg_Spread_STD_1Y')
        return results_by_tenor, currency_avg_metrics, correlation_data, df_all_by_tenor
    
    # ----------------------------- Averaging Spreads -----------------------------
    # ------ Finding Averages of Each Pair Spread (Pair - CCY_Other) for Each Tenor ------
    def find_ave_allCCYallLookback_equalWeights(results_by_tenor, tenors):
        tenor_results = {}  
        for tenor in tenors:
            if tenor in results_by_tenor:
                df = results_by_tenor[tenor]
                all_currency_pairs = set(df['CCY1'].tolist() + df['CCY2'].tolist())
                currency_percentile_summary = []
                for ccy_pair in all_currency_pairs:  # ccy_pair is like 'EURUSD'
                    ccy_pairs = df[df['CCY1'] == ccy_pair]  # All pairs where EURUSD is first
                    if len(ccy_pairs) > 0:
                        avg_percentile_3m = ccy_pairs['Percentile_3M'].mean()
                        avg_percentile_1y = ccy_pairs['Percentile_1Y'].mean()
                        percentile_3y_values = ccy_pairs['Percentile_3Y'].dropna()
                        percentile_5y_values = ccy_pairs['Percentile_5Y'].dropna()
                        avg_percentile_3y = percentile_3y_values.mean() if len(percentile_3y_values) > 0 else None
                        avg_percentile_5y = percentile_5y_values.mean() if len(percentile_5y_values) > 0 else None
                        currency_percentile_summary.append({
                            'Currency_Pair': ccy_pair,  # Updated column name for clarity
                            'Tenor': tenor,
                            'Avg_EquWeight_3M%': round(avg_percentile_3m, 3),
                            'Avg_EquWeight_1Y%': round(avg_percentile_1y, 3),
                            'Avg_EquWeight_3Y%': round(avg_percentile_3y, 3) if avg_percentile_3y is not None else None,
                            'Avg_EquWeight_5Y%': round(avg_percentile_5y, 3) if avg_percentile_5y is not None else None,
                            'Num_Pairs': len(ccy_pairs)})
                percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_EquWeight_5Y%')
                tenor_results[tenor] = percentile_df
        return tenor_results
    
    # ------  Correlation-Weighted Averaging Function ------
    def find_ave_allCCYallLookback_correlationWeights(results_by_tenor, tenors):
        tenor_results = {}  
        for tenor in tenors:
            if tenor in results_by_tenor:
                df = results_by_tenor[tenor]
                all_currency_pairs = set(df['CCY1'].tolist() + df['CCY2'].tolist())
                currency_percentile_summary = []
                for ccy_pair in all_currency_pairs:  # ccy_pair is like 'EURUSD'
                    ccy_pairs = df[df['CCY1'] == ccy_pair].copy()  # All pairs where EURUSD is first
                    if len(ccy_pairs) > 0:
                        ccy_pairs_valid = ccy_pairs.dropna(subset=['Correlation'])
                        if len(ccy_pairs_valid) == 0:
                            weights = np.ones(len(ccy_pairs)) / len(ccy_pairs)
                            ccy_pairs_for_calc = ccy_pairs
                        else:
                            correlation_weights = ccy_pairs_valid['Correlation'].abs()  # Use absolute correlation
                            weights = correlation_weights / correlation_weights.sum()  # Normalize to sum to 1
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
                            weights_3y = weights_3y / weights_3y.sum()  # Renormalize
                            weighted_avg_percentile_3y = np.average(percentile_3y_valid, weights=weights_3y)
                        if len(percentile_5y_valid) > 0:
                            # Get weights for valid 5Y data
                            valid_5y_indices = ccy_pairs_for_calc['Percentile_5Y'].dropna().index
                            weights_5y = weights[ccy_pairs_for_calc.index.isin(valid_5y_indices)]
                            weights_5y = weights_5y / weights_5y.sum()  # Renormalize
                            weighted_avg_percentile_5y = np.average(percentile_5y_valid, weights=weights_5y)
                        avg_correlation = ccy_pairs_valid['Correlation'].mean() if len(ccy_pairs_valid) > 0 else None
                        currency_percentile_summary.append({
                            'Currency_Pair': ccy_pair,  # Updated column name for clarity
                            'Tenor': tenor,
                            'Avg_CorrWeight_3M%': round(weighted_avg_percentile_3m, 3),
                            'Avg_CorrWeight_1Y%': round(weighted_avg_percentile_1y, 3),
                            'Avg_CorrWeight_3Y%': round(weighted_avg_percentile_3y, 3) if weighted_avg_percentile_3y is not None else None,
                            'Avg_CorrWeight_5Y%': round(weighted_avg_percentile_5y, 3) if weighted_avg_percentile_5y is not None else None,
                            'Num_Pairs': len(ccy_pairs)})
                percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_CorrWeight_5Y%')
                tenor_results[tenor] = percentile_df
        return tenor_results
    
    # --------------------- Executing Functions ---------------------
    results_by_tenor, currency_avg_metrics, correlation_data, df_all_by_tenor = xCCY_AllPairs_MultiTenor_Analysis(
        currency_list, tenors)
    
    ave_tenorBuckets_equalBuckets = find_ave_allCCYallLookback_equalWeights(results_by_tenor, tenors)
    ave_tenorBuckets_correlationWeighted = find_ave_allCCYallLookback_correlationWeights(results_by_tenor, tenors)
    
    # Add beta weighting
    ave_tenorBuckets_betaWeighted = find_ave_allCCYallLookback_betaWeights_instantaneous(
        results_by_tenor, df_all_by_tenor, tenors, lookback_days)
    
    return {
        'results_by_tenor': results_by_tenor,
        'ave_results_by_tenor_equalWeights': ave_tenorBuckets_equalBuckets,
        'ave_results_by_tenor_correlationWeighted': ave_tenorBuckets_correlationWeighted,
        'ave_results_by_tenor_betaWeighted': ave_tenorBuckets_betaWeighted,  # New beta results
        'correlation_data': correlation_data
    }








def get_cleanIndividual_pairSpread_Tenors(analysis_results, tenor='1M'):
    data_single = analysis_results["results_by_tenor"][tenor].copy()
    def get_pair_key(ccy1, ccy2):
        return tuple(sorted([ccy1, ccy2]))
    data_single['pair_key'] = data_single.apply(lambda row: get_pair_key(row['CCY1'], row['CCY2']), axis=1)
    deduplicated_data = []
    for pair_key in data_single['pair_key'].unique():
        pair_rows = data_single[data_single['pair_key'] == pair_key]
        if len(pair_rows) > 1:
            chosen_row = pair_rows[pair_rows['CCY1'] == min(pair_rows['CCY1'].tolist())].iloc[0]
            ccy1 = chosen_row['CCY1']
            ccy2 = chosen_row['CCY2']
            ccy1_ccy2_row = pair_rows[(pair_rows['CCY1'] == ccy1) & (pair_rows['CCY2'] == ccy2)]
            ccy2_ccy1_row = pair_rows[(pair_rows['CCY1'] == ccy2) & (pair_rows['CCY2'] == ccy1)]
            ccy1_ccy2_percentile = ccy1_ccy2_row['Percentile_1Y'].iloc[0] if len(ccy1_ccy2_row) > 0 else None
            ccy2_ccy1_percentile = ccy2_ccy1_row['Percentile_1Y'].iloc[0] if len(ccy2_ccy1_row) > 0 else None
            if ccy1_ccy2_percentile is not None and ccy2_ccy1_percentile is not None:
                percentile_range = f"[{ccy1_ccy2_percentile:.3f}, {ccy2_ccy1_percentile:.3f}]"
            elif ccy1_ccy2_percentile is not None:
                percentile_range = f"[{ccy1_ccy2_percentile:.3f}, {100-ccy1_ccy2_percentile:.3f}]"
            else:
                percentile_range = f"[{100-ccy2_ccy1_percentile:.3f}, {ccy2_ccy1_percentile:.3f}]"  
        else:
            chosen_row = pair_rows.iloc[0]
            percentile_value = chosen_row['Percentile_1Y']
            percentile_range = f"[{percentile_value:.3f}, {100-percentile_value:.3f}]"
        chosen_row_dict = chosen_row.to_dict()
        chosen_row_dict['[(1-2)%,(2-1)%]'] = percentile_range
        deduplicated_data.append(chosen_row_dict)
    deduplicated_df = pd.DataFrame(deduplicated_data)
    final_df = deduplicated_df[['CCY1', 'CCY2', 'Current_Spread', 'Spread_ZScore_1Y','[(1-2)%,(2-1)%]', 'Correlation']].reset_index(drop=True)
    return final_df.sort_values('Spread_ZScore_1Y', ascending=False)

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
"""
Each iteration of CCY v CCY Grouped by Tenor
    - Input: currency_list_broad (Full list of Curencies), tenor_interest (Full of Tenors)
"""
def IndividualSpread_AllData_TenorGroup(currency_list_broad, tenor_interest):
    analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenor_interest)
    spreads = {}
    for tenor in tenor_interest:
        spreads[tenor] = analysis_results["results_by_tenor"][tenor].copy()
    return spreads

# tenor_interest = ['1M', '3M', '1Y']
# currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN', 'USDBRL', 'USDCNH']

# spreads_dict = IndividualSpread_AllData_TenorGroup(currency_list_broad, tenor_interest)
# print(spreads_dict['1M'])
# print(spreads_dict['3M'])
# print(spreads_dict['1Y'])






# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
"""
Indvidual CCY Focus vs Rest of CCY        [(EURUSD-USDJPY)(EURUSD-GBPUSD)(EURUSD-USDCHF)...]
    - (1) Spread Percentiles
    - (2) Spread Mean
    - (3) Spread STD

    - Input: ccy_interest (ccy you want compared to rest of ccys), tenor_interest (specific tenor you want to compare spreads on)
"""
def IndivdualCCYSpread_Percentiles_TenorGrouped(ccy_interest, tenor_interest):
    currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
    analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenor_interest)
    tenor = tenor_interest[0]
    data_single = analysis_results["results_by_tenor"][tenor].copy()
    df = data_single[data_single["CCY1"] == ccy_interest]
    df_clean = df[['Pair', 'Tenor', 'Correlation', 'Current_Spread', 'Percentile_3M',  'Percentile_1Y',  'Percentile_3Y',  'Percentile_5Y']]
    return df_clean

def IndivdualCCYSpread_Mean_TenorGrouped(ccy_interest, tenor_interest):
    currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
    analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenor_interest)
    tenor = tenor_interest[0]
    data_single = analysis_results["results_by_tenor"][tenor].copy()
    df = data_single[data_single["CCY1"] == ccy_interest]
    df_clean = df[['Pair', 'Tenor', 'Current_Spread', 'Spread_Mean_3M', 'Spread_Mean_1Y', 'Spread_Mean_3Y', 'Spread_Mean_5Y']]
    df_clean.columns = df_clean.columns.str.replace("Spread_", "")
    return df_clean
def IndivdualCCYSpread_STD_TenorGrouped(ccy_interest, tenor_interest):
    currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
    analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenor_interest)
    tenor = tenor_interest[0]
    data_single = analysis_results["results_by_tenor"][tenor].copy()
    df = data_single[data_single["CCY1"] == ccy_interest]
    df_clean = df[['Pair', 'Tenor', 'Current_Spread', 'Spread_STD_3M', 'Spread_STD_1Y', 'Spread_STD_3Y', 'Spread_STD_5Y']]
    df_clean.columns = df_clean.columns.str.replace("Spread_", "")
    return df_clean

ccy_interest = 'GBPUSD'
tenor_interest = ['1M']
print(IndivdualCCYSpread_Percentiles_TenorGrouped(ccy_interest, tenor_interest))




# tenor_interest = ['3M']
# print(IndivdualCCYSpread_Percentiles_TenorGrouped(ccy_interest, tenor_interest))




# print(IndivdualCCYSpread_Mean_TenorGrouped(ccy_interest, tenor_interest))
# print(IndivdualCCYSpread_STD_TenorGrouped(ccy_interest, tenor_interest))

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------------------
"""
CLEAN OUTPUT OF EACH PAIR FOR TENOR
"""

""" Getting Merged Combos of xCCY Vol Spreads - by tenor Buckets  """

# currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
# tenor_interest = ['1M']

# analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenor_interest)

# data_single = get_cleanIndividual_pairSpread_Tenors(analysis_results, '1M')
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.expand_frame_repr', False)
# print(data_single)









# --------------------------------------------------------------------------------------------------------------

"""
Averaging CCYs over 
# """
# currency_list_broad = [
#     'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 
#     'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN',
#     'USDBRL', 'USDCNH']
# tenors = ['1W']

# analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenors)
# data_weight = analysis_results["ave_results_by_tenor_correlationWeighted"]['1W'].copy()
# print(data_weight)

# data_weight = analysis_results["ave_results_by_tenor_correlationWeighted"]['1W'].copy()
# print(data_weight)


# data_weight = analysis_results["ave_results_by_tenor_betaWeighted"]['1M'].copy()
# print(data_weight)



# """ Average Spread Percentile For CCY - by tenor Buckets --- EVEN WEIGHTED """
# data_weight = analysis_results["ave_results_by_tenor_equalWeights"]['2W'].copy()
# print(data_weight)

# data_weight = analysis_results["ave_results_by_tenor_equalWeights"]['3W'].copy()
# print(data_weight)




# --------------------------------------------------------------------------------------------------------------


# currency_list_broad = [
#     'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 
#     'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN',
#     'USDBRL', 'USDCNH']
# tenors = ['1W', '2W']

# analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenors)



# """ Average Spread Percentile For CCY - by tenor Buckets --- CORR WEIGHTED """



# eurusd_row = data_weight[data_weight["Currency_Pair"] == "EURUSD"]
# print(eurusd_row.T)



























# from pathlib import Path

# def export_report_partials(analysis_results, tenor="1M"):
#     # Directory containing CrossCCYVolSpread.py (your Report folder)
#     script_dir = Path(__file__).resolve().parent

#     # Create ./partials inside that same folder
#     partials = script_dir / "partials"
#     partials.mkdir(parents=True, exist_ok=True)

#     data_single = analysis_results["results_by_tenor"][tenor].copy()
#     data_weight = analysis_results["ave_results_by_tenor_correlationWeighted"][tenor].copy()
#     corr        = analysis_results["correlation_data"][tenor].copy()

#     top10 = (
#         data_single
#         .assign(_key=data_single["Spread_ZScore_1Y"].abs())
#         .sort_values("_key", ascending=False)
#         .head(30)
#         [["Pair","CCY1","CCY2","Current_Spread","Spread_ZScore_1Y","Percentile_1Y","Correlation"]]
#         .round(3)
#     )

#     (partials / f"{tenor}_top10.md").write_text(top10.to_markdown(index=False))
#     (partials / f"{tenor}_corr_weighted.md").write_text(data_weight.round(3).to_markdown(index=False))
#     (partials / f"{tenor}_corr_matrix.md").write_text(corr.round(2).to_markdown())

# if __name__ == "__main__":
#     analysis_results = CheapVolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list, ["1M"])
#     export_report_partials(analysis_results, tenor="1M")

































