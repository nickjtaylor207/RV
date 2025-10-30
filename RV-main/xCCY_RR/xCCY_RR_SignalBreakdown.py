import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xbbg import blp
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from itertools import permutations
from scipy.stats import percentileofscore




def RR25D_VolAdjusted_WindowLookback(currency_list, tenors, lookback_days=252):
    def fetch_RR_and_ATM_data(currency_list: List[str], tenor_list: List[str]):
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365*6)).strftime("%Y-%m-%d")
        base_ccy = sorted(set(currency_list))
        results_by_tenor: Dict[str, pd.DataFrame] = {}
        df_rr_by_tenor: Dict[str, pd.DataFrame] = {}
        df_atm_by_tenor: Dict[str, pd.DataFrame] = {}
        df_rr_adj_by_tenor: Dict[str, pd.DataFrame] = {}  
        
        for tenor in tenor_list:
            rr_data_dict: Dict[str, pd.Series] = {}
            atm_data_dict: Dict[str, pd.Series] = {}
            
            for ccy_pair in base_ccy:
                ticker_RR = f"{ccy_pair}25R{tenor} BGN Curncy"
                ticker_ATM = f"{ccy_pair}V{tenor} BGN Curncy"
                
                try:
                    data_RR = blp.bdh(
                        tickers=ticker_RR,
                        flds="PX_LAST",
                        start_date=start_date,
                        end_date=end_date)
                    data_RR.columns = data_RR.columns.get_level_values(1)
                    rr_data_dict[ccy_pair] = data_RR['PX_LAST']
                    
                    data_ATM = blp.bdh(
                        tickers=ticker_ATM,
                        flds="PX_LAST",
                        start_date=start_date,
                        end_date=end_date)
                    data_ATM.columns = data_ATM.columns.get_level_values(1)
                    atm_data_dict[ccy_pair] = data_ATM['PX_LAST']
                    
                except Exception as e:
                    print(f"Error fetching data for {ccy_pair} {tenor}: {e}")
                    continue
            
            if not rr_data_dict or not atm_data_dict:
                continue
            
            df_rr = pd.DataFrame(rr_data_dict).dropna()
            df_atm = pd.DataFrame(atm_data_dict).dropna()
            
            if df_rr.empty or df_atm.empty:
                continue
            
            df_aligned = pd.concat([df_rr, df_atm], axis=1, keys=['RR', 'ATM']).dropna()
            
            if df_aligned.empty:
                continue
            
            df_rr = df_aligned['RR']
            df_atm = df_aligned['ATM']
            
            # KEEP THE SIGN - this is important for RR interpretation
            df_rr_adj = df_rr.div(df_atm) * 100  
            
            df_rr_by_tenor[tenor] = df_rr
            df_atm_by_tenor[tenor] = df_atm
            df_rr_adj_by_tenor[tenor] = df_rr_adj
            
            latest_date = df_rr_adj.index[-1]
            currency_pairs = list(permutations(base_ccy, 2))
            pair_results = []
            
            for ccy_pair1, ccy_pair2 in currency_pairs:
                if ccy_pair1 not in df_rr_adj.columns or ccy_pair2 not in df_rr_adj.columns:
                    continue
                
                # Spread calculation - SIGNED difference
                current_spread = df_rr_adj.loc[latest_date, ccy_pair1] - df_rr_adj.loc[latest_date, ccy_pair2]
                historical_spreads = df_rr_adj[ccy_pair1] - df_rr_adj[ccy_pair2]
                
                three_months_ago = latest_date - timedelta(days=90)
                one_year_ago = latest_date - timedelta(days=365)
                three_years_ago = latest_date - timedelta(days=365*3)
                five_years_ago = latest_date - timedelta(days=365*5)
                
                past_3_months_spreads = historical_spreads.loc[historical_spreads.index >= three_months_ago]
                past_year_spreads = historical_spreads.loc[historical_spreads.index >= one_year_ago]
                past_3_years_spreads = historical_spreads.loc[historical_spreads.index >= three_years_ago]
                past_5_years_spreads = historical_spreads.loc[historical_spreads.index >= five_years_ago]
                
                if len(past_5_years_spreads) < 2:
                    continue
                
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
                
                # Z-score: SIGNED deviation (not absolute)
                # This tells you if current spread is above or below mean
                z_score_3m = (current_spread - mean_spread_3m) / spread_volatility_3m if spread_volatility_3m and spread_volatility_3m > 0 else 0
                z_score_1y = (current_spread - mean_spread_1y) / spread_volatility_1y if spread_volatility_1y and spread_volatility_1y > 0 else 0
                z_score_3y = ((current_spread - mean_spread_3y) / spread_volatility_3y
                              if (spread_volatility_3y and spread_volatility_3y > 0 and mean_spread_3y is not None) else None)
                z_score_5y = ((current_spread - mean_spread_5y) / spread_volatility_5y
                              if (spread_volatility_5y and spread_volatility_5y > 0 and mean_spread_5y is not None) else None)
                
                current_rr1 = df_rr.loc[latest_date, ccy_pair1]
                current_rr2 = df_rr.loc[latest_date, ccy_pair2]
                current_atm1 = df_atm.loc[latest_date, ccy_pair1]
                current_atm2 = df_atm.loc[latest_date, ccy_pair2]
                
                pair_result = {
                    'Tenor': tenor,
                    'Pair': f"{ccy_pair1}-{ccy_pair2}",
                    'CCY1': ccy_pair1,
                    'CCY2': ccy_pair2,
                    'Current_Spread_VolAdj': round(current_spread, 4),
                    'Current_RR1': round(current_rr1, 4),
                    'Current_RR2': round(current_rr2, 4),
                    'Current_ATM1': round(current_atm1, 4),
                    'Current_ATM2': round(current_atm2, 4),
                    'Current_RR1_VolAdj': round(current_rr1 / current_atm1 * 100, 4),
                    'Current_RR2_VolAdj': round(current_rr2 / current_atm2 * 100, 4),
                    'Spread_Mean_3M': round(mean_spread_3m, 4) if mean_spread_3m is not None else None,
                    'Spread_Mean_1Y': round(mean_spread_1y, 4) if mean_spread_1y is not None else None,
                    'Spread_Mean_3Y': round(mean_spread_3y, 4) if mean_spread_3y is not None else None,
                    'Spread_Mean_5Y': round(mean_spread_5y, 4) if mean_spread_5y is not None else None,
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
                    'Percentile_5Y': round(percentile_5_years, 3) if percentile_5_years is not None else None}
                
                pair_results.append(pair_result)
            
            results_by_tenor[tenor] = pd.DataFrame(pair_results)
        
        return results_by_tenor, df_rr_by_tenor, df_atm_by_tenor, df_rr_adj_by_tenor
    
    def find_ave_allCCY_equalWeights(results_by_tenor, tenors):
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
                        
                        # Also track average Z-scores to see directional bias
                        avg_zscore_1y = ccy_pairs['Spread_ZScore_1Y'].mean()
                        
                        currency_percentile_summary.append({
                            'Currency_Pair': ccy_pair,
                            'Tenor': tenor,
                            'Avg_EquWeight_3M%': round(avg_percentile_3m, 3),
                            'Avg_EquWeight_1Y%': round(avg_percentile_1y, 3),
                            'Avg_EquWeight_3Y%': round(avg_percentile_3y, 3) if avg_percentile_3y is not None else None,
                            'Avg_EquWeight_5Y%': round(avg_percentile_5y, 3) if avg_percentile_5y is not None else None,
                            'Avg_ZScore_1Y': round(avg_zscore_1y, 3),
                            'Num_Pairs': len(ccy_pairs)})
                
                percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_EquWeight_1Y%')
                tenor_results[tenor] = percentile_df
        
        return tenor_results
    
    results_by_tenor, df_rr_by_tenor, df_atm_by_tenor, df_rr_adj_by_tenor = fetch_RR_and_ATM_data(
        currency_list, tenors)
    
    ave_tenorBuckets_equalWeights = find_ave_allCCY_equalWeights(results_by_tenor, tenors)
    
    return {
        'results_by_tenor': results_by_tenor,
        'ave_results_by_tenor_equalWeights': ave_tenorBuckets_equalWeights,
        'df_rr_raw': df_rr_by_tenor,
        'df_atm': df_atm_by_tenor,
        'df_rr_vol_adjusted': df_rr_adj_by_tenor}


tenor_interest = ['1M']
currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN', 'USDBRL', 'USDCNH']

spreads_dict = RR25D_VolAdjusted_WindowLookback(currency_list_broad, tenor_interest)
print(spreads_dict['ave_results_by_tenor_equalWeights']['1M'])



def IndivdualRRSpread_Percentiles_TenorGrouped(ccy_interest, tenor_interest):
    currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
    analysis_results = RR25D_VolAdjusted_WindowLookback(currency_list_broad, tenor_interest)
    tenor = tenor_interest[0]
    data_single = analysis_results["results_by_tenor"][tenor].copy()
    df = data_single[data_single["CCY1"] == ccy_interest]
    df_clean = df[['Pair', 'Tenor', 'Current_Spread_VolAdj', 'Percentile_3M',  'Percentile_1Y',  'Percentile_3Y',  'Percentile_5Y']]
    return df_clean.sort_values('Percentile_1Y', ascending=False)


def IndivdualRRSpread_MeanSTD_TenorGrouped(ccy_interest, tenor_interest):
    currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
    analysis_results = RR25D_VolAdjusted_WindowLookback(currency_list_broad, tenor_interest)
    tenor = tenor_interest[0]
    data_single = analysis_results["results_by_tenor"][tenor].copy()
    df = data_single[data_single["CCY1"] == ccy_interest]
    df_clean = pd.DataFrame({
        'Pair': df['Pair'],
        'Tenor': df['Tenor']})
    periods = ['3M', '1Y', '3Y', '5Y']
    for period in periods:
        df_clean[(period, 'Mean')] = df[f'Spread_Mean_{period}']
        df_clean[(period, 'STD')] = df[f'Spread_STD_{period}']
    single_cols = [('Pair', ''), ('Tenor', '')]
    multi_cols = [(period, stat) for period in periods for stat in ['Mean', 'STD']]
    df_clean.columns = pd.MultiIndex.from_tuples(single_cols + multi_cols)
    
    return df_clean







# ccy = 'EURUSD'
# tenors = ['1W']
# print(IndivdualRRSpread_Percentiles_TenorGrouped(ccy, tenors))
# print(IndivdualRRSpread_MeanSTD_TenorGrouped(ccy, tenors))








