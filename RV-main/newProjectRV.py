from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import permutations
from scipy.stats import percentileofscore
from xbbg import blp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Rectangle


def get_ImplRealVol(currency_list, tenors, days_back):
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    series_dict = {} 
    vol_types = ["H", "V"] 
    for tenor in tenors:
        for ccy in currency_list:
            for vt in vol_types:
                ticker = f"{ccy}{vt}{tenor} Curncy"
                df = blp.bdh(
                    tickers=ticker,
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date)
                df.columns = df.columns.get_level_values(1)
                s = df["PX_LAST"].astype(float).dropna()
                series_dict[(vt, tenor, ccy)] = s
    out = pd.concat(series_dict, axis=1, join="outer").sort_index()
    out.columns = [f"{ccy}_{vt}{tenor}" for (vt, tenor, ccy) in out.columns]
    return out

def get_Allvol_df(ccys, tenors, days_back):
    include_iv=True; include_hv=False
    df = get_ImplRealVol(ccys, tenors, days_back)
    out = {}   
    for ccy in ccys:
        for tenor in tenors:
            vcol = f"{ccy}_V{tenor}"
            hcol = f"{ccy}_H{tenor}"
            has_v = vcol in df.columns
            has_h = hcol in df.columns
            if has_v and has_h:
                pair = df[[vcol, hcol]]
                pair = pair.dropna() 
                spread = pair[vcol].astype(float) - pair[hcol].astype(float)
                out[("VDiff", tenor, ccy)] = spread
                if include_iv and has_v:
                    iv = df[vcol].astype(float).reindex(spread.index)
                    out[("IV", tenor, ccy)] = iv
                if include_hv and has_h:
                    hv = df[hcol].astype(float).reindex(spread.index) 
                    out[("HV", tenor, ccy)] = hv
            else:
                if include_iv and has_v:
                    iv = df[vcol].astype(float)
                    iv = iv.dropna()
                    out[("IV", tenor, ccy)] = iv
    res = pd.concat(out, axis=1).sort_index()
    res.columns = [f"{ccy}_{tenor}_{metric}" for (metric, tenor, ccy) in res.columns]
    return res


def _latest_value_and_percentile(col: pd.Series, exclude_current: bool = True):
    s = col.dropna()
    if s.empty:
        return np.nan, np.nan
    current_value = s.iloc[-1]
    ref = s.iloc[:-1] if exclude_current and len(s) > 1 else s
    pct = percentileofscore(ref.values, current_value, kind="weak")
    return current_value, pct


def summarize_iv_vdiff_percentiles(ccys, tenors, days_back, exclude_current: bool = True, round_value: int = 3, round_pct: int = 1) -> pd.DataFrame:
    panel_df = get_Allvol_df(ccys, tenors, days_back)
    col_tuples = []
    for tenor in tenors:
        col_tuples.extend([
            (tenor, "IV",    "Current"),
            (tenor, "IV",    "%"),
            (tenor, "I-R",   "Current"),
            (tenor, "I-R",   "%"),
            (tenor, "mrkt_rel", "%")])
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["Tenor", "Metric", "Field"])
    rows = {}
    for ccy in ccys:
        row_vals = []
        for tenor in tenors:
            iv_col = f"{ccy}_{tenor}_IV"
            if iv_col in panel_df.columns:
                iv_val, iv_pct = _latest_value_and_percentile(panel_df[iv_col], exclude_current=exclude_current)
            else:
                iv_val, iv_pct = np.nan, np.nan
            vd_col = f"{ccy}_{tenor}_VDiff"
            if vd_col in panel_df.columns:
                vd_val, vd_pct = _latest_value_and_percentile(panel_df[vd_col], exclude_current=exclude_current)
            else:
                vd_val, vd_pct = np.nan, np.nan
            row_vals.extend([
                (round(iv_val, round_value) if pd.notna(iv_val) else np.nan),
                (round(iv_pct, round_pct)   if pd.notna(iv_pct) else np.nan),
                (round(vd_val, round_value) if pd.notna(vd_val) else np.nan),
                (round(vd_pct, round_pct)   if pd.notna(vd_pct) else np.nan),
                np.nan]) 
        rows[ccy] = row_vals
    out = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    out.index.name = "CCY"
    out = out.reindex(columns=columns)
    return out



# ----------------------------------------------------------------------------------------------------------

def ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list, tenors, lookback_days):
    def xCCY_AllPairs_MultiTenor_Analysis(currency_list: List[str], tenor_list: List[str]):
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        base_ccy = sorted(set(currency_list))  
        results_by_tenor: Dict[str, pd.DataFrame] = {}
        currency_avg_metrics: Dict[str, pd.DataFrame] = {}
        correlation_data: Dict[str, pd.DataFrame] = {}  
        df_all_by_tenor: Dict[str, pd.DataFrame] = {} 
        for tenor in tenor_list:
            data_dict: Dict[str, pd.Series] = {}
            for ccy_pair in base_ccy:  
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
            correlation_matrix = df_all.diff().dropna().corr()
            correlation_data[tenor] = correlation_matrix
            latest_date = df_all.index[-1]
            currency_pairs = list(permutations(base_ccy, 2))
            pair_results = []
            currency_volatilities: Dict[str, List[Dict[str, float]]] = {ccy_pair: [] for ccy_pair in base_ccy}
            for ccy_pair1, ccy_pair2 in currency_pairs: 
                if ccy_pair1 not in df_all.columns or ccy_pair2 not in df_all.columns:
                    continue
                current_spread = df_all.loc[latest_date, ccy_pair1] - df_all.loc[latest_date, ccy_pair2]
                historical_spreads = df_all[ccy_pair1] - df_all[ccy_pair2]
                Percentile_lookback = percentileofscore(historical_spreads, current_spread)
                STD_lookback = historical_spreads.std()
                mean_lookback = historical_spreads.mean()
                ZScore_lookback = (current_spread - mean_lookback) / STD_lookback
                correlation = correlation_matrix.loc[ccy_pair1, ccy_pair2] if (ccy_pair1 in correlation_matrix.index and ccy_pair2 in correlation_matrix.columns) else None
                pair_result = {
                    'Tenor': tenor,
                    'Pair': f"{ccy_pair1}-{ccy_pair2}",
                    'CCY1': ccy_pair1,  # Now this is a currency pair like 'EURUSD'
                    'CCY2': ccy_pair2,  # Now this is a currency pair like 'USDJPY'
                    'Correlation': round(correlation, 4) if correlation is not None else None,
                    'Current_Spread': round(current_spread, 4),
                    'Spread_Mean': round(mean_lookback, 4) if mean_lookback is not None else None,
                    'Spread_STD': round(STD_lookback, 4) if STD_lookback is not None else None,
                    'ZScore_current': round(ZScore_lookback, 4) if ZScore_lookback is not None else None,
                    'Percentile_current': round(Percentile_lookback, 3) if Percentile_lookback is not None else None}
                pair_results.append(pair_result)
            results_by_tenor[tenor] = pd.DataFrame(pair_results)
        return results_by_tenor
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
                        weighted_avg_percentile_Lookback = np.average(ccy_pairs_for_calc['Percentile_current'], weights=weights)
                        currency_percentile_summary.append({
                            'Currency_Pair': ccy_pair,  # Updated column name for clarity
                            'Tenor': tenor,
                            'Avg_CorrWeight%': round(weighted_avg_percentile_Lookback, 3),
                            'Num_Pairs': len(ccy_pairs)})
                percentile_df = pd.DataFrame(currency_percentile_summary).sort_values('Avg_CorrWeight%')
                tenor_results[tenor] = percentile_df
        return tenor_results
    results_by_tenor  = xCCY_AllPairs_MultiTenor_Analysis(currency_list, tenors)   
    ave_tenorBuckets_correlationWeighted = find_ave_allCCYallLookback_correlationWeights(results_by_tenor, tenors)
    return {'results_by_tenor': results_by_tenor,
            'ave_results_by_tenor_correlationWeighted': ave_tenorBuckets_correlationWeighted}



def ATM_VolRefferenceRanking_AllCCY_CrossTenor_FullTimePannel(currency_list, tenors, lookback_days):
    def xTenor_SingleCCY_Analysis(currency_list: List[str], tenor_list: List[str]):
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        results_by_ccy: Dict[str, pd.DataFrame] = {}
        correlation_data_by_ccy: Dict[str, pd.DataFrame] = {}
        df_all_by_ccy: Dict[str, pd.DataFrame] = {}
        for ccy in currency_list:
            print(f"Getting {ccy}")
            data_dict: Dict[str, pd.Series] = {}
            for tenor in tenor_list:
                ticker_IV = f"{ccy}V{tenor} BGN Curncy"
                try:
                    data_IV = blp.bdh(
                        tickers=ticker_IV,
                        flds="PX_LAST",
                        start_date=start_date,
                        end_date=end_date)
                    data_IV.columns = data_IV.columns.get_level_values(1)
                    data_dict[tenor] = data_IV['PX_LAST']
                except Exception as e:
                    continue
            if not data_dict:
                continue
            df_all = pd.DataFrame(data_dict).dropna()
            if df_all.empty:
                continue
            df_all_by_ccy[ccy] = df_all
            correlation_matrix = df_all.diff().dropna().corr()
            correlation_data_by_ccy[ccy] = correlation_matrix
            latest_date = df_all.index[-1]
            tenor_pairs = list(permutations(tenor_list, 2))
            pair_results = []
            for tenor1, tenor2 in tenor_pairs:
                if tenor1 not in df_all.columns or tenor2 not in df_all.columns:
                    continue
                current_spread = df_all.loc[latest_date, tenor1] - df_all.loc[latest_date, tenor2]
                historical_spreads = df_all[tenor1] - df_all[tenor2]
                Percentile_lookback = percentileofscore(historical_spreads, current_spread)
                STD_lookback = historical_spreads.std()
                mean_lookback = historical_spreads.mean()
                ZScore_lookback = (current_spread - mean_lookback) / STD_lookback
                correlation = correlation_matrix.loc[tenor1, tenor2] if (
                    tenor1 in correlation_matrix.index and tenor2 in correlation_matrix.columns) else None
                pair_result = {
                    'Currency': ccy,
                    'Pair': f"{tenor1}-{tenor2}",
                    'Tenor1': tenor1,
                    'Tenor2': tenor2,
                    'Correlation': round(correlation, 4) if correlation is not None else None,
                    'Current_Spread': round(current_spread, 4),
                    'Spread_Mean': round(mean_lookback, 4) if mean_lookback is not None else None,
                    'Spread_STD': round(STD_lookback, 4) if STD_lookback is not None else None,
                    'ZScore_current': round(ZScore_lookback, 4) if ZScore_lookback is not None else None,
                    'Percentile_current': round(Percentile_lookback, 3) if Percentile_lookback is not None else None}
                pair_results.append(pair_result)
            results_by_ccy[ccy] = pd.DataFrame(pair_results)
        return results_by_ccy, correlation_data_by_ccy, df_all_by_ccy
    def find_ave_allTenor_correlationWeights(results_by_ccy, currency_list):
        ccy_results = {}
        for ccy in currency_list:
            if ccy in results_by_ccy:
                df = results_by_ccy[ccy]
                all_tenors = set(df['Tenor1'].tolist() + df['Tenor2'].tolist())
                tenor_percentile_summary = []
                for tenor in all_tenors:
                    tenor_pairs = df[df['Tenor1'] == tenor].copy()
                    if len(tenor_pairs) > 0:
                        tenor_pairs_valid = tenor_pairs.dropna(subset=['Correlation'])
                        if len(tenor_pairs_valid) == 0:
                            weights = np.ones(len(tenor_pairs)) / len(tenor_pairs)
                            tenor_pairs_for_calc = tenor_pairs
                        else:
                            correlation_weights = tenor_pairs_valid['Correlation'].abs()
                            weights = correlation_weights / correlation_weights.sum()
                            tenor_pairs_for_calc = tenor_pairs_valid
                        weighted_avg_percentile = np.average(
                            tenor_pairs_for_calc['Percentile_current'], 
                            weights=weights)
                        tenor_percentile_summary.append({
                            'Currency': ccy,
                            'Tenor': tenor,
                            'Avg_CorrWeight%': round(weighted_avg_percentile, 3),
                            'Num_Pairs': len(tenor_pairs)})
                percentile_df = pd.DataFrame(tenor_percentile_summary).sort_values('Avg_CorrWeight%')
                ccy_results[ccy] = percentile_df
        return ccy_results
    results_by_ccy, correlation_data, df_all_data = xTenor_SingleCCY_Analysis(currency_list, tenors)
    ave_ccy_correlationWeighted = find_ave_allTenor_correlationWeights(results_by_ccy, currency_list)
    return {
        'results_by_ccy': results_by_ccy,
        'ave_results_by_ccy_correlationWeighted': ave_ccy_correlationWeighted,
        'correlation_data': correlation_data,
        'raw_data': df_all_data}



def create_combined_score_matrix(ccy_weighted_results, tenor_weighted_results, ccys, tenors):
    combined_scores = pd.DataFrame(index=ccys, columns=tenors)
    for ccy in ccys:
        for tenor in tenors:
            ccy_percentile = None
            if tenor in ccy_weighted_results:
                tenor_df = ccy_weighted_results[tenor]
                ccy_row = tenor_df[tenor_df['Currency_Pair'] == ccy]
                if not ccy_row.empty:
                    ccy_percentile = ccy_row['Avg_CorrWeight%'].values[0]
            tenor_percentile = None
            if ccy in tenor_weighted_results:
                ccy_df = tenor_weighted_results[ccy]
                tenor_row = ccy_df[ccy_df['Tenor'] == tenor]
                if not tenor_row.empty:
                    tenor_percentile = tenor_row['Avg_CorrWeight%'].values[0]
            # Combine scores 
            if ccy_percentile is not None and tenor_percentile is not None:
                combined_avg = (ccy_percentile + tenor_percentile) / 2      # (A) Simple average
                combined_geo = np.sqrt(ccy_percentile * tenor_percentile)   # (B) Geometric mean (emphasizes extremes)
                combined_min = min(ccy_percentile, tenor_percentile)        # (C) Min (most conservative - both dimensions must be high)
                
                ccy_weight = 0.6  # (CURRENT) Weighted combination (can adjust weights)
                tenor_weight = 0.4 # ------ Adjust based on what's more important -------
                combined_weighted = (ccy_weight * ccy_percentile + tenor_weight * tenor_percentile)
                
                # Store the weighted combination
                combined_scores.loc[ccy, tenor] = combined_weighted
    
    return combined_scores.astype(float)



def create_2d_relative_map(ccy_weighted_results, tenor_weighted_results, ccys, tenors):
    scatter_points = []
    for ccy in ccys:
        for tenor in tenors:
            ccy_percentile = None
            if tenor in ccy_weighted_results:
                tenor_df = ccy_weighted_results[tenor]
                ccy_row = tenor_df[tenor_df['Currency_Pair'] == ccy]
                if not ccy_row.empty:
                    ccy_percentile = ccy_row['Avg_CorrWeight%'].values[0]
            tenor_percentile = None
            if ccy in tenor_weighted_results:
                ccy_df = tenor_weighted_results[ccy]
                tenor_row = ccy_df[ccy_df['Tenor'] == tenor]
                if not tenor_row.empty:
                    tenor_percentile = tenor_row['Avg_CorrWeight%'].values[0]
            if ccy_percentile is not None and tenor_percentile is not None:
                scatter_points.append({
                    'Currency': ccy,
                    'Tenor': tenor,
                    'Label': f"{ccy}-{tenor}",
                    'CrossCcy_Percentile': ccy_percentile,
                    'CrossTenor_Percentile': tenor_percentile,
                    'Distance_from_50': np.sqrt((ccy_percentile - 50)**2 + (tenor_percentile - 50)**2)})
    return pd.DataFrame(scatter_points)



def calculate_composite_zscore(cross_ccy_results, cross_tenor_results, ccys, tenors):
    z_matrix = pd.DataFrame(index=ccys, columns=tenors)
    for ccy in ccys:
        for tenor in tenors:
            z_scores = []
            if tenor in cross_ccy_results['results_by_tenor']:
                tenor_df = cross_ccy_results['results_by_tenor'][tenor]
                ccy_pairs = tenor_df[(tenor_df['CCY1'] == ccy) | (tenor_df['CCY2'] == ccy)]
                if not ccy_pairs.empty:
                    avg_zscore_ccy = ccy_pairs['ZScore_current'].abs().mean()
                    z_scores.append(avg_zscore_ccy)
            if ccy in cross_tenor_results['results_by_ccy']:
                ccy_df = cross_tenor_results['results_by_ccy'][ccy]
                tenor_pairs = ccy_df[(ccy_df['Tenor1'] == tenor) | (ccy_df['Tenor2'] == tenor)]
                if not tenor_pairs.empty:
                    avg_zscore_tenor = tenor_pairs['ZScore_current'].abs().mean()
                    z_scores.append(avg_zscore_tenor)
            if z_scores:
                z_matrix.loc[ccy, tenor] = np.mean(z_scores)
    return z_matrix.astype(float)




# -------------------------------------------------------------------------------------------
# ---------------------------------- PLOTTING PRESENTATION ----------------------------------

def create_combined_visualization(combined_matrix, scatter_data):
    fig = plt.figure(figsize=(22, 14))
    # ==================== CORE ANALYSIS (1st Row)====================
    ax1 = plt.subplot(2, 4, (1, 2)) 
    colors = ['#00ff00', '#90EE90', '#FFFF99', '#FFB366', '#ff0000']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('cheap_expensive', colors, N=n_bins)
    sns.heatmap(combined_matrix, annot=True, fmt='.1f', cmap=cmap,
                center=50, vmin=0, vmax=100, 
                cbar_kws={'label': 'Combined Percentile (0=Cheap, 100=Expensive)'},
                linewidths=0.5, linecolor='gray')
    ax1.set_title('COMBINED RELATIVE VALUE HEATMAP', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Tenor', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Currency Pair', fontsize=11, fontweight='bold')
    for i in range(len(combined_matrix.columns)):
        for j in range(len(combined_matrix.index)):
            val = combined_matrix.iloc[j, i]
            if pd.notna(val):
                if val <= 20:
                    ax1.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, 
                                                edgecolor='green', lw=3))
                elif val >= 80:
                    ax1.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, 
                                                edgecolor='red', lw=3))
    # -------- 2D Quadrant Scatter --------
    ax2 = plt.subplot(2, 4, (3, 4))  # Spans 2 columns
    if not scatter_data.empty:
        scatter_data_copy = scatter_data.copy()
        scatter_data_copy['Overall_Score'] = (scatter_data_copy['CrossCcy_Percentile'] + 
                                               scatter_data_copy['CrossTenor_Percentile']) / 2
        scatter = ax2.scatter(scatter_data_copy['CrossCcy_Percentile'], 
                             scatter_data_copy['CrossTenor_Percentile'],
                             s=150, alpha=0.7, 
                             c=scatter_data_copy['Overall_Score'],
                             cmap='RdYlGn_r', vmin=0, vmax=100,
                             edgecolors='black', linewidth=1.5)
        extreme_mask = ((scatter_data_copy['Overall_Score'] <= 25) | # Extreme points only
                        (scatter_data_copy['Overall_Score'] >= 75))
        for _, row in scatter_data_copy[extreme_mask].iterrows():
            ax2.annotate(row['Label'], 
                        (row['CrossCcy_Percentile'], row['CrossTenor_Percentile']),
                        fontsize=9, ha='left', fontweight='bold',
                        xytext=(3, 3), textcoords='offset points')
        ax2.axhline(y=50, color='black', linestyle='-', alpha=0.7, linewidth=2)
        ax2.axvline(x=50, color='black', linestyle='-', alpha=0.7, linewidth=2)
        ax2.text(60, 55, 'EXPENSIVE', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
        ax2.text(25, 85, 'Mixed:\nTenor Rich\nCCY Cheap', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.7))
        ax2.text(75, 15, 'Mixed:\nCCY Rich\nTenor Cheap', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.7))
        ax2.text(40, 45, 'CHEAP', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))
        plt.colorbar(scatter, ax=ax2, label='Overall Score')
    ax2.set_xlabel('Cross-Currency Percentile', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cross-Tenor Percentile', fontsize=11, fontweight='bold')
    ax2.set_title('2D RELATIVE VALUE QUADRANT MAP', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle='--')
    # ==================== RANKINGS & DISTRIBUTION (2nd Row) ====================
    # -------- Top SELLS (Most Expensive)  -------- 
    ax3 = plt.subplot(2, 4, 5)
    ax3.axis('tight')
    ax3.axis('off')
    if not scatter_data.empty:
        scatter_data_copy = scatter_data.copy()
        scatter_data_copy['Overall_Score'] = (scatter_data_copy['CrossCcy_Percentile'] + scatter_data_copy['CrossTenor_Percentile']) / 2
        top_expensive = scatter_data_copy.nlargest(10, 'Overall_Score')[
            ['Label', 'CrossCcy_Percentile', 'CrossTenor_Percentile', 'Overall_Score']].round(1)
        top_expensive.columns = ['Vol', 'CCY %', 'Tenor %', 'Overall %']
        table = ax3.table(cellText=top_expensive.values,
                         colLabels=top_expensive.columns,
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        for i in range(1, len(top_expensive) + 1):
            score = top_expensive.iloc[i-1]['Overall %']
            if score >= 80:
                color = '#ff6666'  # Dark red
            elif score >= 70:
                color = '#ff9999'  # Medium red
            else:
                color = '#ffcccc'  # Light red
            for j in range(4):
                table[(i, j)].set_facecolor(color)
        for j in range(4):
            table[(0, j)].set_facecolor('#cc0000')
            table[(0, j)].set_text_props(weight='bold', color='white')
    ax3.set_title('TOP 10 SELLS (Most Expensive)', 
                  fontsize=12, fontweight='bold', color='darkred')
    # -------- Top BUYS (Cheapest) -------- 
    ax4 = plt.subplot(2, 4, 6)
    ax4.axis('tight')
    ax4.axis('off')
    if not scatter_data.empty:
        top_cheap = scatter_data_copy.nsmallest(10, 'Overall_Score')[
            ['Label', 'CrossCcy_Percentile', 'CrossTenor_Percentile', 'Overall_Score']].round(1)
        top_cheap.columns = ['Vol', 'CCY %', 'Tenor %', 'Overall %']
        table = ax4.table(cellText=top_cheap.values,
                         colLabels=top_cheap.columns,
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        for i in range(1, len(top_cheap) + 1):
            score = top_cheap.iloc[i-1]['Overall %']
            if score <= 20:
                color = '#66ff66'  # Dark green
            elif score <= 30:
                color = '#99ff99'  # Medium green
            else:
                color = '#ccffcc'  # Light green
            for j in range(4):
                table[(i, j)].set_facecolor(color)
        for j in range(4):
            table[(0, j)].set_facecolor('#00cc00')
            table[(0, j)].set_text_props(weight='bold', color='white')
    ax4.set_title('TOP 10 BUYS (Cheapest)', 
                  fontsize=12, fontweight='bold', color='darkgreen')
    # -------- Distribution Histogram --------
    ax5 = plt.subplot(2, 4, 7)
    if not scatter_data.empty:
        ax5.hist(scatter_data_copy['Overall_Score'], bins=20, 
                alpha=0.7, color='steelblue', edgecolor='black')
        ax5.axvline(x=50, color='black', linestyle='--', linewidth=2, label='Fair Value')
        ax5.axvline(x=20, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Cheap Threshold')
        ax5.axvline(x=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Rich Threshold')
        ax5.axvspan(0, 20, alpha=0.2, color='green', label='Buy Zone')
        ax5.axvspan(80, 100, alpha=0.2, color='red', label='Sell Zone')
        ax5.set_xlabel('Combined Percentile Score', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax5.set_title('Distribution of Relative Values', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_xlim(0, 100)
    ax6 = plt.subplot(2, 4, 8)
    ax6.axis('off')
    if not scatter_data.empty:
        stats_text = f"""
MARKET SUMMARY STATISTICS
{'='*40}

Total Volatilities Analyzed: {len(scatter_data_copy)}

PERCENTILE RANGES:
  • Extreme Cheap (0-20%):   {sum(scatter_data_copy['Overall_Score'] <= 20)}
  • Cheap (20-40%):          {sum((scatter_data_copy['Overall_Score'] > 20) & (scatter_data_copy['Overall_Score'] <= 40))}
  • Fair Value (40-60%):     {sum((scatter_data_copy['Overall_Score'] > 40) & (scatter_data_copy['Overall_Score'] <= 60))}
  • Expensive (60-80%):      {sum((scatter_data_copy['Overall_Score'] > 60) & (scatter_data_copy['Overall_Score'] <= 80))}
  • Extreme Expensive (80-100%): {sum(scatter_data_copy['Overall_Score'] > 80)}

PERCENTILE STATISTICS:
  • Mean:    {scatter_data_copy['Overall_Score'].mean():.1f}%
  • Median:  {scatter_data_copy['Overall_Score'].median():.1f}%
  • Std Dev: {scatter_data_copy['Overall_Score'].std():.1f}%
  • Min:     {scatter_data_copy['Overall_Score'].min():.1f}%
  • Max:     {scatter_data_copy['Overall_Score'].max():.1f}%

MARKET BIAS:
• {"SKEWED EXPENSIVE" if scatter_data_copy['Overall_Score'].mean() > 55 else "SKEWED CHEAP" if scatter_data_copy['Overall_Score'].mean() < 45 else "BALANCED"}
• Dispersion: {"HIGH" if scatter_data_copy['Overall_Score'].std() > 20 else "MODERATE" if scatter_data_copy['Overall_Score'].std() > 10 else "LOW"}
        """
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.suptitle('FX VOLATILITY RELATIVE VALUE DASHBOARD', 
                 fontsize=18, fontweight='bold', y=0.98)
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='right', fontsize=8, style='italic')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()







def create_2d_scatter(scatter_data, n_top=10, n_bottom=10):
    if scatter_data.empty:
        print("No data to plot")
        return
    scatter_data_copy = scatter_data.copy()
    scatter_data_copy['Overall_Score'] = (
        scatter_data_copy['CrossCcy_Percentile'] + 
        scatter_data_copy['CrossTenor_Percentile']) / 2
    fig, ax = plt.subplots(figsize=(18, 14))
    scatter = ax.scatter(
        scatter_data_copy['CrossCcy_Percentile'], 
        scatter_data_copy['CrossTenor_Percentile'],
        s=250,
        alpha=0.7, 
        c=scatter_data_copy['Overall_Score'],
        cmap='RdYlGn_r',
        vmin=0, 
        vmax=100,
        edgecolors='black', 
        linewidth=2,
        zorder=3)
    top_n = scatter_data_copy.nlargest(n_top, 'Overall_Score')
    bottom_n = scatter_data_copy.nsmallest(n_bottom, 'Overall_Score')
    extreme_points = pd.concat([top_n, bottom_n]).drop_duplicates()
    ax.scatter(
        extreme_points['CrossCcy_Percentile'], 
        extreme_points['CrossTenor_Percentile'],
        s=300,  # Slightly larger
        facecolors='none',
        edgecolors='gold',
        linewidth=3,
        zorder=4,
        label='Labeled Points')
    texts = []
    if not extreme_points.empty:
        for _, row in extreme_points.iterrows():
            is_expensive = row['Label'] in top_n['Label'].values
            if is_expensive:
                label_color = 'darkred'
                bbox_color = '#ffcccc'
            else:
                label_color = 'darkgreen'
                bbox_color = '#ccffcc'
            text = ax.text(
                row['CrossCcy_Percentile'],
                row['CrossTenor_Percentile'],
                f"{row['Label']}\n{row['Overall_Score']:.1f}%",  # Include score in label
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                color=label_color,
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=bbox_color,
                    edgecolor=label_color,
                    alpha=0.95,
                    linewidth=2),zorder=5)
            texts.append(text)
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle='-',
                color='gray',
                lw=1.5,
                alpha=0.7),
            expand_points=(1.5, 1.5),
            expand_text=(1.2, 1.2),
            force_points=(0.5, 0.5),
            force_text=(0.7, 0.7),
            lim=500)
    ax.axhline(y=50, color='black', linestyle='-', alpha=0.8, linewidth=2.5, zorder=2)
    ax.axvline(x=50, color='black', linestyle='-', alpha=0.8, linewidth=2.5, zorder=2)
    ax.axhline(y=25, color='green', linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)
    ax.axhline(y=75, color='red', linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)
    ax.axvline(x=25, color='green', linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)
    ax.axvline(x=75, color='red', linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)
    ax.axvspan(0, 25, alpha=0.05, color='green', zorder=0)
    ax.axvspan(75, 100, alpha=0.05, color='red', zorder=0)
    ax.axhspan(0, 25, alpha=0.05, color='green', zorder=0)
    ax.axhspan(75, 100, alpha=0.05, color='red', zorder=0)
    ax.text(
        85, 85, 
        'EXPENSIVE\nBoth Dimensions\n→ SELL VOL', 
        ha='center', va='center', 
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#ff9999', alpha=0.8, 
                 edgecolor='darkred', linewidth=2),
        zorder=4)
    ax.text(
        15, 85, 
        'Expensive Tenors\nCheap CCYs', 
        ha='center', va='center', 
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8, 
                 edgecolor='gray', linewidth=1.5),
        zorder=4)
    ax.text(
        85, 15, 
        'Expensive CCYs\nCheap Tenors', 
        ha='center', va='center', 
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8, 
                 edgecolor='gray', linewidth=1.5),
        zorder=4)
    ax.text(
        15, 15, 
        'CHEAP\nBoth Dimensions\n→ BUY VOL', 
        ha='center', va='center', 
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#99ff99', alpha=0.8, 
                 edgecolor='darkgreen', linewidth=2),
        zorder=4)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Overall Percentile Score (0=Cheap, 100=Expensive)', 
                   fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    ax.set_xlabel('Cross-Currency Percentile', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cross-Tenor Percentile', fontsize=14, fontweight='bold')
    ax.set_title(
        f'FX Volatility Relative Value Map\n'
        f'Top {n_top} Most Expensive & Bottom {n_bottom} Cheapest Labeled', 
        fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, zorder=0)
    ax.tick_params(labelsize=11)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    fig.text(
        0.99, 0.01, 
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        ha='right', fontsize=9, style='italic')
    plt.tight_layout()
    plt.show()
    return extreme_points






def create_detailed_RV_monitor(ccy_weighted_results, tenor_weighted_results, 
                                  ccys, tenors, ccy_weight=0.6, tenor_weight=0.4):
    data_dict = {}
    for ccy in ccys:
        row_data = []
        for tenor in tenors:
            ccy_pct = np.nan
            if tenor in ccy_weighted_results:
                tenor_df = ccy_weighted_results[tenor]
                ccy_row = tenor_df[tenor_df['Currency_Pair'] == ccy]
                if not ccy_row.empty:
                    ccy_pct = ccy_row['Avg_CorrWeight%'].values[0]
            tenor_pct = np.nan
            if ccy in tenor_weighted_results:
                ccy_df = tenor_weighted_results[ccy]
                tenor_row = ccy_df[ccy_df['Tenor'] == tenor]
                if not tenor_row.empty:
                    tenor_pct = tenor_row['Avg_CorrWeight%'].values[0]
            if pd.notna(ccy_pct) and pd.notna(tenor_pct):
                combined = ccy_weight * ccy_pct + tenor_weight * tenor_pct
            else:
                combined = np.nan
            row_data.extend([ccy_pct, tenor_pct, combined])
        
        data_dict[ccy] = row_data
    col_tuples = []
    for tenor in tenors:
        col_tuples.extend([
            (tenor, 'vs_CCY_%'),
            (tenor, 'vs_Tenor_%'),
            (tenor, f'Combined_{int(ccy_weight*100)}-{int(tenor_weight*100)}_%')])
    
    columns = pd.MultiIndex.from_tuples(col_tuples)
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=columns)
    df.index.name = 'Currency Pairs'
    return df


def plot_detailed_gamma_monitor_with_gaps(
    df,
    title="Detailed Volatility Relative Value Monitor"):

    

    divider_lw=3.0            # thickness of the line between xTenor and Composite
    non_composite_fade=0.30   # 0..1: how much to fade xCCY/xTenor toward white
    y_order=None              # e.g. ["EURUSD","GBPUSD",...,"USDCNH"]

    # --- enforce desired top-to-bottom order
    if y_order is not None:
        # Be forgiving on case
        df = df.reindex(index=[r for r in y_order if r in df.index])

    fig, ax = plt.subplots(figsize=(26, 12))

    # colormap (0 = cheap / 100 = expensive)
    colors = ['#00ff00', '#90EE90', '#FFFF99', '#FFB366', '#ff0000']
    cmap = mcolors.LinearSegmentedColormap.from_list('cheap_expensive', colors, N=100)
    norm = mcolors.Normalize(vmin=0, vmax=100)

    tenors = df.columns.get_level_values(0).unique()

    gap_width = 0.5
    current_x = 0.0

    # ticks for top (per column) and bottom (per tenor group)
    top_tick_positions, top_tick_labels = [], []
    bottom_tick_positions, bottom_tick_labels = [], []

    def simplify_metric(metric: str) -> str:
        if 'vs_CCY' in metric:
            return 'xCCY'
        if 'vs_Tenor' in metric:
            return 'xTenor'
        if 'Combined' in metric:
            return 'Composite'
        return metric

    def fade_toward_white(rgba, fade):
        # linearly blend RGB toward white; keep alpha unchanged
        r, g, b, a = rgba
        r = 1 - (1 - r) * (1 - fade)
        g = 1 - (1 - g) * (1 - fade)
        b = 1 - (1 - b) * (1 - fade)
        return (r, g, b, a)

    for tenor_idx, tenor in enumerate(tenors):
        tenor_cols = [col for col in df.columns if col[0] == tenor]
        tenor_data = df[tenor_cols]
        group_start_x = current_x

        # Track where Composite starts to draw the divider
        composite_left_edges = []

        for col_idx, col in enumerate(tenor_cols):
            x_pos = current_x + col_idx
            metric = col[1]
            label = simplify_metric(metric)

            # Draw each cell
            for row_idx, ccy in enumerate(df.index):
                value = tenor_data.loc[ccy, col]
                if pd.notna(value):
                    base_rgba = cmap(norm(value))
                    rgba = base_rgba if label == 'Composite' else fade_toward_white(base_rgba, non_composite_fade)

                    rect = Rectangle((x_pos, row_idx), 1, 1, facecolor=rgba,
                                     edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)

                    ax.text(x_pos + 0.5, row_idx + 0.5, f'{value:.1f}',
                            ha='center', va='center', fontsize=9,
                            fontweight=('bold' if label == 'Composite' else 'normal'))

            # Remember top ticks
            top_tick_positions.append(x_pos + 0.5)
            top_tick_labels.append(label)

            # If this column is Composite, mark its left edge for a thick divider
            if label == 'Composite':
                composite_left_edges.append(x_pos)

        # Bottom tick (one per tenor group, centered)
        group_width = len(tenor_cols)
        bottom_tick_positions.append(group_start_x + group_width / 2.0)
        bottom_tick_labels.append(str(tenor))

        # Draw the Composite dividers for this group
        for x_left in composite_left_edges:
            ax.axvline(x=x_left, color='black', linewidth=divider_lw)

        # Group separator (dashed) between tenors
        current_x += group_width + gap_width
        if tenor_idx < len(tenors) - 1:
            ax.axvline(x=current_x - gap_width, color='gray', linewidth=2,
                       linestyle='--', alpha=0.5)

    # Limits
    ax.set_xlim(0, current_x - gap_width)
    ax.set_ylim(0, len(df))

    # Bottom axis: tenor labels
    ax.set_xticks(bottom_tick_positions)
    ax.set_xticklabels(bottom_tick_labels, fontsize=11)
    ax.tick_params(axis='x', pad=8)

    # Top axis: per-column labels
    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(top_tick_positions)
    ax_top.set_xticklabels(top_tick_labels, fontsize=9)
    ax_top.tick_params(axis='x', length=0, pad=8)

    # Y axis: currency pairs (centered)
    ax.set_yticks([i + 0.5 for i in range(len(df))])
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_ylabel('Currency Pairs', fontsize=12, fontweight='bold')

    # Title + colorbar
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Percentile Ranking (0=Cheap, 100=Expensive)', fontsize=11)

    # Timestamp
    fig.text(0.99, 0.98, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='right', va='top', fontsize=8, style='italic')

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    return fig



def create_interactive_monitor(ccy_weighted_results, tenor_weighted_results, 
                               ccys, tenors, ccy_weight=0.6):
    tenor_weight = 1 - ccy_weight
    df = create_detailed_RV_monitor(ccy_weighted_results, tenor_weighted_results,
                                       ccys, tenors, ccy_weight, tenor_weight)
    
    df = df.reindex(index=ccys)
    metric_order = {'vs_CCY': 0, 'vs_Tenor': 1, 'Combined': 2}
    df = df.reindex(
        columns=sorted(df.columns,
                       key=lambda c: (tenors.index(c[0]), metric_order.get(c[1], 99))))
    
    fig = plot_detailed_gamma_monitor_with_gaps(df, 
                                        title=f"Detailed Volatility Monitor (CCY:{ccy_weight*100:.0f}% | Tenor:{tenor_weight*100:.0f}%)")
    
    return df, fig





ccys = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 
        'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
tenors = ['1W', '2W', '1M', '3M', '6M', '1Y']
lookback = 360

# Run the analysis
cross_ccy_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(ccys, tenors, lookback)
cross_tenor_results = ATM_VolRefferenceRanking_AllCCY_CrossTenor_FullTimePannel(ccys, tenors, lookback)

df_monitor, fig = create_interactive_monitor(
    cross_ccy_results['ave_results_by_tenor_correlationWeighted'],
    cross_tenor_results['ave_results_by_ccy_correlationWeighted'],
    ccys, tenors, 
    ccy_weight=0.6
    )








# -------------------------------------------------------------------------------------------
# -------------------------------------- Execution ------------------------------------------


# ccys = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
# tenors = ['1W', '2W', '1M', '3M', '6M', '1Y']
# lookback = 365




# cross_ccy_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(ccys, tenors, lookback)
# cross_tenor_results = ATM_VolRefferenceRanking_AllCCY_CrossTenor_FullTimePannel(ccys, tenors, lookback)


# combined_matrix = create_combined_score_matrix(
#     cross_ccy_results['ave_results_by_tenor_correlationWeighted'],
#     cross_tenor_results['ave_results_by_ccy_correlationWeighted'],
#     ccys, tenors)

# print(combined_matrix)





# scatter_data = create_2d_relative_map(
#     cross_ccy_results['ave_results_by_tenor_correlationWeighted'],
#     cross_tenor_results['ave_results_by_ccy_correlationWeighted'],
#     ccys, tenors)





# create_combined_visualization(combined_matrix, scatter_data)

# create_2d_scatter(scatter_data)








