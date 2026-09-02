from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Optional
from itertools import permutations
from scipy.stats import percentileofscore

from xbbg import blp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



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
                ZScore_lookback = abs(current_spread - mean_lookback) / STD_lookback 
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




def create_combined_summary(currency_list, tenors, days_back):
    summary = summarize_iv_vdiff_percentiles(currency_list, tenors, days_back)
    analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list, tenors, days_back)
    for tenor in tenors:
        if tenor in analysis_results['ave_results_by_tenor_correlationWeighted']:
            rel_df = analysis_results['ave_results_by_tenor_correlationWeighted'][tenor]
            for _, row in rel_df.iterrows():
                ccy = row['Currency_Pair']
                rel_pct = row['Avg_CorrWeight%']
                if ccy in summary.index:
                    summary.loc[ccy, (tenor, "mrkt_rel", "%")] = rel_pct
    return summary, analysis_results






# Plotting Display
def create_fx_volatility_heatmap(currency_list, tenors, days_back):
    combined_summary, analysis_results = create_combined_summary(currency_list, tenors, days_back)
    df = combined_summary.copy()
    fig, axes = plt.subplots(1, len(tenors), figsize=(20, 10), 
                            gridspec_kw={'wspace': 0.05})  
    if len(tenors) == 1:
        axes = [axes]
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    cmap = plt.cm.RdYlGn_r  
    for tenor_idx, tenor in enumerate(tenors):
        ax = axes[tenor_idx]
        tenor_columns = [col for col in df.columns if col[0] == tenor]
        if not tenor_columns:
            continue
        tenor_df = df[tenor_columns]
        n_rows, n_cols = tenor_df.shape
        mrkt_rel_columns = [(tenor, "mrkt_rel", "%")]
        iv_columns = [(tenor, "IV", "Current"), (tenor, "IV", "%")]
        ir_columns = [(tenor, "I-R", "Current"), (tenor, "I-R", "%")]
        mrkt_rel_mask = np.zeros((n_rows, n_cols), dtype=bool)
        iv_mask = np.zeros((n_rows, n_cols), dtype=bool)
        ir_mask = np.zeros((n_rows, n_cols), dtype=bool)
        for col_tuple in mrkt_rel_columns:
            if col_tuple in tenor_df.columns:
                col_idx = tenor_df.columns.get_loc(col_tuple)
                mrkt_rel_mask[:, col_idx] = True
        for col_tuple in iv_columns:
            if col_tuple in tenor_df.columns:
                col_idx = tenor_df.columns.get_loc(col_tuple)
                iv_mask[:, col_idx] = True
        for col_tuple in ir_columns:
            if col_tuple in tenor_df.columns:
                col_idx = tenor_df.columns.get_loc(col_tuple)
                ir_mask[:, col_idx] = True
        for i, row_idx in enumerate(tenor_df.index):
            for j, col_tuple in enumerate(tenor_df.columns):
                value = tenor_df.iloc[i, j]
                if mrkt_rel_mask[i, j] and not pd.isna(value):
                    color = cmap(norm(value))
                elif iv_mask[i, j]:
                    color = "#E8E8E8"
                elif ir_mask[i, j]:
                    color = "#D3D3D3" 
                else:
                    color = "white"
                is_after_iv = False
                is_after_ir = False
                if j == 1: 
                    is_after_iv = True
                elif j == 3:  
                    is_after_ir = True
                if is_after_iv or is_after_ir:
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                    thick_line = plt.Rectangle((j + 1 - 0.05, i), 0.1, 1, facecolor="black", edgecolor="none")
                    ax.add_patch(thick_line)
                else:
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                if pd.isna(value):
                    text = "N/A"
                    fontweight = "normal"
                    fontsize = 8
                elif col_tuple[2] == "%": 
                    text = f"{value:.1f}"
                    fontweight = "bold" if value > 80 or value < 20 else "normal"
                    fontsize = 9
                else:
                    text = f"{value:.3f}"
                    fontweight = "normal"
                    fontsize = 8
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", 
                       fontsize=fontsize, fontweight=fontweight)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        col_labels = []
        label_weights = []
        for col_tuple in tenor_df.columns:
            if col_tuple[2] == "Current":
                col_labels.append(col_tuple[1])
                label_weights.append("normal")
            else:
                col_labels.append(f"{col_tuple[1]} %")
                label_weights.append("bold")  
        ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha='center')
        for i, (label, weight) in enumerate(zip(col_labels, label_weights)):
            if weight == "bold":
                ax.get_xticklabels()[i].set_fontweight('bold')
        if tenor_idx == 0:
            ax.set_yticklabels(tenor_df.index, fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.xaxis.tick_top()
        ax.text(n_cols/2, n_rows + 0.5, tenor, ha="center", va="center", 
               fontsize=14, weight="bold", transform=ax.transData)
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Percentile Ranking", fontsize=12)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    fig.suptitle(
        "Detailed Gamma Monitor", 
        fontsize=16, 
        fontweight="bold", y=0.98)
    axes[0].set_ylabel("Currency Pairs", fontsize=12, fontweight="bold")

    legend_text = (
        "• IV (Implied Volatility): Current volatility levels and 1Y historical percentile ranking\n"
        "• I-R (Implied - Realized): Implied and 30min-Sample Realized volatility with 1Y percentile ranking\n"
        "• mrkt_rel (Market Relationship): Cross-Currency, Correlation-weighted ATM Vol Spread Percentile Measure"
    )
    fig.text(0.5, 0.02, legend_text, fontsize=10, ha='center', va='bottom', 
            transform=fig.transFigure, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.show()
    return fig, axes


# currency_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
# tenors = ['1W', '2W', '3W', '1M']
# days_back = 365

# create_fx_volatility_heatmap(currency_list, tenors, days_back)





# --------------------------------------------------------------------------------------------------
# ------------------------------------Ranking Outputs-----------------------------------------------
# --------------------------------------------------------------------------------------------------

def rank_volatility_opportunities(combined_summary: pd.DataFrame, 
                                 tenors: List[str],
                                 method: str = 'simple_rank',
                                 iv_weight: float = 1.0,
                                 ir_weight: float = 1.0, 
                                 mrkt_weight: float = 1.0) -> Dict[str, pd.DataFrame]:
    top_n = 10
    results = {}
    all_rankings = []
    for tenor in tenors:
        iv_col = (tenor, "IV", "%")
        ir_col = (tenor, "I-R", "%") 
        mrkt_col = (tenor, "mrkt_rel", "%")
        if not all(col in combined_summary.columns for col in [iv_col, ir_col, mrkt_col]):
            continue
        tenor_data = combined_summary[[iv_col, ir_col, mrkt_col]].copy()
        tenor_data = tenor_data.dropna()
        if method == 'simple_rank':
            tenor_data['IV_rank'] = tenor_data[iv_col].rank(ascending=True)
            tenor_data['IR_rank'] = tenor_data[ir_col].rank(ascending=True) 
            tenor_data['Mrkt_rank'] = tenor_data[mrkt_col].rank(ascending=True)
            tenor_data['Buy_Score'] = (tenor_data['IV_rank'] * iv_weight + 
                                     tenor_data['IR_rank'] * ir_weight + 
                                     tenor_data['Mrkt_rank'] * mrkt_weight)
            tenor_data['Buy_Rank'] = tenor_data['Buy_Score'].rank(ascending=True)
            tenor_data['Sell_Score'] = (tenor_data['IV_rank'] * iv_weight + 
                                      tenor_data['IR_rank'] * ir_weight + 
                                      tenor_data['Mrkt_rank'] * mrkt_weight)
            tenor_data['Sell_Rank'] = tenor_data['Sell_Score'].rank(ascending=False)
            
        elif method == 'weighted_score':
            tenor_data['Buy_Score'] = (tenor_data[iv_col] * iv_weight + 
                                     tenor_data[ir_col] * ir_weight + 
                                     tenor_data[mrkt_col] * mrkt_weight) / (iv_weight + ir_weight + mrkt_weight)
            tenor_data['Buy_Rank'] = tenor_data['Buy_Score'].rank(ascending=True)
            tenor_data['Sell_Rank'] = tenor_data['Buy_Score'].rank(ascending=False)
        elif method == 'z_score':
            for col in [iv_col, ir_col, mrkt_col]:
                z_col = f"{col}_z"
                tenor_data[z_col] = (tenor_data[col] - tenor_data[col].mean()) / tenor_data[col].std()
            tenor_data['Buy_Score'] = (tenor_data[f"{iv_col}_z"] * iv_weight + 
                                     tenor_data[f"{ir_col}_z"] * ir_weight + 
                                     tenor_data[f"{mrkt_col}_z"] * mrkt_weight)
            tenor_data['Buy_Rank'] = tenor_data['Buy_Score'].rank(ascending=True)
            tenor_data['Sell_Rank'] = tenor_data['Buy_Score'].rank(ascending=False)
        elif method == 'composite':
            avg_score = (tenor_data[iv_col] + tenor_data[ir_col] + tenor_data[mrkt_col]) / 3  # Simple average
            geom_score = (tenor_data[iv_col] * tenor_data[ir_col] * tenor_data[mrkt_col]) ** (1/3) # Geometric mean
            min_score = tenor_data[[iv_col, ir_col, mrkt_col]].min(axis=1) # Minimum (most conservative - all must be low)
            tenor_data['Buy_Score'] = (avg_score + geom_score + min_score) / 3
            tenor_data['Buy_Rank'] = tenor_data['Buy_Score'].rank(ascending=True)
            tenor_data['Sell_Rank'] = tenor_data['Buy_Score'].rank(ascending=False)
        tenor_data['Tenor'] = tenor
        tenor_data['CCY'] = tenor_data.index
        tenor_data['IV_Pct'] = tenor_data[iv_col]
        tenor_data['IR_Pct'] = tenor_data[ir_col] 
        tenor_data['Mrkt_Pct'] = tenor_data[mrkt_col]
        all_rankings.append(tenor_data)
    if not all_rankings:
        return {'buy_opportunities': pd.DataFrame(), 'sell_opportunities': pd.DataFrame(), 'rankings': pd.DataFrame()}
    full_rankings = pd.concat(all_rankings, ignore_index=True)
    buy_opps = full_rankings.nsmallest(top_n * len(tenors), 'Buy_Rank')[
        ['CCY', 'Tenor', 'IV_Pct', 'IR_Pct', 'Mrkt_Pct', 'Buy_Score', 'Buy_Rank']
    ].head(top_n)
    sell_opps = full_rankings.nsmallest(top_n * len(tenors), 'Sell_Rank')[
        ['CCY', 'Tenor', 'IV_Pct', 'IR_Pct', 'Mrkt_Pct', 'Buy_Score', 'Sell_Rank'] 
    ].head(top_n)
    return {'buy_opportunities': buy_opps, 'sell_opportunities': sell_opps, 'rankings': full_rankings}



# currency_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
# tenors = ['1W', '2W', '3W', '1M']
# days_back = 365

# combined_summary, analysis_results = create_combined_summary(currency_list, tenors, days_back)

# weighted_results = rank_volatility_opportunities(
#     combined_summary, 
#     tenors, 
#     method='weighted_score',
#     iv_weight=1.0,      # Implied Vol Percentile
#     ir_weight=1.0,      # Relative Vol Premia Percentile
#     mrkt_weight=1.0,    # Relative Ranking to Rest of Market (Corr Weighted)
# )

# print("=== TOP 5 BUY OPPORTUNITIES (Simple Ranking) ===")
# print(weighted_results['buy_opportunities'])
# print()
# print(weighted_results['sell_opportunities'])





def IndivdualCCYSpread_Percentiles_TenorGrouped(ccy_interest, tenor_interest):
    currency_list_broad = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
    analysis_results = ATM_VolRefferenceRanking_AllTenorCCY_FullTimePannel(currency_list_broad, tenor_interest, lookback_days=252)
    tenor = tenor_interest[0]
    data_single = analysis_results["results_by_tenor"][tenor].copy()
    df = data_single[data_single["CCY1"] == ccy_interest]
    df_clean = df[['Pair', 'Tenor',  'Current_Spread', 'Percentile_current', 'Correlation', 'Spread_Mean', 'Spread_STD']]
    df_clean.columns = ['Pair','Tenor','spread', 'spread_%', 'Corr', 'spread_mean','spread_std']
    return df_clean


# ccy_interest = 'AUDUSD'
# tenor_interest = ['2W']

# print(IndivdualCCYSpread_Percentiles_TenorGrouped(ccy_interest, tenor_interest))
