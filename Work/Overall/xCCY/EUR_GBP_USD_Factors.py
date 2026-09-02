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
warnings.filterwarnings('ignore')



def get_Vols(ccys, tenors, days):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df_vols = {}
    for ccy in ccys:
        for tenor in tenors:
            ticker_IV = f"{ccy}V{tenor} BGN Curncy"
            data_IV = blp.bdh(
                tickers=[ticker_IV],
                flds="PX_LAST",
                start_date=start_date,
                end_date=end_date)
            if not data_IV.empty:
                col_name = f"{ccy}_{tenor}"
                data_IV.columns = [col_name]
                df_vols[col_name] = data_IV
            else:
                print(f" No data for {ticker_IV}, skipping.")
    if df_vols:
        df_vols_all = pd.concat(df_vols.values(), axis=1)
        return df_vols_all
    else:
        print("No data retrieved for any ticker.")

def get_Vol_Spreads(ccy1, ccy2, tenors, days):
    df_vols = get_Vols([ccy1, ccy2], tenors, days)
    df_spreads = pd.DataFrame(index=df_vols.index)
    for tenor in tenors:
        col1 = f"{ccy1}_{tenor}"
        col2 = f"{ccy2}_{tenor}"
        if col1 in df_vols.columns and col2 in df_vols.columns:
            spread_col = f"{ccy1}_{ccy2}_{tenor}"
            df_spreads[spread_col] = df_vols[col1] - df_vols[col2]
        else:
            print(f" Missing {tenor} data")
    return df_spreads



def USD_GlobalFactors(days, join_method='inner'):
    tickers = ['MOVE', 'USGG2YR', 'USGG10YR', 'SPX', 'VIX']
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    out = pd.DataFrame()
    for t in tickers:
        df = blp.bdh(
            tickers=f"{t} INDEX",
            flds=["PX_LAST"],
            start_date=start_date,
            end_date=end_date)
        df = df.droplevel(0, axis=1)
        df = df.rename(columns={"PX_LAST": t})
        df[f"{t}_DayChg"] = df[t].diff()
        df[f"{t}_PctChg"] = df[t].pct_change()
        df[f"{t}_AbsDayChg"] = df[f"{t}_DayChg"].abs()
        out = df if out.empty else out.join(df, how=join_method)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out["2s10s"] = out["USGG10YR"] - out["USGG2YR"]
    out["2s10s_DayChg"] = out["2s10s"].diff()
    out["2s10s_PctChg"] = out["2s10s"].pct_change()
    out["2s10s_AbsDayChg"] = out["2s10s_DayChg"].abs()
    roll_windows = (5, 21, 63)
    window_labels = {5: '1w', 21: '1m', 63: '3m'}
    for w in roll_windows:
        label = window_labels[w]
        out[f"USGG2YR_RVol_{label}"] = out["USGG2YR_DayChg"].rolling(w).std() * np.sqrt(252)
        out[f"USGG10YR_RVol_{label}"] = out["USGG10YR_DayChg"].rolling(w).std() * np.sqrt(252)
        out[f"2s10s_RVol_{label}"] = out["2s10s_DayChg"].rolling(w).std() * np.sqrt(252)
        out[f"SPX_RVol_{label}"] = out["SPX_PctChg"].rolling(w).std() * np.sqrt(252)
    return out


def EUR_Factors(days, join_method='inner'):
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    out = pd.DataFrame()
    tickers_ECB_OIS = ["EESWE1", "EESWE2", "SX5E", "V2X"]
    for t in tickers_ECB_OIS:
        df = blp.bdh(
            tickers=f"{t} Curncy",
            flds=["PX_LAST"],
            start_date=start_date,
            end_date=end_date)
        df = df.droplevel(0, axis=1)
        df = df.rename(columns={"PX_LAST": t})
        df[f"{t}_DayChg"] = df[t].diff()
        df[f"{t}_PctChg"] = df[t].pct_change()
        df[f"{t}_AbsDayChg"] = df[f"{t}_DayChg"].abs()
        out = df if out.empty else out.join(df, how=join_method)
    sov_tickers = ["GBTPGR10", "GDBR10"]
    for t in sov_tickers:
        df = blp.bdh(
            tickers=f"{t} Index",
            flds=["PX_LAST"],
            start_date=start_date,
            end_date=end_date)
        df = df.droplevel(0, axis=1)
        df = df.rename(columns={"PX_LAST": t})
        df[f"{t}_DayChg"] = df[t].diff()
        out = out.join(df, how=join_method)
    hvol_tickers = {"EURUSDH1W": "1w", "EURUSDH1M": "1m", "EURUSDH3M": "3m"}
    for ticker, label in hvol_tickers.items():
        df = blp.bdh(
            tickers=f"{ticker} BGN Curncy",
            flds=["PX_LAST"],
            start_date=start_date,
            end_date=end_date)
        df = df.droplevel(0, axis=1)
        col_name = f"EURUSD_HVol_{label}"
        df = df.rename(columns={"PX_LAST": col_name})
        df[f"{col_name}_DayChg"] = df[col_name].diff()
        out = out.join(df, how=join_method)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    ois_1y, ois_2y = "EESWE1", "EESWE2"
    out["EUR_OIS_1Y2Y_Slope"] = out[ois_1y] - out[ois_2y]
    out["EUR_OIS_1Y2Y_Slope_DayChg"] = out["EUR_OIS_1Y2Y_Slope"].diff()
    out["EUR_OIS_1Y2Y_Slope_PctChg"] = out["EUR_OIS_1Y2Y_Slope"].pct_change()
    out["EUR_OIS_1Y2Y_Slope_AbsDayChg"] = out["EUR_OIS_1Y2Y_Slope_DayChg"].abs()
    out["BTP_Bund_10Y_Spread"] = out["GBTPGR10"] - out["GDBR10"]
    out["BTP_Bund_10Y_Spread_DayChg"] = out["BTP_Bund_10Y_Spread"].diff()
    out["BTP_Bund_10Y_Spread_AbsDayChg"] = out["BTP_Bund_10Y_Spread_DayChg"].abs()
    roll_windows = {5: '1w', 21: '1m', 63: '3m'}
    for w, label in roll_windows.items():
        out[f"{ois_2y}_DayChg_RollStd_{label}"] = out[f"{ois_2y}_DayChg"].rolling(w).std()
        out[f"EUR_OIS_1Y2Y_Slope_DayChg_RollStd_{label}"] = out["EUR_OIS_1Y2Y_Slope_DayChg"].rolling(w).std()
        out[f"BTP_Bund_10Y_Spread_DayChg_RollStd_{label}"] = out["BTP_Bund_10Y_Spread_DayChg"].rolling(w).std()
        out[f"SX5E_RVol_{label}"] = out["SX5E_PctChg"].rolling(w).std() * np.sqrt(252)
    return out







def GBP_Factors(days, join_method='inner'):
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    out = blp.bdh(
        tickers="GBPUSD Curncy",
        flds=["PX_LAST"],
        start_date=start_date,
        end_date=end_date)
    out = out.droplevel(0, axis=1)
    out = out.rename(columns={"PX_LAST": "GBPUSD"})
    out["GBPUSD_DayChg"] = out["GBPUSD"].diff()
    out["GBPUSD_PctChg"] = out["GBPUSD"].pct_change()
    out["GBPUSD_LogRet"] = np.log(out["GBPUSD"]).diff()
    out["GBPUSD_LogRet_Abs"] = out["GBPUSD_LogRet"].abs()
    hvol_tickers = {"GBPUSDH1W": "1w", "GBPUSDH1M": "1m", "GBPUSDH3M": "3m"}
    for ticker, label in hvol_tickers.items():
        df = blp.bdh(
            tickers=f"{ticker} BGN Curncy",
            flds=["PX_LAST"],
            start_date=start_date,
            end_date=end_date)
        df = df.droplevel(0, axis=1)
        col_name = f"GBPUSD_HVol_{label}"
        df = df.rename(columns={"PX_LAST": col_name})
        df[f"{col_name}_DayChg"] = df[col_name].diff()
        out = out.join(df, how=join_method)
    rate_tickers = ["GUKG2", "GUKG10"]
    df_rates = blp.bdh(
        tickers=[f"{t} Index" for t in rate_tickers],
        flds=["PX_LAST"],
        start_date=start_date,
        end_date=end_date)
    df_rates = df_rates.droplevel(1, axis=1)
    df_rates.columns = [c.split()[0] for c in df_rates.columns]
    out = out.join(df_rates, how=join_method)
    for t in rate_tickers:
        out[f"{t}_DayChg"] = out[t].diff()
        out[f"{t}_AbsDayChg"] = out[f"{t}_DayChg"].abs()
        out[f"{t}_PctChg"] = out[t].pct_change()
    out["UK_2s10s"] = out["GUKG10"] - out["GUKG2"]
    out["UK_2s10s_DayChg"] = out["UK_2s10s"].diff()
    out["UK_2s10s_AbsDayChg"] = out["UK_2s10s_DayChg"].abs()
    eq_tickers = ["UKX", "IVIUK"]
    df_eq = blp.bdh(
        tickers=[f"{t} Index" for t in eq_tickers],
        flds=["PX_LAST"],
        start_date=start_date,
        end_date=end_date)
    df_eq = df_eq.droplevel(1, axis=1)
    df_eq.columns = [c.split()[0] for c in df_eq.columns]
    out = out.join(df_eq, how=join_method)
    out["UKX_DayChg"] = out["UKX"].diff()
    out["UKX_PctChg"] = out["UKX"].pct_change()
    out["UKX_PctChg_Abs"] = out["UKX_PctChg"].abs()
    out["IVIUK_DayChg"] = out["IVIUK"].diff()
    out["IVIUK_PctChg"] = out["IVIUK"].pct_change()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    roll_windows = {5: '1w', 21: '1m', 63: '3m'}
    for w, label in roll_windows.items():
        out[f"GUKG2_DayChg_RollStd_{label}"] = out["GUKG2_DayChg"].rolling(w).std()
        out[f"GUKG10_DayChg_RollStd_{label}"] = out["GUKG10_DayChg"].rolling(w).std()
        out[f"UK_2s10s_DayChg_RollStd_{label}"] = out["UK_2s10s_DayChg"].rolling(w).std()
        out[f"UKX_RVol_{label}"] = out["UKX_PctChg"].rolling(w).std() * np.sqrt(252)
    
    return out







# ----------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------


def build_master_dataset(ccy_pairs, tenors, days, 
                         include_spreads=False,
                         include_usd_factors=True, 
                         include_eur_factors=True,
                         include_gbp_factors=False,
                         exclude_features=None,
                         join_method='inner',
                         verbose=False):
    if verbose:
        print("Building master dataset...")
        print(f"  Join method: {join_method}")
    dfs = []
    
    # FX Vols
    if ccy_pairs and tenors:
        if verbose:
            print(f"  Loading FX vols: {ccy_pairs} @ {tenors}")
        df_vols = get_Vols(ccy_pairs, tenors, days)
        if df_vols is not None and not df_vols.empty:
            dfs.append(df_vols)
            if verbose:
                print(f"    ✓ Loaded {df_vols.shape[1]} vol columns, {len(df_vols)} dates")
        else:
            if verbose:
                print(f"    ⚠️  No vol data retrieved")
    
    # Vol Spreads
    if include_spreads and len(ccy_pairs) >= 2:
        if verbose:
            print(f"  Loading vol spreads...")
        for i in range(len(ccy_pairs)):
            for j in range(i+1, len(ccy_pairs)):
                df_spread = get_Vol_Spreads(ccy_pairs[i], ccy_pairs[j], tenors, days)
                if not df_spread.empty:
                    dfs.append(df_spread)
                    if verbose:
                        print(f"    ✓ Loaded {ccy_pairs[i]}-{ccy_pairs[j]} spreads")
    
    # USD Factors
    if include_usd_factors:
        if verbose:
            print(f"  Loading USD factors...")
        df_usd = USD_GlobalFactors(days, join_method=join_method)
        if not df_usd.empty:
            dfs.append(df_usd)
            if verbose:
                print(f"    ✓ Loaded {df_usd.shape[1]} USD columns, {len(df_usd)} dates")
        else:
            if verbose:
                print(f"    ⚠️  No USD data retrieved")
    
    # EUR Factors
    if include_eur_factors:
        if verbose:
            print(f"  Loading EUR factors...")
        df_eur = EUR_Factors(days, join_method=join_method)
        if not df_eur.empty:
            dfs.append(df_eur)
            if verbose:
                print(f"    ✓ Loaded {df_eur.shape[1]} EUR columns, {len(df_eur)} dates")
        else:
            if verbose:
                print(f"    ⚠️  No EUR data retrieved")
    
    # GBP Factors
    if include_gbp_factors:
        if verbose:
            print(f"  Loading GBP factors...")
        df_gbp = GBP_Factors(days, join_method=join_method)
        if not df_gbp.empty:
            dfs.append(df_gbp)
            if verbose:
                print(f"    ✓ Loaded {df_gbp.shape[1]} GBP columns, {len(df_gbp)} dates")
        else:
            if verbose:
                print(f"    ⚠️  No GBP data retrieved")
    
    # Check if any data
    if not dfs:
        print("⚠️  No data retrieved for any component.")
        return pd.DataFrame()
    
    # Merge
    if verbose:
        print(f"\n  Merging {len(dfs)} dataframes...")
    df_master = pd.concat(dfs, axis=1, join='outer')
    df_master = df_master.sort_index()
    df_master = df_master[~df_master.index.duplicated(keep='last')]
    
    if verbose:
        print(f"    Before cleaning: {df_master.shape[0]} rows × {df_master.shape[1]} cols")
        missing_before = df_master.isnull().sum().sum()
        print(f"    Missing values: {missing_before} ({missing_before/df_master.size*100:.2f}%)")
    
    # Exclude features
    if exclude_features:
        cols_to_keep = [col for col in df_master.columns if col not in exclude_features]
        df_master = df_master[cols_to_keep]
        if verbose:
            print(f"    Excluded {len(exclude_features)} features")
    
    # Drop NaNs
    df_master = df_master.dropna()
    
    if verbose:
        print(f"    After dropna: {df_master.shape[0]} rows × {df_master.shape[1]} cols")
        print(f"    Date range: {df_master.index.min()} to {df_master.index.max()}")
        remaining_nans = df_master.isnull().sum().sum()
        if remaining_nans == 0:
            print(f"    ✓ No NaNs remaining")
        else:
            print(f"    ⚠️  WARNING: {remaining_nans} NaNs still present!")
    
    return df_master




def print_dataset_structure(df):
    print("\n" + "="*80)
    print(" "*25 + "DATASET STRUCTURE")
    print("="*80)
    
    # ========================================================================
    # TARGET VARIABLES (FX Volatility Levels)
    # ========================================================================
    print("\n┌─ FX IMPLIED VOLATILITY")
    print("│")
    
    # EURUSD Vols
    print("├─┬─ EURUSD")
    print("│ │")
    eurusd_tenors = [col for col in df.columns if col.startswith('EURUSD_') and 
                    '_GBPUSD_' not in col and 'HVol' not in col]
    for col in sorted(eurusd_tenors):
        print(f"│ │   • {col}")
    
    # GBPUSD Vols
    print("│ │")
    print("├─┬─ GBPUSD")
    print("│ │")
    gbpusd_tenors = [col for col in df.columns if col.startswith('GBPUSD_') and 
                    '_GBPUSD_' not in col and 'HVol' not in col and
                    col not in ['GBPUSD', 'GBPUSD_DayChg', 'GBPUSD_PctChg', 
                               'GBPUSD_LogRet', 'GBPUSD_LogRet_Abs']]
    for col in sorted(gbpusd_tenors):
        print(f"│ │   • {col}")
    
    # Vol Spreads
    print("│ │")
    print("└─┬─ VOL SPREADS (EURUSD - GBPUSD)")
    print("  │")
    spread_cols = [col for col in df.columns if 'EURUSD_GBPUSD' in col]
    for col in sorted(spread_cols):
        print(f"  │   • {col}")
    
    # ========================================================================
    # HISTORICAL VOLATILITY
    # ========================================================================
    print("\n" + "="*80)
    print("┌─ FX HISTORICAL VOLATILITY")
    print("│")
    
    # EURUSD HVol
    print("├─┬─ EURUSD")
    print("│ │")
    print("│ ├── Levels:")
    for col in ['EURUSD_HVol_1w', 'EURUSD_HVol_1m', 'EURUSD_HVol_3m']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ └── Daily Changes:")
    for col in ['EURUSD_HVol_1w_DayChg', 'EURUSD_HVol_1m_DayChg', 'EURUSD_HVol_3m_DayChg']:
        if col in df.columns:
            print(f"│     • {col}")
    
    # GBPUSD HVol
    print("│")
    print("└─┬─ GBPUSD")
    print("  │")
    print("  ├── Levels:")
    for col in ['GBPUSD_HVol_1w', 'GBPUSD_HVol_1m', 'GBPUSD_HVol_3m']:
        if col in df.columns:
            print(f"  │   • {col}")
    print("  │")
    print("  └── Daily Changes:")
    for col in ['GBPUSD_HVol_1w_DayChg', 'GBPUSD_HVol_1m_DayChg', 'GBPUSD_HVol_3m_DayChg']:
        if col in df.columns:
            print(f"      • {col}")
    
    # ========================================================================
    # USD / GLOBAL FACTORS
    # ========================================================================
    print("\n" + "="*80)
    print("┌─ USD / GLOBAL FACTORS")
    print("│")
    
    # --- US Treasury Rates ---
    print("├─┬─ US TREASURY RATES")
    print("│ │")
    print("│ ├── Levels:")
    for col in ['USGG2YR', 'USGG10YR', '2s10s']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Daily Changes:")
    for col in ['USGG2YR_DayChg', 'USGG10YR_DayChg', '2s10s_DayChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Percent Changes:")
    for col in ['USGG2YR_PctChg', 'USGG10YR_PctChg', '2s10s_PctChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Absolute Daily Changes:")
    for col in ['USGG2YR_AbsDayChg', 'USGG10YR_AbsDayChg', '2s10s_AbsDayChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ └── Realized Volatility:")
    for label in ['1w', '1m', '3m']:
        rvol_cols = [col for col in df.columns if f'_RVol_{label}' in col and 
                    any(x in col for x in ['USGG2YR', 'USGG10YR', '2s10s'])]
        if rvol_cols:
            print(f"│     └─ {label} window:")
            for col in sorted(rvol_cols):
                print(f"│        • {col}")
    
    # --- MOVE ---
    print("│")
    print("├─┬─ MOVE INDEX (Rates Volatility)")
    print("│ │")
    print("│ ├── Level:")
    if 'MOVE' in df.columns:
        print("│ │   • MOVE")
    print("│ │")
    print("│ ├── Daily Change:")
    if 'MOVE_DayChg' in df.columns:
        print("│ │   • MOVE_DayChg")
    print("│ │")
    print("│ ├── Percent Change:")
    if 'MOVE_PctChg' in df.columns:
        print("│ │   • MOVE_PctChg")
    print("│ │")
    print("│ └── Absolute Daily Change:")
    if 'MOVE_AbsDayChg' in df.columns:
        print("│     • MOVE_AbsDayChg")
    
    # --- SPX ---
    print("│")
    print("├─┬─ S&P 500 (US EQUITY)")
    print("│ │")
    print("│ ├── Level:")
    if 'SPX' in df.columns:
        print("│ │   • SPX")
    print("│ │")
    print("│ ├── Daily Change:")
    if 'SPX_DayChg' in df.columns:
        print("│ │   • SPX_DayChg")
    print("│ │")
    print("│ ├── Percent Change:")
    if 'SPX_PctChg' in df.columns:
        print("│ │   • SPX_PctChg")
    print("│ │")
    print("│ ├── Absolute Daily Change:")
    if 'SPX_AbsDayChg' in df.columns:
        print("│ │   • SPX_AbsDayChg")
    print("│ │")
    print("│ └── Realized Volatility:")
    for label in ['1w', '1m', '3m']:
        col = f"SPX_RVol_{label}"
        if col in df.columns:
            print(f"│     • {col}")
    
    # --- VIX ---
    print("│")
    print("└─┬─ VIX (US EQUITY VOLATILITY)")
    print("  │")
    print("  ├── Level:")
    if 'VIX' in df.columns:
        print("  │   • VIX")
    print("  │")
    print("  ├── Daily Change:")
    if 'VIX_DayChg' in df.columns:
        print("  │   • VIX_DayChg")
    print("  │")
    print("  ├── Percent Change:")
    if 'VIX_PctChg' in df.columns:
        print("  │   • VIX_PctChg")
    print("  │")
    print("  └── Absolute Daily Change:")
    if 'VIX_AbsDayChg' in df.columns:
        print("      • VIX_AbsDayChg")
    
    # ========================================================================
    # EUR FACTORS
    # ========================================================================
    print("\n" + "="*80)
    print("┌─ EUR FACTORS")
    print("│")
    
    # --- ECB OIS Rates ---
    print("├─┬─ ECB OIS RATES (Monetary Policy)")
    print("│ │")
    print("│ ├── Levels:")
    for col in ['EESWE1', 'EESWE2', 'EUR_OIS_1Y2Y_Slope']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Daily Changes:")
    for col in ['EESWE1_DayChg', 'EESWE2_DayChg', 'EUR_OIS_1Y2Y_Slope_DayChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Percent Changes:")
    for col in ['EESWE1_PctChg', 'EESWE2_PctChg', 'EUR_OIS_1Y2Y_Slope_PctChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Absolute Daily Changes:")
    for col in ['EESWE1_AbsDayChg', 'EESWE2_AbsDayChg', 'EUR_OIS_1Y2Y_Slope_AbsDayChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ └── Rolling Uncertainty (Std Dev of Changes):")
    for label in ['1w', '1m', '3m']:
        uncertainty_cols = [col for col in df.columns if f'RollStd_{label}' in col and 
                          any(x in col for x in ['EESWE', 'EUR_OIS'])]
        if uncertainty_cols:
            print(f"│     └─ {label} window:")
            for col in sorted(uncertainty_cols):
                print(f"│        • {col}")
    
    # --- Sovereign Risk ---
    print("│")
    print("├─┬─ SOVEREIGN RISK (BTP-Bund Spread)")
    print("│ │")
    print("│ ├── Levels:")
    for col in ['GBTPGR10', 'GDBR10', 'BTP_Bund_10Y_Spread']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Daily Changes:")
    for col in ['GBTPGR10_DayChg', 'GDBR10_DayChg', 'BTP_Bund_10Y_Spread_DayChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Absolute Daily Changes:")
    if 'BTP_Bund_10Y_Spread_AbsDayChg' in df.columns:
        print("│ │   • BTP_Bund_10Y_Spread_AbsDayChg")
    print("│ │")
    print("│ └── Rolling Uncertainty (Std Dev of Changes):")
    for label in ['1w', '1m', '3m']:
        col = f"BTP_Bund_10Y_Spread_DayChg_RollStd_{label}"
        if col in df.columns:
            print(f"│     • {col}")
    
    # --- Euro Stoxx 50 ---
    print("│")
    print("├─┬─ EURO STOXX 50 (EUR EQUITY)")
    print("│ │")
    print("│ ├── Level:")
    if 'SX5E' in df.columns:
        print("│ │   • SX5E")
    print("│ │")
    print("│ ├── Daily Change:")
    if 'SX5E_DayChg' in df.columns:
        print("│ │   • SX5E_DayChg")
    print("│ │")
    print("│ ├── Percent Change:")
    if 'SX5E_PctChg' in df.columns:
        print("│ │   • SX5E_PctChg")
    print("│ │")
    print("│ ├── Absolute Daily Change:")
    if 'SX5E_AbsDayChg' in df.columns:
        print("│ │   • SX5E_AbsDayChg")
    print("│ │")
    print("│ └── Realized Volatility:")
    for label in ['1w', '1m', '3m']:
        col = f"SX5E_RVol_{label}"
        if col in df.columns:
            print(f"│     • {col}")
    
    # --- VSTOXX ---
    print("│")
    print("└─┬─ VSTOXX (EUR EQUITY VOLATILITY)")
    print("  │")
    print("  ├── Level:")
    if 'V2X' in df.columns:
        print("  │   • V2X")
    print("  │")
    print("  ├── Daily Change:")
    if 'V2X_DayChg' in df.columns:
        print("  │   • V2X_DayChg")
    print("  │")
    print("  ├── Percent Change:")
    if 'V2X_PctChg' in df.columns:
        print("  │   • V2X_PctChg")
    print("  │")
    print("  └── Absolute Daily Change:")
    if 'V2X_AbsDayChg' in df.columns:
        print("      • V2X_AbsDayChg")
    
    # ========================================================================
    # GBP FACTORS
    # ========================================================================
    print("\n" + "="*80)
    print("┌─ GBP FACTORS")
    print("│")
    
    # --- GBPUSD Spot ---
    print("├─┬─ GBPUSD SPOT")
    print("│ │")
    print("│ ├── Level:")
    if 'GBPUSD' in df.columns:
        print("│ │   • GBPUSD")
    print("│ │")
    print("│ ├── Daily Change:")
    if 'GBPUSD_DayChg' in df.columns:
        print("│ │   • GBPUSD_DayChg")
    print("│ │")
    print("│ ├── Percent Change:")
    if 'GBPUSD_PctChg' in df.columns:
        print("│ │   • GBPUSD_PctChg")
    print("│ │")
    print("│ ├── Log Return:")
    if 'GBPUSD_LogRet' in df.columns:
        print("│ │   • GBPUSD_LogRet")
    print("│ │")
    print("│ └── Absolute Log Return:")
    if 'GBPUSD_LogRet_Abs' in df.columns:
        print("│     • GBPUSD_LogRet_Abs")
    
    # --- UK Rates ---
    print("│")
    print("├─┬─ UK GILT RATES")
    print("│ │")
    print("│ ├── Levels:")
    for col in ['GUKG2', 'GUKG10', 'UK_2s10s']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Daily Changes:")
    for col in ['GUKG2_DayChg', 'GUKG10_DayChg', 'UK_2s10s_DayChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Percent Changes:")
    for col in ['GUKG2_PctChg', 'GUKG10_PctChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ ├── Absolute Daily Changes:")
    for col in ['GUKG2_AbsDayChg', 'GUKG10_AbsDayChg', 'UK_2s10s_AbsDayChg']:
        if col in df.columns:
            print(f"│ │   • {col}")
    print("│ │")
    print("│ └── Rolling Uncertainty (Std Dev of Changes):")
    for label in ['1w', '1m', '3m']:
        uncertainty_cols = [col for col in df.columns if f'RollStd_{label}' in col and 
                          any(x in col for x in ['GUKG2', 'GUKG10', 'UK_2s10s'])]
        if uncertainty_cols:
            print(f"│     └─ {label} window:")
            for col in sorted(uncertainty_cols):
                print(f"│        • {col}")
    
    # --- FTSE 100 ---
    print("│")
    print("├─┬─ FTSE 100 (UK EQUITY)")
    print("│ │")
    print("│ ├── Level:")
    if 'UKX' in df.columns:
        print("│ │   • UKX")
    print("│ │")
    print("│ ├── Daily Change:")
    if 'UKX_DayChg' in df.columns:
        print("│ │   • UKX_DayChg")
    print("│ │")
    print("│ ├── Percent Change:")
    if 'UKX_PctChg' in df.columns:
        print("│ │   • UKX_PctChg")
    print("│ │")
    print("│ ├── Absolute Percent Change:")
    if 'UKX_PctChg_Abs' in df.columns:
        print("│ │   • UKX_PctChg_Abs")
    print("│ │")
    print("│ └── Realized Volatility:")
    for label in ['1w', '1m', '3m']:
        col = f"UKX_RVol_{label}"
        if col in df.columns:
            print(f"│     • {col}")
    
    # --- FTSE 100 Vol ---
    print("│")
    print("└─┬─ VFTSE (UK EQUITY VOLATILITY)")
    print("  │")
    print("  ├── Level:")
    if 'IVIUK' in df.columns:
        print("  │   • IVIUK")
    print("  │")
    print("  ├── Daily Change:")
    if 'IVIUK_DayChg' in df.columns:
        print("  │   • IVIUK_DayChg")
    print("  │")
    print("  └── Percent Change:")
    if 'IVIUK_PctChg' in df.columns:
        print("      • IVIUK_PctChg")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    eurusd_iv = [c for c in df.columns if c.startswith('EURUSD_') and 
                '_GBPUSD_' not in c and 'HVol' not in c]
    gbpusd_iv = [c for c in df.columns if c.startswith('GBPUSD_') and 
                '_GBPUSD_' not in c and 'HVol' not in c and
                c not in ['GBPUSD', 'GBPUSD_DayChg', 'GBPUSD_PctChg', 
                         'GBPUSD_LogRet', 'GBPUSD_LogRet_Abs']]
    spread_cols = [c for c in df.columns if 'EURUSD_GBPUSD' in c]
    hvol_cols = [c for c in df.columns if 'HVol' in c]
    usd_cols = [c for c in df.columns if any(x in c for x in 
                ['USGG', 'SPX', 'VIX', 'MOVE', '2s10s']) and 'UK' not in c]
    eur_cols = [c for c in df.columns if any(x in c for x in 
                ['EESWE', 'GBTPGR', 'GDBR', 'SX5E', 'V2X', 'BTP', 'EUR_OIS'])]
    gbp_cols = [c for c in df.columns if any(x in c for x in 
                ['GBPUSD', 'GUKG', 'UK_2s10s', 'UKX', 'IVIUK']) and 'EURUSD' not in c]
    
    print(f"Total Columns:       {len(df.columns)}")
    print(f"  • EURUSD IV:       {len(eurusd_iv)}")
    print(f"  • GBPUSD IV:       {len(gbpusd_iv)}")
    print(f"  • Vol Spreads:     {len(spread_cols)}")
    print(f"  • Historical Vol:  {len(hvol_cols)}")
    print(f"  • USD Factors:     {len(usd_cols)}")
    print(f"  • EUR Factors:     {len(eur_cols)}")
    print(f"  • GBP Factors:     {len(gbp_cols)}")
    print(f"\nDate Range:          {df.index.min()} to {df.index.max()}")
    print(f"Observations:        {len(df)}")
    print("="*80 + "\n")

def parse_date_spec(date_spec, reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()
    lookback_pattern = r'(?:(\d+)Y)?(?:(\d+)M)?$'
    match = re.match(lookback_pattern, date_spec.upper())
    if match and (match.group(1) or match.group(2)):
        years = int(match.group(1)) if match.group(1) else 0
        months = int(match.group(2)) if match.group(2) else 0
        return reference_date - relativedelta(years=years, months=months), False
    month_year_pattern = r'^([A-Za-z]{3})(\d{2})$'
    match = re.match(month_year_pattern, date_spec)
    if match:
        month_str = match.group(1)
        year_str = match.group(2)
        year = 2000 + int(year_str)
        date_str = f"1-{month_str}-{year}"
        return pd.to_datetime(date_str, format='%d-%b-%Y'), True  # Flag as month-only
    day_month_year_pattern = r'^(\d{1,2})([A-Za-z]{3})(\d{2})$'
    match = re.match(day_month_year_pattern, date_spec)
    if match:
        day = match.group(1)
        month = match.group(2)
        year = 2000 + int(match.group(3))
        date_str = f"{day}-{month}-{year}"
        return pd.to_datetime(date_str, format='%d-%b-%Y'), False
    raise ValueError(f"Could not parse date specification: {date_spec}")


def find_next_available_date(df, target_date, direction='forward'):
    if direction == 'exact':
        if target_date in df.index:
            return target_date
        else:
            return None
    future_dates = df.index[df.index >= target_date]
    if len(future_dates) > 0:
        return future_dates[0]
    else:
        return None


def parse_period(period_str, df, reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()
    if ':' in period_str:
        start_spec, end_spec = period_str.split(':')
        start_date_raw, start_is_month = parse_date_spec(start_spec, reference_date)
        end_date_raw, end_is_month = parse_date_spec(end_spec, reference_date)
        if end_is_month:
            end_date_raw = end_date_raw + relativedelta(months=1) - timedelta(days=1)
        start_date = find_next_available_date(df, start_date_raw, direction='forward')
        end_date = find_next_available_date(df, end_date_raw, direction='forward')
        if start_date is None or end_date is None:
            return None, None
        if start_date > end_date:
            return None, None
        return start_date, end_date
    else:
        start_date_raw, is_month = parse_date_spec(period_str, reference_date)
        if is_month:
            start_date = find_next_available_date(df, start_date_raw, direction='forward')
            end_date_raw = start_date_raw + relativedelta(months=1) - timedelta(days=1)
            end_date = find_next_available_date(df, end_date_raw, direction='forward')
        else:
            start_date = find_next_available_date(df, start_date_raw, direction='forward')
            end_date = min(df.index.max(), pd.Timestamp(reference_date))
        if start_date is None:
            return None, None
        if end_date is None:
            return None, None
        
        return start_date, end_date


def filter_by_period(df, period_str, verbose=False):
    start_date, end_date = parse_period(period_str, df)
    if start_date is None or end_date is None:
        if verbose:
            print(f"  ⚠️  Could not find valid date range for '{period_str}'")
        return pd.DataFrame()
    if verbose:
        print(f"  {period_str:20s} → {start_date.date()} to {end_date.date()}")
    return df[(df.index >= start_date) & (df.index <= end_date)]





# ---------- Systematic Data Period Formatting ---------- 

# Monthly Period Breakdown
def generate_monthly_periods(start_month, end_month):
    from dateutil.rrule import rrule, MONTHLY
    start, _ = parse_date_spec(start_month)
    end, _ = parse_date_spec(end_month)
    periods = []
    for dt in rrule(MONTHLY, dtstart=start, until=end):
        month_str = dt.strftime('%b%y')
        periods.append(month_str)
    return periods

# Rolling 3m Period Breakdown
def generate_rolling_3month_periods(start_month, end_month):
    from dateutil.rrule import rrule, MONTHLY
    start, _ = parse_date_spec(start_month)
    end, _ = parse_date_spec(end_month)
    periods = []
    current = start
    while current <= end - relativedelta(months=2):  # Need 3 months ahead
        end_of_window = current + relativedelta(months=3) - timedelta(days=1)
        start_str = current.strftime('%b%y')
        end_str = end_of_window.strftime('%b%y')
        period_str = f"{start_str}:{end_str}"
        periods.append(period_str)
        current = current + relativedelta(months=1)
    return periods

# Quarterly Period Breakdown 
def generate_quarterly_periods(start_quarter, end_quarter):
    from dateutil.rrule import rrule, MONTHLY
    start, _ = parse_date_spec(start_quarter)
    end, _ = parse_date_spec(end_quarter)
    quarters = {
        1: ('Jan', 'Mar'),
        2: ('Apr', 'Jun'),
        3: ('Jul', 'Sep'),
        4: ('Oct', 'Dec')}
    periods = []
    current = start
    quarter_num = (current.month - 1) // 3 + 1
    quarter_start_month = (quarter_num - 1) * 3 + 1
    current = current.replace(month=quarter_start_month, day=1)
    while current <= end:
        year_short = current.strftime('%y')
        quarter_num = (current.month - 1) // 3 + 1
        start_month, end_month = quarters[quarter_num]
        period_str = f"{start_month}{year_short}:{end_month}{year_short}"
        periods.append(period_str)
        current = current + relativedelta(months=3)
    return periods










# # ------------ Download Data to CSV File -------------

# df = build_master_dataset(
#     ccy_pairs=['EURUSD', 'GBPUSD'], 
#     tenors=['1W', '1M', '3M', '6M'],
#     days=365 * 15, 
#     include_spreads=True,
#     include_usd_factors=True,
#     include_eur_factors=True,
#     include_gbp_factors=True,
#     verbose=True)


# df.to_csv('eurusd_gbpusd_dataset.csv')