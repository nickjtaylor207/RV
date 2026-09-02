import blpapi

import pdblp
import numpy as np
from xbbg import blp

import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import pandas as pd

import seaborn as sns
import pytz




import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind

def Daily_GetTopCCYs(currency_pairs, years):
    df_ccy = {}
    start_date = (datetime.today() - timedelta(days= years * 356)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    for ticker in currency_pairs:
        data_ccy = blp.bdh(
            tickers=f"{ticker}" ,
            flds=["PX_LAST"],  # Adjust fields as needed
            start_date=start_date,
            end_date=end_date,
            Per="D"  # Daily data
        )
        data_ccy.columns = [ticker]
        df_ccy[ticker] = data_ccy
    df_ccyAll = pd.concat(df_ccy, axis=1)
    df_ccyAll.columns = df_ccyAll.columns.droplevel(0) if isinstance(df_ccyAll.columns, pd.MultiIndex) else df_ccyAll.columns
    df_ccyAll = df_ccyAll.sort_index(ascending=True)


    return df_ccyAll

def main_analysis(df):
    """
    Main function to analyze vol response to spread changes
    """
    
    # Clean data
    df_clean = df.dropna(subset=['Spread_Change', 'AUDUSD_Vol_Change', 'NZDUSD_Vol_Change']).copy()
    
    print("="*70)
    print("ANALYSIS: HOW VOLATILITY CHANGES WITH FORWARD RATE SPREAD CHANGES")
    print("="*70)
    print()
    
    # ===== 1. BASIC STATISTICS =====
    print("1. SPREAD CHANGE STATISTICS")
    print("-" * 50)
    print(f"   Mean Spread Change: {df_clean['Spread_Change'].mean():.4f}")
    print(f"   Std Dev: {df_clean['Spread_Change'].std():.4f}")
    print(f"   25th Percentile: {df_clean['Spread_Change'].quantile(0.25):.4f}")
    print(f"   75th Percentile: {df_clean['Spread_Change'].quantile(0.75):.4f}")
    print(f"   Max: {df_clean['Spread_Change'].max():.4f}")
    print(f"   Min: {df_clean['Spread_Change'].min():.4f}")
    print()
    
    # ===== 2. CORRELATION ANALYSIS =====
    print("2. CORRELATIONS")
    print("-" * 50)
    corr_matrix = df_clean[['Spread_Change', 'AUDUSD_Vol_Change', 'NZDUSD_Vol_Change']].corr()
    print(corr_matrix)
    print()
    print(f"   Interpretation:")
    print(f"   - Spread Change vs AUDUSD Vol Change: {corr_matrix.loc['Spread_Change', 'AUDUSD_Vol_Change']:.3f}")
    print(f"   - Spread Change vs NZDUSD Vol Change: {corr_matrix.loc['Spread_Change', 'NZDUSD_Vol_Change']:.3f}")
    print()
    
    # ===== 3. CATEGORIZE BY SPREAD CHANGE MAGNITUDE =====
    spread_std = df_clean['Spread_Change'].std()
    
    # Create categories
    df_clean['Spread_Regime'] = 'Normal'
    df_clean.loc[df_clean['Spread_Change'] > spread_std, 'Spread_Regime'] = 'Large Widening'
    df_clean.loc[df_clean['Spread_Change'] < -spread_std, 'Spread_Regime'] = 'Large Tightening'
    
    print("3. VOL BEHAVIOR BY SPREAD REGIME (±1 Std Dev)")
    print("-" * 50)
    
    for regime in ['Large Tightening', 'Normal', 'Large Widening']:
        regime_data = df_clean[df_clean['Spread_Regime'] == regime]
        if len(regime_data) > 0:
            print(f"\n   {regime.upper()} ({len(regime_data)} days):")
            print(f"   Spread Change Range: [{regime_data['Spread_Change'].min():.4f}, {regime_data['Spread_Change'].max():.4f}]")
            print(f"   ")
            print(f"   AUDUSD Vol Change:")
            print(f"      Mean: {regime_data['AUDUSD_Vol_Change'].mean():.4f}")
            print(f"      Median: {regime_data['AUDUSD_Vol_Change'].median():.4f}")
            print(f"      Std: {regime_data['AUDUSD_Vol_Change'].std():.4f}")
            print(f"   ")
            print(f"   NZDUSD Vol Change:")
            print(f"      Mean: {regime_data['NZDUSD_Vol_Change'].mean():.4f}")
            print(f"      Median: {regime_data['NZDUSD_Vol_Change'].median():.4f}")
            print(f"      Std: {regime_data['NZDUSD_Vol_Change'].std():.4f}")
    print()
    
    # ===== 4. TOP EVENTS =====
    print("4. TOP 10 SPREAD WIDENING EVENTS AND VOL RESPONSE")
    print("-" * 50)
    top_widening = df_clean.nlargest(10, 'Spread_Change')[
        ['Spread_Change', 'FwdRate_Spread', 'AUDUSD_Vol_Change', 'NZDUSD_Vol_Change', 
         'AUDUSDV1M BGN Curncy', 'NZDUSDV1M BGN Curncy']
    ]
    print(top_widening.to_string())
    print()
    
    print("5. TOP 10 SPREAD TIGHTENING EVENTS AND VOL RESPONSE")
    print("-" * 50)
    top_tightening = df_clean.nsmallest(10, 'Spread_Change')[
        ['Spread_Change', 'FwdRate_Spread', 'AUDUSD_Vol_Change', 'NZDUSD_Vol_Change', 
         'AUDUSDV1M BGN Curncy', 'NZDUSDV1M BGN Curncy']
    ]
    print(top_tightening.to_string())
    print()
    
    # ===== 5. QUINTILE ANALYSIS =====
    df_clean['Spread_Quintile'] = pd.qcut(df_clean['Spread_Change'], 
                                          q=5, 
                                          labels=['Q1 (Tightest)', 'Q2', 'Q3', 'Q4', 'Q5 (Widest)'])
    
    print("6. QUINTILE ANALYSIS")
    print("-" * 50)
    quintile_summary = df_clean.groupby('Spread_Quintile').agg({
        'Spread_Change': ['min', 'max', 'mean', 'count'],
        'AUDUSD_Vol_Change': ['mean', 'median'],
        'NZDUSD_Vol_Change': ['mean', 'median']
    })
    print(quintile_summary.to_string())
    print()
    
    # ===== 6. CREATE VISUALIZATIONS =====
    print("7. Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Scatter - Spread Change vs AUDUSD Vol Change
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(df_clean['Spread_Change'], df_clean['AUDUSD_Vol_Change'], 
                alpha=0.4, s=15, c='blue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Spread Change', fontsize=10)
    ax1.set_ylabel('AUDUSD Vol Change', fontsize=10)
    ax1.set_title('AUDUSD Vol Change vs Spread Change', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add regression line
    z1 = np.polyfit(df_clean['Spread_Change'], df_clean['AUDUSD_Vol_Change'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(sorted(df_clean['Spread_Change']), p1(sorted(df_clean['Spread_Change'])), 
             "r-", alpha=0.6, linewidth=2, label=f'y={z1[0]:.3f}x+{z1[1]:.3f}')
    ax1.legend(fontsize=8)
    
    # Plot 2: Scatter - Spread Change vs NZDUSD Vol Change
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(df_clean['Spread_Change'], df_clean['NZDUSD_Vol_Change'], 
                alpha=0.4, s=15, c='orange')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Spread Change', fontsize=10)
    ax2.set_ylabel('NZDUSD Vol Change', fontsize=10)
    ax2.set_title('NZDUSD Vol Change vs Spread Change', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add regression line
    z2 = np.polyfit(df_clean['Spread_Change'], df_clean['NZDUSD_Vol_Change'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(sorted(df_clean['Spread_Change']), p2(sorted(df_clean['Spread_Change'])), 
             "r-", alpha=0.6, linewidth=2, label=f'y={z2[0]:.3f}x+{z2[1]:.3f}')
    ax2.legend(fontsize=8)
    
    # Plot 3: Box plot by regime
    ax3 = plt.subplot(2, 3, 3)
    regime_order = ['Large Tightening', 'Normal', 'Large Widening']
    data_to_plot_aud = [df_clean[df_clean['Spread_Regime'] == regime]['AUDUSD_Vol_Change'].dropna() 
                        for regime in regime_order]
    bp1 = ax3.boxplot(data_to_plot_aud, labels=regime_order, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('AUDUSD Vol Change', fontsize=10)
    ax3.set_title('AUDUSD Vol Change by Spread Regime', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15, labelsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    # Plot 4: Box plot for NZDUSD
    ax4 = plt.subplot(2, 3, 4)
    data_to_plot_nzd = [df_clean[df_clean['Spread_Regime'] == regime]['NZDUSD_Vol_Change'].dropna() 
                        for regime in regime_order]
    bp2 = ax4.boxplot(data_to_plot_nzd, labels=regime_order, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax4.set_ylabel('NZDUSD Vol Change', fontsize=10)
    ax4.set_title('NZDUSD Vol Change by Spread Regime', fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', rotation=15, labelsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    # Plot 5: Time series
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(df_clean.index, df_clean['Spread_Change'], 
             label='Spread Change', alpha=0.7, linewidth=0.8, color='blue')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax5.axhline(y=spread_std, color='red', linestyle='--', alpha=0.3, linewidth=0.5, label='+1 SD')
    ax5.axhline(y=-spread_std, color='red', linestyle='--', alpha=0.3, linewidth=0.5, label='-1 SD')
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Spread Change', fontsize=10)
    ax5.set_title('Time Series: Spread Changes', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Histogram of spread changes
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(df_clean['Spread_Change'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax6.axvline(x=spread_std, color='orange', linestyle='--', linewidth=1.5, label='+1 SD')
    ax6.axvline(x=-spread_std, color='orange', linestyle='--', linewidth=1.5, label='-1 SD')
    ax6.set_xlabel('Spread Change', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Distribution of Spread Changes', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    print()
    
    # ===== 7. KEY INSIGHTS =====
    print("="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    corr_aud = corr_matrix.loc['Spread_Change', 'AUDUSD_Vol_Change']
    corr_nzd = corr_matrix.loc['Spread_Change', 'NZDUSD_Vol_Change']
    
    print(f"\n1. Correlation Analysis:")
    print(f"   - AUDUSD vol has a {abs(corr_aud):.1%} correlation with spread changes")
    print(f"     ({'positive' if corr_aud > 0 else 'negative'} relationship)")
    print(f"   - NZDUSD vol has a {abs(corr_nzd):.1%} correlation with spread changes")
    print(f"     ({'positive' if corr_nzd > 0 else 'negative'} relationship)")
    
    print(f"\n2. Regime Behavior:")
    for regime in ['Large Tightening', 'Normal', 'Large Widening']:
        regime_data = df_clean[df_clean['Spread_Regime'] == regime]
        if len(regime_data) > 0:
            print(f"   {regime}:")
            print(f"     - AUDUSD vol avg change: {regime_data['AUDUSD_Vol_Change'].mean():.4f}")
            print(f"     - NZDUSD vol avg change: {regime_data['NZDUSD_Vol_Change'].mean():.4f}")
    
    print("\n" + "="*70)
    
    return df_clean, fig







currency_pairs = ['S0159FC 1M1D BCAL Curncy', 'S0198FC 1M1D BCAL Curncy', 'AUDUSDV1M BGN Curncy', 'NZDUSDV1M BGN Curncy']
years = 10

df = Daily_GetTopCCYs(currency_pairs, years)

df['FwdRate_Spread'] = df['S0159FC 1M1D BCAL Curncy'] - df['S0198FC 1M1D BCAL Curncy']


df['FwdRate_Spread_Pct'] = ((df['AUDUSDV1M BGN Curncy'] - df['NZDUSDV1M BGN Curncy']) / 
                             df['NZDUSDV1M BGN Curncy']) * 100

df['Spread_Change'] = df['FwdRate_Spread'].diff()
df['Spread_Change_Pct'] = df['FwdRate_Spread'].pct_change() * 100

df['AUDUSD_Vol_Change'] = df['AUDUSDV1M BGN Curncy'].diff()
df['NZDUSD_Vol_Change'] = df['NZDUSDV1M BGN Curncy'].diff()

df['Vol_Spread'] = df['AUDUSDV1M BGN Curncy'] - df['NZDUSDV1M BGN Curncy']

df[['S0159FC 1M1D BCAL Curncy', 'S0198FC 1M1D BCAL Curncy', 'FwdRate_Spread', 'Vol_Spread', 'AUDUSDV1M BGN Curncy', 'NZDUSDV1M BGN Curncy']]











df_analyzed, fig = main_analysis(df)
plt.show()