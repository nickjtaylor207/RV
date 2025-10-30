from Coint_screen import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns

"""
---- Euro Block ----
ccys_eur = ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURPLN', 'EURHUF', 'EURCZK', 'EURNOK', 'EURSEK']

---- Majors ----
ccys_major = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK']

---- Asia ----
ccys_asia = ['USDJPY', 'USDCNH', 'USDKRW', 'USDSGD', 'USDTWD', 'USDTHB', 'USDIDR', 'USDPHP', 'USDMYR', 'USDINR']

---- EM ----
['USDBRL', 'USDMXN', 'USDZAR', 'USDTRY', 'USDRUB', 'USDCLP', 'USDCOP', 'USDPLN', 'USDHUF']

"""





# currencies  = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK']
# tenor = '2M'
# days = 360 * 5

# analyzer = Screen_MultiPair_Metrics.create_from_currencies(
#                     currencies, tenor, days)


# print(analyzer.get_best_pairs_comprehensive())



# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------




def calculate_vol_adjusted_butterflies(ccys, tenor, days=730):
    tickers = []
    for ccy in ccys:
        tickers.append(f'{ccy}10B{tenor} BGN Curncy')  
        tickers.append(f'{ccy}V{tenor} Curncy')        
    df = Data_DailyCLOSE_Multiple(tickers, days)
    df_result = pd.DataFrame(index=df.index)
    for ccy in ccys:
        bf_ticker = f'{ccy}10B{tenor} BGN Curncy'
        vol_ticker = f'{ccy}V{tenor} Curncy'
        if bf_ticker in df.columns and vol_ticker in df.columns:
            df_result[f'{ccy}_VolAdj_BF'] = (df[bf_ticker] / df[vol_ticker]) * 100
    return df_result





def plot_vol_adjusted_butterflies_comparison(ccys, tenor, days, 
                                             plot_type='overlay', 
                                             normalize=False):

    df = calculate_vol_adjusted_butterflies(ccys, tenor, days)
    if normalize:
        df = (df / df.iloc[0]) * 100
        ylabel = 'Normalized Vol-Adj BF (Start = 100)'
    else:
        ylabel = 'Vol-Adjusted Butterfly (%)'
    
    # ---- Overaly ----
    if plot_type == 'overlay':
        fig, ax = plt.subplots(figsize=(14, 7))
        for col in df.columns:
            ccy_name = col.replace('_VolAdj_BF', '')
            ax.plot(df.index, df[col], label=ccy_name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Vol-Adjusted Butterflies Comparison ({tenor} Tenor)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
    # 
    elif plot_type == 'grid':
        n_ccys = len(ccys)
        n_cols = min(3, n_ccys)
        n_rows = (n_ccys + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten() if n_ccys > 1 else [axes]
        for i, ccy in enumerate(ccys):
            col = f'{ccy}_VolAdj_BF'
            if col in df.columns:
                axes[i].plot(df.index, df[col], linewidth=2, color=f'C{i}')
                axes[i].set_title(f'{ccy}', fontsize=11, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(labelsize=8)
        for i in range(n_ccys, len(axes)):
            axes[i].axis('off')
        plt.suptitle(f'Vol-Adjusted Butterflies Grid ({tenor} Tenor)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        

    elif plot_type == 'heatmap':
        corr_matrix = df.corr()
        labels = [col.replace('_VolAdj_BF', '') for col in corr_matrix.columns]
        corr_matrix.columns = labels
        corr_matrix.index = labels
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title(f'Vol-Adj BF Correlation Matrix ({tenor} Tenor)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        

    elif plot_type == 'pairs':
        if len(ccys) < 2:
            print("Need at least 2 currencies for pairwise comparison")
            return
        
        # Get the base currency (first in list)
        base_ccy = ccys[0]
        other_ccys = ccys[1:]  # All currencies except the first
        
        n_pairs = len(other_ccys)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5*n_rows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, other_ccy in enumerate(other_ccys):
            col1 = f'{base_ccy}_VolAdj_BF'
            col2 = f'{other_ccy}_VolAdj_BF'
            
            if col1 in df.columns and col2 in df.columns:
                # Clean data
                df_clean = df[[col1, col2]].dropna()
                
                # Scatter plot
                axes[i].scatter(df_clean[col1], df_clean[col2], alpha=0.5, s=15)
                
                # Add regression line
                if len(df_clean) > 0:
                    z = np.polyfit(df_clean[col1], df_clean[col2], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(df_clean[col1].min(), df_clean[col1].max(), 100)
                    axes[i].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                    
                    # Calculate regression stats
                    from scipy import stats as scipy_stats
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        df_clean[col1], df_clean[col2])
                    r_squared = r_value ** 2
                
                # Calculate correlation
                corr = df_clean[col1].corr(df_clean[col2])
                
                # Calculate beta (volatility of Y relative to X)
                beta = df_clean[col2].std() / df_clean[col1].std() if df_clean[col1].std() > 0 else 0
                
                # Calculate current z-scores (how far from mean in standard deviations)
                z_score_base = (df_clean[col1].iloc[-1] - df_clean[col1].mean()) / df_clean[col1].std()
                z_score_other = (df_clean[col2].iloc[-1] - df_clean[col2].mean()) / df_clean[col2].std()
                
                # Calculate spread (residual from regression)
                predicted = slope * df_clean[col1] + intercept
                spread = df_clean[col2] - predicted
                spread_zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
                
                # Rolling correlation (30-day)
                rolling_corr = df[col1].rolling(90).corr(df[col2])
                recent_corr = rolling_corr.iloc[-90:].mean() if len(rolling_corr) > 90 else corr
                
                # Create statistics text box
                stats_text = (
                    f'Correlation: {corr:.3f}\n'
                    f'R²: {r_squared:.3f}\n'
                    f'Beta: {beta:.3f}\n'
                    f'Slope: {slope:.3f}\n'
                    f'─────────────\n'
                    f'{base_ccy} Z-score: {z_score_base:+.2f}\n'
                    f'{other_ccy} Z-score: {z_score_other:+.2f}\n'
                    f'Spread Z: {spread_zscore:+.2f}\n'
                    f'90D Corr: {recent_corr:.3f}'
                )
                
                # Add text box to plot
                axes[i].text(0.02, 0.98, stats_text,
                            transform=axes[i].transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            family='monospace')
                
                # Add reference lines for means
                axes[i].axvline(df_clean[col1].mean(), color='blue', 
                            linestyle=':', alpha=0.3, linewidth=1)
                axes[i].axhline(df_clean[col2].mean(), color='orange', 
                            linestyle=':', alpha=0.3, linewidth=1)
                
                # Highlight current point
                axes[i].scatter(df_clean[col1].iloc[-1], df_clean[col2].iloc[-1], 
                            color='red', s=100, marker='*', 
                            edgecolors='black', linewidths=1.5,
                            label='Current', zorder=5)
                
                axes[i].set_xlabel(base_ccy, fontsize=10)
                axes[i].set_ylabel(other_ccy, fontsize=10)
                axes[i].set_title(f'{base_ccy} vs {other_ccy}', 
                                fontsize=11, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(loc='lower right', fontsize=8)
        
        # Hide extra subplots
        for i in range(n_pairs, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{base_ccy} vs All Others - Pairwise Comparisons ({tenor} Tenor)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
    return plt.gcf(), df








ccys = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD']
tenor='1W'
days=365 * 5

fig, df = plot_vol_adjusted_butterflies_comparison(ccys, tenor, days, 
                                                    plot_type='pairs')

plt.show()












def analyze_regime_clustering(ccys, tenor='1M', days=730):
    """
    Analyze and visualize regime clustering in vol-adjusted butterflies
    """
    df = calculate_vol_adjusted_butterflies(ccys, tenor, days)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Get the two currencies for pairwise analysis
    ccy1, ccy2 = ccys[0], ccys[1]
    col1 = f'{ccy1}_VolAdj_BF'
    col2 = f'{ccy2}_VolAdj_BF'
    
    # Clean data
    df_clean = df[[col1, col2]].dropna()
    
    # 1. Time series with regime highlighting
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_clean.index, df_clean[col1], label=ccy1, linewidth=2, alpha=0.7)
    ax1.plot(df_clean.index, df_clean[col2], label=ccy2, linewidth=2, alpha=0.7)
    
    # Highlight high volatility periods
    threshold_1 = df_clean[col1].quantile(0.75)
    threshold_2 = df_clean[col2].quantile(0.75)
    high_vol_periods = (df_clean[col1] > threshold_1) | (df_clean[col2] > threshold_2)
    
    # Shade high vol periods
    for idx in df_clean[high_vol_periods].index:
        ax1.axvspan(idx, idx, alpha=0.1, color='red')
    
    ax1.set_ylabel('Vol-Adjusted BF (%)', fontsize=11)
    ax1.set_title(f'{ccy1} vs {ccy2}: Time Series with High Vol Periods (red shading)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot with density coloring
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(df_clean[col1], df_clean[col2], 
                         c=range(len(df_clean)), cmap='viridis', 
                         alpha=0.6, s=20)
    
    # Add regression line
    z = np.polyfit(df_clean[col1], df_clean[col2], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean[col1].min(), df_clean[col1].max(), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    corr = df_clean[col1].corr(df_clean[col2])
    ax2.set_xlabel(f'{ccy1} Vol-Adj BF', fontsize=11)
    ax2.set_ylabel(f'{ccy2} Vol-Adj BF', fontsize=11)
    ax2.set_title(f'Scatter (colored by time)\nCorr: {corr:.3f}', 
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Time (older → newer)')
    
    # 3. K-means clustering to identify regimes
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean[[col1, col2]])
    
    # Fit K-means with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_scaled)
    
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['blue', 'orange']
    for i in range(2):
        mask = clusters == i
        ax3.scatter(df_clean[col1][mask], df_clean[col2][mask], 
                   c=colors[i], label=f'Regime {i+1}', alpha=0.6, s=20)
    
    # Add cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax3.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
               s=200, edgecolors='black', linewidths=2, label='Centers')
    
    ax3.set_xlabel(f'{ccy1} Vol-Adj BF', fontsize=11)
    ax3.set_ylabel(f'{ccy2} Vol-Adj BF', fontsize=11)
    ax3.set_title('K-Means Clustering (2 Regimes)', fontsize=11, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series colored by regime
    ax4 = fig.add_subplot(gs[2, 0])
    df_clean['regime'] = clusters
    for i in range(2):
        mask = df_clean['regime'] == i
        ax4.scatter(df_clean.index[mask], df_clean[col1][mask], 
                   c=colors[i], label=f'Regime {i+1}', alpha=0.6, s=10)
    ax4.set_ylabel(f'{ccy1} Vol-Adj BF', fontsize=11)
    ax4.set_title(f'{ccy1}: Regime-Colored Time Series', fontsize=11, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 5. Regime statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = f"REGIME ANALYSIS:\n\n"
    for i in range(2):
        mask = clusters == i
        stats_text += f"Regime {i+1} ({colors[i]}):\n"
        stats_text += f"  Observations: {mask.sum()} ({mask.sum()/len(mask)*100:.1f}%)\n"
        stats_text += f"  {ccy1} Mean: {df_clean[col1][mask].mean():.2f}\n"
        stats_text += f"  {ccy1} Std: {df_clean[col1][mask].std():.2f}\n"
        stats_text += f"  {ccy2} Mean: {df_clean[col2][mask].mean():.2f}\n"
        stats_text += f"  {ccy2} Std: {df_clean[col2][mask].std():.2f}\n"
        stats_text += f"  Correlation: {df_clean[col1][mask].corr(df_clean[col2][mask]):.3f}\n\n"
    
    # Calculate transition frequency
    regime_changes = (df_clean['regime'].diff() != 0).sum()
    stats_text += f"Regime Transitions: {regime_changes}\n"
    stats_text += f"Avg Days per Regime: {len(df_clean)/regime_changes:.1f}"
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Regime Clustering Analysis: {ccy1} vs {ccy2} ({tenor} Tenor)', 
                fontsize=14, fontweight='bold')
    
    return fig, df_clean, clusters


# ccys = ['USDJPY', 'NZDUSD']
# tenor = '1M'
# days = 730

# fig, df, clusters = analyze_regime_clustering(ccys, tenor, days)
# plt.show()







def identify_regime_drivers(ccys, tenor='1M', days=730):
    df_bf = calculate_vol_adjusted_butterflies(ccys, tenor, days)
    market_indicators = [
        'VIX Index',           # Equity vol
        'MOVE Index',          # Bond vol
        'USDJPY Curncy',       # Spot rate
        'GT2 Govt',            # 2Y US yield
        'GTJPY2Y Govt'           # 2Y Japan yield
    ]
    
    try:
        df_market = Data_DailyCLOSE_Multiple(market_indicators, days)
        
        # Merge datasets
        df_combined = pd.concat([df_bf, df_market], axis=1).dropna()
        
        # Calculate rate differential
        if 'GT2 Govt' in df_combined.columns and 'GTJPY2Y Govt' in df_combined.columns:
            df_combined['Rate_Diff'] = df_combined['GT2 Govt'] - df_combined['GTJPY2Y Govt']
        
        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # Plot 1: Vol-adj BFs
        for col in df_bf.columns:
            ccy = col.replace('_VolAdj_BF', '')
            axes[0].plot(df_combined.index, df_combined[col], label=ccy, linewidth=2)
        axes[0].set_ylabel('Vol-Adj BF', fontsize=10)
        axes[0].set_title('Vol-Adjusted Butterflies', fontsize=11, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: VIX (risk sentiment)
        if 'VIX Index' in df_combined.columns:
            axes[1].plot(df_combined.index, df_combined['VIX Index'], 
                        color='red', linewidth=2)
            axes[1].set_ylabel('VIX', fontsize=10)
            axes[1].set_title('Equity Volatility (Risk Sentiment)', fontsize=11, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: USDJPY
        if 'USDJPY Curncy' in df_combined.columns:
            axes[2].plot(df_combined.index, df_combined['USDJPY Curncy'], 
                        color='green', linewidth=2)
            axes[2].set_ylabel('USDJPY', fontsize=10)
            axes[2].set_title('USDJPY Spot Rate', fontsize=11, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Rate differential
        if 'Rate_Diff' in df_combined.columns:
            axes[3].plot(df_combined.index, df_combined['Rate_Diff'], 
                        color='purple', linewidth=2)
            axes[3].set_ylabel('Rate Diff (bp)', fontsize=10)
            axes[3].set_title('US-Japan 2Y Rate Differential', fontsize=11, fontweight='bold')
            axes[3].grid(True, alpha=0.3)
        
        axes[3].set_xlabel('Date', fontsize=11)
        plt.suptitle(f'Regime Drivers Analysis ({tenor} Tenor)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig, df_combined
        
    except Exception as e:
        print(f"Could not fetch market indicators: {e}")
        return None, None



# fig, df_market = identify_regime_drivers(['USDJPY', 'AUDUSD', 'NZDUSD'], tenor='1M', days=730)
# if fig:
#     plt.show()









