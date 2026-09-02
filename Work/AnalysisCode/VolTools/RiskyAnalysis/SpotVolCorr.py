import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xbbg import blp
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression





class RR_Performance_Analyzer:
    def __init__(self, currency_pairs: List[str], tenors: List[str], delta: str = '25'):
        self.currency_pairs = currency_pairs
        self.tenors = tenors
        self.delta = delta
        self.df_data = None
        self.correlation_models = {}  # Store empirical models


    def _tenor_to_days(self, tenor: str) -> int:
        tenor_map = {
            '1W': 5,
            '2W': 10,
            '3W': 15,
            '1M': 21,
            '6W': 30,
            '2M': 42,
            '3M': 63,
            '6M': 126,
            '9M': 189,
            '1Y': 252,
            '2Y': 504}
        
        if tenor not in tenor_map:
            raise ValueError(f"Unknown tenor: {tenor}. Valid tenors: {list(tenor_map.keys())}")
        
        return tenor_map[tenor]
        
    def _get_spot_data(self, lookback_days: int = 730) -> pd.DataFrame:
        start_date = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        ticker_list = [f"{pair} Curncy" for pair in self.currency_pairs]
        data_all = blp.bdh(
            tickers=ticker_list,
            flds=["PX_LAST"],
            start_date=start_date,
            end_date=end_date,
            Per="D")
        if data_all.empty:
            print("No spot data returned")
            return pd.DataFrame()
        df_spot = pd.DataFrame()
        for pair in self.currency_pairs:
            ticker_full = f"{pair} Curncy"
            if ticker_full in data_all.columns.get_level_values(0):
                df_spot[f"{pair}"] = data_all[ticker_full].iloc[:, 0]
            else:
                print(f"No data for {pair}, skipping.")
        return df_spot
    
    def _get_vol_rr_bf_data(self, lookback_days: int = 730) -> pd.DataFrame:
        start_date = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        ticker_list = []
        ticker_map = {}
        for pair in self.currency_pairs:
            for tenor in self.tenors:
                # ATM Volatility
                ticker_IV = f"{pair}V{tenor} BGN Curncy"
                ticker_list.append(ticker_IV)
                ticker_map[ticker_IV] = (pair, tenor, 'VOL')
                # Risk Reversal
                ticker_RR = f"{pair}{self.delta}R{tenor} BGN Curncy"
                ticker_list.append(ticker_RR)
                ticker_map[ticker_RR] = (pair, tenor, 'RR')
                # Butterfly
                ticker_BF = f"{pair}{self.delta}B{tenor} BGN Curncy"
                ticker_list.append(ticker_BF)
                ticker_map[ticker_BF] = (pair, tenor, 'BF')

        data_all = blp.bdh(
            tickers=ticker_list,
            flds="PX_LAST",
            start_date=start_date,
            end_date=end_date)
        if data_all.empty:
            print("No vol/RR/BF data returned")
            return pd.DataFrame()
        df_vol_rr_bf = pd.DataFrame()
        for ticker, (pair, tenor, data_type) in ticker_map.items():
            if ticker in data_all.columns.get_level_values(0):
                if data_type == 'VOL':
                    column_name = f"{pair}_{tenor}_VOL"
                elif data_type == 'RR':
                    column_name = f"{pair}_{tenor}_RR"
                else:  # BF
                    column_name = f"{pair}_{tenor}_BF"
                df_vol_rr_bf[column_name] = data_all[ticker].iloc[:, 0]
            else:
                print(f"No data for {ticker}, skipping.")
        return df_vol_rr_bf
    
    def load_all_data(self, lookback_days: int = 1260):
        """Load spot, vol, RR, and BF data"""
        df_spot = self._get_spot_data(lookback_days)
        df_vol_rr_bf = self._get_vol_rr_bf_data(lookback_days)
        self.df_data = pd.concat([df_spot, df_vol_rr_bf], axis=1)
        self.df_data = self.df_data.sort_index()
        return self.df_data



    def calculate_realized_correlation_multi_window(self, 
                                                   pairs: List[str] = None, 
                                                   tenors: List[str] = None,
                                                   windows: List[str] = ['1W', '1M', '3M', '6M']) -> pd.DataFrame:
        if self.df_data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        if pairs is None:
            pairs = self.currency_pairs
        if tenors is None:
            tenors = self.tenors
        df_results = pd.DataFrame(index=self.df_data.index)
        for pair in pairs:
            for tenor in tenors:
                spot_col = pair
                vol_col = f"{pair}_{tenor}_VOL"
                if spot_col not in self.df_data.columns or vol_col not in self.df_data.columns:
                    print(f"Warning: Missing data for {pair} {tenor}, skipping.")
                    continue
                spot_returns = self.df_data[spot_col].pct_change()
                vol_changes = self.df_data[vol_col].diff()
                for window_tenor in windows:
                    window_days = self._tenor_to_days(window_tenor)
                    col_name = f'{tenor}_{pair}_{window_tenor}corr'
                    df_results[col_name] = spot_returns.rolling(window=window_days).corr(vol_changes)
        return df_results



    def calculate_realized_correlation_metrics(self, pair: str, tenor: str, 
                                            roll_windows: List[int] = [21, 63, 126, 252]) -> pd.DataFrame:
        """
        Calculate detailed correlation metrics for a single pair/tenor
        Returns DataFrame with implied RR, realized correlations at multiple windows, and errors
        """
        if self.df_data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        tenor_days = self._tenor_to_days(tenor)
        
        spot_col = pair
        vol_col = f"{pair}_{tenor}_VOL"
        rr_col = f"{pair}_{tenor}_RR"
        
        if spot_col not in self.df_data.columns or vol_col not in self.df_data.columns:
            raise ValueError(f"Missing data for {pair} {tenor}")
        
        df = pd.DataFrame({
            'spot': self.df_data[spot_col],
            'atm_vol': self.df_data[vol_col],
            'implied_rr': self.df_data[rr_col]
        }).dropna()
        
        # Calculate forward changes
        df['spot_return_fwd'] = np.log(df['spot'].shift(-tenor_days) / df['spot'])
        df['vol_change_fwd'] = df['atm_vol'].shift(-tenor_days) - df['atm_vol']
        
        # Calculate correlations at multiple windows
        for window in roll_windows:
            # Realized correlation
            col_name = f'realized_corr_{window}d'
            df[col_name] = df['spot_return_fwd'].rolling(window).corr(df['vol_change_fwd'])
            
            # Normalize to RR-equivalent (multiply by std of vol changes)
            vol_std = df['vol_change_fwd'].rolling(window).std()
            df[f'realized_rr_{window}d'] = df[col_name] * vol_std
            
            # Calculate error (implied - realized)
            df[f'rr_error_{window}d'] = df['implied_rr'] - df[f'realized_rr_{window}d']
            
            # Check if sign is correct
            df[f'correct_sign_{window}d'] = (np.sign(df[col_name]) == np.sign(df['implied_rr']))
            
            # Rolling hit rate
            df[f'hit_rate_{window}d'] = df[f'correct_sign_{window}d'].rolling(window).mean() * 100
        
        # Add pair/tenor labels
        df['pair'] = pair
        df['tenor'] = tenor
        
        return df
    




    def get_correlation_summary_table(self, pairs: List[str] = None, 
                                    tenors: List[str] = None,
                                    window: int = 63) -> pd.DataFrame:
        """
        Generate summary table comparing correlation metrics across pairs and tenors
        """
        if pairs is None:
            pairs = self.currency_pairs
        if tenors is None:
            tenors = self.tenors
        
        results = []
        
        for pair in pairs:
            for tenor in tenors:
                try:
                    df = self.calculate_realized_correlation_metrics(pair, tenor, roll_windows=[window])
                    
                    latest = df.iloc[-1]
                    
                    # Calculate statistics
                    avg_corr = df[f'realized_corr_{window}d'].mean()
                    std_corr = df[f'realized_corr_{window}d'].std()
                    hit_rate = df[f'correct_sign_{window}d'].mean() * 100
                    avg_error = df[f'rr_error_{window}d'].mean()
                    abs_error = np.abs(df[f'rr_error_{window}d']).mean()
                    
                    results.append({
                        'pair': pair,
                        'tenor': tenor,
                        'current_implied_rr': latest['implied_rr'],
                        'current_realized_corr': latest[f'realized_corr_{window}d'],
                        'avg_realized_corr': avg_corr,
                        'std_realized_corr': std_corr,
                        'hit_rate_pct': hit_rate,
                        'current_error': latest[f'rr_error_{window}d'],
                        'avg_abs_error': abs_error,
                        'current_atm_vol': latest['atm_vol']
                    })
                except Exception as e:
                    print(f"Error processing {pair} {tenor}: {e}")
                    continue
        
        df_summary = pd.DataFrame(results)
        
        # Sort by absolute error to highlight mispricings
        df_summary = df_summary.sort_values('current_error', key=abs, ascending=False)
        
        return df_summary






    def plot_realized_correlation(self, pair: str, tenor: str, 
                                windows: List[int] = [21, 63, 126, 252],
                                figsize=(16, 12)):
        """
        Comprehensive visualization of correlation analysis for a single pair/tenor
        """
        df = self.calculate_realized_correlation_metrics(pair, tenor, roll_windows=windows)
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Plot 1: Implied RR vs Realized Correlations (multiple windows)
        ax1 = axes[0]
        ax1.plot(df.index, df['implied_rr'], label='Implied RR', 
                color='black', linewidth=2, alpha=0.8)
        
        colors = ['red', 'orange', 'green', 'blue']
        for window, color in zip(windows, colors):
            ax1.plot(df.index, df[f'realized_rr_{window}d'], 
                    label=f'Realized RR ({window}d)', 
                    color=color, linewidth=1.5, alpha=0.7)
        
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('RR / Correlation (vol pts)', fontsize=11)
        ax1.set_title(f'{pair} {tenor} Risk Reversal: Implied vs Realized', 
                    fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correlation Values (easier to read than RR-normalized)
        ax2 = axes[1]
        for window, color in zip(windows, colors):
            ax2.plot(df.index, df[f'realized_corr_{window}d'], 
                    label=f'{window}d correlation', 
                    color=color, linewidth=1.5, alpha=0.7)
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Correlation Coefficient', fontsize=11)
        ax2.set_title(f'{pair} {tenor} Realized Spot/Vol Correlation', 
                    fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RR Pricing Error (using 63d window as reference)
        ax3 = axes[2]
        error_col = f'rr_error_{windows[1]}d'  # Use 63d as default
        colors_bar = ['green' if x > 0 else 'red' for x in df[error_col]]
        ax3.bar(df.index, df[error_col], color=colors_bar, alpha=0.5, width=1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('RR Error (vol pts)', fontsize=11)
        ax3.set_title(f'{pair} {tenor} RR Pricing Error ({windows[1]}d window)\n'
                    f'Green = RR overpriced, Red = RR underpriced', 
                    fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Hit Rate Over Time (multiple windows)
        ax4 = axes[3]
        for window, color in zip(windows, colors):
            ax4.plot(df.index, df[f'hit_rate_{window}d'], 
                    label=f'{window}d hit rate', 
                    color=color, linewidth=1.5, alpha=0.7)
        
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax4.set_ylabel('Hit Rate (%)', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_title(f'{pair} {tenor} RR Prediction Accuracy (Sign Hit Rate)', 
                    fontsize=13, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()







# analyzer = RR_Performance_Analyzer(
#     currency_pairs=['EURUSD'],
#     tenors=['1W', '1M', '3M'],
#     delta='25'
# )
# analyzer.load_all_data(lookback_days=1260)

# df_corr = analyzer.calculate_realized_correlation_metrics('EURUSD', '1W')



# # analyzer.plot_realized_correlation('EURUSD', '1W')



# summary = analyzer.get_correlation_summary_table()
# print(summary)



