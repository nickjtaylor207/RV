import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

class VolatilityScreenerAnalyzer:
    def __init__(self, excel_path, weighting_method='equal', selected_tenors=None):

        self.excel_path = excel_path
        self.weighting_method = weighting_method.lower()

        self.selected_tenors = selected_tenors
        valid_methods = ['equal', 'correlation', 'cointegration_1', 'cointegration_2']
        if self.weighting_method not in valid_methods:
            raise ValueError(f"weighting_method must be one of {valid_methods}, got '{weighting_method}'")
        self.summary_eq = None
        self.summary_corr = None
        self.summary_coint_1 = None
        self.summary_coint_2 = None
        self.trades_eq = None
        self.trades_corr = None
        self.trades_coint_1 = None
        self.trades_coint_2 = None
        self.load_data()
        self.g10_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 
                         'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK']
        self.em_pairs = ['USDMXN', 'USDBRL', 'USDCNH', 'USDZAR']
    

    # ============================================================================
    # UTILITY METHODS - Added missing methods
    # ============================================================================

    def print_section_header(self, title):
        """Print a formatted section header."""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
        
    def get_active_tenors(self):
        """Get list of active tenors based on selected_tenors or all available."""
        if self.selected_tenors is not None:
            return self.selected_tenors
        return sorted(self.trades['tenor'].unique().tolist())
    def _identify_drawdown_periods(self, trades_df):
        """Identify and analyze specific drawdown periods."""
        print("\n\n📉 MAJOR DRAWDOWN PERIODS")
        print("-"*60)
        major_dds = trades_df[trades_df['drawdown'] < -5].copy()
        if len(major_dds) == 0:
            print("No major drawdown periods (>5%) identified")
            return
        major_dds['dd_group'] = (major_dds['drawdown'] == 0).cumsum()
        for group_id in major_dds['dd_group'].unique():
            group = major_dds[major_dds['dd_group'] == group_id]
            start_date = group['signal_date'].iloc[0]
            end_date = group['signal_date'].iloc[-1]
            max_dd = group['drawdown'].min()
            duration = (end_date - start_date).days
            print(f"\nDrawdown Period:")
            print(f"  Start: {start_date.strftime('%Y-%m-%d')}")
            print(f"  End: {end_date.strftime('%Y-%m-%d')}")
            print(f"  Duration: {duration} days")
            print(f"  Max Drawdown: {max_dd:.1f}%")


    # ============================================================================
    # DATA LOADING & PROCESSING
    # ============================================================================

    def load_data(self):
        try:
            xls = pd.ExcelFile(self.excel_path)
            available_sheets = xls.sheet_names
            if 'Summary_EqualWeights' in available_sheets:
                self.summary_eq = pd.read_excel(self.excel_path, sheet_name='Summary_EqualWeights', index_col=0)
                self.trades_eq = pd.read_excel(self.excel_path, sheet_name='Trades_EqualWeights')
                self.trades_eq = self._process_trades_df(self.trades_eq)
            if 'Summary_CorrWeights' in available_sheets:
                self.summary_corr = pd.read_excel(self.excel_path, sheet_name='Summary_CorrWeights', index_col=0)
                self.trades_corr = pd.read_excel(self.excel_path, sheet_name='Trades_CorrWeights')
                self.trades_corr = self._process_trades_df(self.trades_corr)
            if 'Summary_CointWeights_1st' in available_sheets:
                self.summary_coint_1 = pd.read_excel(self.excel_path, sheet_name='Summary_CointWeights_1st', index_col=0)
                self.trades_coint_1 = pd.read_excel(self.excel_path, sheet_name='Trades_CointWeights_1st')
                self.trades_coint_1 = self._process_trades_df(self.trades_coint_1)
            if 'Summary_CointWeights_2nd' in available_sheets:
                self.summary_coint_2 = pd.read_excel(self.excel_path, sheet_name='Summary_CointWeights_2nd', index_col=0)
                self.trades_coint_2 = pd.read_excel(self.excel_path, sheet_name='Trades_CointWeights_2nd')
                self.trades_coint_2 = self._process_trades_df(self.trades_coint_2)
            if self.weighting_method == 'equal' and self.summary_eq is None:
                raise ValueError("Equal weights data not found in Excel file")
            elif self.weighting_method == 'correlation' and self.summary_corr is None:
                raise ValueError("Correlation weights data not found in Excel file")
            elif self.weighting_method == 'cointegration_1' and self.summary_coint_1 is None:
                raise ValueError("Cointegration_1 weights data not found in Excel file")
            elif self.weighting_method == 'cointegration_2' and self.summary_coint_2 is None:
                raise ValueError("Cointegration_2 weights data not found in Excel file")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    def _process_trades_df(self, trades_df):
        """Process and filter trades DataFrame."""
        trades_df = trades_df[trades_df['tenor'].notna()].copy()
        trades_df['signal_date'] = pd.to_datetime(trades_df['signal_date'])
        trades_df['future_date'] = pd.to_datetime(trades_df['future_date'])
        if self.selected_tenors is not None:
            trades_df = trades_df[trades_df['tenor'].isin(self.selected_tenors)]
            if len(trades_df) == 0:
                available_tenors = trades_df['tenor'].unique().tolist()
                raise ValueError(f"No data found for selected tenors {self.selected_tenors}. "
                               f"Available tenors: {available_tenors}")
        return trades_df
    @property
    def summary(self):
        """Returns the active summary dataset based on the weighting method."""
        if self.weighting_method == 'correlation':
            return self.summary_corr
        elif self.weighting_method == 'cointegration_1':
            return self.summary_coint_1
        elif self.weighting_method == 'cointegration_2':
            return self.summary_coint_2
        else: 
            return self.summary_eq

    @property
    def trades(self):
        """Returns the active trades dataset based on the weighting method."""
        if self.weighting_method == 'correlation':
            return self.trades_corr
        elif self.weighting_method == 'cointegration_1':
            return self.trades_coint_1
        elif self.weighting_method == 'cointegration_2':
            return self.trades_coint_2
        else:
            return self.trades_eq
    @property
    def weighting_method_display(self):
        """Returns a formatted string describing the current weighting method."""
        method_names = {
            'equal': "Equal Weights",
            'correlation': "Correlation Weights",
            'cointegration_1': "Cointegration Weights (Version 1)",
            'cointegration_2': "Cointegration Weights (Version 2)"
        }
        return method_names.get(self.weighting_method, self.weighting_method)



    def data_tenorPairSignal_trades(self, tenor=None, currency_pair=None, signal_type=None, start_date=None, end_date=None):
        trades_df = self.trades.copy()
        if start_date is not None or end_date is not None:
            if 'date' in trades_df.columns:
                date_col = 'date'
            elif 'trade_date' in trades_df.columns:
                date_col = 'trade_date'
            elif 'signal_date' in trades_df.columns:
                date_col = 'signal_date'
            elif 'timestamp' in trades_df.columns:
                date_col = 'timestamp'
            else:
                datetime_cols = trades_df.select_dtypes(include=['datetime64']).columns
                date_col = datetime_cols[0] if len(datetime_cols) > 0 else None
            if date_col is not None:
                if not pd.api.types.is_datetime64_any_dtype(trades_df[date_col]):
                    trades_df[date_col] = pd.to_datetime(trades_df[date_col])
                if start_date is not None:
                    start_date = pd.to_datetime(start_date)
                    trades_df = trades_df[trades_df[date_col] >= start_date]
                if end_date is not None:
                    end_date = pd.to_datetime(end_date)
                    trades_df = trades_df[trades_df[date_col] <= end_date]
        if tenor is not None:
            if isinstance(tenor, str):
                tenor = [tenor]  
            trades_df = trades_df[trades_df['tenor'].isin(tenor)].copy()
            if len(trades_df) == 0:
                print(f'No trades found for tenor(s): {tenor}')
                return pd.DataFrame()
        if currency_pair is not None:
            if isinstance(currency_pair, str):
                currency_pair = [currency_pair]  
            trades_df = trades_df[trades_df['currency_pair'].isin(currency_pair)].copy()
            if len(trades_df) == 0:
                print(f'No trades found for currency pair(s): {currency_pair}')
                return pd.DataFrame()
        if signal_type is not None:
            if signal_type.lower() in ['e', 'expensive']:
                signal_filter = 'expensive'
            elif signal_type.lower() in ['c', 'cheap']:
                signal_filter = 'cheap'
            else:
                print(f"Invalid signal_type: {signal_type}. Use 'e'/'expensive' or 'c'/'cheap'")
                return pd.DataFrame()
            trades_df = trades_df[trades_df['signal_type'] == signal_filter].copy()
            if len(trades_df) == 0:
                print(f'No {signal_filter} trades found')
                return pd.DataFrame()
        trades_df = trades_df[['signal_date', 'future_date',
                        'currency_pair', 'tenor', 'signal_type', 'signal_percentile',
                        'implied_vol', 'realized_vol', 'vol_diff', 'vol_diff_pct', 'trade_success']]
        return trades_df.sort_values('signal_date').reset_index(drop=True)





    # ----------------------------------------------------------------------------------------------------
    #       Tenor Grouping
    # ----------------------------------------------------------------------------------------------------

    def print_TenorGeneralRiskRewards(self, start_date=None, end_date=None):
        self.print_section_header("RISK-REWARD ANALYSIS BY TENOR")
        filtered_trades = self.trades.copy()
        date_filter_info = ""
        if start_date is not None or end_date is not None:
            if 'date' in filtered_trades.columns:
                date_col = 'date'
            elif 'trade_date' in filtered_trades.columns:
                date_col = 'trade_date'
            elif 'timestamp' in filtered_trades.columns:
                date_col = 'timestamp'
            else:
                datetime_cols = filtered_trades.select_dtypes(include=['datetime64']).columns
                if len(datetime_cols) > 0:
                    date_col = datetime_cols[0]
                else:
                    print("Warning: No datetime column found for date filtering")
                    date_col = None
            if date_col is not None:
                if not pd.api.types.is_datetime64_any_dtype(filtered_trades[date_col]):
                    filtered_trades[date_col] = pd.to_datetime(filtered_trades[date_col])
                if start_date is not None:
                    start_date = pd.to_datetime(start_date)
                    filtered_trades = filtered_trades[filtered_trades[date_col] >= start_date]
                    date_filter_info += f" FROM {start_date.strftime('%Y-%m-%d')}"
                if end_date is not None:
                    end_date = pd.to_datetime(end_date)
                    filtered_trades = filtered_trades[filtered_trades[date_col] <= end_date]
                    date_filter_info += f" TO {end_date.strftime('%Y-%m-%d')}"
        tenors = self.get_active_tenors()
        for tenor in tenors:
            if tenor in self.summary.columns:
                print(f"\n{tenor.upper()} TENOR:")
                print("="*30)
                tenor_trades = filtered_trades[filtered_trades['tenor'] == tenor]
                for signal_type in ['expensive', 'cheap']:
                    signal_trades = tenor_trades[tenor_trades['signal_type'] == signal_type]
                    if len(signal_trades) > 0:
                        successful = signal_trades[signal_trades['trade_success'] == True]
                        failed = signal_trades[signal_trades['trade_success'] == False]
                        total_trades = len(signal_trades)
                        if signal_type == 'expensive':
                            hit_rate = self.summary.loc['expensive_hit_rate', tenor] * 100
                            avg_vol_diff = self.summary.loc['expensive_avg_vol_diff', tenor] * 100
                            avg_vol_diff_pct = self.summary.loc['expensive_avg_vol_diff_pct', tenor] * 100
                            max_drawdown = self.summary.loc['expensive_max_drawdown', tenor] * 100
                            max_drawdown_pct = self.summary.loc['expensive_max_drawdown_pct', tenor] * 100
                        else:  # cheap
                            hit_rate = self.summary.loc['cheap_hit_rate', tenor] * 100
                            avg_vol_diff = self.summary.loc['cheap_avg_vol_diff', tenor] * 100
                            avg_vol_diff_pct = self.summary.loc['cheap_avg_vol_diff_pct', tenor] * 100
                            max_drawdown = self.summary.loc['cheap_max_drawdown', tenor] * 100
                            max_drawdown_pct = self.summary.loc['cheap_max_drawdown_pct', tenor] * 100
                        print(f"\n  {signal_type.capitalize()} Vol Signals:")
                        print(f"    Total Trades: {total_trades}")
                        print(f"    Hit Rate: {hit_rate:.1f}%")
                        if len(successful) > 0 and len(failed) > 0:
                            avg_win = abs(successful['vol_diff_pct'].mean()) * 100
                            avg_loss = abs(failed['vol_diff_pct'].mean()) * 100
                            win_rate = len(successful) / len(signal_trades)
                            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
                            print(f"    Avg Win: {avg_win:.2f}%")
                            print(f"    Avg Loss: {avg_loss:.2f}%")
                            print(f"    Risk-Reward Ratio: {risk_reward_ratio:.2f}:1")
                            print(f"    Expectancy: {expectancy:+.2f}%")
                        print(f"    Avg Vol Diff: {avg_vol_diff:.2f}%")
                        print(f"    Avg Vol Diff (Pct): {avg_vol_diff_pct:.2f}%")
                        print(f"    Max Drawdown: {max_drawdown:.2f}%")
                        print(f"    Max Drawdown (Pct): {max_drawdown_pct:.2f}%")
                    else:
                        print(f"\n  {signal_type.capitalize()} Vol Signals: No trades")
                exp_hr = self.summary.loc['expensive_hit_rate', tenor] * 100
                cheap_hr = self.summary.loc['cheap_hit_rate', tenor] * 100
                performance_gap = exp_hr - cheap_hr
                print(f"\n  Performance Gap (Exp - Cheap): {performance_gap:+.1f}%")
                print("-"*30)





    # ----------------------------------------------------------------------------------------------------
    #       Different Spread Percentile Weightings Comparisons  (MORE WORK NEEDED)
    # ----------------------------------------------------------------------------------------------------
    def print_GEN_WeightingCompare(self, start_date=None, end_date=None):
        print("\n" + "="*80)
        print(" COMPREHENSIVE WEIGHTING METHOD COMPARISON")
        print("="*80)
        methods_data = []
        date_filter_info = ""
        def filter_trades_by_date(trades):
            filtered_trades = trades.copy()
            filter_info = ""
            if start_date is not None or end_date is not None:
                if 'date' in filtered_trades.columns:
                    date_col = 'date'
                elif 'trade_date' in filtered_trades.columns:
                    date_col = 'trade_date'
                elif 'timestamp' in filtered_trades.columns:
                    date_col = 'timestamp'
                else:
                    datetime_cols = filtered_trades.select_dtypes(include=['datetime64']).columns
                    date_col = datetime_cols[0] if len(datetime_cols) > 0 else None
                if date_col is not None:
                    if not pd.api.types.is_datetime64_any_dtype(filtered_trades[date_col]):
                        filtered_trades[date_col] = pd.to_datetime(filtered_trades[date_col])
                    if start_date is not None:
                        start_dt = pd.to_datetime(start_date)
                        filtered_trades = filtered_trades[filtered_trades[date_col] >= start_dt]
                        filter_info += f" FROM {start_dt.strftime('%Y-%m-%d')}"
                    if end_date is not None:
                        end_dt = pd.to_datetime(end_date)
                        filtered_trades = filtered_trades[filtered_trades[date_col] <= end_dt]
                        filter_info += f" TO {end_dt.strftime('%Y-%m-%d')}"
                else:
                    print("Warning: No datetime column found for date filtering")
            return filtered_trades, filter_info
        if self.summary_eq is not None:
            filtered_trades_eq, date_filter_info = filter_trades_by_date(self.trades_eq)
            methods_data.append(('Equal', self.summary_eq, filtered_trades_eq))
        if self.summary_corr is not None:
            filtered_trades_corr, _ = filter_trades_by_date(self.trades_corr)
            methods_data.append(('Correlation', self.summary_corr, filtered_trades_corr))
        if self.summary_coint_1 is not None:
            filtered_trades_coint_1, _ = filter_trades_by_date(self.trades_coint_1)
            methods_data.append(('Cointegration_1', self.summary_coint_1, filtered_trades_coint_1))
        if self.summary_coint_2 is not None:
            filtered_trades_coint_2, _ = filter_trades_by_date(self.trades_coint_2)
            methods_data.append(('Cointegration_2', self.summary_coint_2, filtered_trades_coint_2))
        if len(methods_data) < 2:
            print("\nInsufficient data for comparison. Need at least 2 weighting methods.")
            return
        print(f"\nAvailable methods: {[m[0] for m in methods_data]}")
        print(f"Selected Tenors: {', '.join(self.get_active_tenors())}")
        if date_filter_info:
            print(f"Date Range: {date_filter_info.strip()}")
        print(f"\nFiltered Trade Counts:")
        for method_name, _, trades in methods_data:
            print(f"  {method_name}: {len(trades)} trades")
        print("-"*80)
        print("\n📊 OVERALL PERFORMANCE METRICS")
        print("-"*60)
        comparison_results = []
        for method_name, summary, trades in methods_data:
            trades_filtered = trades.copy()
            if self.selected_tenors:
                trades_filtered = trades_filtered[trades_filtered['tenor'].isin(self.selected_tenors)]
            if len(trades_filtered) > 0:
                overall_hr = trades_filtered['trade_success'].mean() * 100
                exp_trades = trades_filtered[trades_filtered['signal_type'] == 'expensive']
                cheap_trades = trades_filtered[trades_filtered['signal_type'] == 'cheap']
                exp_hr = exp_trades['trade_success'].mean() * 100 if len(exp_trades) > 0 else 0
                cheap_hr = cheap_trades['trade_success'].mean() * 100 if len(cheap_trades) > 0 else 0
                avg_vol_diff = trades_filtered['vol_diff_pct'].mean() * 100
                exp_vol_diff = exp_trades['vol_diff_pct'].mean() * 100 if len(exp_trades) > 0 else 0
                cheap_vol_diff = cheap_trades['vol_diff_pct'].mean() * 100 if len(cheap_trades) > 0 else 0
                wins = trades_filtered[trades_filtered['trade_success'] == True]
                losses = trades_filtered[trades_filtered['trade_success'] == False]
                avg_win = wins['vol_diff_pct'].mean() * 100 if len(wins) > 0 else 0
                avg_loss = losses['vol_diff_pct'].mean() * 100 if len(losses) > 0 else 0
                max_drawdown = trades_filtered['vol_diff_pct'].min() * 100
                vol_std = trades_filtered['vol_diff_pct'].std() * 100
                sharpe_ratio = avg_vol_diff / vol_std if vol_std > 0 else 0
                win_rate = len(wins) / len(trades_filtered)
                expectancy = (win_rate * abs(avg_win)) - ((1 - win_rate) * abs(avg_loss))
                comparison_results.append({
                    'Method': method_name,
                    'Total Trades': len(trades_filtered),
                    'Overall Hit Rate': overall_hr,
                    'Expensive Hit Rate': exp_hr,
                    'Cheap Hit Rate': cheap_hr,
                    'Expensive Count': len(exp_trades),
                    'Cheap Count': len(cheap_trades),
                    'Avg Vol Diff': avg_vol_diff,
                    'Exp Vol Diff': exp_vol_diff,
                    'Cheap Vol Diff': cheap_vol_diff,
                    'Avg Win': avg_win,
                    'Avg Loss': avg_loss,
                    'Max Drawdown': max_drawdown,
                    'Vol Std': vol_std,
                    'Sharpe Ratio': sharpe_ratio,
                    'Expectancy': expectancy,
                    'Win Rate': win_rate * 100})
        comparison_df = pd.DataFrame(comparison_results)
        for _, row in comparison_df.iterrows():
            print(f"\n{row['Method'].upper()} WEIGHTS:")
            print(f"  Total Trades: {row['Total Trades']:.0f}")
            print(f"  Overall Hit Rate: {row['Overall Hit Rate']:.1f}%")
            print(f"  ├─ Expensive: {row['Expensive Hit Rate']:.1f}% ({row['Expensive Count']:.0f} trades)")
            print(f"  └─ Cheap: {row['Cheap Hit Rate']:.1f}% ({row['Cheap Count']:.0f} trades)")
            print(f"  Average Vol Diff: {row['Avg Vol Diff']:+.2f}%")
            print(f"  ├─ Expensive: {row['Exp Vol Diff']:+.2f}%")
            print(f"  └─ Cheap: {row['Cheap Vol Diff']:+.2f}%")
            print(f"  Risk-Adjusted Return: {row['Sharpe Ratio']:.3f}")
            print(f"  Expectancy: {row['Expectancy']:+.2f}%")
            print(f"  Max Drawdown: {row['Max Drawdown']:.2f}%")
        print("\n\n🏆 PERFORMANCE RANKINGS")
        print("-"*60)
        ranking_metrics = [
            ('Overall Hit Rate', True),  # Higher is better
            ('Expensive Hit Rate', True),
            ('Cheap Hit Rate', True),
            ('Expectancy', True),
            ('Sharpe Ratio', True),
            ('Max Drawdown', False),  # Lower (less negative) is better
            ('Vol Std', False)]
        rankings = {}
        for metric, higher_better in ranking_metrics:
            if metric in comparison_df.columns:
                sorted_df = comparison_df.sort_values(metric, ascending=not higher_better)
                print(f"\n{metric}:")
                for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
                    value = row[metric]
                    symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    print(f"  {symbol} {i}. {row['Method']}: {value:.2f}{'%' if 'Rate' in metric or 'Expectancy' in metric else ''}")
                for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
                    if row['Method'] not in rankings:
                        rankings[row['Method']] = []
                    rankings[row['Method']].append(i)
        avg_rankings = {method: np.mean(ranks) for method, ranks in rankings.items()}
        best_overall_method = min(avg_rankings, key=avg_rankings.get)
        print(f"\n📈 OVERALL RANKING (Average Position):")
        for method, avg_rank in sorted(avg_rankings.items(), key=lambda x: x[1]):
            print(f"  {method}: {avg_rank:.2f}")
        if self.selected_tenors and len(self.selected_tenors) > 1:
            print("\n\n📅 TENOR-SPECIFIC COMPARISON")
            print("-"*60)
            for tenor in self.selected_tenors:
                print(f"\n{tenor} Tenor Performance:")
                tenor_comparison = []
                for method_name, _, trades in methods_data:
                    tenor_trades = trades[trades['tenor'] == tenor]
                    if len(tenor_trades) > 0:
                        hr = tenor_trades['trade_success'].mean() * 100
                        exp_hr = tenor_trades[tenor_trades['signal_type'] == 'expensive']['trade_success'].mean() * 100
                        cheap_hr = tenor_trades[tenor_trades['signal_type'] == 'cheap']['trade_success'].mean() * 100
                        tenor_comparison.append({
                            'Method': method_name,
                            'Hit Rate': hr,
                            'Exp HR': exp_hr,
                            'Cheap HR': cheap_hr,
                            'Count': len(tenor_trades)})
                if tenor_comparison:
                    best_for_tenor = max(tenor_comparison, key=lambda x: x['Hit Rate'])
                    for comp in tenor_comparison:
                        indicator = "✅" if comp['Method'] == best_for_tenor['Method'] else "  "
                        print(f"  {indicator} {comp['Method']}: {comp['Hit Rate']:.1f}% "
                            f"(Exp: {comp['Exp HR']:.1f}%, Cheap: {comp['Cheap HR']:.1f}%, "
                            f"n={comp['Count']})")
        print("\n\n💱 CURRENCY-SPECIFIC PERFORMANCE")
        print("-"*60)
        all_currencies = set()
        for _, _, trades in methods_data:
            all_currencies.update(trades['currency_pair'].unique())
        currency_performance = {}
        for currency in all_currencies:
            currency_performance[currency] = {}
            for method_name, _, trades in methods_data:
                curr_trades = trades[trades['currency_pair'] == currency]
                if self.selected_tenors:
                    curr_trades = curr_trades[curr_trades['tenor'].isin(self.selected_tenors)]
                if len(curr_trades) > 0:
                    currency_performance[currency][method_name] = {
                        'hit_rate': curr_trades['trade_success'].mean() * 100,
                        'count': len(curr_trades)}
        print("\nTop 5 Currencies by Method:")
        for method_name, _, _ in methods_data:
            method_currencies = []
            for currency, methods in currency_performance.items():
                if method_name in methods and methods[method_name]['count'] >= 10:
                    method_currencies.append((currency, methods[method_name]['hit_rate']))
            method_currencies.sort(key=lambda x: x[1], reverse=True)
            print(f"\n{method_name}:")
            for i, (currency, hit_rate) in enumerate(method_currencies[:5], 1):
                print(f"  {i}. {currency}: {hit_rate:.1f}%")
        print("\n\n📊 PERFORMANCE DIFFERENCES")
        print("-"*60)
        method_names = [m[0] for m in methods_data]
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                perf1 = comparison_df[comparison_df['Method'] == method1].iloc[0]
                perf2 = comparison_df[comparison_df['Method'] == method2].iloc[0]
                print(f"\n{method1} vs {method2}:")
                hr_diff = perf1['Overall Hit Rate'] - perf2['Overall Hit Rate']
                print(f"  Hit Rate Difference: {hr_diff:+.1f}%")
                exp_hr_diff = perf1['Expensive Hit Rate'] - perf2['Expensive Hit Rate']
                print(f"  Expensive HR Difference: {exp_hr_diff:+.1f}%")
                exp_diff = perf1['Expectancy'] - perf2['Expectancy']
                print(f"  Expectancy Difference: {exp_diff:+.2f}%")
                sharpe_diff = perf1['Sharpe Ratio'] - perf2['Sharpe Ratio']
                print(f"  Risk-Adjusted Return Diff: {sharpe_diff:+.3f}")
                if abs(hr_diff) > 2:  # Significant difference threshold
                    winner = method1 if hr_diff > 0 else method2
                    print(f"  ➡️ {winner} performs better overall")
        print("\n\n💡 RECOMMENDATIONS")
        print("-"*60)
        best_exp = comparison_df.loc[comparison_df['Expensive Hit Rate'].idxmax()]
        print(f"\nBest for Expensive Vol: {best_exp['Method']} ({best_exp['Expensive Hit Rate']:.1f}%)")
        best_cheap = comparison_df.loc[comparison_df['Cheap Hit Rate'].idxmax()]
        print(f"Best for Cheap Vol: {best_cheap['Method']} ({best_cheap['Cheap Hit Rate']:.1f}%)")
        best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        print(f"Best Risk-Adjusted: {best_sharpe['Method']} (Sharpe: {best_sharpe['Sharpe Ratio']:.3f})")
        most_consistent = comparison_df.loc[comparison_df['Vol Std'].idxmin()]
        print(f"Most Consistent: {most_consistent['Method']} (Std: {most_consistent['Vol Std']:.2f}%)")
        
        print(f"\n🎯 OVERALL RECOMMENDATION:")
        print(f"  Based on comprehensive analysis across all metrics,")
        print(f"  {best_overall_method} weighting provides the best overall performance")
        print(f"  with an average ranking of {avg_rankings[best_overall_method]:.2f}")
        
        for coint_version in ['Cointegration_1', 'Cointegration_2']:
            if coint_version in method_names:
                coint_perf = comparison_df[comparison_df['Method'] == coint_version].iloc[0]
                other_methods = comparison_df[comparison_df['Method'] != coint_version]
                avg_other_hr = other_methods['Overall Hit Rate'].mean()
                coint_improvement = coint_perf['Overall Hit Rate'] - avg_other_hr
                print(f"\n📈 {coint_version} Weighting Analysis:")
                if coint_improvement > 0:
                    print(f"  ✅ Improves hit rate by {coint_improvement:.1f}% vs other methods")
                    print(f"  ✅ Particularly strong for: ", end="")
                    strengths = []
                    if coint_perf['Expensive Hit Rate'] == comparison_df['Expensive Hit Rate'].max():
                        strengths.append("expensive vol")
                    if coint_perf['Sharpe Ratio'] == comparison_df['Sharpe Ratio'].max():
                        strengths.append("risk-adjusted returns")
                    if coint_perf['Expectancy'] == comparison_df['Expectancy'].max():
                        strengths.append("expectancy")
                    print(", ".join(strengths) if strengths else "general performance")
                else:
                    print(f"  ⚠️ Underperforms by {-coint_improvement:.1f}% vs other methods")
                    print(f"  Consider using {best_overall_method} weighting instead")
    def get_weighting_comparison_df(self, columns_int=None, start_date=None, end_date=None, include_differences=True):
        def filter_trades_by_date(trades):
            filtered_trades = trades.copy()
            if start_date is not None or end_date is not None:
                if 'date' in filtered_trades.columns:
                    date_col = 'date'
                elif 'trade_date' in filtered_trades.columns:
                    date_col = 'trade_date'
                elif 'timestamp' in filtered_trades.columns:
                    date_col = 'timestamp'
                else:
                    datetime_cols = filtered_trades.select_dtypes(include=['datetime64']).columns
                    date_col = datetime_cols[0] if len(datetime_cols) > 0 else None
                
                if date_col is not None:
                    if not pd.api.types.is_datetime64_any_dtype(filtered_trades[date_col]):
                        filtered_trades[date_col] = pd.to_datetime(filtered_trades[date_col])
                    if start_date is not None:
                        start_dt = pd.to_datetime(start_date)
                        filtered_trades = filtered_trades[filtered_trades[date_col] >= start_dt]
                    if end_date is not None:
                        end_dt = pd.to_datetime(end_date)
                        filtered_trades = filtered_trades[filtered_trades[date_col] <= end_dt]
            return filtered_trades
        methods_data = []
        if self.summary_eq is not None:
            filtered_trades_eq = filter_trades_by_date(self.trades_eq)
            methods_data.append(('Equal', self.summary_eq, filtered_trades_eq))
        if self.summary_corr is not None:
            filtered_trades_corr = filter_trades_by_date(self.trades_corr)
            methods_data.append(('Correlation', self.summary_corr, filtered_trades_corr))
        if self.summary_coint_1 is not None:
            filtered_trades_coint_1 = filter_trades_by_date(self.trades_coint_1)
            methods_data.append(('Cointegration_1', self.summary_coint_1, filtered_trades_coint_1))
        if self.summary_coint_2 is not None:
            filtered_trades_coint_2 = filter_trades_by_date(self.trades_coint_2)
            methods_data.append(('Cointegration_2', self.summary_coint_2, filtered_trades_coint_2))     
        if len(methods_data) < 1:
            print("No weighting method data available.")
            return pd.DataFrame()
        all_data = {}
        for method_name, summary, trades in methods_data:
            trades_filtered = trades.copy()
            if self.selected_tenors:
                trades_filtered = trades_filtered[trades_filtered['tenor'].isin(self.selected_tenors)]
            if len(trades_filtered) > 0:
                exp_trades = trades_filtered[trades_filtered['signal_type'] == 'expensive']
                cheap_trades = trades_filtered[trades_filtered['signal_type'] == 'cheap']
                exp_hr = exp_trades['trade_success'].mean() * 100 if len(exp_trades) > 0 else 0
                cheap_hr = cheap_trades['trade_success'].mean() * 100 if len(cheap_trades) > 0 else 0
                exp_vol_diff = exp_trades['vol_diff_pct'].mean() * 100 if len(exp_trades) > 0 else 0
                cheap_vol_diff = cheap_trades['vol_diff_pct'].mean() * 100 if len(cheap_trades) > 0 else 0
                exp_wins = exp_trades[exp_trades['trade_success'] == True]
                exp_losses = exp_trades[exp_trades['trade_success'] == False]
                cheap_wins = cheap_trades[cheap_trades['trade_success'] == True]
                cheap_losses = cheap_trades[cheap_trades['trade_success'] == False]
                exp_avg_win = abs(exp_wins['vol_diff_pct'].mean() * 100) if len(exp_wins) > 0 else 0
                exp_avg_loss = abs(exp_losses['vol_diff_pct'].mean() * 100) if len(exp_losses) > 0 else 0
                cheap_avg_win = cheap_wins['vol_diff_pct'].mean() * 100 if len(cheap_wins) > 0 else 0
                cheap_avg_loss = abs(cheap_losses['vol_diff_pct'].mean() * 100) if len(cheap_losses) > 0 else 0
                exp_vol_std = exp_trades['vol_diff_pct'].std() * 100 if len(exp_trades) > 0 else 0
                cheap_vol_std = cheap_trades['vol_diff_pct'].std() * 100 if len(cheap_trades) > 0 else 0
                exp_sharpe = exp_vol_diff / exp_vol_std if exp_vol_std > 0 else 0
                cheap_sharpe = cheap_vol_diff / cheap_vol_std if cheap_vol_std > 0 else 0
                exp_win_rate = len(exp_wins) / len(exp_trades) if len(exp_trades) > 0 else 0
                cheap_win_rate = len(cheap_wins) / len(cheap_trades) if len(cheap_trades) > 0 else 0
                exp_expectancy = (exp_win_rate * exp_avg_win) - ((1 - exp_win_rate) * exp_avg_loss) if len(exp_trades) > 0 else 0
                cheap_expectancy = (cheap_win_rate * cheap_avg_win) - ((1 - cheap_win_rate) * cheap_avg_loss) if len(cheap_trades) > 0 else 0
                exp_rr = exp_avg_win / exp_avg_loss if exp_avg_loss != 0 else np.inf
                cheap_rr = cheap_avg_win / cheap_avg_loss if cheap_avg_loss != 0 else np.inf
                all_data[(method_name, 'Expensive')] = {
                    'Count': len(exp_trades),
                    'Hit_Rate_%': round(exp_hr, 2),
                    'Avg_Vol_Diff_%': round(exp_vol_diff, 2),
                    'Avg_Win_%': round(exp_avg_win, 2),
                    'Avg_Loss_%': round(exp_avg_loss, 2),
                    'Risk_Reward_Ratio': round(exp_rr, 2),
                    'Vol_Std_%': round(exp_vol_std, 2),
                    'Sharpe_Ratio': round(exp_sharpe, 3),
                    'Expectancy_%': round(exp_expectancy, 2),}
                all_data[(method_name, 'Cheap')] = {
                    'Count': len(cheap_trades),
                    'Hit_Rate_%': round(cheap_hr, 2),
                    'Avg_Vol_Diff_%': round(cheap_vol_diff, 2),
                    'Avg_Win_%': round(cheap_avg_win, 2),
                    'Avg_Loss_%': round(cheap_avg_loss, 2),
                    'Risk_Reward_Ratio': round(cheap_rr, 2),
                    'Vol_Std_%': round(cheap_vol_std, 2),
                    'Sharpe_Ratio': round(cheap_sharpe, 3),
                    'Expectancy_%': round(cheap_expectancy, 2),}
                if include_differences:
                    all_data[(method_name, 'Difference')] = {
                        'Count': len(exp_trades) - len(cheap_trades),
                        'Hit_Rate_%': round(exp_hr - cheap_hr, 2),
                        'Avg_Vol_Diff_%': round(exp_vol_diff - cheap_vol_diff, 2),
                        'Avg_Win_%': round(exp_avg_win - cheap_avg_win, 2),
                        'Avg_Loss_%': round(exp_avg_loss - cheap_avg_loss, 2),
                        'Risk_Reward_Ratio': round(exp_rr - cheap_rr, 2),
                        'Vol_Std_%': round(exp_vol_std - cheap_vol_std, 2),
                        'Sharpe_Ratio': round(exp_sharpe - cheap_sharpe, 3),
                        'Expectancy_%': round(exp_expectancy - cheap_expectancy, 2),}
        comparison_df = pd.DataFrame(all_data)
        if columns_int is not None:
            available_rows = comparison_df.index.tolist()
            valid_rows = [row for row in columns_int if row in available_rows]
            if valid_rows:
                comparison_df = comparison_df.loc[valid_rows]
            else:
                print(f"Warning: No matching rows found for {columns_int}")
                print(f"Available rows: {available_rows}")
        return comparison_df









    # ----------------------------------------------------------------------------------------------------
    #       Currency Grouping
    # ----------------------------------------------------------------------------------------------------

    # ----------------------------- Display General Output -----------------------------

    def _calculate_enhanced_currency_stats(self, curr_trades, exp_trades, cheap_trades):
        stats = {}
        stats['total'] = len(curr_trades)
        stats['exp_count'] = len(exp_trades)
        stats['cheap_count'] = len(cheap_trades)
        stats['exp_hit_rate'] = (exp_trades['trade_success'].sum() / len(exp_trades) * 100) if len(exp_trades) > 0 else 0
        stats['cheap_hit_rate'] = (cheap_trades['trade_success'].sum() / len(cheap_trades) * 100) if len(cheap_trades) > 0 else 0
        stats['overall_hit_rate'] = (curr_trades['trade_success'].sum() / len(curr_trades) * 100) if len(curr_trades) > 0 else 0
        stats['exp_avg_diff'] = exp_trades['vol_diff_pct'].mean() * 100 if len(exp_trades) > 0 else 0
        stats['cheap_avg_diff'] = cheap_trades['vol_diff_pct'].mean() * 100 if len(cheap_trades) > 0 else 0
        stats['overall_avg_diff'] = curr_trades['vol_diff_pct'].mean() * 100 if len(curr_trades) > 0 else 0
        for signal_type, trades in [('exp', exp_trades), ('cheap', cheap_trades)]:
            if len(trades) > 0:
                successful = trades[trades['trade_success'] == True]
                failed = trades[trades['trade_success'] == False]
                if len(successful) > 0:
                    if signal_type == 'exp':
                        stats[f'{signal_type}_avg_win'] = abs(successful['vol_diff_pct'].mean()) * 100
                        stats[f'{signal_type}_max_win'] = abs(successful['vol_diff_pct'].min()) * 100  # Most negative = best win
                    else:  # cheap
                        stats[f'{signal_type}_avg_win'] = successful['vol_diff_pct'].mean() * 100
                        stats[f'{signal_type}_max_win'] = successful['vol_diff_pct'].max() * 100  # Most positive = best win
                    stats[f'{signal_type}_win_count'] = len(successful)
                else:
                    stats[f'{signal_type}_avg_win'] = 0
                    stats[f'{signal_type}_max_win'] = 0
                    stats[f'{signal_type}_win_count'] = 0
                if len(failed) > 0:
                    if signal_type == 'exp':
                        stats[f'{signal_type}_avg_loss'] = failed['vol_diff_pct'].mean() * 100
                        stats[f'{signal_type}_max_loss'] = failed['vol_diff_pct'].max() * 100  # Most positive = worst loss
                    else:  # cheap
                        stats[f'{signal_type}_avg_loss'] = abs(failed['vol_diff_pct'].mean()) * 100
                        stats[f'{signal_type}_max_loss'] = abs(failed['vol_diff_pct'].min()) * 100  # Most negative = worst loss
                    stats[f'{signal_type}_loss_count'] = len(failed)
                else:
                    stats[f'{signal_type}_avg_loss'] = 0
                    stats[f'{signal_type}_max_loss'] = 0
                    stats[f'{signal_type}_loss_count'] = 0
                if stats[f'{signal_type}_avg_loss'] > 0:
                    stats[f'{signal_type}_risk_reward_ratio'] = stats[f'{signal_type}_avg_win'] / stats[f'{signal_type}_avg_loss']
                else:
                    stats[f'{signal_type}_risk_reward_ratio'] = float('inf') if stats[f'{signal_type}_avg_win'] > 0 else 0
                win_rate = len(successful) / len(trades) if len(trades) > 0 else 0
                stats[f'{signal_type}_expectancy'] = (win_rate * stats[f'{signal_type}_avg_win']) - ((1 - win_rate) * stats[f'{signal_type}_avg_loss'])
        if len(curr_trades) > 0:
            stats['vol_diff_std'] = curr_trades['vol_diff_pct'].std() * 100
            stats['vol_diff_median'] = curr_trades['vol_diff_pct'].median() * 100
            if stats['vol_diff_std'] > 0:
                stats['return_vol_ratio'] = stats['overall_avg_diff'] / stats['vol_diff_std']
            else:
                stats['return_vol_ratio'] = 0
        stats['performance_gap'] = stats['exp_hit_rate'] - stats['cheap_hit_rate']
        if len(curr_trades) > 0:
            curr_trades_sorted = curr_trades.sort_values('date' if 'date' in curr_trades.columns else curr_trades.columns[0])
            success_sequence = curr_trades_sorted['trade_success'].astype(int)
            current_win_streak = 0
            current_loss_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            for success in success_sequence:
                if success == 1:
                    current_win_streak += 1
                    current_loss_streak = 0
                    max_win_streak = max(max_win_streak, current_win_streak)
                else:
                    current_loss_streak += 1
                    current_win_streak = 0
                    max_loss_streak = max(max_loss_streak, current_loss_streak)
            stats['max_win_streak'] = max_win_streak
            stats['max_loss_streak'] = max_loss_streak
        return stats
    def _print_enhanced_currency_analysis(self, currency, stats):
        print(f"\n{currency}:")
        print(f"  Total Trades: {stats['total']}")
        print(f"  Overall Hit Rate: {stats['overall_hit_rate']:.1f}%")
        print(f"  Overall Avg Return: {stats['overall_avg_diff']:+.2f}%")
        print(f"  Return Volatility: {stats.get('vol_diff_std', 0):.2f}%")
        print(f"  Return/Vol Ratio: {stats.get('return_vol_ratio', 0):.2f}")
        if stats['exp_count'] > 0:
            print(f"\n  📈 EXPENSIVE Vol Signals ({stats['exp_count']} trades):")
            print(f"    Hit Rate: {stats['exp_hit_rate']:.1f}%")
            print(f"    Avg Win: {stats['exp_avg_win']:.2f}% | Avg Loss: {stats['exp_avg_loss']:+.2f}%")
            print(f"    Best Win: {stats['exp_max_win']:.2f}% | Worst Loss: {stats['exp_max_loss']:+.2f}%")
            print(f"    Risk-Reward: {stats['exp_risk_reward_ratio']:.2f}:1")
            print(f"    Expectancy: {stats['exp_expectancy']:+.2f}%")
            print(f"    W/L Record: {stats['exp_win_count']}-{stats['exp_loss_count']}")
        if stats['cheap_count'] > 0:
            print(f"\n  📉 CHEAP Vol Signals ({stats['cheap_count']} trades):")
            print(f"    Hit Rate: {stats['cheap_hit_rate']:.1f}%")
            print(f"    Avg Win: {stats['cheap_avg_win']:+.2f}% | Avg Loss: {stats['cheap_avg_loss']:.2f}%")
            print(f"    Best Win: {stats['cheap_max_win']:+.2f}% | Worst Loss: {stats['cheap_max_loss']:.2f}%")
            print(f"    Risk-Reward: {stats['cheap_risk_reward_ratio']:.2f}:1")
            print(f"    Expectancy: {stats['cheap_expectancy']:+.2f}%")
            print(f"    W/L Record: {stats['cheap_win_count']}-{stats['cheap_loss_count']}")
        print(f"\n  📊 PERFORMANCE METRICS:")
        print(f"    Performance Gap (Exp-Cheap): {stats['performance_gap']:+.1f}%")
        print(f"    Median Return: {stats.get('vol_diff_median', 0):+.2f}%")
        if 'max_win_streak' in stats:
            print(f"    Max Win Streak: {stats['max_win_streak']} | Max Loss Streak: {stats['max_loss_streak']}")
        if 'avg_coint_score' in stats:
            print(f"    Avg Cointegration Score: {stats['avg_coint_score']:.3f}")
        print("-" * 50)
    

    def _print_enhanced_aggregate_analysis(self, currency_stats):
        g10_currencies = {k: v for k, v in currency_stats.items() if v['type'] == 'G10'}
        em_currencies = {k: v for k, v in currency_stats.items() if v['type'] == 'EM'}
        print("\n" + "="*60)
        print("📈 AGGREGATE PERFORMANCE SUMMARY")
        print("="*60)
        for currency_type, currencies in [('G10', g10_currencies), ('EM', em_currencies)]:
            if currencies:
                print(f"\n{currency_type} CURRENCIES SUMMARY:")
                print("-" * 30)
                total_trades = sum(stats['total'] for stats in currencies.values())
                total_wins = sum(stats['total'] * stats['overall_hit_rate'] / 100 for stats in currencies.values())
                avg_hit_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
                weighted_avg_return = sum(stats['total'] * stats['overall_avg_diff'] for stats in currencies.values()) / total_trades if total_trades > 0 else 0
                best_performer = max(currencies.items(), key=lambda x: x[1]['overall_hit_rate'])
                worst_performer = min(currencies.items(), key=lambda x: x[1]['overall_hit_rate'])
                print(f"  Total Trades: {total_trades}")
                print(f"  Average Hit Rate: {avg_hit_rate:.1f}%")
                print(f"  Weighted Avg Return: {weighted_avg_return:+.2f}%")
                print(f"  Best Performer: {best_performer[0]} ({best_performer[1]['overall_hit_rate']:.1f}%)")
                print(f"  Worst Performer: {worst_performer[0]} ({worst_performer[1]['overall_hit_rate']:.1f}%)")
                exp_hit_rates = [stats['exp_hit_rate'] for stats in currencies.values() if stats['exp_count'] > 0]
                cheap_hit_rates = [stats['cheap_hit_rate'] for stats in currencies.values() if stats['cheap_count'] > 0]
                if exp_hit_rates and cheap_hit_rates:
                    avg_exp_hr = sum(exp_hit_rates) / len(exp_hit_rates)
                    avg_cheap_hr = sum(cheap_hit_rates) / len(cheap_hit_rates)
                    print(f"  Avg Expensive Hit Rate: {avg_exp_hr:.1f}%")
                    print(f"  Avg Cheap Hit Rate: {avg_cheap_hr:.1f}%")
                    print(f"  Avg Performance Gap: {avg_exp_hr - avg_cheap_hr:+.1f}%")

    def print_ALLcurrency_enhanced(self, start_date=None, end_date=None):
        self.print_section_header("ENHANCED CURRENCY-SPECIFIC PERFORMANCE ANALYSIS")
        trades_df = self.trades.copy()
        date_filter_info = ""
        if start_date is not None or end_date is not None:
            if 'date' in trades_df.columns:
                date_col = 'date'
            elif 'trade_date' in trades_df.columns:
                date_col = 'trade_date'
            elif 'timestamp' in trades_df.columns:
                date_col = 'timestamp'
            else:
                datetime_cols = trades_df.select_dtypes(include=['datetime64']).columns
                date_col = datetime_cols[0] if len(datetime_cols) > 0 else None
            if date_col is not None:
                if not pd.api.types.is_datetime64_any_dtype(trades_df[date_col]):
                    trades_df[date_col] = pd.to_datetime(trades_df[date_col])
                if start_date is not None:
                    start_date = pd.to_datetime(start_date)
                    trades_df = trades_df[trades_df[date_col] >= start_date]
                    date_filter_info += f" FROM {start_date.strftime('%Y-%m-%d')}"
                if end_date is not None:
                    end_date = pd.to_datetime(end_date)
                    trades_df = trades_df[trades_df[date_col] <= end_date]
                    date_filter_info += f" TO {end_date.strftime('%Y-%m-%d')}"
        print(f" Total Filtered Trades: {len(trades_df)}")
        if date_filter_info:
            print(f" Date Range: {date_filter_info.strip()}")
        print()
        currency_stats = {}
        for currency in trades_df['currency_pair'].unique():
            curr_trades = trades_df[trades_df['currency_pair'] == currency]
            exp_trades = curr_trades[curr_trades['signal_type'] == 'expensive']
            cheap_trades = curr_trades[curr_trades['signal_type'] == 'cheap']
            stats = self._calculate_enhanced_currency_stats(curr_trades, exp_trades, cheap_trades)
            stats['type'] = 'G10' if currency in self.g10_pairs else 'EM'

            if self.weighting_method in ['cointegration_1', 'cointegration_2'] and 'coint_score' in curr_trades.columns:
                stats['avg_coint_score'] = curr_trades['coint_score'].mean()

            currency_stats[currency] = stats
        sorted_currencies = sorted(currency_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        print("\n🌍 G10 CURRENCIES:")
        print("=" * 60)
        for curr, stats in sorted_currencies:
            if stats['type'] == 'G10':
                self._print_enhanced_currency_analysis(curr, stats)
        print("\n🌏 EMERGING MARKET CURRENCIES:")
        print("=" * 60)
        for curr, stats in sorted_currencies:
            if stats['type'] == 'EM':
                self._print_enhanced_currency_analysis(curr, stats)
        self._print_enhanced_aggregate_analysis(currency_stats)






    # ----------------------------- Display Currney and Tenor Specific -----------------------------
    def data_ccyTenor_trades(self, currency_pair, tenor, start_date=None, end_date=None):
        trades_df = self.trades.copy()
        trades_df = trades_df[(trades_df['currency_pair'] == currency_pair) & 
                            (trades_df['tenor'] == tenor)]
        if len(trades_df) == 0:
            return pd.DataFrame()
        if start_date is not None or end_date is not None:
            if start_date is not None:
                if len(str(start_date)) == 7:  # Format: YYYY-MM
                    start_date = pd.to_datetime(start_date + '-01')
                else:
                    start_date = pd.to_datetime(start_date)
                trades_df = trades_df[trades_df['signal_date'] >= start_date]
            if end_date is not None:
                if len(str(end_date)) == 7:  # Format: YYYY-MM
                    end_date_temp = pd.to_datetime(end_date + '-01')
                    end_date = end_date_temp + pd.offsets.MonthEnd(0)
                else:
                    end_date = pd.to_datetime(end_date)
                trades_df = trades_df[trades_df['signal_date'] <= end_date]
        trades_df = trades_df.sort_values('signal_date').reset_index(drop=True)
        return trades_df


    def print_SingleccyTenoranalysis(self, currency_pair, tenor, start_date=None, end_date=None, 
                                    show_details=True ):
        export_csv=False
        trades_df = self.data_ccyTenor_trades(currency_pair, tenor, start_date, end_date)
        if len(trades_df) == 0:
            print(f"No trades found for {currency_pair} {tenor}")
            return
        date_filter_info = ""
        if start_date is not None:
            if len(str(start_date)) == 7:
                date_filter_info += f" FROM {start_date}"
            else:
                date_filter_info += f" FROM {pd.to_datetime(start_date).strftime('%Y-%m')}"
        if end_date is not None:
            if len(str(end_date)) == 7:
                date_filter_info += f" TO {end_date}"
            else:
                date_filter_info += f" TO {pd.to_datetime(end_date).strftime('%Y-%m')}"
        print("\n" + "="*80)
        print(f" TRADE ANALYSIS: {currency_pair} {tenor}")
        if date_filter_info:
            print(f" Date Range:{date_filter_info}")
        print(f" Weighting Method: {self.weighting_method_display}")
        print("="*80)
        total_trades = len(trades_df)
        overall_hit_rate = trades_df['trade_success'].mean() * 100
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Overall Hit Rate: {overall_hit_rate:.1f}%")
        print(f"  Average Vol Diff: {trades_df['vol_diff_pct'].mean() * 100:+.2f}%")
        exp_trades = trades_df[trades_df['signal_type'] == 'expensive']
        cheap_trades = trades_df[trades_df['signal_type'] == 'cheap']
        print("\n" + "-"*80)
        print(f"{'METRIC':<30} {'EXPENSIVE SIGNALS':>20} {'CHEAP SIGNALS':>20}")
        print("-"*80)
        metrics = []
        metrics.append(('Trade Count', len(exp_trades), len(cheap_trades), '.0f', ''))
        exp_hr = exp_trades['trade_success'].mean() * 100 if len(exp_trades) > 0 else np.nan
        cheap_hr = cheap_trades['trade_success'].mean() * 100 if len(cheap_trades) > 0 else np.nan
        metrics.append(('Hit Rate', exp_hr, cheap_hr, '.1f', '%'))
        exp_avg_diff = exp_trades['vol_diff_pct'].mean() * 100 if len(exp_trades) > 0 else np.nan
        cheap_avg_diff = cheap_trades['vol_diff_pct'].mean() * 100 if len(cheap_trades) > 0 else np.nan
        metrics.append(('Avg Vol Diff', exp_avg_diff, cheap_avg_diff, '+.2f', '%'))
        exp_std = exp_trades['vol_diff_pct'].std() * 100 if len(exp_trades) > 0 else np.nan
        cheap_std = cheap_trades['vol_diff_pct'].std() * 100 if len(cheap_trades) > 0 else np.nan
        metrics.append(('Vol Diff Std Dev', exp_std, cheap_std, '.2f', '%'))
        if len(exp_trades) > 0:
            exp_best = exp_trades['vol_diff_pct'].min() * 100  # Most negative
            exp_worst = exp_trades['vol_diff_pct'].max() * 100  # Most positive
        else:
            exp_best = exp_worst = np.nan
        if len(cheap_trades) > 0:
            cheap_best = cheap_trades['vol_diff_pct'].max() * 100  # Most positive
            cheap_worst = cheap_trades['vol_diff_pct'].min() * 100  # Most negative
        else:
            cheap_best = cheap_worst = np.nan
        metrics.append(('Best Trade', exp_best, cheap_best, '+.2f', '%'))
        metrics.append(('Worst Trade', exp_worst, cheap_worst, '+.2f', '%'))
        exp_wins = exp_trades[exp_trades['trade_success'] == True] if len(exp_trades) > 0 else pd.DataFrame()
        exp_losses = exp_trades[exp_trades['trade_success'] == False] if len(exp_trades) > 0 else pd.DataFrame()
        cheap_wins = cheap_trades[cheap_trades['trade_success'] == True] if len(cheap_trades) > 0 else pd.DataFrame()
        cheap_losses = cheap_trades[cheap_trades['trade_success'] == False] if len(cheap_trades) > 0 else pd.DataFrame()
        exp_avg_win = abs(exp_wins['vol_diff_pct'].mean() * 100) if len(exp_wins) > 0 else np.nan
        cheap_avg_win = cheap_wins['vol_diff_pct'].mean() * 100 if len(cheap_wins) > 0 else np.nan
        metrics.append(('Avg Win Size', exp_avg_win, cheap_avg_win, '.2f', '%'))
        exp_avg_loss = abs(exp_losses['vol_diff_pct'].mean() * 100) if len(exp_losses) > 0 else np.nan
        cheap_avg_loss = abs(cheap_losses['vol_diff_pct'].mean() * 100) if len(cheap_losses) > 0 else np.nan
        metrics.append(('Avg Loss Size', exp_avg_loss, cheap_avg_loss, '.2f', '%'))
        if len(exp_trades) > 0 and len(exp_wins) > 0 and len(exp_losses) > 0:
            exp_win_rate = len(exp_wins) / len(exp_trades)
            exp_expectancy = (exp_win_rate * exp_avg_win) - ((1 - exp_win_rate) * exp_avg_loss)
        else:
            exp_expectancy = np.nan
        if len(cheap_trades) > 0 and len(cheap_wins) > 0 and len(cheap_losses) > 0:
            cheap_win_rate = len(cheap_wins) / len(cheap_trades)
            cheap_expectancy = (cheap_win_rate * cheap_avg_win) - ((1 - cheap_win_rate) * cheap_avg_loss)
        else:
            cheap_expectancy = np.nan
        metrics.append(('Expectancy', exp_expectancy, cheap_expectancy, '+.2f', '%'))
        for metric_name, exp_val, cheap_val, fmt, suffix in metrics:
            if pd.isna(exp_val):
                exp_str = "N/A"
            else:
                exp_str = f"{exp_val:{fmt}}{suffix}"
            if pd.isna(cheap_val):
                cheap_str = "N/A"
            else:
                cheap_str = f"{cheap_val:{fmt}}{suffix}"
            print(f"{metric_name:<30} {exp_str:>20} {cheap_str:>20}")
        print("\n" + "-"*80)
        print("PERFORMANCE COMPARISON:")
        if not pd.isna(exp_hr) and not pd.isna(cheap_hr):
            hr_diff = exp_hr - cheap_hr
            print(f"  Hit Rate Difference (Exp - Cheap): {hr_diff:+.1f}%")
            if exp_hr > cheap_hr:
                print(f"  ✅ Expensive signals perform better by {hr_diff:.1f}%")
            else:
                print(f"  ⚠️ Cheap signals perform better by {abs(hr_diff):.1f}%")
        if show_details and total_trades > 0:
            print("\n" + "="*80)
            print("DETAILED TRADE LIST:")
            print("-"*80)
            vol_columns = []
            if 'current_vol' in trades_df.columns and 'future_vol' in trades_df.columns:
                vol_columns = ['current_vol', 'future_vol']
            elif 'spot_vol' in trades_df.columns and 'realized_vol' in trades_df.columns:
                vol_columns = ['spot_vol', 'realized_vol']
            elif 'initial_vol' in trades_df.columns and 'final_vol' in trades_df.columns:
                vol_columns = ['initial_vol', 'final_vol']
            for idx, (_, trade) in enumerate(trades_df.iterrows(), 1):
                print(f"\nTrade #{idx}:")
                print(f"  Date: {trade['signal_date'].strftime('%Y-%m-%d')} → {trade['future_date'].strftime('%Y-%m-%d')}")
                print(f"  Type: {trade['signal_type'].upper()} (Percentile: {trade['signal_percentile']:.1f})")

                if self.weighting_method == 'correlation' and 'corr_score' in trade.index:
                    print(f"  Correlation Score: {trade['corr_score']:.3f}")
                elif self.weighting_method in ['cointegration_1', 'cointegration_2'] and 'coint_score' in trade.index:
                    print(f"  Cointegration Score: {trade['coint_score']:.3f}")

                if len(vol_columns) == 2:
                    print(f"  Volatility: {trade[vol_columns[0]]:.1f}% → {trade[vol_columns[1]]:.1f}%")
                print(f"  Vol Change: {trade['vol_diff']:+.2f}% ({trade['vol_diff_pct']*100:+.2f}%)")
                result_emoji = "✅" if trade['trade_success'] else "❌"
                result_text = 'SUCCESS' if trade['trade_success'] else 'FAILURE'
                print(f"  Result: {result_emoji} {result_text}")
        if export_csv:
            filename = f"{currency_pair}_{tenor}_trades_{self.weighting_method}"
            if start_date:
                filename += f"_from_{start_date.replace('-', '')}"
            if end_date:
                filename += f"_to_{end_date.replace('-', '')}"
            filename += ".csv"
            trades_df.to_csv(filename, index=False)
            print(f"\n✅ Trades exported to: {filename}")
        print("\n" + "="*80)


    def get_MultipleccyTenorComparison_df(self, currency_pairs, tenor, start_date=None, end_date=None):
        if not isinstance(currency_pairs, list):
            currency_pairs = [currency_pairs]
        all_data = {}
        for currency_pair in currency_pairs:
            trades_df = self.data_ccyTenor_trades(currency_pair, tenor, start_date, end_date)
            if len(trades_df) == 0:
                all_data[(currency_pair, 'Expensive')] = {
                    'Trade Count': 0,
                    'Hit Rate (%)': np.nan,
                    'Avg Vol Diff (%)': np.nan,
                    'Vol Diff Std Dev (%)': np.nan,
                    'Best Trade (%)': np.nan,
                    'Worst Trade (%)': np.nan,
                    'Avg Win Size (%)': np.nan,
                    'Avg Loss Size (%)': np.nan,
                    'Expectancy (%)': np.nan}
                all_data[(currency_pair, 'Cheap')] = {
                    'Trade Count': 0,
                    'Hit Rate (%)': np.nan,
                    'Avg Vol Diff (%)': np.nan,
                    'Vol Diff Std Dev (%)': np.nan,
                    'Best Trade (%)': np.nan,
                    'Worst Trade (%)': np.nan,
                    'Avg Win Size (%)': np.nan,
                    'Avg Loss Size (%)': np.nan,
                    'Expectancy (%)': np.nan}
                continue
            exp_trades = trades_df[trades_df['signal_type'] == 'expensive']
            cheap_trades = trades_df[trades_df['signal_type'] == 'cheap']
            if len(exp_trades) > 0:
                exp_wins = exp_trades[exp_trades['trade_success'] == True]
                exp_losses = exp_trades[exp_trades['trade_success'] == False]
                exp_hit_rate = exp_trades['trade_success'].mean() * 100
                exp_avg_vol_diff = exp_trades['vol_diff_pct'].mean() * 100
                exp_vol_std = exp_trades['vol_diff_pct'].std() * 100
                exp_best_trade = exp_trades['vol_diff_pct'].min() * 100
                exp_worst_trade = exp_trades['vol_diff_pct'].max() * 100
                exp_avg_win = abs(exp_wins['vol_diff_pct'].mean() * 100) if len(exp_wins) > 0 else np.nan
                exp_avg_loss = abs(exp_losses['vol_diff_pct'].mean() * 100) if len(exp_losses) > 0 else np.nan
                if len(exp_wins) > 0 and len(exp_losses) > 0:
                    exp_win_rate = len(exp_wins) / len(exp_trades)
                    exp_expectancy = (exp_win_rate * exp_avg_win) - ((1 - exp_win_rate) * exp_avg_loss)
                else:
                    exp_expectancy = np.nan
                all_data[(currency_pair, 'Expensive')] = {
                    'Trade Count': len(exp_trades),
                    'Hit Rate (%)': round(exp_hit_rate, 2),
                    'Avg Vol Diff (%)': round(exp_avg_vol_diff, 2),
                    'Vol Diff Std Dev (%)': round(exp_vol_std, 2),
                    'Best Trade (%)': round(exp_best_trade, 2),
                    'Worst Trade (%)': round(exp_worst_trade, 2),
                    'Avg Win Size (%)': round(exp_avg_win, 2) if not pd.isna(exp_avg_win) else np.nan,
                    'Avg Loss Size (%)': round(exp_avg_loss, 2) if not pd.isna(exp_avg_loss) else np.nan,
                    'Expectancy (%)': round(exp_expectancy, 2) if not pd.isna(exp_expectancy) else np.nan}
            else:
                all_data[(currency_pair, 'Expensive')] = {
                    'Trade Count': 0,
                    'Hit Rate (%)': np.nan,
                    'Avg Vol Diff (%)': np.nan,
                    'Vol Diff Std Dev (%)': np.nan,
                    'Best Trade (%)': np.nan,
                    'Worst Trade (%)': np.nan,
                    'Avg Win Size (%)': np.nan,
                    'Avg Loss Size (%)': np.nan,
                    'Expectancy (%)': np.nan}
            if len(cheap_trades) > 0:
                cheap_wins = cheap_trades[cheap_trades['trade_success'] == True]
                cheap_losses = cheap_trades[cheap_trades['trade_success'] == False]
                cheap_hit_rate = cheap_trades['trade_success'].mean() * 100
                cheap_avg_vol_diff = cheap_trades['vol_diff_pct'].mean() * 100
                cheap_vol_std = cheap_trades['vol_diff_pct'].std() * 100
                cheap_best_trade = cheap_trades['vol_diff_pct'].max() * 100
                cheap_worst_trade = cheap_trades['vol_diff_pct'].min() * 100
                cheap_avg_win = cheap_wins['vol_diff_pct'].mean() * 100 if len(cheap_wins) > 0 else np.nan
                cheap_avg_loss = abs(cheap_losses['vol_diff_pct'].mean() * 100) if len(cheap_losses) > 0 else np.nan

                if len(cheap_wins) > 0 and len(cheap_losses) > 0:
                    cheap_win_rate = len(cheap_wins) / len(cheap_trades)
                    cheap_expectancy = (cheap_win_rate * cheap_avg_win) - ((1 - cheap_win_rate) * cheap_avg_loss)
                else:
                    cheap_expectancy = np.nan
                all_data[(currency_pair, 'Cheap')] = {
                    'Trade Count': len(cheap_trades),
                    'Hit Rate (%)': round(cheap_hit_rate, 2),
                    'Avg Vol Diff (%)': round(cheap_avg_vol_diff, 2),
                    'Vol Diff Std Dev (%)': round(cheap_vol_std, 2),
                    'Best Trade (%)': round(cheap_best_trade, 2),
                    'Worst Trade (%)': round(cheap_worst_trade, 2),
                    'Avg Win Size (%)': round(cheap_avg_win, 2) if not pd.isna(cheap_avg_win) else np.nan,
                    'Avg Loss Size (%)': round(cheap_avg_loss, 2) if not pd.isna(cheap_avg_loss) else np.nan,
                    'Expectancy (%)': round(cheap_expectancy, 2) if not pd.isna(cheap_expectancy) else np.nan}
            else:
                all_data[(currency_pair, 'Cheap')] = {
                    'Trade Count': 0,
                    'Hit Rate (%)': np.nan,
                    'Avg Vol Diff (%)': np.nan,
                    'Vol Diff Std Dev (%)': np.nan,
                    'Best Trade (%)': np.nan,
                    'Worst Trade (%)': np.nan,
                    'Avg Win Size (%)': np.nan,
                    'Avg Loss Size (%)': np.nan,
                    'Expectancy (%)': np.nan}
        df = pd.DataFrame(all_data)
        
        return df








    #------------------------------------------------------------------------------------------------------------------------------ 
   
    #------------------------------------------------------------------------------------------------------------------------------


    def data_monthlyBreakdown_GeneralPerformance(self, start_date=None, end_date=None):
        trades_df = self.trades.copy()
        trades_df['month'] = trades_df['signal_date'].dt.to_period('M')
        if start_date or end_date:
            original_count = len(trades_df)
            if start_date:
                start_period = pd.Period(start_date, freq='M')
                trades_df = trades_df[trades_df['month'] >= start_period]
            if end_date:
                end_period = pd.Period(end_date, freq='M')
                trades_df = trades_df[trades_df['month'] <= end_period]
            filtered_count = len(trades_df)
            if filtered_count == 0:
                print("No trades found in the specified date range!")
                return None
        months = sorted(trades_df['month'].unique())
        monthly_analysis = []
        for month in months:
            month_trades = trades_df[trades_df['month'] == month]
            month_str = str(month)
            total_trades = len(month_trades)
            overall_hit_rate = month_trades['trade_success'].mean() * 100
            exp_trades = month_trades[month_trades['signal_type'] == 'expensive']
            cheap_trades = month_trades[month_trades['signal_type'] == 'cheap']
            month_stats = {
                'Month': month_str,
                'Total_Trades': total_trades,
                'Overall_Hit_Rate': overall_hit_rate}
            if len(exp_trades) > 0:
                exp_metrics = self._calculate_signal_metrics(exp_trades, 'Exp')
                month_stats.update(exp_metrics)
            else:
                month_stats.update(self._get_empty_metrics('Exp'))
            if len(cheap_trades) > 0:
                cheap_metrics = self._calculate_signal_metrics(cheap_trades, 'Cheap')
                month_stats.update(cheap_metrics)
            else:
                month_stats.update(self._get_empty_metrics('Cheap'))
            top_currencies = month_trades.groupby('currency_pair')['trade_success'].agg(['mean', 'count'])
            top_currencies = top_currencies[top_currencies['count'] >= 3].sort_values('mean', ascending=False)
            if len(top_currencies) > 0:
                month_stats['Best_Currency'] = top_currencies.index[0]
                month_stats['Best_Currency_HR'] = top_currencies.iloc[0]['mean'] * 100
                if len(top_currencies) > 1:
                    month_stats['Worst_Currency'] = top_currencies.index[-1]
                    month_stats['Worst_Currency_HR'] = top_currencies.iloc[-1]['mean'] * 100
            tenor_performance = month_trades.groupby('tenor')['trade_success'].mean() * 100
            if len(tenor_performance) > 0:
                month_stats['Best_Tenor'] = tenor_performance.idxmax()
                month_stats['Best_Tenor_HR'] = tenor_performance.max()
            monthly_analysis.append(month_stats)
        monthly_df = pd.DataFrame(monthly_analysis)

        return monthly_df


    def _calculate_signal_metrics(self, trades, prefix):
        metrics = {}
        
        # Basic counts and rates
        metrics[f'{prefix}_Count'] = len(trades)
        metrics[f'{prefix}_Hit_Rate'] = trades['trade_success'].mean() * 100
        
        # Volatility difference metrics (raw)
        metrics[f'{prefix}_Avg_Vol_Diff'] = trades['vol_diff'].mean()
        metrics[f'{prefix}_Std_Vol_Diff'] = trades['vol_diff'].std()
        
        # Volatility difference metrics (percentage)
        metrics[f'{prefix}_Avg_Vol_Pct'] = trades['vol_diff_pct'].mean() * 100
        metrics[f'{prefix}_Std_Vol_Pct'] = trades['vol_diff_pct'].std() * 100
        
        # Extremes
        metrics[f'{prefix}_Max_Win'] = trades['vol_diff_pct'].max() * 100
        metrics[f'{prefix}_Max_Loss'] = trades['vol_diff_pct'].min() * 100
        
        # Signal strength
        metrics[f'{prefix}_Avg_Signal'] = trades['signal_percentile'].mean()
        metrics[f'{prefix}_Std_Signal'] = trades['signal_percentile'].std()
        
        # Win/Loss breakdown
        wins = trades[trades['trade_success'] == True]
        losses = trades[trades['trade_success'] == False]
        
        if len(wins) > 0:
            metrics[f'{prefix}_Win_Count'] = len(wins)
            metrics[f'{prefix}_Avg_Win'] = wins['vol_diff_pct'].mean() * 100
            metrics[f'{prefix}_Median_Win'] = wins['vol_diff_pct'].median() * 100
        else:
            metrics[f'{prefix}_Win_Count'] = 0
            metrics[f'{prefix}_Avg_Win'] = np.nan
            metrics[f'{prefix}_Median_Win'] = np.nan
        
        if len(losses) > 0:
            metrics[f'{prefix}_Loss_Count'] = len(losses)
            metrics[f'{prefix}_Avg_Loss'] = losses['vol_diff_pct'].mean() * 100
            metrics[f'{prefix}_Median_Loss'] = losses['vol_diff_pct'].median() * 100
        else:
            metrics[f'{prefix}_Loss_Count'] = 0
            metrics[f'{prefix}_Avg_Loss'] = np.nan
            metrics[f'{prefix}_Median_Loss'] = np.nan
        
        # Risk metrics
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(trades)
            metrics[f'{prefix}_Expectancy'] = (win_rate * abs(metrics[f'{prefix}_Avg_Win'])) - \
                                            ((1 - win_rate) * abs(metrics[f'{prefix}_Avg_Loss']))
            metrics[f'{prefix}_Payoff_Ratio'] = abs(metrics[f'{prefix}_Avg_Win']) / abs(metrics[f'{prefix}_Avg_Loss'])
        else:
            metrics[f'{prefix}_Expectancy'] = np.nan
            metrics[f'{prefix}_Payoff_Ratio'] = np.nan
        
        # Sharpe-like ratio
        mean_return = trades['vol_diff_pct'].mean()
        std_return = trades['vol_diff_pct'].std()
        metrics[f'{prefix}_Sharpe'] = (mean_return / std_return) if std_return > 0 else np.nan
        
        # Consecutive streaks
        max_wins, max_losses = self._max_consecutive_streaks(trades['trade_success'])
        metrics[f'{prefix}_Max_Win_Streak'] = max_wins
        metrics[f'{prefix}_Max_Loss_Streak'] = max_losses
        
        # Statistical significance
        if len(trades) > 0:
            try:
                pval = stats.binomtest(trades['trade_success'].sum(), 
                                    n=len(trades), 
                                    p=0.5, 
                                    alternative='greater').pvalue
                metrics[f'{prefix}_P_Value'] = pval
            except:
                metrics[f'{prefix}_P_Value'] = np.nan
        
        # Big winners/losers (>2% moves)
        metrics[f'{prefix}_Pct_Big_Winners'] = (trades['vol_diff_pct'] > 0.02).mean() * 100
        metrics[f'{prefix}_Pct_Big_Losers'] = (trades['vol_diff_pct'] < -0.02).mean() * 100
        
        return metrics

    def _get_empty_metrics(self, prefix):
        """Return empty metrics structure for when no trades exist."""
        return {
            f'{prefix}_Count': 0,
            f'{prefix}_Hit_Rate': np.nan,
            f'{prefix}_Avg_Vol_Diff': np.nan,
            f'{prefix}_Std_Vol_Diff': np.nan,
            f'{prefix}_Avg_Vol_Pct': np.nan,
            f'{prefix}_Std_Vol_Pct': np.nan,
            f'{prefix}_Max_Win': np.nan,
            f'{prefix}_Max_Loss': np.nan,
            f'{prefix}_Avg_Signal': np.nan,
            f'{prefix}_Std_Signal': np.nan,
            f'{prefix}_Win_Count': 0,
            f'{prefix}_Avg_Win': np.nan,
            f'{prefix}_Median_Win': np.nan,
            f'{prefix}_Loss_Count': 0,
            f'{prefix}_Avg_Loss': np.nan,
            f'{prefix}_Median_Loss': np.nan,
            f'{prefix}_Expectancy': np.nan,
            f'{prefix}_Payoff_Ratio': np.nan,
            f'{prefix}_Sharpe': np.nan,
            f'{prefix}_Max_Win_Streak': 0,
            f'{prefix}_Max_Loss_Streak': 0,
            f'{prefix}_P_Value': np.nan,
            f'{prefix}_Pct_Big_Winners': np.nan,
            f'{prefix}_Pct_Big_Losers': np.nan
        }

    def _max_consecutive_streaks(self, success_bool_series):
        """Calculate maximum consecutive winning and losing streaks."""
        max_w = max_l = cur_w = cur_l = 0
        for s in success_bool_series.astype(bool):
            if s:
                cur_w += 1
                max_w = max(max_w, cur_w)
                cur_l = 0
            else:
                cur_l += 1
                max_l = max(max_l, cur_l)
                cur_w = 0
        return max_w, max_l

    def _print_detailed_month_analysis(self, month_data):
        """Print detailed analysis for a single month with side-by-side comparison."""
        print(f"\n{'='*80}")
        print(f" {month_data['Month']}")
        print(f"{'='*80}")
        
        # Overall performance header
        overall_hr = month_data.get('Overall_Hit_Rate', np.nan)
        total_trades = month_data.get('Total_Trades', 0)
        
        # Performance grade
        if overall_hr >= 80:
            grade = "A+ 🔥"; status = "EXCEPTIONAL"
        elif overall_hr >= 70:
            grade = "A ✅"; status = "STRONG"
        elif overall_hr >= 60:
            grade = "B ⚠️"; status = "MODERATE"  
        elif overall_hr >= 50:
            grade = "C 📊"; status = "WEAK"
        else:
            grade = "D ❌"; status = "POOR"
        
        print(f"\nOVERALL: {status} (Grade: {grade}) | Total Trades: {total_trades} | Hit Rate: {overall_hr:.1f}%")
        
        # Best performers if available
        if 'Best_Currency' in month_data:
            print(f"Best Performer: {month_data['Best_Currency']} ({month_data['Best_Currency_HR']:.1f}%) | ", end="")
        if 'Best_Tenor' in month_data:
            print(f"Best Tenor: {month_data['Best_Tenor']} ({month_data['Best_Tenor_HR']:.1f}%)")
        
        # Side-by-side comparison table
        print("\n" + "-"*80)
        print(f"{'METRIC':<25} {'EXPENSIVE VOL':>25} {'CHEAP VOL':>25}")
        print("-"*80)
        
        # Define metrics to display with formatting
        metrics = [
            ('VOLUME', '', ''),
            ('Trade Count', 'Exp_Count', 'Cheap_Count', '.0f', ''),
            ('Hit Rate', 'Exp_Hit_Rate', 'Cheap_Hit_Rate', '.1f', '%'),
            ('', '', ''),
            ('PROFITABILITY', '', ''),
            ('Avg Vol Diff', 'Exp_Avg_Vol_Pct', 'Cheap_Avg_Vol_Pct', '+.2f', '%'),
            ('Vol Std Dev', 'Exp_Std_Vol_Pct', 'Cheap_Std_Vol_Pct', '.2f', '%'),
            ('Avg Win', 'Exp_Avg_Win', 'Cheap_Avg_Win', '+.2f', '%'),
            ('Avg Loss', 'Exp_Avg_Loss', 'Cheap_Avg_Loss', '+.2f', '%'),
            ('Median Win', 'Exp_Median_Win', 'Cheap_Median_Win', '+.2f', '%'),
            ('Median Loss', 'Exp_Median_Loss', 'Cheap_Median_Loss', '+.2f', '%'),
            ('', '', ''),
            ('EXTREMES', '', ''),
            ('Max Vol Return', 'Exp_Max_Win', 'Cheap_Max_Win', '+.2f', '%'),
            ('Min Vol Return', 'Exp_Max_Loss', 'Cheap_Max_Loss', '+.2f', '%'),
            ('', '', ''),
            ('RISK METRICS', '', ''),
            ('Expectancy', 'Exp_Expectancy', 'Cheap_Expectancy', '+.2f', '%'),
            ('Payoff Ratio', 'Exp_Payoff_Ratio', 'Cheap_Payoff_Ratio', '.2f', ':1'),
            ('Sharpe Ratio', 'Exp_Sharpe', 'Cheap_Sharpe', '.3f', ''),
            ('P-value (vs 50%)', 'Exp_P_Value', 'Cheap_P_Value', '.4f', ''),
            ('', '', ''),
            ('STREAKS', '', ''),
            ('Max Win Streak', 'Exp_Max_Win_Streak', 'Cheap_Max_Win_Streak', '.0f', ''),
            ('Max Loss Streak', 'Exp_Max_Loss_Streak', 'Cheap_Max_Loss_Streak', '.0f', ''),
            ('', '', ''),
            ('SIGNALS', '', ''),
            ('Avg Signal Strength', 'Exp_Avg_Signal', 'Cheap_Avg_Signal', '.1f', 'th'),
            ('Signal Std Dev', 'Exp_Std_Signal', 'Cheap_Std_Signal', '.1f', ''),
            ('', '', ''),
            ('BIG MOVES', '', ''),
            ('% Trades >2%', 'Exp_Pct_Big_Winners', 'Cheap_Pct_Big_Winners', '.1f', '%'),
            ('% Trades <-2%', 'Exp_Pct_Big_Losers', 'Cheap_Pct_Big_Losers', '.1f', '%'),
        ]
        
        for metric in metrics:
            if len(metric) == 3:  # Section header or separator
                metric_name, _, _ = metric
                if metric_name:
                    print(f"\n{metric_name}")
                    print("-"*80)
            else:  # Data row
                metric_name, exp_key, cheap_key, fmt, suffix = metric
                
                # Get values
                exp_val = month_data.get(exp_key, np.nan)
                cheap_val = month_data.get(cheap_key, np.nan)
                
                # Format values
                if pd.isna(exp_val):
                    exp_str = "N/A"
                elif suffix == ':1' and not pd.isna(exp_val):
                    exp_str = f"{exp_val:{fmt}}:1"
                else:
                    exp_str = f"{exp_val:{fmt}}{suffix}"
                
                if pd.isna(cheap_val):
                    cheap_str = "N/A"
                elif suffix == ':1' and not pd.isna(cheap_val):
                    cheap_str = f"{cheap_val:{fmt}}:1"
                else:
                    cheap_str = f"{cheap_val:{fmt}}{suffix}"
                
                # Add indicators for notable values
                if 'Hit_Rate' in exp_key and not pd.isna(exp_val):
                    if exp_val >= 80:
                        exp_str = "🔥 " + exp_str
                    elif exp_val >= 70:
                        exp_str = "✅ " + exp_str
                
                if 'Hit_Rate' in cheap_key and not pd.isna(cheap_val):
                    if cheap_val >= 60:
                        cheap_str = "✅ " + cheap_str
                    elif cheap_val >= 50:
                        cheap_str = "⚠️ " + cheap_str
                
                print(f"{metric_name:<25} {exp_str:>25} {cheap_str:>25}")

 
































    #------------------------------------------------------------------------------------------------------------------------------ 
    #------------------------------------------------------------------------------------------------------------------------------
    def analyze_performance_cycles(self):
        """Identify cyclical patterns and regime changes in strategy performance."""
        self.print_section_header("PERFORMANCE CYCLES & REGIME ANALYSIS")
        trades_df = self.trades.copy()
        trades_df = trades_df.sort_values('signal_date')
        window_sizes = [20, 50, 100]  # Number of trades
        print("\n📊 ROLLING PERFORMANCE ANALYSIS")
        print("-"*60)
        for window in window_sizes:
            if len(trades_df) < window:
                continue
            trades_df[f'rolling_hr_{window}'] = trades_df['trade_success'].rolling(
                window=window, min_periods=window//2).mean() * 100 # Rolling hit rate
            # Separate by signal type
            for signal_type in ['expensive', 'cheap']:
                signal_trades = trades_df[trades_df['signal_type'] == signal_type].copy()
                if len(signal_trades) >= window:
                    signal_trades[f'rolling_hr_{window}'] = signal_trades['trade_success'].rolling(
                        window=min(window, len(signal_trades)), min_periods=min(window//2, len(signal_trades)//2)).mean() * 100
                    rolling_hr = signal_trades[f'rolling_hr_{window}'].dropna()
                    if len(rolling_hr) > 0:
                        print(f"\n{window}-Trade Rolling Window ({signal_type.upper()}):")
                        print(f"  Current Performance: {rolling_hr.iloc[-1]:.1f}%")
                        print(f"  Historical Range: {rolling_hr.min():.1f}% - {rolling_hr.max():.1f}%")
                        percentiles = [25, 50, 75]
                        thresholds = np.percentile(rolling_hr, percentiles)
                        print(f"  Performance Quartiles:")
                        for p, t in zip(percentiles, thresholds):
                            print(f"    {p}th percentile: {t:.1f}%")
        
        self._detect_regime_changes(trades_df) # Detect regime changes
        self._analyze_performance_trends(trades_df) # Analyze performance trends


    def _detect_regime_changes(self, trades_df):
        """Detect significant changes in strategy performance regimes."""
        print("\n\n🔄 REGIME CHANGE DETECTION")
        print("-"*60)
        window = 50
        if len(trades_df) < window:
            print("Insufficient data for regime detection")
            return
        
        # Overall regime changes
        trades_df['rolling_hr'] = trades_df['trade_success'].rolling(
            window=window, min_periods=window//2).mean() * 100
        trades_df['hr_change'] = trades_df['rolling_hr'].diff()
        
        threshold = 10
        significant_changes = trades_df[abs(trades_df['hr_change']) > threshold].copy()
        
        if len(significant_changes) > 0:
            print(f"\nSignificant OVERALL regime changes (>{threshold}% change):")
            for _, row in significant_changes.tail(5).iterrows():
                date = row['signal_date'].strftime('%Y-%m-%d')
                change = row['hr_change']
                new_hr = row['rolling_hr']
                direction = "IMPROVEMENT ↑" if change > 0 else "DETERIORATION ↓"
                print(f"  {date}: {direction} {abs(change):.1f}% (new level: {new_hr:.1f}%)")
        
        # Regime changes by signal type
        for signal_type in ['expensive', 'cheap']:
            signal_trades = trades_df[trades_df['signal_type'] == signal_type].copy()
            if len(signal_trades) >= window:
                signal_trades['rolling_hr'] = signal_trades['trade_success'].rolling(
                    window=min(window, len(signal_trades)), 
                    min_periods=min(window//2, len(signal_trades)//2)).mean() * 100
                signal_trades['hr_change'] = signal_trades['rolling_hr'].diff()
                
                sig_changes = signal_trades[abs(signal_trades['hr_change']) > threshold]
                if len(sig_changes) > 0:
                    print(f"\n{signal_type.upper()} regime changes:")
                    for _, row in sig_changes.tail(3).iterrows():
                        date = row['signal_date'].strftime('%Y-%m-%d')
                        change = row['hr_change']
                        new_hr = row['rolling_hr']
                        direction = "IMPROVEMENT ↑" if change > 0 else "DETERIORATION ↓"
                        print(f"  {date}: {direction} {abs(change):.1f}% (new: {new_hr:.1f}%)")

    def _analyze_performance_trends(self, trades_df):
        """Analyze long-term performance trends."""
        print("\n\n📈 TREND ANALYSIS")
        print("-"*60)
        
        trades_df['month'] = trades_df['signal_date'].dt.to_period('M')
        
        # Overall trend
        monthly_performance = trades_df.groupby('month')['trade_success'].mean() * 100
        
        if len(monthly_performance) < 3:
            print("Insufficient data for trend analysis")
            return
        
        # Linear regression for overall trend
        x = np.arange(len(monthly_performance))
        y = monthly_performance.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        print(f"\nOVERALL Linear Trend:")
        print(f"  Direction: {'IMPROVING' if slope > 0 else 'DETERIORATING'}")
        print(f"  Monthly Change: {slope:.2f}% per month")
        print(f"  R-squared: {r_value**2:.3f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.3f})")
        
        # Trend by signal type
        for signal_type in ['expensive', 'cheap']:
            signal_monthly = trades_df[trades_df['signal_type'] == signal_type].groupby('month')['trade_success'].mean() * 100
            
            if len(signal_monthly) >= 3:
                x_sig = np.arange(len(signal_monthly))
                y_sig = signal_monthly.values
                slope_sig, _, r_sig, p_sig, _ = stats.linregress(x_sig, y_sig)
                
                print(f"\n{signal_type.upper()} Trend:")
                print(f"  Direction: {'IMPROVING' if slope_sig > 0 else 'DETERIORATING'}")
                print(f"  Monthly Change: {slope_sig:.2f}% per month")
                print(f"  R-squared: {r_sig**2:.3f}")
        
        # Recent vs historical
        if len(monthly_performance) >= 6:
            recent_3m = monthly_performance.tail(3).mean()
            historical = monthly_performance.iloc[:-3].mean()
            change = recent_3m - historical
            
            print(f"\n📊 Recent vs Historical:")
            print(f"  Last 3 months: {recent_3m:.1f}%")
            print(f"  Historical avg: {historical:.1f}%")
            print(f"  Difference: {change:+.1f}%")
            
            if abs(change) > 5:
                if change > 0:
                    print("  📈 Strategy showing significant recent improvement")
                else:
                    print("  📉 Strategy showing significant recent deterioration")

    def analyze_drawdown_periods(self):
        """Analyze periods of poor performance and recovery patterns."""
        self.print_section_header("DRAWDOWN & RECOVERY ANALYSIS")
        
        trades_df = self.trades.copy()
        trades_df = trades_df.sort_values('signal_date')
        
        # Overall drawdown analysis
        print("\n📉 OVERALL DRAWDOWN ANALYSIS")
        print("-"*60)
        
        trades_df['cumulative_wins'] = trades_df['trade_success'].cumsum()
        trades_df['cumulative_trades'] = range(1, len(trades_df) + 1)
        trades_df['cumulative_hr'] = trades_df['cumulative_wins'] / trades_df['cumulative_trades'] * 100
        trades_df['peak_hr'] = trades_df['cumulative_hr'].cummax()
        trades_df['drawdown'] = trades_df['cumulative_hr'] - trades_df['peak_hr']
        
        current_dd = trades_df['drawdown'].iloc[-1]
        max_dd = trades_df['drawdown'].min()
        
        print(f"\nCurrent Status:")
        print(f"  Current Drawdown: {current_dd:.1f}%")
        print(f"  Maximum Historical Drawdown: {max_dd:.1f}%")
        
        if current_dd < -1:
            dd_start_idx = trades_df[trades_df['drawdown'] == 0].index[-1] if any(trades_df['drawdown'] == 0) else 0
            dd_start_date = trades_df.loc[dd_start_idx, 'signal_date']
            days_in_dd = (trades_df['signal_date'].iloc[-1] - dd_start_date).days
            print(f"  ⚠️ Currently in drawdown for {days_in_dd} days")
            print(f"  Drawdown started: {dd_start_date.strftime('%Y-%m-%d')}")
        else:
            print(f"  ✅ Strategy at or near peak performance")
        
        # Drawdown by signal type
        for signal_type in ['expensive', 'cheap']:
            signal_trades = trades_df[trades_df['signal_type'] == signal_type].copy()
            if len(signal_trades) > 10:
                signal_trades = signal_trades.reset_index(drop=True)
                signal_trades['cumulative_wins'] = signal_trades['trade_success'].cumsum()
                signal_trades['cumulative_trades'] = range(1, len(signal_trades) + 1)
                signal_trades['cumulative_hr'] = signal_trades['cumulative_wins'] / signal_trades['cumulative_trades'] * 100
                signal_trades['peak_hr'] = signal_trades['cumulative_hr'].cummax()
                signal_trades['drawdown'] = signal_trades['cumulative_hr'] - signal_trades['peak_hr']
                
                current_dd_sig = signal_trades['drawdown'].iloc[-1]
                max_dd_sig = signal_trades['drawdown'].min()
                
                print(f"\n{signal_type.upper()} Drawdown:")
                print(f"  Current: {current_dd_sig:.1f}%")
                print(f"  Maximum: {max_dd_sig:.1f}%")
        
        # Identify significant drawdown periods
        self._identify_drawdown_periods(trades_df)
        
        # Recovery analysis
        self._analyze_recovery_patterns(trades_df)

    def _analyze_recovery_patterns(self, trades_df):
        """Analyze how quickly the strategy recovers from drawdowns."""
        print("\n\n🔄 RECOVERY PATTERN ANALYSIS")
        print("-"*60)
        
        # Analyze recovery velocity after losses
        losing_streaks = []
        winning_streaks = []
        current_streak = 0
        streak_type = None
        
        for success in trades_df['trade_success']:
            if success:
                if streak_type == 'loss' and current_streak > 0:
                    losing_streaks.append(current_streak)
                current_streak = 1 if streak_type != 'win' else current_streak + 1
                streak_type = 'win'
            else:
                if streak_type == 'win' and current_streak > 0:
                    winning_streaks.append(current_streak)
                current_streak = 1 if streak_type != 'loss' else current_streak + 1
                streak_type = 'loss'
        
        if losing_streaks:
            print(f"\nLosing Streak Statistics:")
            print(f"  Average Length: {np.mean(losing_streaks):.1f} trades")
            print(f"  Maximum Length: {max(losing_streaks)} trades")
            print(f"  Total Streaks: {len(losing_streaks)}")
        
        if winning_streaks:
            print(f"\nWinning Streak Statistics:")
            print(f"  Average Length: {np.mean(winning_streaks):.1f} trades")
            print(f"  Maximum Length: {max(winning_streaks)} trades")
            print(f"  Total Streaks: {len(winning_streaks)}")
        
        # Analyze bounce-back rate after losses
        print("\n📈 Bounce-Back Analysis:")
        
        for signal_type in ['expensive', 'cheap']:
            signal_trades = trades_df[trades_df['signal_type'] == signal_type].copy()
            if len(signal_trades) > 20:
                losses_idx = signal_trades[~signal_trades['trade_success']].index
                bounce_backs = 0
                
                for idx in losses_idx:
                    if idx + 1 in signal_trades.index:
                        if signal_trades.loc[idx + 1, 'trade_success']:
                            bounce_backs += 1
                
                if len(losses_idx) > 0:
                    bounce_rate = bounce_backs / len(losses_idx) * 100
                    print(f"  {signal_type.upper()}: {bounce_rate:.1f}% immediate recovery after loss")

    def analyze_seasonal_patterns(self):
        self.print_section_header("SEASONAL & CALENDAR EFFECTS")
        trades_df = self.trades.copy()
        trades_df['month'] = trades_df['signal_date'].dt.month
        trades_df['quarter'] = trades_df['signal_date'].dt.quarter
        trades_df['day_of_week'] = trades_df['signal_date'].dt.dayofweek
        trades_df['week_of_month'] = (trades_df['signal_date'].dt.day - 1) // 7 + 1
        
        print("\n📅 MONTHLY SEASONALITY")
        print("-"*60)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Overall monthly performance
        monthly_stats = trades_df.groupby('month').agg({
            'trade_success': ['mean', 'count'],
            'vol_diff_pct': 'mean'
        })
        monthly_stats.columns = ['hit_rate', 'count', 'avg_diff']
        monthly_stats['hit_rate'] *= 100
        monthly_stats['avg_diff'] *= 100
        
        print("\nOverall Performance by Calendar Month:")
        for month_num in range(1, 13):
            if month_num in monthly_stats.index:
                row = monthly_stats.loc[month_num]
                if row['count'] >= 10:
                    month_name = months[month_num - 1]
                    status = "🟢" if row['hit_rate'] >= 70 else "🟡" if row['hit_rate'] >= 60 else "🔴"
                    print(f"  {status} {month_name}: {row['hit_rate']:.1f}% "
                        f"({row['count']:.0f} trades, {row['avg_diff']:+.2f}% avg)")
        
        # Performance by signal type and month
        print("\n📊 SIGNAL TYPE SEASONALITY")
        print("-"*60)
        
        for signal_type in ['expensive', 'cheap']:
            signal_monthly = trades_df[trades_df['signal_type'] == signal_type].groupby('month')['trade_success'].agg(['mean', 'count'])
            signal_monthly['mean'] *= 100
            
            print(f"\n{signal_type.upper()} by Month:")
            best_months = []
            worst_months = []
            
            for month_num in range(1, 13):
                if month_num in signal_monthly.index:
                    row = signal_monthly.loc[month_num]
                    if row['count'] >= 5:
                        if row['mean'] >= (80 if signal_type == 'expensive' else 60):
                            best_months.append(months[month_num - 1])
                        elif row['mean'] < (60 if signal_type == 'expensive' else 40):
                            worst_months.append(months[month_num - 1])
            
            if best_months:
                print(f"  Best months: {', '.join(best_months)}")
            if worst_months:
                print(f"  Worst months: {', '.join(worst_months)}")
        
        # Quarterly analysis
        print("\n\n📊 QUARTERLY PATTERNS")
        print("-"*60)
        
        quarterly_stats = trades_df.groupby(['quarter', 'signal_type']).agg({
            'trade_success': ['mean', 'count']
        })
        
        for quarter in range(1, 5):
            print(f"\nQ{quarter}:")
            for signal_type in ['expensive', 'cheap']:
                if (quarter, signal_type) in quarterly_stats.index:
                    stats = quarterly_stats.loc[(quarter, signal_type)]
                    hr = stats[('trade_success', 'mean')] * 100
                    count = stats[('trade_success', 'count')]
                    if count >= 10:
                        print(f"  {signal_type.capitalize()}: {hr:.1f}% ({count:.0f} trades)")
        
        # Day of week effects
        print("\n\n📆 DAY-OF-WEEK EFFECTS")
        print("-"*60)
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dow_stats = trades_df[trades_df['day_of_week'] < 5].groupby('day_of_week').agg({
            'trade_success': ['mean', 'count']
        })
        
        print("\nPerformance by Day of Week:")
        for dow in range(5):
            if dow in dow_stats.index:
                stats = dow_stats.loc[dow]
                hr = stats[('trade_success', 'mean')] * 100
                count = stats[('trade_success', 'count')]
                if count >= 10:
                    status = "🟢" if hr >= 70 else "🟡" if hr >= 60 else "🔴"
                    print(f"  {status} {days[dow]}: {hr:.1f}% ({count:.0f} trades)")

    def run_temporal_analysis_suite(self, start_date=None, end_date=None):
        """Run complete temporal analysis suite."""
        print("\n" + "="*80)
        print(" COMPLETE TEMPORAL ANALYSIS SUITE")
        print("="*80)
        
        # 1. Monthly detailed breakdown
        monthly_df = self.analyze_temporal_patterns_detailed(start_date, end_date)
        
        # 2. Performance cycles and regimes
        self.analyze_performance_cycles()
        
        # 3. Drawdown analysis
        self.analyze_drawdown_periods()
        
        # 4. Seasonal patterns
        self.analyze_seasonal_patterns()
        
        print("\n" + "="*80)
        print(" END OF TEMPORAL ANALYSIS")
        print("="*80)
        
        return monthly_df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
















def create_visual_analysis(analyzer):
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Volatility Screener Strategy Analysis ({analyzer.weighting_method})', fontsize=16, fontweight='bold')
    trades_df = analyzer.trades.copy()
    ax = axes[0, 0]
    signal_performance = trades_df.groupby('signal_type')['trade_success'].mean() * 100
    bars = ax.bar(signal_performance.index, signal_performance.values, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Hit Rate by Signal Type', fontweight='bold')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, signal_performance.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    ax = axes[0, 1] # 2. Performance by Currency Pair
    currency_exp = trades_df[trades_df['signal_type'] == 'expensive'].groupby('currency_pair')['trade_success'].mean() * 100
    currency_exp = currency_exp.sort_values(ascending=False)
    colors = ['#2ecc71' if pair in analyzer.em_pairs else '#3498db' for pair in currency_exp.index]
    bars = ax.barh(range(len(currency_exp)), currency_exp.values, color=colors)
    ax.set_yticks(range(len(currency_exp)))
    ax.set_yticklabels(currency_exp.index, fontsize=9)
    ax.set_title('Expensive Vol Hit Rate by Currency', fontweight='bold')
    ax.set_xlabel('Hit Rate (%)')
    ax.set_xlim(0, 105)
    legend_elements = [Patch(facecolor='#2ecc71', label='EM'),
                      Patch(facecolor='#3498db', label='G10')]
    ax.legend(handles=legend_elements, loc='lower right')
    ax = axes[0, 2] # 3. Performance by Tenor
    tenor_data = []
    for tenor in trades_df['tenor'].unique().tolist():
        for signal_type in ['expensive', 'cheap']:
            subset = trades_df[(trades_df['tenor'] == tenor) & (trades_df['signal_type'] == signal_type)]
            if len(subset) > 0:
                tenor_data.append({
                    'Tenor': tenor,
                    'Type': signal_type.capitalize(),
                    'Hit Rate': subset['trade_success'].mean() * 100})
    tenor_df = pd.DataFrame(tenor_data)
    pivot_tenor = tenor_df.pivot(index='Tenor', columns='Type', values='Hit Rate')
    pivot_tenor.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
    ax.set_title('Hit Rate by Tenor and Signal Type', fontweight='bold')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_xlabel('Tenor')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Signal Type')
    ax.set_ylim(0, 100)
    ax = axes[1, 0] # 4. Percentile vs Hit Rate (Expensive)
    exp_trades = trades_df[trades_df['signal_type'] == 'expensive'].copy()
    exp_trades['percentile_bin'] = pd.cut(exp_trades['signal_percentile'], 
                                          bins=[85, 87, 90, 93, 95, 100],
                                          labels=['85-87', '87-90', '90-93', '93-95', '95-100'])
    percentile_performance = exp_trades.groupby('percentile_bin')['trade_success'].mean() * 100
    bars = ax.bar(range(len(percentile_performance)), percentile_performance.values, color='#2ecc71')
    ax.set_xticks(range(len(percentile_performance)))
    ax.set_xticklabels(percentile_performance.index)
    ax.set_title('Expensive Vol: Hit Rate by Percentile Band', fontweight='bold')
    ax.set_xlabel('Percentile Range')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_ylim(0, 100)
    for i, (bar, val) in enumerate(zip(bars, percentile_performance.values)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    ax = axes[1, 1] # 5. Volatility Difference Distribution
    exp_vols = trades_df[trades_df['signal_type'] == 'expensive']['vol_diff_pct'] * 100
    cheap_vols = trades_df[trades_df['signal_type'] == 'cheap']['vol_diff_pct'] * 100
    ax.hist([exp_vols, cheap_vols], bins=30, alpha=0.7, label=['Expensive', 'Cheap'], 
            color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_title('Distribution of Volatility Differences', fontweight='bold')
    ax.set_xlabel('Vol Diff (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.text(0.02, 0.95, f'Expensive mean: {exp_vols.mean():.1f}%', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.90, f'Cheap mean: {cheap_vols.mean():.1f}%', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax = axes[1, 2] # 6. Monthly Performance Trend
    trades_df['month'] = pd.to_datetime(trades_df['signal_date']).dt.to_period('M')
    monthly_exp = trades_df[trades_df['signal_type'] == 'expensive'].groupby('month')['trade_success'].mean() * 100
    monthly_cheap = trades_df[trades_df['signal_type'] == 'cheap'].groupby('month')['trade_success'].mean() * 100
    months_str = [str(m) for m in monthly_exp.index]
    ax.plot(range(len(monthly_exp)), monthly_exp.values, marker='o', linewidth=2, 
            label='Expensive', color='#2ecc71', markersize=8)
    if len(monthly_cheap) > 0:
        ax.plot(range(len(monthly_cheap)), monthly_cheap.values, marker='s', linewidth=2, 
                label='Cheap', color='#e74c3c', markersize=8)
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels([m[-2:] for m in months_str], rotation=45)
    ax.set_title('Monthly Hit Rate Trend', fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Hit Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'volatility_screener_analysis_{analyzer.weighting_method.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Visualization saved as 'volatility_screener_analysis_{analyzer.weighting_method.lower().replace(' ', '_')}.png'")


def generate_production_config(analyzer):
    """Generate production-ready configuration based on analysis."""
    trades_df = analyzer.trades.copy()
    config = {
        'weighting_method': analyzer.weighting_method,
        'thresholds': {
            'expensive': {},
            'cheap': {}
        },
        'position_sizing': {},
        'blacklist': [],
        'special_rules': {}}
    for currency in trades_df['currency_pair'].unique():
        curr_trades = trades_df[trades_df['currency_pair'] == currency]
        exp_trades = curr_trades[curr_trades['signal_type'] == 'expensive']
        cheap_trades = curr_trades[curr_trades['signal_type'] == 'cheap']
        if len(exp_trades) >= 10:
            if exp_trades['trade_success'].mean() >= 0.9:
                config['thresholds']['expensive'][currency] = 80  # More aggressive
            elif exp_trades['trade_success'].mean() >= 0.75:
                config['thresholds']['expensive'][currency] = 85  # Standard
            else:
                config['thresholds']['expensive'][currency] = 90  # Conservative
        if len(cheap_trades) >= 10:
            if cheap_trades['trade_success'].mean() >= 0.6:
                config['thresholds']['cheap'][currency] = 10  # Decent performance
            elif cheap_trades['trade_success'].mean() >= 0.45:
                config['thresholds']['cheap'][currency] = 5   # Tighter threshold
            else:
                config['blacklist'].append(f"{currency}_cheap")  # Don't trade
        if len(exp_trades) >= 20:
            win_rate = exp_trades['trade_success'].mean()
            if win_rate >= 0.9:
                config['position_sizing'][f"{currency}_expensive"] = 1.5  # 150% of base size
            elif win_rate >= 0.75:
                config['position_sizing'][f"{currency}_expensive"] = 1.0  # 100% of base size
            else:
                config['position_sizing'][f"{currency}_expensive"] = 0.5  # 50% of base size
    for currency in trades_df['currency_pair'].unique():
        curr_trades = trades_df[trades_df['currency_pair'] == currency]
        exp_hr = curr_trades[curr_trades['signal_type'] == 'expensive']['trade_success'].mean()
        cheap_hr = curr_trades[curr_trades['signal_type'] == 'cheap']['trade_success'].mean()
        if cheap_hr > exp_hr + 0.1 and len(curr_trades) >= 20:
            config['special_rules'][currency] = 'INVERTED'
    return config
















