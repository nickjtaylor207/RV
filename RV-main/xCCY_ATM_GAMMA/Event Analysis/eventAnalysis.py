
from xbbg import blp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from get_events import EconomicDataManager



class EventAnalyzer:
    WEIGHTING_METHODS = {
        'equal': 'Trades_EqualWeights',
        'correlation': 'Trades_CorrWeights',
        'cointegration': 'Trades_CointWeights'}
    def __init__(self, excel_file_path, weighting_method, trades_data=None, summary_data=None):
        self.weighting_method = weighting_method.lower()
        if self.weighting_method not in self.WEIGHTING_METHODS:
            available_methods = list(self.WEIGHTING_METHODS.keys())
            raise ValueError(f"Invalid weighting method '{weighting_method}'. Available methods: {available_methods}")
        if trades_data is not None:
            self.trades = trades_data.copy()
            self.summary = summary_data
            self.source_file = "provided_data"
        else:
            file_path = excel_file_path or self.DEFAULT_FILE_PATH
            self._load_data_from_excel(file_path)
        self.g10_pairs = self._get_g10_pairs()
        self._prepare_data()
        print(f"EventAnalyzer initialized with {self.weighting_method} weighting")
        print(f"Loaded {len(self.trades)} trades from {self.source_file}")
        print(f"Date range: {self.trades[self.date_col].min()} to {self.trades[self.date_col].max()}")
        print(f"Available currencies: {sorted(self.trades['currency_pair'].unique())}")
        print(f"Available tenors: {sorted(self.trades['tenor'].unique())}")
    def _load_data_from_excel(self, excel_file_path):
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
        self.source_file = excel_file_path
        sheet_name = self.WEIGHTING_METHODS[self.weighting_method]
        try:
            print(f"Loading {self.weighting_method} weighting data from sheet: {sheet_name}")
            self.trades = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            if self.trades is None or len(self.trades) == 0:
                raise ValueError(f"No data found in sheet '{sheet_name}'")
            print(f"Successfully loaded {len(self.trades)} trades")
            required_columns = ['currency_pair', 'tenor', 'signal_type', 'trade_success', 'vol_diff_pct']
            missing_columns = [col for col in required_columns if col not in self.trades.columns]
            if missing_columns:
                print(f"Warning: Missing expected columns: {missing_columns}")
            self._clean_trades_data()
            try:
                self.summary = pd.read_excel(excel_file_path, sheet_name='summary')
                print(f"Summary data loaded with shape: {self.summary.shape}")
            except:
                print("No summary data found - continuing without it")
                self.summary = None
        except Exception as e:
            print(f"Error loading data from Excel: {e}")
            raise
    def _clean_trades_data(self):
        original_count = len(self.trades)
        critical_columns = ['currency_pair', 'tenor', 'signal_type']
        for col in critical_columns:
            if col in self.trades.columns:
                self.trades = self.trades.dropna(subset=[col])
        if 'currency_pair' in self.trades.columns:
            self.trades['currency_pair'] = self.trades['currency_pair'].astype(str).str.strip().str.upper()
        if 'signal_type' in self.trades.columns:
            self.trades['signal_type'] = self.trades['signal_type'].astype(str).str.strip().str.lower()
            signal_mapping = {
                'exp': 'expensive',
                'expensive': 'expensive',
                'cheap': 'cheap',
                'chp': 'cheap'}
            self.trades['signal_type'] = self.trades['signal_type'].map(signal_mapping).fillna(self.trades['signal_type'])
        if 'trade_success' in self.trades.columns:
            if self.trades['trade_success'].dtype == 'object':
                bool_mapping = {
                    'true': True, 'True': True, 'TRUE': True, '1': True, 1: True,
                    'false': False, 'False': False, 'FALSE': False, '0': False, 0: False}
                self.trades['trade_success'] = self.trades['trade_success'].map(bool_mapping)
        numeric_columns = ['vol_diff_pct', 'implied_vol', 'realized_vol']
        for col in numeric_columns:
            if col in self.trades.columns:
                self.trades[col] = pd.to_numeric(self.trades[col], errors='coerce')
        if len(self.trades) < original_count:
            print(f"Data cleaning: {original_count} -> {len(self.trades)} trades")
    def _prepare_data(self):
        if 'date' in self.trades.columns:
            self.date_col = 'date'
        elif 'trade_date' in self.trades.columns:
            self.date_col = 'trade_date'
        elif 'timestamp' in self.trades.columns:
            self.date_col = 'timestamp'
        else:
            datetime_cols = self.trades.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                self.date_col = datetime_cols[0]
            else:
                self._detect_and_convert_dates()
        if not pd.api.types.is_datetime64_any_dtype(self.trades[self.date_col]):
            self.trades[self.date_col] = pd.to_datetime(self.trades[self.date_col])
        self.trades = self.trades.sort_values(self.date_col).reset_index(drop=True)
    def _detect_and_convert_dates(self):
        date_columns = ['date', 'trade_date', 'timestamp', 'entry_date']
        for col in date_columns:
            if col in self.trades.columns:
                try:
                    self.trades[col] = pd.to_datetime(self.trades[col], errors='coerce')
                    self.date_col = col
                    print(f"Converted column '{col}' to datetime")
                    return
                except:
                    continue
        for col in self.trades.columns:
            if self.trades[col].dtype == 'object':
                sample_values = self.trades[col].dropna().head(5)
                if len(sample_values) > 0:
                    try:
                        pd.to_datetime(sample_values.iloc[0])
                        self.trades[col] = pd.to_datetime(self.trades[col], errors='coerce')
                        self.date_col = col
                        print(f"Detected and converted date column: {col}")
                        return
                    except:
                        continue
        raise ValueError("No datetime column found in trades data")
    
    def _get_g10_pairs(self):
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 
            'AUDUSD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
            'EURCHF', 'EURAUD', 'EURNZD', 'GBPCHF', 'GBPAUD',
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'NZDCAD',
            'NZDCHF', 'NZDJPY', 'CADJPY', 'CADCHF', 'CHFJPY']
    @staticmethod
    def get_available_weighting_methods():
        """Get list of available weighting methods."""
        return list(EventAnalyzer.WEIGHTING_METHODS.keys())
    @staticmethod
    def set_default_file_path(file_path):
        """Set default file path."""
        EventAnalyzer.DEFAULT_FILE_PATH = file_path
        print(f"Default file path set to: {file_path}")
    def switch_weighting_method(self, new_weighting_method):
        print(f"Switching from {self.weighting_method} to {new_weighting_method} weighting...")
        new_analyzer = EventAnalyzer(
            weighting_method=new_weighting_method,
            excel_file_path=self.source_file if self.source_file != "provided_data" else None)
        self.weighting_method = new_analyzer.weighting_method
        self.trades = new_analyzer.trades
        self.summary = new_analyzer.summary
        self.date_col = new_analyzer.date_col
        print(f"Successfully switched to {self.weighting_method} weighting")
    # -------------------------------------------------------------------------------------------------------
    def _filter_trades(self, start_date=None, end_date=None, currencies=None, tenors=None):
        filtered = self.trades.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered = filtered[filtered[self.date_col] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered = filtered[filtered[self.date_col] <= end_dt]
        if currencies:
            filtered = filtered[filtered['currency_pair'].isin(currencies)]
        if tenors:
            filtered = filtered[filtered['tenor'].isin(tenors)]
        return filtered
    def _compare_periods(self, period_data):
        print(f"\n📊 PERIOD COMPARISON:")
        print("-" * 40)
        comparison_results = []
        for period_name, trades in period_data.items():
            if len(trades) > 0:
                result = {
                    'Period': period_name,
                    'Trades': len(trades),
                    'Hit Rate': trades['trade_success'].mean() * 100,
                    'Avg Return': trades['vol_diff_pct'].mean() * 100,
                    'Volatility': trades['vol_diff_pct'].std() * 100,
                    'Best Trade': trades['vol_diff_pct'].max() * 100,
                    'Worst Trade': trades['vol_diff_pct'].min() * 100}
                comparison_results.append(result)
        for result in comparison_results:
            print(f"\n{result['Period']}:")
            print(f"  Trades: {result['Trades']}")
            print(f"  Hit Rate: {result['Hit Rate']:.1f}%")
            print(f"  Avg Return: {result['Avg Return']:+.2f}%")
            print(f"  Volatility: {result['Volatility']:.2f}%")
            print(f"  Best/Worst: {result['Best Trade']:+.2f}% / {result['Worst Trade']:+.2f}%")
    def _print_weighting_comparison_summary(self, results, event_name):
        print(f"\n{'='*80}")
        print(f"SUMMARY: {event_name.upper()} - WEIGHTING METHOD COMPARISON")
        print(f"{'='*80}")
        summary_data = []
        for method, result in results.items():
            if result and result['event_trades'] is not None and len(result['event_trades']) > 0:
                trades = result['event_trades']
                summary_data.append({
                    'Method': method.capitalize(),
                    'Trades': len(trades),
                    'Hit Rate': trades['trade_success'].mean() * 100,
                    'Avg Return': trades['vol_diff_pct'].mean() * 100,
                    'Volatility': trades['vol_diff_pct'].std() * 100})
        if summary_data:
            print(f"\nEvent Period Performance:")
            print("-" * 60)
            for data in sorted(summary_data, key=lambda x: x['Hit Rate'], reverse=True):
                print(f"{data['Method']:>12}: {data['Hit Rate']:>6.1f}% hit rate, "
                      f"{data['Avg Return']:>+6.2f}% avg return, {data['Trades']:>3} trades")
        else:
            print("No valid results to compare")
    # ---------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------



    def analyze_tenor_aware_event_impact(self, event_data, currencies=None, tenors=None):
        print(f"\n{'='*80}")
        print(f"TENOR-AWARE EVENT IMPACT ANALYSIS")
        print(f"Weighting Method: {self.weighting_method.upper()}")
        print(f"{'='*80}")
        if 'ReleaseDate' not in event_data.columns:
            raise ValueError("Event data must have a 'ReleaseDate' column")
        event_clean = event_data.dropna(subset=['ReleaseDate']).copy()
        event_clean = event_clean.sort_values('ReleaseDate')
        if currencies is None:
            currencies = sorted(self.trades['currency_pair'].unique())
        if tenors is None:
            tenors = sorted(self.trades['tenor'].unique())
        print(f"Analyzing {len(event_clean)} events for:")
        print(f"  Currencies: {currencies}")
        print(f"  Tenors: {tenors}")
        tenor_to_days = {
            '1W': 7,
            '2W': 14,
            '1M': 30,
            '2M': 60,
            '3M': 90,
            '6M': 180,
            '1Y': 365
        }
        
        results = {}
        
        for idx, event_row in event_clean.iterrows():
            release_date = event_row['ReleaseDate']
            event_key = release_date.strftime('%Y-%m-%d')
            
            print(f"\n{'='*60}")
            print(f"EVENT: {release_date.strftime('%Y-%m-%d')}")
            
            # Add event-specific info if available
            if 'Actual' in event_row and 'SMed' in event_row:
                actual = event_row['Actual']
                expected = event_row['SMed']
                surprise = actual - expected
                print(f"Actual: {actual:.1f} | Expected: {expected:.1f} | Surprise: {surprise:+.2f}")
            
            print(f"{'='*60}")
            
            event_results = {
                'release_date': release_date,
                'currency_tenor_analysis': {}
            }
            
            # Add event details if available
            if 'Actual' in event_row and 'SMed' in event_row:
                event_results['actual'] = float(event_row['Actual'])
                event_results['expected'] = float(event_row['SMed'])
                event_results['surprise'] = float(event_row['Actual'] - event_row['SMed'])
            
            # Analyze each currency-tenor combination
            for currency in currencies:
                for tenor in tenors:
                    
                    # Calculate the trade selection window based on tenor
                    if tenor in tenor_to_days:
                        tenor_days = tenor_to_days[tenor]
                    else:
                        # Try to parse tenor if not in mapping (e.g., '3W' -> 21 days)
                        try:
                            if tenor.endswith('W'):
                                tenor_days = int(tenor[:-1]) * 7
                            elif tenor.endswith('M'):
                                tenor_days = int(tenor[:-1]) * 30
                            elif tenor.endswith('Y'):
                                tenor_days = int(tenor[:-1]) * 365
                            else:
                                print(f"Warning: Unknown tenor format '{tenor}', skipping")
                                continue
                        except:
                            print(f"Warning: Could not parse tenor '{tenor}', skipping")
                            continue
                    
                    # Calculate trade selection window
                    # Trades selected from (release_date - tenor_days) to (release_date - 1)
                    trade_start = release_date - timedelta(days=tenor_days)
                    trade_end = release_date - timedelta(days=1)  # Exclude release date itself
                    
                    print(f"\n{currency} {tenor}:")
                    print(f"  Trade window: {trade_start.strftime('%Y-%m-%d')} to {trade_end.strftime('%Y-%m-%d')}")
                    print(f"  (Tenor incorporates event on {release_date.strftime('%Y-%m-%d')})")
                    
                    # Filter trades for this currency-tenor combination in the relevant window
                    relevant_trades = self._filter_trades(
                        start_date=trade_start,
                        end_date=trade_end,
                        currencies=[currency],
                        tenors=[tenor]
                    )
                    
                    if len(relevant_trades) == 0:
                        # Skip silently - no output for currency-tenor combinations with no trades
                        continue
                    
                    # Separate expensive and cheap trades
                    expensive_trades = relevant_trades[relevant_trades['signal_type'] == 'expensive']
                    cheap_trades = relevant_trades[relevant_trades['signal_type'] == 'cheap']
                    
                    # Calculate statistics for each signal type
                    def calculate_signal_stats(trades, signal_type):
                        if len(trades) == 0:
                            return None
                        
                        successful = trades[trades['trade_success'] == True]
                        failed = trades[trades['trade_success'] == False]
                        
                        # For expensive vol: wins are negative (realized < implied)
                        # For cheap vol: wins are positive (realized > implied)
                        if signal_type == 'expensive':
                            avg_win = abs(successful['vol_diff_pct'].mean()) * 100 if len(successful) > 0 else 0
                            best_win = abs(successful['vol_diff_pct'].min()) * 100 if len(successful) > 0 else 0  # Most negative
                            avg_loss = failed['vol_diff_pct'].mean() * 100 if len(failed) > 0 else 0
                            worst_loss = failed['vol_diff_pct'].max() * 100 if len(failed) > 0 else 0  # Most positive
                        else:  # cheap
                            avg_win = successful['vol_diff_pct'].mean() * 100 if len(successful) > 0 else 0
                            best_win = successful['vol_diff_pct'].max() * 100 if len(successful) > 0 else 0  # Most positive
                            avg_loss = abs(failed['vol_diff_pct'].mean()) * 100 if len(failed) > 0 else 0
                            worst_loss = abs(failed['vol_diff_pct'].min()) * 100 if len(failed) > 0 else 0  # Most negative
                        
                        return {
                            'total_trades': len(trades),
                            'wins': len(successful),
                            'losses': len(failed),
                            'hit_rate': (len(successful) / len(trades)) * 100,
                            'avg_return': trades['vol_diff_pct'].mean() * 100,
                            'avg_win': avg_win,
                            'avg_loss': avg_loss,
                            'best_win': best_win,
                            'worst_loss': worst_loss,
                            'volatility': trades['vol_diff_pct'].std() * 100,
                            'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf')
                        }
                    
                    expensive_stats = calculate_signal_stats(expensive_trades, 'expensive')
                    cheap_stats = calculate_signal_stats(cheap_trades, 'cheap')
                    
                    # Add detailed trade information
                    def get_trade_details(trades, signal_type):
                        """Get detailed information about individual trades."""
                        if len(trades) == 0:
                            return []
                        
                        trade_details = []
                        for _, trade in trades.iterrows():
                            trade_info = {
                                'date': trade[self.date_col],
                                'success': trade['trade_success'],
                                'vol_diff_pct': trade['vol_diff_pct'] * 100,
                                'signal_type': signal_type
                            }
                            
                            # Add essential columns for trade detail display
                            essential_cols = ['signal_percentile', 'implied_vol', 'realized_vol']
                            for col in essential_cols:
                                if col in trade.index:
                                    trade_info[col] = trade[col]
                                else:
                                    trade_info[col] = None  # Will show as N/A
                            
                            # Add additional optional columns
                            optional_cols = ['entry_price', 'exit_price', 'notional', 'pnl', 'trade_id']
                            for col in optional_cols:
                                if col in trade.index:
                                    trade_info[col] = trade[col]
                            
                            trade_details.append(trade_info)
                        
                        return sorted(trade_details, key=lambda x: x['date'])
                    
                    expensive_trade_details = get_trade_details(expensive_trades, 'expensive')
                    cheap_trade_details = get_trade_details(cheap_trades, 'cheap')
                    
                    # Store all trades for this currency-tenor-event combination
                    all_trade_details = expensive_trade_details + cheap_trade_details
                    
                    # Print detailed analysis with enhanced trade information
                    print(f"  Total Trades: {len(relevant_trades)}")
                    
                    def print_trade_details(trade_details, signal_name):
                        """Print formatted trade details."""
                        if not trade_details:
                            return
                        
                        print(f"    Individual {signal_name} Trades:")
                        print(f"    {'Date':<12} {'Sig%':<8} {'Impl':<8} {'Real':<8} {'Diff%':<10} {'Result'}")
                        print(f"    {'-'*60}")
                        
                        for i, trade in enumerate(trade_details, 1):
                            date_str = trade['date'].strftime('%Y-%m-%d')
                            
                            # Format signal percentile
                            sig_pct = f"{trade['signal_percentile']:.1f}" if trade['signal_percentile'] is not None else "N/A"
                            
                            # Format implied vol
                            impl_vol = f"{trade['implied_vol']:.2f}" if trade['implied_vol'] is not None else "N/A"
                            
                            # Format realized vol
                            real_vol = f"{trade['realized_vol']:.2f}" if trade['realized_vol'] is not None else "N/A"
                            
                            # Format vol diff
                            vol_diff = f"{trade['vol_diff_pct']:+.2f}%"
                            
                            # Result
                            result = "WIN" if trade['success'] else "LOSS"
                            
                            print(f"    {date_str:<12} {sig_pct:<8} {impl_vol:<8} {real_vol:<8} {vol_diff:<10} {result}")
                    
                    if expensive_stats:
                        print(f"  EXPENSIVE Vol ({expensive_stats['total_trades']} trades):")
                        print(f"    Hit Rate: {expensive_stats['hit_rate']:.1f}%")
                        print(f"    W/L: {expensive_stats['wins']}-{expensive_stats['losses']}")
                        print(f"    Avg Win: {expensive_stats['avg_win']:.2f}% | Best Win: {expensive_stats['best_win']:.2f}%")
                        print(f"    Avg Loss: {expensive_stats['avg_loss']:+.2f}% | Worst Loss: {expensive_stats['worst_loss']:+.2f}%")
                        print(f"    Risk/Reward: {expensive_stats['risk_reward_ratio']:.2f}:1")
                        
                        # Show individual expensive trades in table format
                        print_trade_details(expensive_trade_details, "Expensive")
                    
                    if cheap_stats:
                        print(f"  CHEAP Vol ({cheap_stats['total_trades']} trades):")
                        print(f"    Hit Rate: {cheap_stats['hit_rate']:.1f}%")
                        print(f"    W/L: {cheap_stats['wins']}-{cheap_stats['losses']}")
                        print(f"    Avg Win: {cheap_stats['avg_win']:+.2f}% | Best Win: {cheap_stats['best_win']:+.2f}%")
                        print(f"    Avg Loss: {cheap_stats['avg_loss']:.2f}% | Worst Loss: {cheap_stats['worst_loss']:.2f}%")
                        print(f"    Risk/Reward: {cheap_stats['risk_reward_ratio']:.2f}:1")
                        
                        # Show individual cheap trades in table format
                        print_trade_details(cheap_trade_details, "Cheap")
                    
                    # Show additional insights if available
                    if len(expensive_trades) > 0 and len(cheap_trades) > 0:
                        exp_avg_return = expensive_trades['vol_diff_pct'].mean() * 100
                        cheap_avg_return = cheap_trades['vol_diff_pct'].mean() * 100
                        print(f"    Strategy Comparison:")
                        print(f"      Expensive avg: {exp_avg_return:+.2f}% vs Cheap avg: {cheap_avg_return:+.2f}%")
                        better_strategy = "Expensive" if exp_avg_return > cheap_avg_return else "Cheap"
                        print(f"      Better performer: {better_strategy} vol strategy")
                    
                    # Store results
                    currency_tenor_key = f"{currency}_{tenor}"
                    event_results['currency_tenor_analysis'][currency_tenor_key] = {
                        'currency': currency,
                        'tenor': tenor,
                        'tenor_days': tenor_days,
                        'trade_window_start': trade_start,
                        'trade_window_end': trade_end,
                        'total_trades': len(relevant_trades),
                        'expensive_stats': expensive_stats,
                        'cheap_stats': cheap_stats,
                        'trades_data': relevant_trades,  # Full trade data
                        'expensive_trade_details': expensive_trade_details,  # Formatted trade details
                        'cheap_trade_details': cheap_trade_details,  # Formatted trade details
                        'all_trade_details': all_trade_details  # Combined trade details
                    }
            
            results[event_key] = event_results
        
        # Print summary analysis
        self._print_tenor_aware_summary(results, currencies, tenors)
        
        return results

    def _print_tenor_aware_summary(self, results, currencies, tenors):
        """Print summary analysis across all events for each currency-tenor combination."""
        print(f"\n{'='*80}")
        print(f"SUMMARY: TENOR-AWARE EVENT IMPACT ANALYSIS")
        print(f"{'='*80}")
        summary_stats = {}
        for currency in currencies:
            for tenor in tenors:
                currency_tenor_key = f"{currency}_{tenor}"
                expensive_data = []
                cheap_data = []
                total_trades = 0
                for event_date, event_result in results.items():
                    if currency_tenor_key in event_result['currency_tenor_analysis']:
                        analysis = event_result['currency_tenor_analysis'][currency_tenor_key]
                        total_trades += analysis['total_trades']
                        if analysis['expensive_stats']:
                            expensive_data.append(analysis['expensive_stats'])
                        if analysis['cheap_stats']:
                            cheap_data.append(analysis['cheap_stats'])
                if expensive_data or cheap_data:
                    summary_stats[currency_tenor_key] = {
                        'currency': currency,
                        'tenor': tenor,
                        'events_with_data': len([1 for event_date, event_result in results.items() 
                                                if currency_tenor_key in event_result['currency_tenor_analysis']]),
                        'total_trades_across_events': total_trades,
                        'expensive_summary': self._calculate_summary_stats(expensive_data) if expensive_data else None,
                        'cheap_summary': self._calculate_summary_stats(cheap_data) if cheap_data else None}
        for currency_tenor_key, summary in summary_stats.items():
            currency = summary['currency']
            tenor = summary['tenor']
            print(f"\n{currency} {tenor} - ACROSS ALL EVENTS:")
            print(f"  Events with data: {summary['events_with_data']}")
            print(f"  Total trades: {summary['total_trades_across_events']}")
            if summary['expensive_summary']:
                exp_sum = summary['expensive_summary']
                print(f"  EXPENSIVE Vol Summary:")
                print(f"    Avg Hit Rate: {exp_sum['avg_hit_rate']:.1f}%")
                print(f"    Avg Risk/Reward: {exp_sum['avg_risk_reward']:.2f}:1")
                print(f"    Best Event Hit Rate: {exp_sum['best_hit_rate']:.1f}%")
                print(f"    Worst Event Hit Rate: {exp_sum['worst_hit_rate']:.1f}%")
            if summary['cheap_summary']:
                cheap_sum = summary['cheap_summary']
                print(f"  CHEAP Vol Summary:")
                print(f"    Avg Hit Rate: {cheap_sum['avg_hit_rate']:.1f}%")
                print(f"    Avg Risk/Reward: {cheap_sum['avg_risk_reward']:.2f}:1")
                print(f"    Best Event Hit Rate: {cheap_sum['best_hit_rate']:.1f}%")
                print(f"    Worst Event Hit Rate: {cheap_sum['worst_hit_rate']:.1f}%")

    def get_event_trades_dataframe(self, results, event_date=None, currency=None, tenor=None, signal_type=None):
        all_trades = []
        events_to_process = []
        if event_date:
            if event_date in results:
                events_to_process = [event_date]
            else:
                print(f"Event date {event_date} not found in results")
                return pd.DataFrame()
        else:
            events_to_process = list(results.keys())
        for event_key in events_to_process:
            event_result = results[event_key]
            for currency_tenor_key, analysis in event_result['currency_tenor_analysis'].items():
                if currency and analysis['currency'] != currency:
                    continue
                if tenor and analysis['tenor'] != tenor:
                    continue
                trade_details_list = []
                if signal_type == 'expensive':
                    trade_details_list = analysis.get('expensive_trade_details', [])
                elif signal_type == 'cheap':
                    trade_details_list = analysis.get('cheap_trade_details', [])
                else:
                    trade_details_list = analysis.get('all_trade_details', [])
                for trade_detail in trade_details_list:
                    trade_row = trade_detail.copy()
                    trade_row['event_date'] = event_result['release_date']
                    trade_row['currency'] = analysis['currency']
                    trade_row['tenor'] = analysis['tenor']
                    trade_row['trade_window_start'] = analysis['trade_window_start']
                    trade_row['trade_window_end'] = analysis['trade_window_end']
                    if 'actual' in event_result:
                        trade_row['event_actual'] = event_result['actual']
                    if 'expected' in event_result:
                        trade_row['event_expected'] = event_result['expected']
                    if 'surprise' in event_result:
                        trade_row['event_surprise'] = event_result['surprise']
                    all_trades.append(trade_row)
        if not all_trades:
            print("No trades found matching the specified criteria")
            return pd.DataFrame()
        trades_df = pd.DataFrame(all_trades)
        column_order = [
            'event_date', 'currency', 'tenor', 'date', 'signal_type', 
            'signal_percentile', 'implied_vol', 'realized_vol', 'vol_diff_pct',
            'success', 'event_actual', 'event_expected', 'event_surprise']
        available_columns = [col for col in column_order if col in trades_df.columns]
        remaining_columns = [col for col in trades_df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        trades_df = trades_df[final_columns]
        trades_df = trades_df.sort_values(['event_date', 'currency', 'tenor', 'date'])
        print(f"Retrieved {len(trades_df)} trades matching criteria:")
        if event_date:
            print(f"  Event: {event_date}")
        if currency:
            print(f"  Currency: {currency}")
        if tenor:
            print(f"  Tenor: {tenor}")
        if signal_type:
            print(f"  Signal Type: {signal_type}")
        return trades_df.reset_index(drop=True)

    def _calculate_summary_stats(self, stats_list):
        """Calculate summary statistics across multiple events."""
        if not stats_list:
            return None
        hit_rates = [s['hit_rate'] for s in stats_list]
        risk_rewards = [s['risk_reward_ratio'] for s in stats_list if s['risk_reward_ratio'] != float('inf')]
        return {
            'avg_hit_rate': np.mean(hit_rates),
            'best_hit_rate': max(hit_rates),
            'worst_hit_rate': min(hit_rates),
            'avg_risk_reward': np.mean(risk_rewards) if risk_rewards else 0,
            'events_analyzed': len(stats_list)}


















pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)


manager = EconomicDataManager(blp)
cpi_data = manager.get_pastData('FDTR Index', 360)

# Your EventAnalyzer
path = r'C:\Users\ntaylor\Desktop\Work\AnalysisCode\VolTools\xCCyVol_ATM_GAMMA\backtest_resultsEqualCorrCoint_24Sep_5yBack.xlsx'
analyzer = EventAnalyzer(path, 'cointegration')

# NEW: Tenor-aware analysis
results = analyzer.analyze_tenor_aware_event_impact(
    event_data=cpi_data,
    currencies=['EURUSD'], 
    tenors=['1W', '2W']
)







































