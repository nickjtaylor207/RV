import pandas as pd
import numpy as np
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import table
from scipy import stats
import re
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset, MonthEnd, Week, Day
from scipy.stats import percentileofscore
from typing import List, Dict, Optional, Tuple
import pdblp
from xbbg import blp


from dataGather import FXEventAnalyzer


def get_current_on_vol_metrics(currency_pairs, lookback_days=30):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days + 10)
    results = []
    for ccy in currency_pairs:
        try:
            on_ticker = f'{ccy}VON BGN Curncy'
            w1_ticker = f'{ccy}V1W BGN Curncy'
            on_data = blp.bdh(
                tickers=on_ticker,flds=['PX_LAST'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),Per='D')
            w1_data = blp.bdh(
                tickers=w1_ticker,flds=['PX_LAST'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),Per='D')
            if on_data.empty or w1_data.empty:
                print(f"Warning: No data available for {ccy}")
                continue
            if isinstance(on_data.columns, pd.MultiIndex):
                on_data.columns = on_data.columns.droplevel(0)
            if isinstance(w1_data.columns, pd.MultiIndex):
                w1_data.columns = w1_data.columns.droplevel(0)
            on_data.columns = ['ON_Vol']
            w1_data.columns = ['W1_Vol']
            current_on_vol = on_data['ON_Vol'].iloc[-1]
            current_1w_vol = w1_data['W1_Vol'].iloc[-1]
            on_vols_hist = on_data['ON_Vol'].iloc[:-1].copy()
            mean = on_vols_hist.mean()
            std = on_vols_hist.std()
            on_vols_clean = on_vols_hist[
                (on_vols_hist < mean + 3*std) & 
                (on_vols_hist > mean - 3*std)]
            # X Days back Mean (Excluding Outliers)
            baseline_historical = on_vols_clean.mean()
            # Baseline_ON = sqrt((1W_vol² × 7 - ON_vol² × 1) / 6)
            var_1w = (current_1w_vol / 100) ** 2
            var_on = (current_on_vol / 100) ** 2
            var_baseline = (var_1w * 7 - var_on * 1) / 6
            if var_baseline > 0:
                baseline_interpolated = np.sqrt(var_baseline) * 100
            else:
                baseline_interpolated = baseline_historical
            # Weighted combination (30% historical, 70% interpolated)
            baseline_weighted = 0.3 * baseline_historical + 0.7 * baseline_interpolated
            # Premium vs baseline
            event_premium = current_on_vol - baseline_weighted
            results.append({
                'Currency_Pair': ccy,
                'Date': on_data.index[-1],
                'Current_ON_Vol': current_on_vol,
                'Current_1W_Vol': current_1w_vol,
                'Hist30dAve_ON': baseline_historical,
                'ONv1W_Calander_Interp_ON': baseline_interpolated,
                'Baseline_Weighted_ON_30hist/70cal': baseline_weighted,
                'EventON_vs_WeightedBaselineON_diff': event_premium})
        except Exception as e:
            print(f"Error processing {ccy}: {str(e)}")
            continue
    return pd.DataFrame(results)



def get_historical_event_on_metrics(indicator_ticker, fx_tickers, 
                                   event_time='08:30:00',
                                   lookback_days=365,
                                   num_events=None,
                                   baseline_lookback_days=30):
    analyzer = FXEventAnalyzer(blp, pdblp_port=8194, pdblp_timeout=5000)
    event_data = analyzer.get_pastData(indicator_ticker, lookback_days)
    if event_data.empty:
        print(f"No event data found for {indicator_ticker}")
        return pd.DataFrame()
    today = pd.Timestamp.now().normalize()
    event_data['ReleaseDate'] = pd.to_datetime(event_data['ReleaseDate'])
    event_data = event_data[event_data['ReleaseDate'] < today]
    if event_data.empty:
        print(f"No past events found for {indicator_ticker}")
        return pd.DataFrame()
    if num_events is not None:
        event_data = event_data.tail(num_events)
    try:
        time_parts = event_time.split(':')
        fixed_hour = int(time_parts[0])
        fixed_minute = int(time_parts[1])
        fixed_second = int(time_parts[2]) if len(time_parts) > 2 else 0
    except Exception as e:
        print(f"Error parsing event_time: {str(e)}")
        fixed_hour, fixed_minute, fixed_second = 8, 30, 0
    results = []
    for idx, row in event_data.iterrows():
        event_date = pd.to_datetime(row['ReleaseDate'])
        event_datetime = event_date.replace(
            hour=fixed_hour,
            minute=fixed_minute,
            second=fixed_second)
        for fx_ticker in fx_tickers:
            try:
                # Calculate baseline metrics
                baseline_metrics = analyzer.calculate_baseline_vols(
                    fx_ticker, event_date, lookback_days=baseline_lookback_days)
                # Calculate event premium metrics
                premium_metrics = analyzer.calculate_event_premium_metrics(
                    fx_ticker, 
                    event_datetime,
                    baseline_metrics['baseline_weighted'],
                    baseline_metrics['event_on_vol'],
                    interval=10)
                results.append({
                    'EventName': row.get('NAME', indicator_ticker),
                    'EventDate': event_date,
                    'FX_Pair': fx_ticker,
                    'Actual': row.get('Actual', np.nan),
                    'SMed': row.get('SMed', np.nan),
                    'Surprise': (row.get('Actual', np.nan) - row.get('SMed', np.nan) 
                               if pd.notna(row.get('Actual')) and pd.notna(row.get('SMed')) 
                               else np.nan),
                    'Event_ON_Vol': baseline_metrics['event_on_vol'],
                    'Baseline_Weighted_ON': baseline_metrics['baseline_weighted'],
                    'Event_RV_5pm_10am': premium_metrics['event_realized_vol'],
                    'Event_ONimplied_vs_ONrealized_diff': premium_metrics['event_impact'],
                    'Premium_Efficiency_%': premium_metrics['premium_efficiency'],
                    'VolSeller_PnL': premium_metrics['vol_seller_pnl']})
            except Exception as e:
                print(f"  Error processing {fx_ticker}: {str(e)}")
                continue
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        print("No results generated")
        return pd.DataFrame()


def create_event_premium_comparison(indicator_ticker, fx_tickers, 
                                   event_time='08:30:00',
                                   lookback_days=365,
                                   num_historical_events=10,
                                   baseline_lookback_days=30):
    # 1. Get historical event metrics
    historical = get_historical_event_on_metrics(
        indicator_ticker=indicator_ticker,
        fx_tickers=fx_tickers,
        event_time=event_time,
        lookback_days=lookback_days,
        num_events=num_historical_events,
        baseline_lookback_days=baseline_lookback_days)
    
    # 2. Get current ON vol metrics
    current = get_current_on_vol_metrics(fx_tickers, lookback_days=baseline_lookback_days)
    
    # 3. Create comparison summary
    comparison_data = []
    
    for ccy in fx_tickers:
        hist_ccy = historical[historical['FX_Pair'] == ccy]
        curr_ccy = current[current['Currency_Pair'] == ccy]
        if hist_ccy.empty or curr_ccy.empty:
            continue
        hist_stats = {
            'ON_implied_vs_realized': {
                'mean': hist_ccy['Event_ONimplied_vs_ONrealized_diff'].mean(),
                'median': hist_ccy['Event_ONimplied_vs_ONrealized_diff'].median(),
                'std': hist_ccy['Event_ONimplied_vs_ONrealized_diff'].std(),
                'min': hist_ccy['Event_ONimplied_vs_ONrealized_diff'].min(),
                'max': hist_ccy['Event_ONimplied_vs_ONrealized_diff'].max()},
            'premium_efficiency': {
                'mean': hist_ccy['Premium_Efficiency_%'].mean(),
                'median': hist_ccy['Premium_Efficiency_%'].median(),
                'std': hist_ccy['Premium_Efficiency_%'].std(),
                'min': hist_ccy['Premium_Efficiency_%'].min(),
                'max': hist_ccy['Premium_Efficiency_%'].max()},
            'vol_seller_pnl': {
                'mean': hist_ccy['VolSeller_PnL'].mean(),
                'median': hist_ccy['VolSeller_PnL'].median(),
                'std': hist_ccy['VolSeller_PnL'].std(),
                'min': hist_ccy['VolSeller_PnL'].min(),
                'max': hist_ccy['VolSeller_PnL'].max(),
                'win_rate': (hist_ccy['VolSeller_PnL'] > 0).mean() * 100},
            'event_on_vol': {
                'mean': hist_ccy['Event_ON_Vol'].mean(),
                'median': hist_ccy['Event_ON_Vol'].median(),
                'std': hist_ccy['Event_ON_Vol'].std()},
            'event_premium': {
                'mean': (hist_ccy['Event_ON_Vol'] - hist_ccy['Baseline_Weighted_ON']).mean(),
                'median': (hist_ccy['Event_ON_Vol'] - hist_ccy['Baseline_Weighted_ON']).median(),
                'std': (hist_ccy['Event_ON_Vol'] - hist_ccy['Baseline_Weighted_ON']).std(),
                'min': (hist_ccy['Event_ON_Vol'] - hist_ccy['Baseline_Weighted_ON']).min(),
                'max': (hist_ccy['Event_ON_Vol'] - hist_ccy['Baseline_Weighted_ON']).max()}}
        
        # Current values
        curr_on_vol = curr_ccy['Current_ON_Vol'].iloc[0]
        curr_baseline = curr_ccy['Baseline_Weighted_ON_30hist/70cal'].iloc[0]
        curr_event_prem = curr_ccy['EventON_vs_WeightedBaselineON_diff'].iloc[0]

        # Calculate z-scores and percentiles
        event_prem_zscore = ((curr_event_prem - hist_stats['event_premium']['mean']) / 
                            hist_stats['event_premium']['std'] if hist_stats['event_premium']['std'] > 0 else 0)
        on_vol_zscore = ((curr_on_vol - hist_stats['event_on_vol']['mean']) / 
                        hist_stats['event_on_vol']['std'] if hist_stats['event_on_vol']['std'] > 0 else 0)
        
        # Determine if current setup is attractive
        # High event premium + historically profitable = attractive
        event_prem_percentile = percentileofscore(
            hist_ccy['Event_ON_Vol'] - hist_ccy['Baseline_Weighted_ON'], 
            curr_event_prem)
        
        comparison_data.append({
            'Currency': ccy,
            '--- CURRENT SETUP ---': '',
            'Current_ON_Vol': curr_on_vol,
            'Current_Baseline': curr_baseline,
            'Current_Event_Premium': curr_event_prem,
            'Event_Prem_Z-Score': event_prem_zscore,
            'Event_Prem_Percentile': event_prem_percentile,
            
            '--- HISTORICAL AVG ---': '',
            'Hist_Avg_Event_ON': hist_stats['event_on_vol']['mean'],
            'Hist_Avg_Event_Premium': hist_stats['event_premium']['mean'],
            'Hist_Avg_VolSeller_PnL': hist_stats['vol_seller_pnl']['mean'],
            'Hist_VolSeller_WinRate_%': hist_stats['vol_seller_pnl']['win_rate'],
            
            '--- HISTORICAL RANGE ---': '',
            'Hist_Min_Event_Premium': hist_stats['event_premium']['min'],
            'Hist_Max_Event_Premium': hist_stats['event_premium']['max'],
            'Hist_Avg_Premium_Efficiency_%': hist_stats['premium_efficiency']['mean'],
            
            '--- TRADE SIGNAL ---': '',
            'Attractiveness_Score': event_prem_zscore if hist_stats['vol_seller_pnl']['mean'] > 0 else -event_prem_zscore,
            'Signal': 'SELL VOL' if (curr_event_prem > hist_stats['event_premium']['mean'] and 
                                     hist_stats['vol_seller_pnl']['mean'] > 0) else 'AVOID',
            'n_historical_events': len(hist_ccy)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df, historical, current






currency_pairs = ['EURUSD', 'USDJPY', 'GBPUSD',  'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
metrics = get_current_on_vol_metrics(currency_pairs, lookback_days=30)


print(metrics.T)





















# fx_tickers = ['GBPUSD']

# comparison, historical, current = create_event_premium_comparison(
#     indicator_ticker='UKIPIMOM Index',
#     fx_tickers=fx_tickers,
#     event_time='08:30:00',
#     lookback_days=365,
#     num_historical_events=6,
#     baseline_lookback_days=30)



# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.expand_frame_repr', False)

# print(comparison.T)


















