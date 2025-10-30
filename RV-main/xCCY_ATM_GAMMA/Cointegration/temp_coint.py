

import numpy as np
from xbbg import blp
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import pandas as pd
import statsmodels.api as sm

import pytz


import blpapi
import pdblp
from numpy import unique
import numpy as np
import pandas as pd
from xbbg import blp, pipeline
from datetime import datetime, timedelta, time
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.api import OLS, add_constant
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from itertools import combinations
import matplotlib.dates as mdates






def Data_DailyCLOSE_Multiple(tickers, days):
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    df_all = {}
    for ticker in tickers:
        data_single = blp.bdh(
            tickers=f"{ticker}" ,
            flds=["PX_LAST"],  # Adjust fields as needed
            start_date=start_date,
            end_date=end_date,
            Per="D")
        data_single.columns = [ticker]
        df_all[ticker] = data_single
    df_all = pd.concat(df_all, axis=1)
    df_all.columns = df_all.columns.droplevel(0) if isinstance(df_all.columns, pd.MultiIndex) else df_all.columns
    df_all = df_all.sort_index(ascending=True)
    return df_all



# Individual Pair Screening Class
class Screen_SinglePair_Metrics:
    def __init__(self, tickers, days):
        self.ticker1 = tickers[0]
        self.ticker2 = tickers[1]
        self.df =  Data_DailyCLOSE_Multiple(tickers, days).dropna()
        self.log_df = np.log(self.df)
        self.spread = None
        self.alpha = None
        self.beta = None


    # Calculating Dollar-Neutral spread through OLS fitting fo entire data
    def spread_calc(self):
        x = self.log_df[self.ticker1]; y = self.log_df[self.ticker2]
        X0 = sm.add_constant(x); ols = sm.OLS(y, X0).fit()
        beta = ols.params[self.ticker1]; alpha = ols.params['const']
        spread = y - (alpha + beta * x)
        self.spread = spread
        self.alpha = alpha; self.beta = beta


    # Summary Statistics of the Spread
    def spread_stats(self):
        if self.spread is None:
            self.spread_calc()
        spread = self.spread
        stats = {
        'spread_current': spread.iloc[-1],
        'spread_current_zscore': (spread.iloc[-1] - spread.mean()) / spread.std(),
        'spread_current_percentile': (spread <= spread.iloc[-1]).mean() * 100,
        'spread_mean': spread.mean(),
        'spread_std': spread.std(),
        'spread_min': spread.min(),
        'spread_max': spread.max(),
        'spread_skewness': spread.skew(),
        'spread_kurtosis': spread.kurtosis(),
        'spread_25th_percentile': spread.quantile(0.25),
        'spread_75th_percentile': spread.quantile(0.75),
        'spread_iqr': spread.quantile(0.75) - spread.quantile(0.25),
        }
        df_stats = pd.DataFrame([stats])
        return df_stats

    # -----------------------------------------------------------------------------------------
    # Cointegration Stregth - Tests Long Term Relationship Between Assets
    # -----------------------------------------------------------------------------------------
    # JOHANSEN TEST - Confirms Pairs with long-term Stationarity over days
    def johansen_test(self, det_order=1, maxlags=10, ic='aic'):
        all_results = []
        data = self.log_df[[self.ticker1, self.ticker2]].dropna()
        var_model = VAR(data)
        lag_order_results = var_model.select_order(maxlags=maxlags)
        optimal_lag = getattr(lag_order_results, ic)
        k_ar_diff = max(optimal_lag - 1, 1)
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        trace_stat = result.lr1[0]
        max_eigen_stat = result.lr2[0]
        trace_crit_5 = result.cvt[0, 1]
        max_eigen_crit_5 = result.cvm[0, 1]
        trace_reject = trace_stat > trace_crit_5
        max_eigen_reject = max_eigen_stat > max_eigen_crit_5
        result_dict = {
            'pair': f"{self.ticker1} / {self.ticker2}",
            'trace_statistic': trace_stat,
            'trace_critical_5pct': trace_crit_5,
            'trace_reject_h0': trace_reject,               # True
            'max_eigen_statistic': max_eigen_stat,
            'max_eigen_critical_5pct': max_eigen_crit_5,
            'max_eigen_reject_h0': max_eigen_reject,       # True   
            'eigenvalue_1': result.eig[0],
            # 'cointegrating_vector': result.evec[:, 0],
            'optimal_lag': optimal_lag,
            'k_ar_diff': k_ar_diff}
        all_results.append(result_dict)
        df = pd.DataFrame(all_results)
        return df.sort_values('trace_statistic', ascending=False).T


    def engle_granger_Test(self):
        x = self.log_df[self.ticker1]; y = self.log_df[self.ticker2]
        eg_t, eg_p, _ = coint(x, y)
        X0 = sm.add_constant(x); ols = sm.OLS(y, X0).fit()
        alpha = ols.params['const']; beta = ols.params[self.ticker1]
        results = {
        'eg_t_stat': eg_t, 'eg_p_value': eg_p,
        'intercept': alpha, 'hedge_ratio': beta,
        'ols_r_squared': ols.rsquared}
        df_egTest = pd.DataFrame([results])
        return df_egTest.T
    

    # Hypothesis testing of spread stationarity (ADF) 
    def adf_SNR_spread(self):
        if self.spread is None:
            self.spread_calc()
        spread = self.spread
        s_lag = spread.shift(1); ds = spread - s_lag
        df = pd.concat([ds, s_lag], axis=1).dropna(); df.columns = ['ds','s_lag']
        X1 = sm.add_constant(df['s_lag']); arres = sm.OLS(df['ds'], X1).fit()
        phi = arres.params['s_lag']; sigma = arres.resid.std(); snr = (1 - phi) / sigma
        stat, p, _, _, crit, _ = adfuller(spread.dropna(), autolag='AIC')
        results = {
            'adf_statistic': stat, 'adf_p_value': p,
            'adf_critical_1pct': crit['1%'], 'adf_critical_5pct': crit['5%'], 'adf_critical_10pct': crit['10%'],
            'adf_reject_5pct': stat < crit['5%'], 'adf_reject_1pct': stat < crit['1%'],
            'snr_ratio': snr}
        df_adfTest = pd.DataFrame([results])
        return df_adfTest.T
    

    def halfLife_estimate(self):
        if self.spread is None:
            self.spread_calc()
        spread_lag = self.spread.shift(1)
        spread_diff = self.spread.diff()
        spread_lag_const = add_constant(spread_lag.dropna())
        spread_diff_aligned = spread_diff.dropna()
        common_index = spread_lag_const.index.intersection(spread_diff_aligned.index)
        model = OLS(spread_diff_aligned.loc[common_index], 
                spread_lag_const.loc[common_index]).fit()
        theta = model.params.iloc[1]
        theta_pvalue = model.pvalues.iloc[1]
        confidence_interval = model.conf_int().iloc[1].values
        halflife = -np.log(2) / theta if theta < 0 else np.inf
        CI = np.round(confidence_interval, 3).tolist()
        results = {'halflife': halflife,'phi': theta,
                'phi_pvalue': theta_pvalue,'confidence_interval': [CI],
                    'r_squared': model.rsquared}
        return pd.DataFrame(results)
    

    def hurst_exponent(self, lags_range=(2, 100)):
        def calculate_hurst(ts):
            max_lag = min(lags_range[1], len(ts) // 4)
            lags = range(lags_range[0], max_lag if max_lag >= lags_range[0] else lags_range[0] + 1)
            if len(list(lags)) < 2:
                return np.nan
            tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
            if np.sum(np.isfinite(tau)) < 2:
                return np.nan
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return 2.0 * poly[0]
        px1 = self.df[self.ticker1].dropna()
        px2 = self.df[self.ticker2].dropna()
        logp1 = np.log(px1.values)
        logp2 = np.log(px2.values)
        hurst1 = calculate_hurst(logp1)
        hurst2 = calculate_hurst(logp2)
        if self.spread is None:
            self.spread_calc()
        spread_level = self.spread.dropna().values  # level, not pct_change
        hurst_spread = calculate_hurst(spread_level)
        results = {
            'pair': f"{self.ticker1} / {self.ticker2}",
            'hurst_ticker1': hurst1,
            'hurst_ticker2': hurst2,
            'hurst_spread': hurst_spread,
            'mean_hurst': np.nanmean([hurst1, hurst2]),
            'both_mean_reverting': (hurst1 < 0.5) and (hurst2 < 0.5),
            'spread_mean_reverting': hurst_spread < 0.5}
        return pd.DataFrame([results])

    # -----------------------------------------------------------------------------------------
    # Mean Reversion Dynamics - Tests the speed of reversion between assets
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    # Volatilty Structure - Stability and Relative Volatility for hedge sizing
    # -----------------------------------------------------------------------------------------
    def volatility_analysis(self):
        returns1 = self.df[self.ticker1].pct_change()
        returns2 = self.df[self.ticker2].pct_change()
        if self.spread is None:
            self.spread_calc()
        spread_returns = self.spread.pct_change()
        results = {
            f'{self.ticker1}_volatility': returns1.std() * np.sqrt(252),
            f'{self.ticker2}_volatility': returns2.std() * np.sqrt(252),
            'spread_volatility': spread_returns.std() * np.sqrt(252),
            'volatility_ratio': (spread_returns.std() / 
                            ((returns1.std() + returns2.std()) / 2)),
            'rolling_volatility_correlation': returns1.rolling(30).std().corr(
                returns2.rolling(30).std())}
        return pd.DataFrame([results])


    # -----------------------------------------------------------------------------------------
    # Returns Current Correlation of different windows
    def correlationWindow_current(self, windows=[30, 60, 252]):
        returns1 = self.log_df[self.ticker1].pct_change()
        returns2 = self.log_df[self.ticker2].pct_change()
        results = {}
        for window in windows:
            corr = returns1.rolling(window).corr(returns2)
            results[f'window_{window}'] = {'current': corr.iloc[-1]}
        corr_df = pd.DataFrame(results)
        return corr_df
    # Returns Statistics on Rolling Correlations for the different Windows
    def correlationWindows_breakdown(self, windows=[30, 60, 252]):
        returns1 = self.log_df[self.ticker1].pct_change()
        returns2 = self.log_df[self.ticker2].pct_change()
        results = {}
        for window in windows:
            corr = returns1.rolling(window).corr(returns2)
            mean_corr = corr.mean()
            std_corr = corr.std()
            stability = 1 / (1 + (std_corr / abs(mean_corr))) if mean_corr != 0 else 0
            results[f'window_{window}'] = {
                'current': corr.iloc[-1],
                'mean': corr.mean(),
                'std': corr.std(),
                'min': corr.min(),
                'max': corr.max(),
                'stability': stability}
        corr_df = pd.DataFrame(results)
        return corr_df
    
    # -----------------------------------------------------------------------------------------


    # -----------------------------------------------------------------------------------------
    # --------------------------- Plotting ----------------------------------------------------
    
    # 2 Individual time series plots
    def plot_rawPrices_timeSeries(self):
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(self.df.index, self.df[self.ticker1], color='darkorange', label=self.ticker1)
        ax1.set_ylabel(f'{self.ticker1} Price', color='darkorange')
        ax1.tick_params(axis='y', labelcolor='darkorange')
        ax2 = ax1.twinx()
        ax2.plot(self.df.index, self.df[self.ticker2], color='steelblue', label=self.ticker2)
        ax2.set_ylabel(f'{self.ticker2} Price', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        plt.title(f'{self.ticker1} vs {self.ticker2} (Raw Prices)')
        plt.grid(True)
        fig.tight_layout()
        plt.show()
    def plot_logPrices_timeSeries(self):
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(self.log_df.index, self.log_df[self.ticker1], color='darkorange', label=f'Log({self.ticker1})')
        ax1.set_ylabel(f'Log({self.ticker1})', color='darkorange')
        ax1.tick_params(axis='y', labelcolor='darkorange')
        ax2 = ax1.twinx()
        ax2.plot(self.log_df.index, self.log_df[self.ticker2], color='steelblue', label=f'Log({self.ticker2})')
        ax2.set_ylabel(f'Log({self.ticker2})', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        plt.title(f'{self.ticker1} vs {self.ticker2} (Log Prices)')
        plt.grid(True)
        fig.tight_layout()
        plt.show()

    # Spread Time Series and Distributions
    def plot_spread_timeSeries(self):
        if self.spread is None:
            self.spread_calc()
        plt.figure(figsize=(14, 5))
        plt.plot(self.spread.index, self.spread, label='Spread', color='navy', linewidth=2)
        plt.title(f'{self.ticker2} and {self.ticker1} Dollar Neutral Spread')
        plt.xlabel('Date')
        plt.ylabel('Log Spread (Roughly % Deviation from Equilibrium Value)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    def plot_spread_distribution(self, bins=50, kde=True):
        if self.spread is None:
            self.spread_calc()
        plt.figure(figsize=(10, 5))
        sns.histplot(self.spread.dropna(), bins=bins, kde=kde, color='navy', edgecolor='white')
        plt.axvline(self.spread.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean ≈ α = {self.alpha:.4f}')
        plt.title(f'Distribution of Spread: {self.ticker2} - β·{self.ticker1}')
        plt.xlabel('Log Spread (Roughly % Deviation from Equilibrium Value)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # CORRELATIONS
    # Plot rolling correlations for different window sizes
    def plot_rolling_correlation(self, windows=[30, 60, 252, 400], figsize=(12, 8), title=None):
        returns1 = self.log_df[self.ticker1].pct_change()
        returns2 = self.log_df[self.ticker2].pct_change()
        plt.figure(figsize=figsize)
        for window in windows:
            corr = returns1.rolling(window).corr(returns2)
            plt.plot(corr.index, corr.values, label=f'{window}-day correlation', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title(title or f'Rolling Correlation: {self.ticker1} vs {self.ticker2}', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Correlation', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-1, 1)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    # Correlation Breakdown 
    def plot_correlation_heatmap(self, windows=[30, 60, 252], figsize=(10, 6)):
        corr_breakdown = self.correlationWindows_breakdown(windows)
        corr_breakdown_T = corr_breakdown.T
        plt.figure(figsize=figsize)
        sns.heatmap(corr_breakdown_T, annot=True, cmap='RdYlBu_r', center=0, 
                    fmt='.3f', cbar_kws={'label': 'Correlation Value'})
        plt.title(f'Correlation Statistics Heatmap: {self.ticker1} vs {self.ticker2}', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Statistics', fontsize=12)
        plt.ylabel('Window Size', fontsize=12)
        plt.tight_layout()
        plt.show()
    # Plotting the Distributions of different windows
    def plot_correlation_distributions(self, windows=[30, 60, 252], figsize=(12, 8)):
        returns1 = self.log_df[self.ticker1].pct_change()
        returns2 = self.log_df[self.ticker2].pct_change()
        n = len(windows)
        ncols = 2
        nrows = (n + 1) // 2  # ensures enough rows for 2 columns
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
        axes = axes.flatten()
        for i, window in enumerate(windows):
            corr = returns1.rolling(window).corr(returns2).dropna()
            mean_val = corr.mean()
            median_val = corr.median()
            std_val = corr.std()
            ax = axes[i]
            ax.hist(corr.values, bins=30, alpha=0.7, color='skyblue',
                    edgecolor='black', density=True)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                    label=f'Median: {median_val:.2f}')
            ax.axvline(mean_val + std_val, color='blue', linestyle=':', linewidth=2,
                    label=f'+1 STD: {mean_val + std_val:.2f}')
            ax.axvline(mean_val - std_val, color='blue', linestyle=':', linewidth=2,
                    label=f'-1 STD: {mean_val - std_val:.2f}')
            ax.set_title(f'{window}-Day Window', fontweight='bold')
            ax.set_xlabel('Correlation')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        axes[0].set_ylabel('Density')
        plt.suptitle(f'Rolling Correlation Distributions: {self.ticker1} vs {self.ticker2}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


class Screen_MultiPair_Metrics:
    def __init__(self, tickers, days):
        self.tickers = tickers
        self.days = days
        self.pairs = list(combinations(tickers, 2))
        self.screeners = {}
        for pair in self.pairs:
            pair_key = f"{pair[0]} / {pair[1]}"
            self.screeners[pair_key] = Screen_SinglePair_Metrics(list(pair), days) 
    
    def get_all_pairs(self):
        return self.pairs
    
    # -----------------------------------------------------------------------------------------    
    def johansen_test_all(self, det_order=0, maxlags=10, ic='aic'):
        all_results = []
        for pair_key, screener in self.screeners.items():
            result = screener.johansen_test(det_order=det_order, maxlags=maxlags, ic=ic)
            if isinstance(result, pd.DataFrame):
                result = result.squeeze()  
            if 'pair' in result.index:
                result = result.drop('pair')
            result.name = pair_key
            all_results.append(result)
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('trace_statistic', ascending=False)

    def engle_granger_Test_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            result = screener.engle_granger_Test()
            if isinstance(result, pd.DataFrame):
                result = result.squeeze()  
            result.name = pair_key
            all_results.append(result)
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('eg_t_stat', ascending=True)

    def adf_SNR_spread_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            result = screener.adf_SNR_spread()
            if isinstance(result, pd.DataFrame):
                result = result.squeeze()  
            result.name = pair_key
            all_results.append(result)
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('adf_statistic')

    def half_life_analysis_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            result = screener.halfLife_estimate()
            if isinstance(result, pd.DataFrame):
                result = result.squeeze()  
            if 'pair' in result.index:
                result = result.drop('pair')
            result.name = pair_key
            all_results.append(result)
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('halflife', ascending=False)

    def hurst_exponent_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            result = screener.hurst_exponent()
            if isinstance(result, pd.DataFrame):
                result = result.squeeze()  
            if 'pair' in result.index:
                result = result.drop('pair')
            result.name = pair_key
            all_results.append(result)
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('hurst_spread')

    # -----------------------------------------------------------------------------------------
    # Returns Current Correlation of different windows for all pairs
    def correlationWindow_current_all(self, windows=[30, 60, 252]):
        all_results = {}
        for pair_key, screener in self.screeners.items():
            pair_name = screener.ticker1 + " / " + screener.ticker2
            result = screener.correlationWindow_current(windows)
            all_results[pair_name] = result.iloc[0].to_dict()
        df = pd.DataFrame(all_results).T
        return df.sort_values('window_252', ascending=False)

    # -----------------------------------------------------------------------------------------
    def comprehensive_analysis_all(self, det_order=1, maxlags=10, ic='aic', windows=[30, 60, 252]):
        all_results = []
        for pair_key, screener in self.screeners.items():
            result_dict = {}
            try:
                eg_result = screener.engle_granger_Test()
                if isinstance(eg_result, pd.DataFrame):
                    eg_result = eg_result.squeeze()
                for key in eg_result.index:
                    if key != 'pair':
                        result_dict[f'eg_{key}'] = eg_result[key]
            except Exception as e:
                print(f"Error in Engle-Granger test for {pair_key}: {e}")
            try:
                adf_result = screener.adf_SNR_spread()
                if isinstance(adf_result, pd.DataFrame):
                    adf_result = adf_result.squeeze()
                for key in adf_result.index:
                    if key != 'pair':
                        result_dict[key] = adf_result[key]
            except Exception as e:
                print(f"Error in ADF/SNR test for {pair_key}: {e}")
            try:
                johansen_result = screener.johansen_test(det_order=det_order, maxlags=maxlags, ic=ic)
                if isinstance(johansen_result, pd.DataFrame):
                    johansen_result = johansen_result.squeeze()
                for key in johansen_result.index:
                    if key != 'pair':
                        result_dict[f'johansen_{key}'] = johansen_result[key]
            except Exception as e:
                print(f"Error in Johansen test for {pair_key}: {e}")
            try:
                halflife_result = screener.halfLife_estimate()
                if isinstance(halflife_result, pd.DataFrame):
                    halflife_result = halflife_result.squeeze()
                for key in halflife_result.index:
                    if key != 'pair':
                        result_dict[f'halflife_{key}'] = halflife_result[key]
            except Exception as e:
                print(f"Error in half-life analysis for {pair_key}: {e}")
            try:
                hurst_result = screener.hurst_exponent()
                if isinstance(hurst_result, pd.DataFrame):
                    hurst_result = hurst_result.squeeze()
                for key in hurst_result.index:
                    if key != 'pair':
                        result_dict[f'hurst_{key}'] = hurst_result[key]
            except Exception as e:
                print(f"Error in Hurst analysis for {pair_key}: {e}")
            try:
                corr_result = screener.correlationWindows_breakdown(windows)
                if isinstance(corr_result, pd.DataFrame):
                    for window in windows:
                        col_name = f'window_{window}'
                        if col_name in corr_result.columns:
                            for row_name in corr_result.index:
                                result_dict[f'corr_{col_name}_{row_name}'] = corr_result.loc[row_name, col_name]
            except Exception as e:
                print(f"Error in correlation analysis for {pair_key}: {e}")
            result = pd.Series(result_dict)
            result.name = pair_key
            all_results.append(result)
        df = pd.concat(all_results, axis=1).T
        try:
            if 'eg_p_value' in df.columns:
                return df.sort_values('eg_p_value')
            elif 'adf_p_value' in df.columns:
                return df.sort_values('adf_p_value')
            elif 'adf_statistic' in df.columns:
                return df.sort_values('adf_statistic')
            else:
                return df
        except:
            return df


    @staticmethod
    def format_ticker_with_tenor(currency, tenor):
        tenor_map = {
            '1W': 'V1W',
            '2W': 'V2W', 
            '1M': 'V1M',
            '2M': 'V2M',
            '3M': 'V3M',
            '6M': 'V6M',
            '1Y': 'V1Y'}
        if tenor not in tenor_map:
            raise ValueError(f"Unsupported tenor: {tenor}. Supported tenors: {list(tenor_map.keys())}")
        return f"{currency}{tenor_map[tenor]} Curncy"

    @classmethod
    def create_from_currencies(cls, currencies, tenor='1M', days=730):
        formatted_tickers = [cls.format_ticker_with_tenor(curr, tenor) for curr in currencies]
        return cls(formatted_tickers, days)


    def analyze_currencies(currencies, tenor='1M', days=730):
        analyzer = Screen_MultiPair_Metrics.create_from_currencies(currencies, tenor, days)
        return analyzer.get_best_pairs_comprehensive()
    
    

    def get_best_pairs_comprehensive(self, det_order=0, maxlags=10, ic='aic'):
        results = self.comprehensive_analysis_all(det_order=det_order, maxlags=maxlags, ic=ic)
        if results.empty:
            return pd.DataFrame()
        results = results.copy()
        results['cointegration_score'] = 0.0
        if 'eg_eg_p_value' in results.columns:
            results.loc[results['eg_eg_p_value'] < 0.001, 'cointegration_score'] += 10  # Very strong
            results.loc[results['eg_eg_p_value'] < 0.01, 'cointegration_score'] += 7   # Strong
            results.loc[results['eg_eg_p_value'] < 0.05, 'cointegration_score'] += 4   # Significant
            results.loc[results['eg_eg_p_value'] < 0.10, 'cointegration_score'] += 2   # Marginal
        if 'eg_ols_r_squared' in results.columns:
            results.loc[results['eg_ols_r_squared'] > 0.8, 'cointegration_score'] += 3
            results.loc[results['eg_ols_r_squared'] > 0.6, 'cointegration_score'] += 2
            results.loc[results['eg_ols_r_squared'] > 0.4, 'cointegration_score'] += 1
        if 'adf_reject_5pct' in results.columns:
            results.loc[results['adf_reject_5pct'] == True, 'cointegration_score'] += 8
        if 'adf_reject_1pct' in results.columns:
            results.loc[results['adf_reject_1pct'] == True, 'cointegration_score'] += 5  # Additional bonus for 1%
        if 'adf_statistic' in results.columns:
            adf_percentiles = results['adf_statistic'].quantile([0.25, 0.5, 0.75])
            results.loc[results['adf_statistic'] < adf_percentiles[0.25], 'cointegration_score'] += 4
            results.loc[results['adf_statistic'] < adf_percentiles[0.5], 'cointegration_score'] += 2
        if 'johansen_trace_reject_h0' in results.columns:
            results.loc[results['johansen_trace_reject_h0'] == True, 'cointegration_score'] += 6
        if 'johansen_max_eigen_reject_h0' in results.columns:
            results.loc[results['johansen_max_eigen_reject_h0'] == True, 'cointegration_score'] += 4
        if 'johansen_eigenvalue_1' in results.columns:
            eigen_median = results['johansen_eigenvalue_1'].median()
            results.loc[results['johansen_eigenvalue_1'] > eigen_median * 1.5, 'cointegration_score'] += 3
            results.loc[results['johansen_eigenvalue_1'] > eigen_median, 'cointegration_score'] += 2
        if 'halflife_halflife' in results.columns:
            results.loc[(results['halflife_halflife'] >= 2) & (results['halflife_halflife'] <= 30), 'cointegration_score'] += 4
            results.loc[(results['halflife_halflife'] >= 1) & (results['halflife_halflife'] <= 60), 'cointegration_score'] += 2
            results.loc[results['halflife_halflife'] > 365, 'cointegration_score'] -= 2
        if 'halflife_phi_pvalue' in results.columns:
            results.loc[results['halflife_phi_pvalue'] < 0.01, 'cointegration_score'] += 3
            results.loc[results['halflife_phi_pvalue'] < 0.05, 'cointegration_score'] += 2
        if 'hurst_hurst_spread' in results.columns:
            results.loc[results['hurst_hurst_spread'] < 0.4, 'cointegration_score'] += 4  # Strong mean reversion
            results.loc[results['hurst_hurst_spread'] < 0.45, 'cointegration_score'] += 3
            results.loc[results['hurst_hurst_spread'] < 0.5, 'cointegration_score'] += 2
            results.loc[results['hurst_hurst_spread'] > 0.6, 'cointegration_score'] -= 2
        if 'hurst_spread_mean_reverting' in results.columns:
            results.loc[results['hurst_spread_mean_reverting'] == True, 'cointegration_score'] += 2
        if 'snr_ratio' in results.columns:
            snr_percentiles = results['snr_ratio'].quantile([0.5, 0.75, 0.9])
            results.loc[results['snr_ratio'] > snr_percentiles[0.9], 'cointegration_score'] += 4
            results.loc[results['snr_ratio'] > snr_percentiles[0.75], 'cointegration_score'] += 3
            results.loc[results['snr_ratio'] > snr_percentiles[0.5], 'cointegration_score'] += 2
        if 'corr_window_252_stability' in results.columns:
            results.loc[results['corr_window_252_stability'] > 0.8, 'cointegration_score'] += 2
            results.loc[results['corr_window_252_stability'] > 0.6, 'cointegration_score'] += 1
        if 'corr_window_252_current' in results.columns:
            results.loc[abs(results['corr_window_252_current']) > 0.7, 'cointegration_score'] += 1
            results.loc[abs(results['corr_window_252_current']) > 0.8, 'cointegration_score'] += 1
        agreement_count = 0
        if 'eg_eg_p_value' in results.columns:
            agreement_count += (results['eg_eg_p_value'] < 0.05).astype(int)
        if 'adf_reject_5pct' in results.columns:
            agreement_count += results['adf_reject_5pct'].astype(int)
        if 'johansen_trace_reject_h0' in results.columns:
            agreement_count += results['johansen_trace_reject_h0'].astype(int)
        results.loc[agreement_count >= 3, 'cointegration_score'] += 5  # All tests agree
        results.loc[agreement_count == 2, 'cointegration_score'] += 3  # Two tests agree
        results['cointegration_rank'] = results['cointegration_score'].rank(ascending=False, method='dense')
        results['cointegration_percentile'] = results['cointegration_score'].rank(pct=True) * 100
        def interpret_score(score):
            if score >= 25:
                return "Very Strong"
            elif score >= 18:
                return "Strong" 
            elif score >= 12:
                return "Moderate"
            elif score >= 6:
                return "Weak"
            else:
                return "Very Weak"
        results['cointegration_strength'] = results['cointegration_score'].apply(interpret_score)
        results_sorted = results.sort_values('cointegration_score', ascending=False)
        return results_sorted




# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def analyze_multiple_tenors_to_excel(currencies, tenors, days=730, output_file='cointegration_analysis.xlsx'):
    all_results = {}
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for tenor in tenors:
            try:
                analyzer = Screen_MultiPair_Metrics.create_from_currencies(
                    currencies, tenor, days)
                results = analyzer.get_best_pairs_comprehensive()
                if not results.empty:
                    output_columns = ['eg_eg_t_stat', 'eg_eg_p_value', 'eg_intercept', 'eg_hedge_ratio',
                                        'eg_ols_r_squared', 'adf_statistic', 'adf_p_value', 'adf_critical_1pct',
                                        'adf_critical_5pct', 'adf_critical_10pct', 'adf_reject_5pct',
                                        'adf_reject_1pct', 'snr_ratio', 'johansen_trace_statistic',
                                        'johansen_trace_critical_5pct', 'johansen_trace_reject_h0',
                                        'johansen_max_eigen_statistic', 'johansen_max_eigen_critical_5pct',
                                        'johansen_max_eigen_reject_h0', 'johansen_eigenvalue_1',
                                        'johansen_optimal_lag', 'johansen_k_ar_diff', 'halflife_halflife',
                                        'halflife_phi', 'halflife_phi_pvalue', 'halflife_confidence_interval',
                                        'halflife_r_squared', 'hurst_hurst_ticker1', 'hurst_hurst_ticker2',
                                        'hurst_hurst_spread', 'hurst_mean_hurst', 'hurst_both_mean_reverting',
                                        'hurst_spread_mean_reverting', 'corr_window_30_current',
                                        'corr_window_30_mean', 'corr_window_30_std', 'corr_window_30_min',
                                        'corr_window_30_max', 'corr_window_30_stability',
                                        'corr_window_60_current', 'corr_window_60_mean', 'corr_window_60_std',
                                        'corr_window_60_min', 'corr_window_60_max', 'corr_window_60_stability',
                                        'corr_window_252_current', 'corr_window_252_mean',
                                        'corr_window_252_std', 'corr_window_252_min', 'corr_window_252_max',
                                        'corr_window_252_stability', 'cointegration_score',
                                        'cointegration_rank', 'cointegration_percentile',
                                        'cointegration_strength']
                    available_columns = [col for col in output_columns if col in results.columns]
                    results_filtered = results[available_columns].copy()
                    numerical_cols = results_filtered.select_dtypes(include=[np.number]).columns
                    results_filtered[numerical_cols] = results_filtered[numerical_cols].round(4)
                    all_results[tenor] = results_filtered
                    sheet_name = f'{tenor}'
                    results_filtered.to_excel(writer, sheet_name=sheet_name)
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                    print(f"Top 5 pairs for {tenor}:")
                    top_5 = results_filtered.head(5)
                    for idx, (pair, row) in enumerate(top_5.iterrows(), 1):
                        print(f"  {idx}. {pair}")
                        print(f"     Score: {row.get('cointegration_score', 'N/A'):.2f}")
                        print(f"     Strength: {row.get('cointegration_strength', 'N/A')}")
                        if 'eg_eg_p_value' in row:
                            print(f"     EG p-value: {row['eg_eg_p_value']:.4f}")
                else:
                    print(f"  No results for tenor {tenor}") 
            except Exception as e:
                print(f"  Error analyzing tenor {tenor}: {e}")
                continue
    print(f"\nAnalysis complete! Results saved to: {output_file}")
    return all_results



def create_summary_sheet(results_dict, output_file='cointegration_analysis.xlsx'):
    summary_data = []
    for tenor, df in results_dict.items():
        if not df.empty:
            # Get top 3 pairs for each tenor
            top_pairs = df.head(3)
            for rank, (pair, row) in enumerate(top_pairs.iterrows(), 1):
                summary_data.append({
                    'Tenor': tenor,
                    'Rank': rank,
                    'Pair': pair,
                    'Score': row.get('cointegration_score', np.nan),
                    'Strength': row.get('cointegration_strength', 'N/A'),
                    'EG p-value': row.get('eg_eg_p_value', np.nan),
                    'R-squared': row.get('eg_ols_r_squared', np.nan)})
    summary_df = pd.DataFrame(summary_data)
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        if 'Summary' in writer.book.sheetnames:
            del writer.book['Summary']
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        wb = writer.book
        wb.move_sheet('Summary', offset=-len(wb.sheetnames)+1)
        worksheet = writer.sheets['Summary']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width



# Example usage function
def run_multi_tenor_analysis(currencies=None, tenors=None, days=730, output_file='cointegration_results.xlsx'):
    if currencies is None:
        currencies = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
                     'NZDUSD', 'USDCHF', 'USDMXN', 'USDBRL']
    if tenors is None:
        tenors = ['1W', '1M', '3M', '6M', '1Y']
    print("=" * 60)
    print("MULTI-TENOR COINTEGRATION ANALYSIS")
    print("=" * 60)
    print(f"Currencies: {', '.join(currencies)}")
    print(f"Tenors: {', '.join(tenors)}")
    print(f"Historical days: {days}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    results = analyze_multiple_tenors_to_excel(
        currencies=currencies,
        tenors=tenors,
        days=days,
        output_file=output_file)
    if results:
        create_summary_sheet(results, output_file)
        print(f"\nSummary sheet added to: {output_file}")
    return results