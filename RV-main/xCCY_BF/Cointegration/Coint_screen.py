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
from xbbg import blp, pipeline
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.api import OLS, add_constant
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import matplotlib.dates as mdates
import warnings




def Data_DailyCLOSE_Multiple(tickers, days):
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    df_all = {}
    for ticker in tickers:
        data_single = blp.bdh(
            tickers=f"{ticker}",
            flds=["PX_LAST"],
            start_date=start_date,
            end_date=end_date,
            Per="D")
        data_single.columns = [ticker]
        df_all[ticker] = data_single
    df_all = pd.concat(df_all, axis=1)
    df_all.columns = df_all.columns.droplevel(0) if isinstance(df_all.columns, pd.MultiIndex) else df_all.columns
    df_all = df_all.sort_index(ascending=True)
    return df_all

def plot2TimeSeries(df, col1, col2):
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(df.index, df[f'{col1}'], label=f'{col1}', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{col1}', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df[f'{col2}'], label=f'{col2}', linewidth=2, linestyle='--', color='orange')
    ax2.set_ylabel(f'{col2}', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title(f'{col1} (left) and {col2} (right)')
    plt.grid(True)
    return plt.show()



class Screen_SinglePair_Metrics:
    def __init__(self, tickers, days):
        self.ticker1 = tickers[0]
        self.ticker2 = tickers[1]
        raw_df = Data_DailyCLOSE_Multiple(tickers, days)
        
        # Clean the data
        raw_df = raw_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check data quality
        if len(raw_df) < 100:
            raise ValueError(f"Insufficient data for {tickers}: only {len(raw_df)} observations")
        
        # For butterflies, work in levels (not logs)
        self.df = raw_df.copy()
        self.log_df = self.df.copy()  # Keep same for compatibility, but no log transform
        
        self.spread = None
        self.alpha = None
        self.beta = None
        self.split_point = int(len(self.df) * 0.7)

    def spread_calc(self):
        """Calculate spread using levels (appropriate for butterflies)"""
        x = self.df[self.ticker1]
        y = self.df[self.ticker2]
        
        # Check for valid data
        if x.isna().any() or y.isna().any():
            raise ValueError("Data contains NaN values")
        
        X0 = sm.add_constant(x)
        ols = sm.OLS(y, X0).fit()
        beta = ols.params[self.ticker1]
        alpha = ols.params['const']
        spread = y - (alpha + beta * x)
        self.spread = spread
        self.alpha = alpha
        self.beta = beta

    def spread_stats(self):
        """Summary statistics of the spread"""
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

    def engle_granger_Test(self):
        """Engle-Granger cointegration test"""
        x = self.df[self.ticker1]
        y = self.df[self.ticker2]
        
        # Suppress collinearity warning (common with similar butterflies)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            eg_t, eg_p, _ = coint(x, y)
        
        X0 = sm.add_constant(x)
        ols = sm.OLS(y, X0).fit()
        alpha = ols.params['const']
        beta = ols.params[self.ticker1]
        results = {
            'eg_t_stat': eg_t,
            'eg_p_value': eg_p,
            'intercept': alpha,
            'hedge_ratio': beta,
            'ols_r_squared': ols.rsquared
        }
        df_egTest = pd.DataFrame([results])
        return df_egTest.T

    def adf_SNR_spread(self):
        """ADF test and SNR on spread"""
        if self.spread is None:
            self.spread_calc()
        spread = self.spread
        s_lag = spread.shift(1)
        ds = spread - s_lag
        df = pd.concat([ds, s_lag], axis=1).dropna()
        df.columns = ['ds', 's_lag']
        X1 = sm.add_constant(df['s_lag'])
        arres = sm.OLS(df['ds'], X1).fit()
        phi = arres.params['s_lag']
        sigma = arres.resid.std()
        snr = (1 - phi) / sigma if sigma > 0 else 0
        stat, p, _, _, crit, _ = adfuller(spread.dropna(), autolag='AIC')
        results = {
            'adf_statistic': stat,
            'adf_p_value': p,
            'adf_critical_1pct': crit['1%'],
            'adf_critical_5pct': crit['5%'],
            'adf_critical_10pct': crit['10%'],
            'adf_reject_5pct': stat < crit['5%'],
            'adf_reject_1pct': stat < crit['1%'],
            'snr_ratio': snr
        }
        df_adfTest = pd.DataFrame([results])
        return df_adfTest.T

    def halfLife_estimate(self):
        """Estimate mean-reversion half-life"""
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
        results = {
            'halflife': halflife,
            'phi': theta,
            'phi_pvalue': theta_pvalue,
            'confidence_interval': [CI],
            'r_squared': model.rsquared
        }
        return pd.DataFrame(results)

    def hurst_exponent(self, lags_range=(2, 100)):
        """Calculate Hurst exponent for mean-reversion analysis"""
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
        
        # Work with levels for butterflies
        hurst1 = calculate_hurst(px1.values)
        hurst2 = calculate_hurst(px2.values)
        
        if self.spread is None:
            self.spread_calc()
        spread_level = self.spread.dropna().values
        hurst_spread = calculate_hurst(spread_level)
        
        results = {
            'pair': f"{self.ticker1} / {self.ticker2}",
            'hurst_ticker1': hurst1,
            'hurst_ticker2': hurst2,
            'hurst_spread': hurst_spread,
            'mean_hurst': np.nanmean([hurst1, hurst2]),
            'both_mean_reverting': (hurst1 < 0.5) and (hurst2 < 0.5),
            'spread_mean_reverting': hurst_spread < 0.5
        }
        return pd.DataFrame([results])

    def out_of_sample_validation(self):
        """Test if cointegration holds out-of-sample"""
        in_sample = self.df.iloc[:self.split_point]
        X_in = sm.add_constant(in_sample[self.ticker1])
        model_in = sm.OLS(in_sample[self.ticker2], X_in).fit()
        
        out_sample = self.df.iloc[self.split_point:]
        spread_out = (out_sample[self.ticker2] -
                      (model_in.params['const'] +
                       model_in.params[self.ticker1] * out_sample[self.ticker1]))
        
        adf_stat_out, p_val_out, _, _, crit_out, _ = adfuller(spread_out.dropna(), autolag='AIC')
        
        in_sample_spread = (in_sample[self.ticker2] -
                           (model_in.params['const'] +
                            model_in.params[self.ticker1] * in_sample[self.ticker1]))
        
        vol_ratio = spread_out.std() / in_sample_spread.std() if in_sample_spread.std() > 0 else np.inf
        
        results = {
            'oos_adf_statistic': adf_stat_out,
            'oos_adf_pvalue': p_val_out,
            'oos_adf_critical_5pct': crit_out['5%'],
            'oos_stationary': p_val_out < 0.05,
            'oos_spread_std': spread_out.std(),
            'is_spread_std': in_sample_spread.std(),
            'oos_volatility_ratio': vol_ratio,
            'oos_stable': vol_ratio < 1.5
        }
        return pd.DataFrame([results])

    def variance_ratio_test(self, lags=[2, 5, 10]):
        """Variance ratio test for random walk hypothesis"""
        if self.spread is None:
            self.spread_calc()
        
        spread_returns = self.spread.diff().dropna()
        var_1 = spread_returns.var()
        
        results = {}
        for lag in lags:
            lag_returns = self.spread.diff(lag).dropna()
            var_lag = lag_returns.var()
            
            vr = var_lag / (lag * var_1) if var_1 > 0 else np.nan
            
            n = len(spread_returns)
            variance_vr = 2 * (2 * lag - 1) * (lag - 1) / (3 * lag * n)
            z_stat = (vr - 1) / np.sqrt(variance_vr) if variance_vr > 0 else np.nan
            
            results[f'vr_lag_{lag}'] = vr
            results[f'vr_z_stat_{lag}'] = z_stat
            results[f'vr_reject_rw_{lag}'] = abs(z_stat) > 1.96
        
        return pd.DataFrame([results])

    def volatility_analysis(self):
        """Analyze volatility structure"""
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
                returns2.rolling(30).std())
        }
        return pd.DataFrame([results])




class Screen_MultiPair_Metrics:
    def __init__(self, tickers, days):
        self.tickers = tickers
        self.days = days
        self.pairs = list(combinations(tickers, 2))
        self.screeners = {}
        
        # Build screeners with error handling
        for pair in self.pairs:
            pair_key = f"{pair[0]} / {pair[1]}"
            try:
                self.screeners[pair_key] = Screen_SinglePair_Metrics(list(pair), days)
            except Exception as e:
                print(f"Warning: Could not create screener for {pair_key}: {e}")
                continue
    
    def get_all_pairs(self):
        return self.pairs
    
    def engle_granger_Test_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            try:
                result = screener.engle_granger_Test()
                if isinstance(result, pd.DataFrame):
                    result = result.squeeze()
                result.name = pair_key
                all_results.append(result)
            except Exception as e:
                print(f"Error in EG test for {pair_key}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('eg_t_stat', ascending=True)
    
    def adf_SNR_spread_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            try:
                result = screener.adf_SNR_spread()
                if isinstance(result, pd.DataFrame):
                    result = result.squeeze()
                result.name = pair_key
                all_results.append(result)
            except Exception as e:
                print(f"Error in ADF test for {pair_key}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('adf_statistic')
    
    def half_life_analysis_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            try:
                result = screener.halfLife_estimate()
                if isinstance(result, pd.DataFrame):
                    result = result.squeeze()
                if 'pair' in result.index:
                    result = result.drop('pair')
                result.name = pair_key
                all_results.append(result)
            except Exception as e:
                print(f"Error in half-life for {pair_key}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('halflife', ascending=False)
    
    def hurst_exponent_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            try:
                result = screener.hurst_exponent()
                if isinstance(result, pd.DataFrame):
                    result = result.squeeze()
                if 'pair' in result.index:
                    result = result.drop('pair')
                result.name = pair_key
                all_results.append(result)
            except Exception as e:
                print(f"Error in Hurst for {pair_key}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('hurst_spread')
    
    def out_of_sample_validation_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            try:
                result = screener.out_of_sample_validation()
                if isinstance(result, pd.DataFrame):
                    result = result.squeeze()
                result.name = pair_key
                all_results.append(result)
            except Exception as e:
                print(f"Error in OOS validation for {pair_key}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        df = pd.concat(all_results, axis=1).T
        return df.sort_values('oos_adf_pvalue')
    
    def variance_ratio_test_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            try:
                result = screener.variance_ratio_test()
                if isinstance(result, pd.DataFrame):
                    result = result.squeeze()
                result.name = pair_key
                all_results.append(result)
            except Exception as e:
                print(f"Error in VR test for {pair_key}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        df = pd.concat(all_results, axis=1).T
        return df
    
    def comprehensive_analysis_all(self):
        all_results = []
        for pair_key, screener in self.screeners.items():
            result_dict = {}
            
            # Engle-Granger test
            try:
                eg_result = screener.engle_granger_Test()
                if isinstance(eg_result, pd.DataFrame):
                    eg_result = eg_result.squeeze()
                for key in eg_result.index:
                    if key != 'pair':
                        result_dict[f'eg_{key}'] = eg_result[key]
            except Exception as e:
                print(f"Error in Engle-Granger test for {pair_key}: {e}")
            
            # ADF and SNR
            try:
                adf_result = screener.adf_SNR_spread()
                if isinstance(adf_result, pd.DataFrame):
                    adf_result = adf_result.squeeze()
                for key in adf_result.index:
                    if key != 'pair':
                        result_dict[key] = adf_result[key]
            except Exception as e:
                print(f"Error in ADF/SNR test for {pair_key}: {e}")
            
            # Half-life
            try:
                halflife_result = screener.halfLife_estimate()
                if isinstance(halflife_result, pd.DataFrame):
                    halflife_result = halflife_result.squeeze()
                for key in halflife_result.index:
                    if key != 'pair':
                        result_dict[f'halflife_{key}'] = halflife_result[key]
            except Exception as e:
                print(f"Error in half-life analysis for {pair_key}: {e}")
            
            # Hurst exponent
            try:
                hurst_result = screener.hurst_exponent()
                if isinstance(hurst_result, pd.DataFrame):
                    hurst_result = hurst_result.squeeze()
                for key in hurst_result.index:
                    if key != 'pair':
                        result_dict[f'hurst_{key}'] = hurst_result[key]
            except Exception as e:
                print(f"Error in Hurst analysis for {pair_key}: {e}")
            
            # Out-of-sample validation
            try:
                oos_result = screener.out_of_sample_validation()
                if isinstance(oos_result, pd.DataFrame):
                    oos_result = oos_result.squeeze()
                for key in oos_result.index:
                    result_dict[f'oos_{key}'] = oos_result[key]
            except Exception as e:
                print(f"Error in OOS validation for {pair_key}: {e}")
            
            # Variance ratio test
            try:
                vr_result = screener.variance_ratio_test()
                if isinstance(vr_result, pd.DataFrame):
                    vr_result = vr_result.squeeze()
                for key in vr_result.index:
                    result_dict[f'vr_{key}'] = vr_result[key]
            except Exception as e:
                print(f"Error in variance ratio test for {pair_key}: {e}")

            # Only add if we got some results
            if result_dict:
                result = pd.Series(result_dict)
                result.name = pair_key
                all_results.append(result)

        if not all_results:
            return pd.DataFrame()
        
        df = pd.concat(all_results, axis=1).T
        return df

    @staticmethod
    def format_ticker_with_tenor(currency, tenor):
        tenor_map_bf = {
            '1W': '10B1W',
            '2W': '10B2W',
            '1M': '10B1M',
            '2M': '10B2M',
            '3M': '10B3M',
            '6M': '10B6M',
            '1Y': '10B1Y'}
        if tenor not in tenor_map_bf:
            raise ValueError(f"Unsupported tenor: {tenor}. Supported tenors: {list(tenor_map_bf.keys())}")
        return f"{currency}{tenor_map_bf[tenor]} BGN Curncy"
    
    @classmethod
    def create_from_currencies(cls, currencies, tenor='1M', days=730, metric_type='ATM'):
        formatted_tickers = [cls.format_ticker_with_tenor(curr, tenor) for curr in currencies]
        return cls(formatted_tickers, days)

    def get_best_pairs_comprehensive(self):
        results = self.comprehensive_analysis_all()
        if results.empty:
            print("Warning: No results to score")
            return pd.DataFrame()
        
        results = results.copy()
        results['cointegration_score'] = 0.0

        # PRIMARY TEST: Engle-Granger (Weight: 35%)
        if 'eg_eg_p_value' in results.columns:
            results.loc[results['eg_eg_p_value'] < 0.001, 'cointegration_score'] += 17.5
            results.loc[results['eg_eg_p_value'] < 0.01, 'cointegration_score'] += 14.0
            results.loc[results['eg_eg_p_value'] < 0.05, 'cointegration_score'] += 10.5
            results.loc[results['eg_eg_p_value'] < 0.10, 'cointegration_score'] += 7.0
        
        if 'eg_ols_r_squared' in results.columns and 'eg_eg_p_value' in results.columns:
            sig_mask = results['eg_eg_p_value'] < 0.05
            results.loc[sig_mask & (results['eg_ols_r_squared'] > 0.7), 'cointegration_score'] += 3.5
            results.loc[sig_mask & (results['eg_ols_r_squared'] > 0.5), 'cointegration_score'] += 2.0
        
        # STATIONARITY: ADF Test on Spread (Weight: 30%)
        if 'adf_p_value' in results.columns:
            results.loc[results['adf_p_value'] < 0.001, 'cointegration_score'] += 15.0
            results.loc[results['adf_p_value'] < 0.01, 'cointegration_score'] += 12.0
            results.loc[results['adf_p_value'] < 0.05, 'cointegration_score'] += 9.0
            results.loc[results['adf_p_value'] < 0.10, 'cointegration_score'] += 6.0
        
        if 'snr_ratio' in results.columns:
            snr_75th = results['snr_ratio'].quantile(0.75)
            snr_90th = results['snr_ratio'].quantile(0.90)
            results.loc[results['snr_ratio'] > snr_90th, 'cointegration_score'] += 3.0
            results.loc[results['snr_ratio'] > snr_75th, 'cointegration_score'] += 1.5
        
        # OUT-OF-SAMPLE VALIDATION (Weight: 20%)
        if 'oos_oos_stationary' in results.columns:
            results.loc[results['oos_oos_stationary'] == True, 'cointegration_score'] += 12.0
        
        if 'oos_oos_adf_pvalue' in results.columns:
            results.loc[results['oos_oos_adf_pvalue'] < 0.01, 'cointegration_score'] += 4.0
            results.loc[results['oos_oos_adf_pvalue'] < 0.05, 'cointegration_score'] += 2.0
        
        if 'oos_oos_volatility_ratio' in results.columns:
            results.loc[results['oos_oos_volatility_ratio'] > 2.0, 'cointegration_score'] -= 4.0
            results.loc[results['oos_oos_volatility_ratio'] < 1.3, 'cointegration_score'] += 2.0
        
        # MEAN REVERSION SPEED: Half-Life (Weight: 10%)
        if 'halflife_halflife' in results.columns:
            hl = results['halflife_halflife']
            results.loc[(hl >= 5) & (hl <= 60), 'cointegration_score'] += 6.0
            results.loc[(hl > 60) & (hl <= 120), 'cointegration_score'] += 3.0
            results.loc[hl > 250, 'cointegration_score'] -= 3.0
            results.loc[hl < 2, 'cointegration_score'] -= 2.0
        
        if 'halflife_phi_pvalue' in results.columns:
            results.loc[results['halflife_phi_pvalue'] < 0.01, 'cointegration_score'] += 2.0
            results.loc[results['halflife_phi_pvalue'] < 0.05, 'cointegration_score'] += 1.0
        
        # MEAN REVERSION CHARACTER: Hurst Exponent (Weight: 5%)
        if 'hurst_hurst_spread' in results.columns:
            results.loc[results['hurst_hurst_spread'] < 0.35, 'cointegration_score'] += 3.0
            results.loc[results['hurst_hurst_spread'] < 0.42, 'cointegration_score'] += 2.5
            results.loc[results['hurst_hurst_spread'] < 0.5, 'cointegration_score'] += 1.5
            results.loc[results['hurst_hurst_spread'] > 0.6, 'cointegration_score'] -= 2.0
        
        if 'hurst_spread_mean_reverting' in results.columns:
            results.loc[results['hurst_spread_mean_reverting'] == True, 'cointegration_score'] += 1.0
        
        # VARIANCE RATIO TEST
        vr_reject_count = 0
        for lag in [2, 5, 10]:
            col_name = f'vr_vr_reject_rw_{lag}'
            if col_name in results.columns:
                vr_reject_count += results[col_name].astype(int)
        if isinstance(vr_reject_count, pd.Series):
            results.loc[vr_reject_count >= 3, 'cointegration_score'] += 3.0
            results.loc[vr_reject_count == 2, 'cointegration_score'] += 1.5
        
        # TEST AGREEMENT BONUS
        agreement_count = 0
        if 'eg_eg_p_value' in results.columns:
            agreement_count += (results['eg_eg_p_value'] < 0.05).astype(int)
        if 'adf_p_value' in results.columns:
            agreement_count += (results['adf_p_value'] < 0.05).astype(int)
        if 'oos_oos_stationary' in results.columns:
            agreement_count += results['oos_oos_stationary'].astype(int)
        
        results.loc[agreement_count >= 3, 'cointegration_score'] += 5.0
        results.loc[agreement_count == 2, 'cointegration_score'] += 2.5

                # FINAL RANKINGS AND INTERPRETATIONS
        results['cointegration_rank'] = results['cointegration_score'].rank(
            ascending=False, method='dense')
        results['cointegration_percentile'] = results['cointegration_score'].rank(pct=True) * 100
        
        def interpret_score(score):
            if score >= 50:
                return "Excellent"
            elif score >= 40:
                return "Very Strong"
            elif score >= 30:
                return "Strong"
            elif score >= 20:
                return "Moderate"
            elif score >= 10:
                return "Weak"
            else:
                return "Very Weak"
        
        results['cointegration_strength'] = results['cointegration_score'].apply(interpret_score)
        
        results_sorted = results.sort_values('cointegration_score', ascending=False)
        return results_sorted


def analyze_multiple_tenors_to_excel(currencies, tenors, days=730, 
                                     output_file='cointegration_analysis.xlsx'):
    """
    Analyze multiple currency pairs across different tenors and export to Excel
    """
    all_results = {}

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for tenor in tenors:
            print(f"\nAnalyzing tenor: {tenor}")
            try:
                analyzer = Screen_MultiPair_Metrics.create_from_currencies(
                    currencies, tenor, days)
                
                # Skip if no valid screeners were created
                if not analyzer.screeners:
                    print(f"  No valid data for tenor {tenor}")
                    continue
                
                results = analyzer.get_best_pairs_comprehensive()

                if not results.empty:
                    # Define output columns in logical order
                    output_columns = [
                        # Scoring
                        'cointegration_score', 'cointegration_rank', 
                        'cointegration_percentile', 'cointegration_strength',
                        
                        # Engle-Granger
                        'eg_eg_t_stat', 'eg_eg_p_value', 'eg_intercept', 
                        'eg_hedge_ratio', 'eg_ols_r_squared',
                        
                        # ADF and SNR
                        'adf_statistic', 'adf_p_value', 'adf_critical_1pct',
                        'adf_critical_5pct', 'adf_critical_10pct', 
                        'adf_reject_5pct', 'adf_reject_1pct', 'snr_ratio',
                        
                        # Out-of-sample
                        'oos_oos_adf_statistic', 'oos_oos_adf_pvalue', 
                        'oos_oos_stationary', 'oos_oos_spread_std', 
                        'oos_is_spread_std', 'oos_oos_volatility_ratio', 
                        'oos_oos_stable',
                        
                        # Half-life
                        'halflife_halflife', 'halflife_phi', 'halflife_phi_pvalue',
                        'halflife_confidence_interval', 'halflife_r_squared',
                        
                        # Hurst
                        'hurst_hurst_ticker1', 'hurst_hurst_ticker2',
                        'hurst_hurst_spread', 'hurst_mean_hurst',
                        'hurst_both_mean_reverting', 'hurst_spread_mean_reverting',
                        
                        # Variance Ratio
                        'vr_vr_lag_2', 'vr_vr_z_stat_2', 'vr_vr_reject_rw_2',
                        'vr_vr_lag_5', 'vr_vr_z_stat_5', 'vr_vr_reject_rw_5',
                        'vr_vr_lag_10', 'vr_vr_z_stat_10', 'vr_vr_reject_rw_10'
                    ]

                    available_columns = [col for col in output_columns if col in results.columns]
                    results_filtered = results[available_columns].copy()

                    # Round numerical columns
                    numerical_cols = results_filtered.select_dtypes(include=[np.number]).columns
                    results_filtered[numerical_cols] = results_filtered[numerical_cols].round(4)

                    all_results[tenor] = results_filtered

                    # Write to Excel
                    sheet_name = f'{tenor}'
                    results_filtered.to_excel(writer, sheet_name=sheet_name)

                    # Format Excel sheet
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

                    # Print top 5 pairs
                    print(f"  Top 5 pairs for {tenor}:")
                    top_5 = results_filtered.head(5)
                    for idx, (pair, row) in enumerate(top_5.iterrows(), 1):
                        print(f"    {idx}. {pair}")
                        print(f"       Score: {row.get('cointegration_score', 'N/A'):.2f}")
                        print(f"       Strength: {row.get('cointegration_strength', 'N/A')}")
                        if 'eg_eg_p_value' in row:
                            print(f"       EG p-value: {row['eg_eg_p_value']:.4f}")
                        if 'oos_oos_stationary' in row:
                            print(f"       OOS Stationary: {row['oos_oos_stationary']}")
                else:
                    print(f"  No results for tenor {tenor}")

            except Exception as e:
                print(f"  Error analyzing tenor {tenor}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\nAnalysis complete! Results saved to: {output_file}")
    return all_results


def create_summary_sheet(results_dict, output_file='cointegration_analysis.xlsx'):
    """
    Create a summary sheet showing top pairs across all tenors
    """
    summary_data = []

    for tenor, df in results_dict.items():
        if not df.empty:
            # Get top 5 pairs for each tenor
            top_pairs = df.head(5)
            for rank, (pair, row) in enumerate(top_pairs.iterrows(), 1):
                summary_data.append({
                    'Tenor': tenor,
                    'Rank': rank,
                    'Pair': pair,
                    'Score': row.get('cointegration_score', np.nan),
                    'Strength': row.get('cointegration_strength', 'N/A'),
                    'EG p-value': row.get('eg_eg_p_value', np.nan),
                    'ADF p-value': row.get('adf_p_value', np.nan),
                    'OOS Stationary': row.get('oos_oos_stationary', np.nan),
                    'Half-life': row.get('halflife_halflife', np.nan),
                    'Hurst': row.get('hurst_hurst_spread', np.nan)
                })

    summary_df = pd.DataFrame(summary_data)

    # Append to existing Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        # Remove existing Summary sheet if present
        if 'Summary' in writer.book.sheetnames:
            del writer.book['Summary']

        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Move Summary to first position
        wb = writer.book
        wb.move_sheet('Summary', offset=-len(wb.sheetnames) + 1)

        # Format columns
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


def run_multi_tenor_analysis(currencies=None, tenors=None, days=730,
                             output_file='cointegration_results.xlsx'):
    """
    Main function to run multi-tenor cointegration analysis
    """
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


def run_multi_group_tenor_analysis(currency_groups, tenors, days=730,
                                   output_file='cointegration_results.xlsx'):
    """
    Analyze multiple currency groups across different tenors with separate tabs for each combination
    
    Parameters:
    -----------
    currency_groups : dict
        Dictionary where keys are group names and values are lists of currency pairs
        Example: {'EUR': ['EURUSD', 'EURGBP', ...], 'Majors': ['EURUSD', 'GBPUSD', ...]}
    tenors : list
        List of tenors to analyze (e.g., ['1W', '2W', '1M'])
    days : int
        Number of historical days to analyze
    output_file : str
        Output Excel file name
    
    Returns:
    --------
    dict : Nested dictionary with results for each group and tenor
    """
    all_results = {}
    
    print("=" * 60)
    print("MULTI-GROUP MULTI-TENOR COINTEGRATION ANALYSIS")
    print("=" * 60)
    print(f"Currency Groups: {list(currency_groups.keys())}")
    print(f"Tenors: {tenors}")
    print(f"Historical days: {days}")
    print(f"Output file: {output_file}")
    print("=" * 60)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for tenor in tenors:
            for group_name, currencies in currency_groups.items():
                sheet_name = f"{tenor}_{group_name}"
                print(f"\nAnalyzing: {sheet_name}")
                print(f"  Currencies: {', '.join(currencies)}")
                
                try:
                    # Create analyzer for this group and tenor
                    analyzer = Screen_MultiPair_Metrics.create_from_currencies(
                        currencies, tenor, days)
                    
                    # Skip if no valid screeners were created
                    if not analyzer.screeners:
                        print(f"  No valid data for {sheet_name}")
                        continue
                    
                    # Get comprehensive results
                    results = analyzer.get_best_pairs_comprehensive()

                    if not results.empty:
                        # Define output columns
                        output_columns = [
                            # Scoring
                            'cointegration_score', 'cointegration_rank', 
                            'cointegration_percentile', 'cointegration_strength',
                            
                            # Engle-Granger
                            'eg_eg_t_stat', 'eg_eg_p_value', 'eg_intercept', 
                            'eg_hedge_ratio', 'eg_ols_r_squared',
                            
                            # ADF and SNR
                            'adf_statistic', 'adf_p_value', 'adf_critical_1pct',
                            'adf_critical_5pct', 'adf_critical_10pct', 
                            'adf_reject_5pct', 'adf_reject_1pct', 'snr_ratio',
                            
                            # Out-of-sample
                            'oos_oos_adf_statistic', 'oos_oos_adf_pvalue', 
                            'oos_oos_stationary', 'oos_oos_spread_std', 
                            'oos_is_spread_std', 'oos_oos_volatility_ratio', 
                            'oos_oos_stable',
                            
                            # Half-life
                            'halflife_halflife', 'halflife_phi', 'halflife_phi_pvalue',
                            'halflife_confidence_interval', 'halflife_r_squared',
                            
                            # Hurst
                            'hurst_hurst_ticker1', 'hurst_hurst_ticker2',
                            'hurst_hurst_spread', 'hurst_mean_hurst',
                            'hurst_both_mean_reverting', 'hurst_spread_mean_reverting',
                            
                            # Variance Ratio
                            'vr_vr_lag_2', 'vr_vr_z_stat_2', 'vr_vr_reject_rw_2',
                            'vr_vr_lag_5', 'vr_vr_z_stat_5', 'vr_vr_reject_rw_5',
                            'vr_vr_lag_10', 'vr_vr_z_stat_10', 'vr_vr_reject_rw_10'
                        ]

                        available_columns = [col for col in output_columns if col in results.columns]
                        results_filtered = results[available_columns].copy()

                        # Round numerical columns
                        numerical_cols = results_filtered.select_dtypes(include=[np.number]).columns
                        results_filtered[numerical_cols] = results_filtered[numerical_cols].round(4)

                        # Store results
                        if tenor not in all_results:
                            all_results[tenor] = {}
                        all_results[tenor][group_name] = results_filtered

                        # Write to Excel with truncated sheet name (Excel limit is 31 chars)
                        sheet_name_truncated = sheet_name[:31]
                        results_filtered.to_excel(writer, sheet_name=sheet_name_truncated)

                        # Format Excel sheet
                        worksheet = writer.sheets[sheet_name_truncated]
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

                        # Print top 3 pairs
                        print(f"  Top 3 pairs for {sheet_name}:")
                        top_3 = results_filtered.head(3)
                        for idx, (pair, row) in enumerate(top_3.iterrows(), 1):
                            print(f"    {idx}. {pair}")
                            print(f"       Score: {row.get('cointegration_score', 'N/A'):.2f} | "
                                  f"Strength: {row.get('cointegration_strength', 'N/A')}")
                    else:
                        print(f"  No results for {sheet_name}")

                except Exception as e:
                    print(f"  Error analyzing {sheet_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results saved to: {output_file}")
    print(f"{'=' * 60}")
    return all_results










if __name__ == "__main__":
    # Define currency groups
    currency_groups = {
        'EUR': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURPLN', 'EURHUF', 'EURCZK', 'EURNOK', 'EURSEK'],
        'Majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK'],
        'Asia': ['USDJPY', 'USDCNH', 'USDKRW', 'USDSGD', 'USDTWD', 'USDTHB', 'USDIDR', 'USDPHP', 'USDMYR', 'USDINR'],
        'EM': ['USDBRL', 'USDMXN', 'USDZAR', 'USDTRY', 'USDRUB', 'USDCLP', 'USDCOP', 'USDPLN', 'USDHUF']
    }
    
    # Define tenors
    tenors = ['1W', '2W', '3W', '1M', '3M', '6M', '1Y']
    
    # Run analysis
    results = run_multi_group_tenor_analysis(
        currency_groups=currency_groups,
        tenors=tenors,
        days=365 * 5,
        output_file='BF_Cointegration_All_5Y_3Oct.xlsx'
    )



























































