import pdblp
import pandas as pd
import numpy as np
import pytz
from typing import Dict, Tuple, List
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from xbbg import blp
from datetime import datetime, timedelta


class CORR_IR_Value_Plot:
    def __init__(self, cross_pairs: List[Tuple[str, str, str]], tenors: List[str]):
        self.cross_pairs = cross_pairs
        self.tenors = tenors
        self.df_correlations = None
        self.all_ccys = None
        self.major_pairs_only = None
        self._extract_currency_pairs()
    
    def _extract_currency_pairs(self):
        all_ccys = set()
        major_pairs_only = set()
        for major1, major2, cross in self.cross_pairs:
            all_ccys.add(major1)
            all_ccys.add(major2)
            all_ccys.add(cross)
            major_pairs_only.add(major1)
            major_pairs_only.add(major2)
        self.all_ccys = sorted(list(all_ccys))
        self.major_pairs_only = sorted(list(major_pairs_only))
    
    def _get_daily_spot_batch(self, currency_pairs: List[str], years: int = 5) -> pd.DataFrame:
        df_ccy = {}
        start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        ticker_list = [f"{ticker} Curncy" for ticker in currency_pairs]
        data_all = blp.bdh(
            tickers=ticker_list,
            flds=["PX_LAST"], 
            start_date=start_date,
            end_date=end_date,
            Per="D")
        if data_all.empty:
            print("No data returned for any currency pairs")
            return pd.DataFrame()
        for ticker in currency_pairs:
            ticker_full = f"{ticker} Curncy"
            if ticker_full in data_all.columns.get_level_values(0):
                data_ccy = data_all[ticker_full]
                data_ccy.columns = [ticker]
                df_ccy[ticker] = data_ccy
            else:
                print(f"No data for {ticker}, skipping.")
        df_ccyAll = pd.concat(df_ccy.values(), axis=1)
        df_ccyAll = df_ccyAll.sort_index(ascending=True)
        return df_ccyAll
    
    def _get_daily_vols_batch(self, ccys: List[str], tenors: List[str]) -> pd.DataFrame:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        ticker_list = []
        ticker_map = {}
        for ccy in ccys:
            for tenor in tenors:
                ticker_IV = f"{ccy}V{tenor} BGN Curncy"
                ticker_list.append(ticker_IV)
                ticker_map[ticker_IV] = (ccy, tenor)
        data_all = blp.bdh(
            tickers=ticker_list,
            flds="PX_LAST",
            start_date=start_date,
            end_date=end_date)
        if data_all.empty:
            print("No volatility data returned")
            return pd.DataFrame()
        df_vols = {}
        for ticker_IV, (ccy, tenor) in ticker_map.items():
            if ticker_IV in data_all.columns.get_level_values(0):
                column_name = f"{ccy}_{tenor}"
                data_IV = data_all[ticker_IV].copy()
                data_IV.columns = [column_name]
                df_vols[column_name] = data_IV
            else:
                print(f"No data for {ticker_IV}, skipping.")
        df_vols_all = pd.concat(df_vols.values(), axis=1)
        return df_vols_all
    
    def _get_historical_vols_batch(self, ccys: List[str], tenors: List[str], 
                                   lookback_days: int) -> pd.DataFrame:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        ticker_list = []
        ticker_map = {}
        for ccy in ccys:
            for tenor in tenors:
                ticker_IV = f"{ccy}V{tenor} BGN Curncy"
                ticker_list.append(ticker_IV)
                ticker_map[ticker_IV] = (ccy, tenor)
        data_all = blp.bdh(
            tickers=ticker_list,
            flds="PX_LAST",
            start_date=start_date,
            end_date=end_date)
        if data_all.empty:
            print("No historical volatility data returned")
            return pd.DataFrame()
        df_vols_hist = pd.DataFrame()
        for ticker_IV, (ccy, tenor) in ticker_map.items():
            if ticker_IV in data_all.columns.get_level_values(0):
                column_name = f"{ccy}_{tenor}"
                data_IV = data_all[ticker_IV].copy()
                df_vols_hist[column_name] = data_IV.iloc[:, 0]
        return df_vols_hist
    
    @staticmethod
    def _determine_usd_position(ccy_pair: str) -> str:
        if ccy_pair[:3] == 'USD':
            return 'CCY1'
        elif ccy_pair[3:6] == 'USD':
            return 'CCY2'
        else:
            return None
    


    def _calculate_implied_correlation(self, df_vols: pd.DataFrame, 
                                      major1: str, major2: str, 
                                      cross: str, tenors: List[str]) -> pd.DataFrame:
        usd_pos1 = self._determine_usd_position(major1)
        usd_pos2 = self._determine_usd_position(major2)
        if usd_pos1 is None or usd_pos2 is None:
            print(f"Warning: Non-USD pair detected in {major1} or {major2}")
            use_division_formula = True
        else:
            use_division_formula = (usd_pos1 == usd_pos2)
        results = {}
        for tenor in tenors:
            col_major1 = f"{major1}_{tenor}"
            col_major2 = f"{major2}_{tenor}"
            col_cross = f"{cross}_{tenor}"
            if all(col in df_vols.columns for col in [col_major1, col_major2, col_cross]):
                vol_major1 = df_vols[col_major1].iloc[-1]
                vol_major2 = df_vols[col_major2].iloc[-1]
                vol_cross = df_vols[col_cross].iloc[-1]
                if use_division_formula:
                    # Division: Z = X / Y -> σ_Z² = σ_X² + σ_Y² - 2ρσ_Xσ_Y
                    implied_corr = (vol_major1**2 + vol_major2**2 - vol_cross**2) / (2 * vol_major1 * vol_major2)
                else:
                    # Multiplication: Z = X * Y -> σ_Z² = σ_X² + σ_Y² + 2ρσ_Xσ_Y
                    implied_corr = (vol_cross**2 - vol_major1**2 - vol_major2**2) / (2 * vol_major1 * vol_major2)
                results[f"ICorr_{tenor}"] = round(implied_corr, 4)
            else:
                print(f"Missing data for tenor {tenor}")
        cross_name = f"{major1}_{major2}"
        df_corr = pd.DataFrame([results], index=[cross_name])
        return df_corr
    
    def _calculate_historical_implied_corr_series(self, df_vols_hist: pd.DataFrame,
                                                  major1: str, major2: str, 
                                                  cross: str, tenor: str) -> pd.Series:
        usd_pos1 = self._determine_usd_position(major1)
        usd_pos2 = self._determine_usd_position(major2)
        if usd_pos1 is None or usd_pos2 is None:
            use_division_formula = True
        else:
            use_division_formula = (usd_pos1 == usd_pos2)
        col_major1 = f"{major1}_{tenor}"
        col_major2 = f"{major2}_{tenor}"
        col_cross = f"{cross}_{tenor}"
        if all(col in df_vols_hist.columns for col in [col_major1, col_major2, col_cross]):
            vol_major1 = df_vols_hist[col_major1]
            vol_major2 = df_vols_hist[col_major2]
            vol_cross = df_vols_hist[col_cross]
            if use_division_formula:
                implied_corr = (vol_major1**2 + vol_major2**2 - vol_cross**2) / (2 * vol_major1 * vol_major2)
            else:
                implied_corr = (vol_cross**2 - vol_major1**2 - vol_major2**2) / (2 * vol_major1 * vol_major2)
            return implied_corr
        else:
            return pd.Series()
    
    def _calculate_hist_corr(self, ccy1: str, ccy2: str, 
                            daily_spot_data: pd.DataFrame, 
                            tenors: List[str]) -> pd.DataFrame:
        df_prices = daily_spot_data[[ccy1, ccy2]]
        df_returns = np.log(df_prices / df_prices.shift(1)).dropna()
        all_periods = {
            '1W': 7, '2W': 14, '1M': 30, '2M': 60, '3M': 90,
            '6M': 180, '1Y': 365, '2Y': 730, '3Y': 1095, '5Y': 1825}
        periods = {}
        for tenor in tenors:
            if tenor in all_periods:
                periods[tenor] = all_periods[tenor]
        if '1Y' not in periods:
            periods['1Y'] = all_periods['1Y']
        if '5Y' not in periods:
            periods['5Y'] = all_periods['5Y']
        max_date = df_returns.index.max()
        correlations = {}
        for period_name, days in periods.items():
            cutoff_date = max_date - timedelta(days=days)
            recent_returns = df_returns[df_returns.index > cutoff_date]
            if len(recent_returns) > 1:  # Need at least 2 data points for correlation
                corr = recent_returns.iloc[:, 0].corr(recent_returns.iloc[:, 1])
                correlations[f'RCorr_{period_name}'] = corr
            else:
                correlations[f'RCorr_{period_name}'] = np.nan
        df_corr = pd.DataFrame([correlations], index=[f'{ccy1}_{ccy2}'])
        return round(df_corr, 4)
    
    def _calculate_historical_realized_corr_series(self, ccy1: str, ccy2: str,
                                                   daily_spot_data: pd.DataFrame,
                                                   tenor: str, lookback_days: int) -> pd.Series:
        df_prices = daily_spot_data[[ccy1, ccy2]]
        df_returns = np.log(df_prices / df_prices.shift(1)).dropna()
        tenor_days = {
            '1W': 7, '2W': 14, '1M': 30, '2M': 60, '3M': 90,
            '6M': 180, '1Y': 365, '2Y': 730, '3Y': 1095, '5Y': 1825}
        if tenor not in tenor_days:
            return pd.Series()
        window_days = tenor_days[tenor]
        cutoff_date = df_returns.index.max() - timedelta(days=lookback_days)
        df_returns_period = df_returns[df_returns.index > cutoff_date]
        corr_series = []
        dates = []
        for date in df_returns_period.index:
            window_start = date - timedelta(days=window_days)
            window_data = df_returns[(df_returns.index > window_start) & (df_returns.index <= date)]
            if len(window_data) > 1:
                corr = window_data.iloc[:, 0].corr(window_data.iloc[:, 1])
                corr_series.append(corr)
                dates.append(date)
        return pd.Series(corr_series, index=dates)
    
    
    def calculate_correlations(self, verbose: bool = False) -> pd.DataFrame:
        df_all_vols = self._get_daily_vols_batch(self.all_ccys, self.tenors)
        daily_spot_data = self._get_daily_spot_batch(self.all_ccys, years=5)
        all_implied_corr = []
        all_hist_corr = []
        for major1, major2, cross in self.cross_pairs:
            try:
                df_implied = self._calculate_implied_correlation(
                    df_all_vols, major1, major2, cross, self.tenors)
                all_implied_corr.append(df_implied)
                df_hist = self._calculate_hist_corr(
                    major1, major2, daily_spot_data, self.tenors)
                all_hist_corr.append(df_hist)
            except Exception as e:
                print(f"Error processing {major1}/{major2}/{cross}: {e}")
                continue
        df_all_implied = pd.concat(all_implied_corr)
        df_all_hist = pd.concat(all_hist_corr)
        self.df_correlations = pd.concat([df_all_implied, df_all_hist], axis=1)
        return self.df_correlations
    
    def calculate_correlation_percentiles(self, lookback_days: int = 365, 
                                         verbose: bool = False) -> pd.DataFrame:
        df_vols_hist = self._get_historical_vols_batch(self.all_ccys, self.tenors, lookback_days)
        df_vols_current = self._get_daily_vols_batch(self.all_ccys, self.tenors)
        daily_spot_data = self._get_daily_spot_batch(self.all_ccys, years=5)
        all_tenors = list(self.tenors)
        if '1Y' not in all_tenors:
            all_tenors.append('1Y')
        if '5Y' not in all_tenors:
            all_tenors.append('5Y')
        results = []
        for major1, major2, cross in self.cross_pairs:
            cross_name = f"{major1}_{major2}"
            row_data = {'Pair': cross_name}
            for tenor in all_tenors:
                try:
                    # --- Implied Correlation ---
                    if tenor in self.tenors:  # Only calculate ICorr for main tenors
                        df_implied_current = self._calculate_implied_correlation(
                            df_vols_current, major1, major2, cross, [tenor])
                        current_icorr = df_implied_current[f"ICorr_{tenor}"].iloc[0]
                        hist_icorr_series = self._calculate_historical_implied_corr_series(
                            df_vols_hist, major1, major2, cross, tenor)
                        if len(hist_icorr_series) > 0:
                            percentile = (hist_icorr_series < current_icorr).sum() / len(hist_icorr_series) * 100
                            row_data[f'ICorr_{tenor}_Current'] = current_icorr
                            row_data[f'ICorr_{tenor}_Pct'] = round(percentile, 1)
                        else:
                            row_data[f'ICorr_{tenor}_Current'] = current_icorr
                            row_data[f'ICorr_{tenor}_Pct'] = np.nan
                    
                    # --- Realized Correlation ---
                    df_rcorr_current = self._calculate_hist_corr(
                        major1, major2, daily_spot_data, [tenor])
                    current_rcorr = df_rcorr_current[f"RCorr_{tenor}"].iloc[0]
                    hist_rcorr_series = self._calculate_historical_realized_corr_series(
                        major1, major2, daily_spot_data, tenor, lookback_days)
                    if len(hist_rcorr_series) > 0:
                        percentile = (hist_rcorr_series < current_rcorr).sum() / len(hist_rcorr_series) * 100
                        row_data[f'RCorr_{tenor}_Current'] = current_rcorr
                        row_data[f'RCorr_{tenor}_Pct'] = round(percentile, 1)
                    else:
                        row_data[f'RCorr_{tenor}_Current'] = current_rcorr
                        row_data[f'RCorr_{tenor}_Pct'] = np.nan
                except Exception as e:
                    if verbose:
                        print(f"Error processing {cross_name} - {tenor}: {e}")
                    continue
            results.append(row_data)
        df_results = pd.DataFrame(results)
        df_results.set_index('Pair', inplace=True)
        
        print("Percentile calculation complete!")
        return df_results
    
    def calculate_correlation_zscores(self, lookback_days: int = 365, 
                                 verbose: bool = False) -> pd.DataFrame:
        df_vols_hist = self._get_historical_vols_batch(self.all_ccys, self.tenors, lookback_days)
        df_vols_current = self._get_daily_vols_batch(self.all_ccys, self.tenors)
        daily_spot_data = self._get_daily_spot_batch(self.all_ccys, years=5)
        all_tenors = list(self.tenors)
        if '1Y' not in all_tenors:
            all_tenors.append('1Y')
        if '5Y' not in all_tenors:
            all_tenors.append('5Y')
        results = []
        for major1, major2, cross in self.cross_pairs:
            cross_name = f"{major1}_{major2}"
            row_data = {'Pair': cross_name}
            for tenor in all_tenors:
                try:
                    # --- Implied Correlation ---
                    if tenor in self.tenors:  # Only calculate ICorr for main tenors
                        df_implied_current = self._calculate_implied_correlation(
                            df_vols_current, major1, major2, cross, [tenor])
                        current_icorr = df_implied_current[f"ICorr_{tenor}"].iloc[0]
                        hist_icorr_series = self._calculate_historical_implied_corr_series(
                            df_vols_hist, major1, major2, cross, tenor)
                        if len(hist_icorr_series) > 0:
                            mean_icorr = hist_icorr_series.mean()
                            std_icorr = hist_icorr_series.std()
                            if std_icorr > 0:
                                zscore = (current_icorr - mean_icorr) / std_icorr
                            else:
                                zscore = 0.0
                            row_data[f'ICorr_{tenor}_Current'] = current_icorr
                            row_data[f'ICorr_{tenor}_ZScore'] = round(zscore, 2)
                        else:
                            row_data[f'ICorr_{tenor}_Current'] = current_icorr
                            row_data[f'ICorr_{tenor}_ZScore'] = np.nan
                    # --- Realized Correlation ---
                    df_rcorr_current = self._calculate_hist_corr(
                        major1, major2, daily_spot_data, [tenor])
                    current_rcorr = df_rcorr_current[f"RCorr_{tenor}"].iloc[0]
                    hist_rcorr_series = self._calculate_historical_realized_corr_series(
                        major1, major2, daily_spot_data, tenor, lookback_days)
                    if len(hist_rcorr_series) > 0:
                        mean_rcorr = hist_rcorr_series.mean()
                        std_rcorr = hist_rcorr_series.std()
                        if std_rcorr > 0:
                            zscore = (current_rcorr - mean_rcorr) / std_rcorr
                        else:
                            zscore = 0.0
                        row_data[f'RCorr_{tenor}_Current'] = current_rcorr
                        row_data[f'RCorr_{tenor}_ZScore'] = round(zscore, 2)
                    else:
                        row_data[f'RCorr_{tenor}_Current'] = current_rcorr
                        row_data[f'RCorr_{tenor}_ZScore'] = np.nan
                except Exception as e:
                    if verbose:
                        print(f"Error processing {cross_name} - {tenor}: {e}")
                    continue
            results.append(row_data)
        df_results = pd.DataFrame(results)
        df_results.set_index('Pair', inplace=True)
        return df_results
    




cross_pairs = [
    ('EURUSD', 'USDJPY', 'EURJPY'),
    ('AUDUSD', 'USDJPY', 'AUDJPY'),
    ('EURUSD', 'USDCHF', 'EURCHF'),
    ('EURUSD', 'AUDUSD', 'EURAUD'),
    ('EURUSD', 'USDCAD', 'EURCAD'),
    ('AUDUSD', 'USDCHF', 'AUDCHF'),
]

tenors = ['1M', '2M']




analyzer = CORR_IR_Value_Plot(cross_pairs, tenors)

# df = analyzer.calculate_correlations()
# print(df)


df = analyzer.calculate_correlation_percentiles(lookback_days=365)
print(df)
