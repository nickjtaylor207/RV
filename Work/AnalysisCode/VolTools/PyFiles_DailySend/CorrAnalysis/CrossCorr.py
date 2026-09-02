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

    # --------------------------------------------------------------------------------------------------
    # -------------------------------- PLOTTING FUNCTIONS ----------------------------------------------
    def plot_heatmap(self, df_correlations: pd.DataFrame = None, 
                    figsize_per_plot: int = 4) -> Tuple:
        if df_correlations is None:
            if self.df_correlations is None:
                raise ValueError("No correlation data available. Run calculate_correlations() first.")
            df_correlations = self.df_correlations
        additional_tenors = []
        if '1Y' not in self.tenors:
            additional_tenors.append('1Y')
        if '5Y' not in self.tenors:
            additional_tenors.append('5Y')
        num_main_plots = len(self.tenors)
        num_additional = len(additional_tenors)
        total_plots = num_main_plots + num_additional
        fig, axes = plt.subplots(1, total_plots, 
                                figsize=(figsize_per_plot * total_plots, 10), 
                                gridspec_kw={'wspace': 0.05})
        if total_plots == 1:
            axes = [axes]
        for tenor_idx, tenor in enumerate(self.tenors):
            ax = axes[tenor_idx]
            rcorr_col = f'RCorr_{tenor}'
            icorr_col = f'ICorr_{tenor}'
            cols_to_plot = []
            if rcorr_col in df_correlations.columns:
                cols_to_plot.append(rcorr_col)
            if icorr_col in df_correlations.columns:
                cols_to_plot.append(icorr_col)
            if not cols_to_plot:
                continue
            tenor_df = df_correlations[cols_to_plot]
            n_rows, n_cols = tenor_df.shape
            for i, row_idx in enumerate(tenor_df.index):
                for j, col_name in enumerate(tenor_df.columns):
                    value = tenor_df.iloc[i, j]
                    if 'RCorr' in col_name:
                        color = "#E8E8E8"  # Light gray for RCorr
                    elif 'ICorr' in col_name:
                        color = "#D3D3D3"  # Medium gray for ICorr
                    else:
                        color = "white"
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, 
                                        edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                    if j == 0 and n_cols > 1:  # After RCorr column
                        thick_line = plt.Rectangle((j + 1 - 0.05, i), 0.1, 1, 
                                                  facecolor="black", edgecolor="none")
                        ax.add_patch(thick_line)
                    if pd.isna(value):
                        text = "N/A"
                        fontsize = 8
                    else:
                        text = f"{value:.3f}"
                        fontsize = 9
                    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                           fontsize=fontsize)
            ax.set_xlim(0, n_cols)
            ax.set_ylim(0, n_rows)
            ax.set_xticks(np.arange(n_cols) + 0.5)
            ax.set_yticks(np.arange(n_rows) + 0.5)
            col_labels = []
            for col_name in tenor_df.columns:
                if 'RCorr' in col_name:
                    col_labels.append('RCorr')
                elif 'ICorr' in col_name:
                    col_labels.append('ICorr')
                else:
                    col_labels.append(col_name.split('_')[-1])
            ax.set_xticklabels(col_labels, fontsize=10, rotation=0, ha='center')
            if tenor_idx == 0:
                ax.set_yticklabels(tenor_df.index, fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.xaxis.tick_top()
            ax.text(n_cols/2, n_rows + 0.5, tenor, ha="center", va="center",
                   fontsize=14, weight="bold", transform=ax.transData)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
        for add_idx, add_tenor in enumerate(additional_tenors):
            ax = axes[num_main_plots + add_idx]
            rcorr_col = f'RCorr_{add_tenor}'
            if rcorr_col not in df_correlations.columns:
                continue
            add_df = df_correlations[[rcorr_col]]
            n_rows, n_cols = add_df.shape
            for i, row_idx in enumerate(add_df.index):
                for j, col_name in enumerate(add_df.columns):
                    value = add_df.iloc[i, j]
                    color = "#E8E8E8"
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, 
                                        edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                    if pd.isna(value):
                        text = "N/A"
                        fontsize = 8
                    else:
                        text = f"{value:.3f}"
                        fontsize = 9
                    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                           fontsize=fontsize)
            ax.set_xlim(0, n_cols)
            ax.set_ylim(0, n_rows)
            ax.set_xticks(np.arange(n_cols) + 0.5)
            ax.set_yticks(np.arange(n_rows) + 0.5)
            ax.set_xticklabels(['RCorr'], fontsize=10, rotation=0, ha='center')
            ax.set_yticklabels([])
            ax.xaxis.tick_top()
            ax.text(n_cols/2, n_rows + 0.5, add_tenor, ha="center", va="center",
                   fontsize=14, weight="bold", transform=ax.transData)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle("Cross-Currency Correlation Monitor", 
                    fontsize=16, fontweight="bold", y=0.98)
        axes[0].set_ylabel("Currency Pairs", fontsize=12, fontweight="bold")
        legend_text = (
            "• RCorr (Realized Correlation): Rolling realized correlation from daily returns\n"
            "• ICorr (Implied Correlation): Triangular correlation implied from ATM Majors/Crosses\n")
        fig.text(0.5, 0.02, legend_text, fontsize=10, ha='center', va='bottom',
                transform=fig.transFigure, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        plt.show()
        return fig, axes
    
    def plot_percentile_heatmap(self, df_percentiles: pd.DataFrame = None,
                               lookback_days: int = 365,
                               figsize_per_plot: int = 5) -> Tuple:
        if df_percentiles is None:
            df_percentiles = self.calculate_correlation_percentiles(lookback_days)
        additional_tenors = []
        if '1Y' not in self.tenors:
            additional_tenors.append('1Y')
        if '5Y' not in self.tenors:
            additional_tenors.append('5Y')
        num_main_plots = len(self.tenors)
        num_additional = len(additional_tenors)
        total_plots = num_main_plots + num_additional
        fig, axes = plt.subplots(1, total_plots, 
                                figsize=(figsize_per_plot * total_plots, 10), 
                                gridspec_kw={'wspace': 0.05})
        if total_plots == 1:
            axes = [axes]
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
        cmap = plt.cm.RdYlGn_r
        for tenor_idx, tenor in enumerate(self.tenors):
            ax = axes[tenor_idx]
            cols_to_plot = [
                f'RCorr_{tenor}_Pct',
                f'RCorr_{tenor}_Current',
                f'ICorr_{tenor}_Current',
                f'ICorr_{tenor}_Pct']
            cols_to_plot = [col for col in cols_to_plot if col in df_percentiles.columns]
            if not cols_to_plot:
                continue
            tenor_df = df_percentiles[cols_to_plot]
            n_rows, n_cols = tenor_df.shape
            for i, row_idx in enumerate(tenor_df.index):
                for j, col_name in enumerate(tenor_df.columns):
                    value = tenor_df.iloc[i, j]
                    if '_Pct' in col_name and not pd.isna(value):
                        color = cmap(norm(value))
                    elif 'RCorr' in col_name:
                        color = "#E8E8E8"  # Light gray for RCorr values
                    elif 'ICorr' in col_name:
                        color = "#D3D3D3"  # Medium gray for ICorr values
                    else:
                        color = "white"
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, 
                                        edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                    if j == 1:  # After RCorr, before ICorr
                        thick_line = plt.Rectangle((j + 1 - 0.05, i), 0.1, 1, 
                                                facecolor="black", edgecolor="none")
                        ax.add_patch(thick_line)
                    if pd.isna(value):
                        text = "N/A"
                        fontsize = 8
                        fontweight = "normal"
                    elif '_Pct' in col_name:
                        text = f"{value:.1f}"
                        fontsize = 9
                        fontweight = "bold" if value > 85 or value < 15 else "normal"
                    else:
                        text = f"{value:.3f}"
                        fontsize = 9
                        fontweight = "normal"
                    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                        fontsize=fontsize, fontweight=fontweight)
            ax.set_xlim(0, n_cols)
            ax.set_ylim(0, n_rows)
            ax.set_xticks(np.arange(n_cols) + 0.5)
            ax.set_yticks(np.arange(n_rows) + 0.5)
            col_labels = []
            for col_name in tenor_df.columns:
                if 'RCorr' in col_name and '_Pct' in col_name:
                    col_labels.append('R %')
                elif 'RCorr' in col_name:
                    col_labels.append('RCorr')
                elif 'ICorr' in col_name and '_Pct' in col_name:
                    col_labels.append('I %')
                elif 'ICorr' in col_name:
                    col_labels.append('ICorr')
                else:
                    col_labels.append(col_name)
            ax.set_xticklabels(col_labels, fontsize=10, rotation=0, ha='center')
            if tenor_idx == 0:
                ax.set_yticklabels(tenor_df.index, fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.xaxis.tick_top()
            ax.text(n_cols/2, n_rows + 0.5, tenor, ha="center", va="center",
                fontsize=14, weight="bold", transform=ax.transData)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
        for add_idx, add_tenor in enumerate(additional_tenors):
            ax = axes[num_main_plots + add_idx]
            cols_to_plot = [
                f'RCorr_{add_tenor}_Pct',
                f'RCorr_{add_tenor}_Current']
            cols_to_plot = [col for col in cols_to_plot if col in df_percentiles.columns]
            if not cols_to_plot:
                continue
            add_df = df_percentiles[cols_to_plot]
            n_rows, n_cols = add_df.shape
            for i, row_idx in enumerate(add_df.index):
                for j, col_name in enumerate(add_df.columns):
                    value = add_df.iloc[i, j]
                    if '_Pct' in col_name and not pd.isna(value):
                        color = cmap(norm(value))
                    else:
                        color = "#E8E8E8"
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, 
                                        edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                    if pd.isna(value):
                        text = "N/A"
                        fontsize = 8
                        fontweight = "normal"
                    elif '_Pct' in col_name:
                        text = f"{value:.1f}"
                        fontsize = 9
                        fontweight = "bold" if value > 85 or value < 15 else "normal"
                    else:
                        text = f"{value:.3f}"
                        fontsize = 9
                        fontweight = "normal"
                    
                    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                        fontsize=fontsize, fontweight=fontweight)
            ax.set_xlim(0, n_cols)
            ax.set_ylim(0, n_rows)
            ax.set_xticks(np.arange(n_cols) + 0.5)
            ax.set_yticks(np.arange(n_rows) + 0.5)
            ax.set_xticklabels(['R %', 'RCorr'], fontsize=10, rotation=0, ha='center')
            ax.set_yticklabels([])
            ax.xaxis.tick_top()
            ax.text(n_cols/2, n_rows + 0.5, add_tenor, ha="center", va="center",
                fontsize=14, weight="bold", transform=ax.transData)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Percentile Ranking", fontsize=12)
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        fig.suptitle("Cross-Currency Correlation Monitor with Percentiles", 
                    fontsize=16, fontweight="bold", y=0.98)
        axes[0].set_ylabel("Currency Pairs", fontsize=12, fontweight="bold")
        legend_text = (
            f"• Percentiles calculated over {lookback_days}-day lookback period\n"
            "• RCorr: Realized correlation | ICorr: Implied correlation\n")
        fig.text(0.5, 0.02, legend_text, fontsize=10, ha='center', va='bottom',
                transform=fig.transFigure, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        plt.show()
        return fig, axes
    
    def plot_zscore_heatmap(self, df_zscores: pd.DataFrame = None,
                        lookback_days: int = 365,
                        figsize_per_plot: int = 5) -> Tuple:
        if df_zscores is None:
            df_zscores = self.calculate_correlation_zscores(lookback_days)
        additional_tenors = []
        if '1Y' not in self.tenors:
            additional_tenors.append('1Y')
        if '5Y' not in self.tenors:
            additional_tenors.append('5Y')
        num_main_plots = len(self.tenors)
        num_additional = len(additional_tenors)
        total_plots = num_main_plots + num_additional
        fig, axes = plt.subplots(1, total_plots, 
                                figsize=(figsize_per_plot * total_plots, 10), 
                                gridspec_kw={'wspace': 0.05})
        if total_plots == 1:
            axes = [axes]
        norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        cmap = plt.cm.RdYlGn_r  # Red for negative, green for positive
        for tenor_idx, tenor in enumerate(self.tenors):
            ax = axes[tenor_idx]
            cols_to_plot = [
                f'RCorr_{tenor}_ZScore',
                f'RCorr_{tenor}_Current',
                f'ICorr_{tenor}_Current',
                f'ICorr_{tenor}_ZScore']
            cols_to_plot = [col for col in cols_to_plot if col in df_zscores.columns]
            if not cols_to_plot:
                continue
            tenor_df = df_zscores[cols_to_plot]
            n_rows, n_cols = tenor_df.shape
            for i, row_idx in enumerate(tenor_df.index):
                for j, col_name in enumerate(tenor_df.columns):
                    value = tenor_df.iloc[i, j]
                    if '_ZScore' in col_name and not pd.isna(value):
                        clamped_value = np.clip(value, -3, 3)
                        color = cmap(norm(clamped_value))
                    elif 'RCorr' in col_name:
                        color = "#E8E8E8"  # Light gray for RCorr values
                    elif 'ICorr' in col_name:
                        color = "#D3D3D3"  # Medium gray for ICorr values
                    else:
                        color = "white"
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, 
                                        edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                    if j == 1:  # After RCorr, before ICorr
                        thick_line = plt.Rectangle((j + 1 - 0.05, i), 0.1, 1, 
                                                facecolor="black", edgecolor="none")
                        ax.add_patch(thick_line)
                    if pd.isna(value):
                        text = "N/A"
                        fontsize = 8
                        fontweight = "normal"
                    elif '_ZScore' in col_name:
                        text = f"{value:.2f}"
                        fontsize = 9
                        fontweight = "bold" if abs(value) > 1.5 else "normal"
                    else:
                        text = f"{value:.3f}"
                        fontsize = 9
                        fontweight = "normal"
                    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                        fontsize=fontsize, fontweight=fontweight)
            ax.set_xlim(0, n_cols)
            ax.set_ylim(0, n_rows)
            ax.set_xticks(np.arange(n_cols) + 0.5)
            ax.set_yticks(np.arange(n_rows) + 0.5)
            col_labels = []
            for col_name in tenor_df.columns:
                if 'RCorr' in col_name and '_ZScore' in col_name:
                    col_labels.append('R Z')
                elif 'RCorr' in col_name:
                    col_labels.append('RCorr')
                elif 'ICorr' in col_name and '_ZScore' in col_name:
                    col_labels.append('I Z')
                elif 'ICorr' in col_name:
                    col_labels.append('ICorr')
                else:
                    col_labels.append(col_name)
            ax.set_xticklabels(col_labels, fontsize=10, rotation=0, ha='center')
            if tenor_idx == 0:
                ax.set_yticklabels(tenor_df.index, fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.xaxis.tick_top()
            ax.text(n_cols/2, n_rows + 0.5, tenor, ha="center", va="center",
                fontsize=14, weight="bold", transform=ax.transData)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
        for add_idx, add_tenor in enumerate(additional_tenors):
            ax = axes[num_main_plots + add_idx]
            cols_to_plot = [
                f'RCorr_{add_tenor}_ZScore',
                f'RCorr_{add_tenor}_Current']
            cols_to_plot = [col for col in cols_to_plot if col in df_zscores.columns]
            if not cols_to_plot:
                continue
            add_df = df_zscores[cols_to_plot]
            n_rows, n_cols = add_df.shape
            for i, row_idx in enumerate(add_df.index):
                for j, col_name in enumerate(add_df.columns):
                    value = add_df.iloc[i, j]
                    if '_ZScore' in col_name and not pd.isna(value):
                        clamped_value = np.clip(value, -3, 3)
                        color = cmap(norm(clamped_value))
                    else:
                        color = "#E8E8E8"
                    rect = plt.Rectangle((j, i), 1, 1, facecolor=color, 
                                        edgecolor="black", linewidth=0.5)
                    ax.add_patch(rect)
                    if pd.isna(value):
                        text = "N/A"
                        fontsize = 8
                        fontweight = "normal"
                    elif '_ZScore' in col_name:
                        text = f"{value:.2f}"
                        fontsize = 9
                        fontweight = "bold" if abs(value) > 1.5 else "normal"
                    else:
                        text = f"{value:.3f}"
                        fontsize = 9
                        fontweight = "normal"
                    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                        fontsize=fontsize, fontweight=fontweight)
            ax.set_xlim(0, n_cols)
            ax.set_ylim(0, n_rows)
            ax.set_xticks(np.arange(n_cols) + 0.5)
            ax.set_yticks(np.arange(n_rows) + 0.5)
            ax.set_xticklabels(['R Z', 'RCorr'], fontsize=10, rotation=0, ha='center')
            ax.set_yticklabels([])
            ax.xaxis.tick_top()
            ax.text(n_cols/2, n_rows + 0.5, add_tenor, ha="center", va="center",
                fontsize=14, weight="bold", transform=ax.transData)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Z-Score", fontsize=12)
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cbar.set_ticklabels(['-3', '-2', '-1', '0', '1', '2', '3'])
        fig.suptitle("Cross-Currency Correlation Monitor with Z-Scores", 
                    fontsize=16, fontweight="bold", y=0.98)
        axes[0].set_ylabel("Currency Pairs", fontsize=12, fontweight="bold")
        legend_text = (
            f"• Z-scores calculated over {lookback_days}-day lookback period\n"
            "• RCorr: Realized correlation | ICorr: Implied correlation\n")
        fig.text(0.5, 0.02, legend_text, fontsize=10, ha='center', va='bottom',
                transform=fig.transFigure, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        plt.show()
        return fig, axes











cross_pairs = [
    ('EURUSD', 'USDJPY', 'EURJPY'),
    # ('USDCHF', 'USDJPY', 'CHFJPY'),
    ('AUDUSD', 'USDJPY', 'AUDJPY'),
    # ('GBPUSD', 'USDJPY', 'GBPJPY'),
    # ('AUDUSD', 'USDCAD', 'AUDCAD'),

    ('EURUSD', 'GBPUSD', 'EURGBP'),
    ('EURUSD', 'USDCHF', 'EURCHF'),
    ('EURUSD', 'AUDUSD', 'EURAUD'),
    # ('EURUSD', 'NZDUSD', 'EURNZD'),
    ('EURUSD', 'USDCAD', 'EURCAD'),
    # ('EURUSD', 'USDNOK', 'EURNOK'),
    # ('EURUSD', 'USDSEK', 'EURSEK'),

    ('AUDUSD', 'USDCHF', 'AUDCHF'),

    ('GBPUSD', 'USDCHF', 'GBPCHF'),
    # ('AUDUSD', 'NZDUSD', 'AUDNZD')

]

tenors = ['1M', '2M', '3M', '6M']




analyzer = CORR_IR_Value_Plot(cross_pairs, tenors)



# analyzer.plot_zscore_heatmap(lookback_days=365)








# analyzer.plot_heatmap()                    # Simple values only
analyzer.plot_percentile_heatmap()         # Values + Percentiles
# analyzer.plot_zscore_heatmap()             # Values + Z-Scores






























