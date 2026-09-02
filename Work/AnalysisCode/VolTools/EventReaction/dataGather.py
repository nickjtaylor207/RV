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

def daily_GetTickers(currency_pairs, start_date, end_date):
    df_ccy = {}
    for ticker in currency_pairs:
        data_ccy = blp.bdh(
            tickers=f"{ticker} Curncy" ,
            flds=["PX_LAST"],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            Per="D")
        data_ccy.columns = [ticker]
        df_ccy[ticker] = data_ccy
    df_ccyAll = pd.concat(df_ccy, axis=1)
    df_ccyAll.columns = df_ccyAll.columns.droplevel(0) if isinstance(df_ccyAll.columns, pd.MultiIndex) else df_ccyAll.columns
    df_ccyAll = df_ccyAll.sort_index(ascending=True)
    return df_ccyAll


class EconomicDataManager:
    DEFAULT_FIELDS = [
        "NAME",
        "TIME", 
        "ECO_RELEASE_TIME",
        "ECO_RELEASE_DT",
        "ACTUAL_RELEASE",
        "BN_SURVEY_MEDIAN",
        "BN_SURVEY_AVERAGE", 
        "BN_SURVEY_HIGH",
        "BN_SURVEY_LOW",
        "FORECAST_STANDARD_DEVIATION",
        "BN_SURVEY_NUMBER_OBSERVATIONS"]
    FUTURE_FIELDS = ["ECO_RELEASE_DT", "ECO_RELEASE_TIME", "NAME"]
    FIELD_RENAME_MAP = {
        "ECO_RELEASE_DT": "ReleaseDate",
        "ECO_RELEASE_TIME": "ReleaseTime",
        "ACTUAL_RELEASE": "Actual", 
        "BN_SURVEY_MEDIAN": "SMed",
        "BN_SURVEY_AVERAGE": "SAve",
        "BN_SURVEY_HIGH": "SHigh",
        "BN_SURVEY_LOW": "SLow",
        "FORECAST_STANDARD_DEVIATION": "ForecastSDTv",
        "BN_SURVEY_NUMBER_OBSERVATIONS": "NumbSurvey"}
    
    def __init__(self, bloomberg_api, indicators_dict: Optional[Dict] = None):
        self.blp = bloomberg_api
        self.indicators = indicators_dict or {}
    
    def add_indicators(self, currency_code: str, indicators: List[Dict]) -> None:
        self.indicators[currency_code] = indicators
    
    def get_futureData(self, ticker: str, days, 
                    custom_fields: Optional[List[str]] = None) -> pd.DataFrame:
        fields = custom_fields or self.FUTURE_FIELDS
        today = datetime.today().date()
        target_date = today + timedelta(days=days)
        try:
            data = self.blp.bdh(
                tickers=ticker,
                flds=fields,
                start_date=today.strftime('%Y-%m-%d'),
                end_date=target_date.strftime('%Y-%m-%d'))
            if data.empty:
                return pd.DataFrame()
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            data = data.rename(columns=self.FIELD_RENAME_MAP)
            if 'ReleaseDate' in data.columns:
                data['ReleaseDate'] = pd.to_datetime(
                    data['ReleaseDate'].astype(str).str.split('.').str[0],
                    format='%Y%m%d',
                    errors='coerce')
                data = data.dropna(subset=['ReleaseDate'])
                data = data[data['ReleaseDate'].dt.date < target_date]
                data = data.sort_values('ReleaseDate')
            return data
        except Exception as e:
            print(f"Error getting future data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_pastData(self, ticker: str, days: int, 
                    custom_fields: Optional[List[str]] = None) -> pd.DataFrame:
        fields = custom_fields or self.DEFAULT_FIELDS
        today = datetime.today().date()
        cutoff_date = today - timedelta(days=days)
        try:
            data = self.blp.bdh(
                tickers=ticker,
                flds=fields,
                start_date=(today - timedelta(days=days*2)).strftime('%Y-%m-%d'),
                end_date=today.strftime('%Y-%m-%d'))
            if data.empty:
                return pd.DataFrame()
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            data = data.rename(columns=self.FIELD_RENAME_MAP)
            if 'ReleaseDate' in data.columns:
                data['ReleaseDate'] = pd.to_datetime(
                    data['ReleaseDate'].astype(str).str.split('.').str[0],
                    format='%Y%m%d',
                    errors='coerce')
                data = data.dropna(subset=['ReleaseDate'])
                data = data[data['ReleaseDate'].dt.date > cutoff_date]
                data = data.sort_values('ReleaseDate')
            return data
        except Exception as e:
            print(f"Error getting past data for {ticker}: {str(e)}")
            return pd.DataFrame()


class FXEventAnalyzer(EconomicDataManager):
    def __init__(self, bloomberg_api, indicators_dict: Optional[Dict] = None, 
                 pdblp_port: int = 8194, pdblp_timeout: int = 5000):
        super().__init__(bloomberg_api, indicators_dict)
        self.pdblp_con = None
        self.pdblp_port = pdblp_port
        self.pdblp_timeout = pdblp_timeout
        self._init_pdblp_connection()
    
    def _init_pdblp_connection(self):
        try:
            self.pdblp_con = pdblp.BCon(debug=False, port=self.pdblp_port, timeout=self.pdblp_timeout)
            self.pdblp_con.start()
        except Exception as e:
            print(f"Warning: Could not establish pdblp connection: {str(e)}")
            self.pdblp_con = None
    
    #  Tenor Parsing 
    @staticmethod
    def parse_tenor_to_days(tenor_str: str) -> int:
        if 'W' in tenor_str:
            weeks = int(tenor_str.replace('W', ''))
            return weeks * 7
        elif 'M' in tenor_str:
            months = int(tenor_str.replace('M', ''))
            return months * 30
        elif 'D' in tenor_str:
            days = int(tenor_str.replace('D', ''))
            return days
        elif 'Y' in tenor_str:
            years = int(tenor_str.replace('Y', ''))
            return years * 365
        return 30
    @staticmethod
    def parse_tenor_to_offset(tenor_str: str):
        if 'W' in tenor_str:
            weeks = int(tenor_str.replace('W', ''))
            return DateOffset(weeks=weeks)
        elif 'M' in tenor_str:
            months = int(tenor_str.replace('M', ''))
            return DateOffset(months=months)
        elif 'D' in tenor_str:
            days = int(tenor_str.replace('D', ''))
            return DateOffset(days=days)
        elif 'Y' in tenor_str:
            years = int(tenor_str.replace('Y', ''))
            return DateOffset(years=years)
        return DateOffset(months=1)
    
    # ------------- Spot Data -------------
    def get_fx_spot_data(self, fx_ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            data = self.blp.bdh(
                tickers=f'{fx_ticker} Curncy',
                flds=['PX_LAST'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'))
            if data.empty:
                return pd.DataFrame()
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            data = data.rename(columns={'PX_LAST': 'Spot'})
            return data
        except Exception as e:
            print(f"Error getting FX spot data for {fx_ticker}: {str(e)}")
            return pd.DataFrame()
    def get_fx_around_event(self, fx_ticker: str, event_date: datetime) -> pd.DataFrame:
        days_before = 15
        days_after = 7
        start_date = event_date - timedelta(days=days_before+1)
        end_date = event_date + timedelta(days=days_after)
        spot_data = self.get_fx_spot_data(fx_ticker, start_date, end_date)
        if spot_data.empty:
            return pd.DataFrame()
        spot_data.index = pd.to_datetime(spot_data.index)
        event_ts = pd.to_datetime(event_date)
        spot_data['EventDate'] = event_ts
        spot_data["DaysFromEvent"] = (spot_data.index - event_ts).days
        spot_data['Daily_LogRet'] = np.log(spot_data['Spot'] / spot_data['Spot'].shift(1))
        mean_ret = spot_data['Daily_LogRet'].mean()
        std_ret = spot_data['Daily_LogRet'].std()
        spot_data['Z_Score'] = (spot_data['Daily_LogRet'] - mean_ret) / std_ret
        spot_data['Abnormal_Flag'] = (abs(spot_data['Z_Score']) > 1.3).astype(int)
        if (spot_data.index >= event_ts).any():
            event_spot = spot_data.loc[spot_data.index >= event_ts, "Spot"].iloc[0]
        else:
            event_spot = spot_data["Spot"].iloc[-1]
        spot_data["IndexedSpot"] = (spot_data["Spot"] / event_spot) * 100
        mask = (spot_data["DaysFromEvent"] >= -days_before) & (spot_data["DaysFromEvent"] <= days_after)
        spot_data = spot_data.loc[mask]
        spot_data = spot_data.sort_values("DaysFromEvent")
        return spot_data
    # ------------- VOLATILITY DATA PULLING FUNCTIONS -------------
    def get_implied_vol(self, vol_ticker: str, start_date: datetime, 
                       end_date: datetime, tenor: str = '1M') -> pd.DataFrame:
        if 'Curncy' in vol_ticker:
            vol_ticker = vol_ticker.replace(' Curncy', '')
        iv_ticker = f"{vol_ticker}V{tenor} Curncy"
        try:
            data = self.blp.bdh(
                tickers=iv_ticker,
                flds=['PX_LAST'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'))
            if data.empty:
                return pd.DataFrame()
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            data = data.rename(columns={'PX_LAST': f'IV_{tenor}'})
            return data
        except Exception as e:
            print(f"Error getting implied vol for {iv_ticker}: {str(e)}")
            return pd.DataFrame()
    def get_realized_vol(self, vol_ticker: str, start_date: datetime, 
                        end_date: datetime, tenor: str = '1M') -> pd.DataFrame:
        if 'Curncy' in vol_ticker:
            vol_ticker = vol_ticker.replace(' Curncy', '')
        rv_ticker = f"{vol_ticker}H{tenor} Curncy"
        try:
            data = self.blp.bdh(
                tickers=rv_ticker,
                flds=['PX_LAST'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'))
            if data.empty:
                return pd.DataFrame()
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            data = data.rename(columns={'PX_LAST': f'RV_{tenor}'})
            return data
        except Exception as e:
            print(f"Error getting realized vol for {rv_ticker}: {str(e)}")
            return pd.DataFrame()
    @staticmethod
    def get_nearest_rv(target_date: pd.Timestamp, rv_series: pd.Series, max_days: int = 15):
        if pd.isna(target_date):
            return np.nan
        if target_date in rv_series.index and pd.notna(rv_series[target_date]):
            return rv_series[target_date]
        for offset in range(1, max_days + 1):
            past_date = target_date - pd.Timedelta(days=offset)
            if past_date in rv_series.index and pd.notna(rv_series[past_date]):
                return rv_series[past_date]
            future_date = target_date + pd.Timedelta(days=offset)
            if future_date in rv_series.index and pd.notna(rv_series[future_date]):
                return rv_series[future_date]
        return np.nan
    # ------------- INTRADAY REALIZED VOL CALCULATION -------------
    def get_intraday_spot_data(self, fx_ticker: str, start_datetime: datetime, 
                               end_datetime: datetime, interval: int = 60) -> pd.DataFrame:
        if self.pdblp_con is None:
            print("Warning: pdblp connection not available")
            return pd.DataFrame()
        if 'Curncy' in fx_ticker:
            fx_ticker = fx_ticker.replace(' Curncy', '')
        try:
            eastern = pytz.timezone("US/Eastern")
            if start_datetime.tzinfo is None:
                start_time_et = eastern.localize(start_datetime)
            else:
                start_time_et = start_datetime.astimezone(eastern)
            
            if end_datetime.tzinfo is None:
                end_time_et = eastern.localize(end_datetime)
            else:
                end_time_et = end_datetime.astimezone(eastern)
            start_time_gmt = start_time_et.astimezone(pytz.utc)
            end_time_gmt = end_time_et.astimezone(pytz.utc)
            df = self.pdblp_con.bdib(
                ticker=f"{fx_ticker} Curncy",
                start_datetime=start_time_gmt,
                end_datetime=end_time_gmt,
                event_type="TRADE",
                interval=interval)
            if df.empty:
                return pd.DataFrame()
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
            df = df.reset_index()
            df['date'] = df['time'].dt.date
            df['time_close'] = df['time'].dt.time
            df = df.drop(columns=['time'])
            df = df[['date', 'time_close'] + [col for col in df.columns if col not in ['date', 'time_close']]]
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time_close'].astype(str))
            df.set_index('datetime', inplace=True)
            df.drop(columns=['date', 'time_close'], inplace=True)
            if 'close' in df.columns:
                price_df = df[['close']].copy()
            elif 'last_price' in df.columns:
                price_df = df[['last_price']].copy()
                price_df.columns = ['close']
            else:
                print(f"Warning: No price column found. Available: {df.columns.tolist()}")
                return pd.DataFrame()
            return price_df
        except Exception as e:
            print(f"Error getting intraday data for {fx_ticker}: {str(e)}")
            return pd.DataFrame()
    def calculate_realized_vol_intraday(self, fx_ticker: str, start_datetime: datetime, 
                                       end_datetime: datetime, 
                                       interval: int = 60,
                                       annual_factor: int = 252) -> float:
        intraday_data = self.get_intraday_spot_data(
            fx_ticker, start_datetime, end_datetime, interval)
        if intraday_data.empty or len(intraday_data) < 2:
            return np.nan, np.nan
        returns = np.log(intraday_data['close'] / intraday_data['close'].shift(1))
        logReturns = returns.sum() * 100
        returns = returns.dropna()
        if len(returns) < 2:
            return np.nan, np.nan
        sum_squared_returns = (returns ** 2).sum()
        intervals_per_day = (24 * 60) / interval
        intervals_per_year = intervals_per_day * annual_factor
        num_intervals = len(returns)
        annualized_variance = sum_squared_returns * (intervals_per_year / num_intervals)
        annualized_vol = np.sqrt(annualized_variance) * 100
        return annualized_vol, logReturns
    def calculate_event_intraday_volsRet(self, fx_ticker: str, event_datetime: datetime) -> Dict[str, float]:
        results = {}
        start_12h = event_datetime - timedelta(hours=12)
        rv_12h_before, logRet_12h_before = self.calculate_realized_vol_intraday(
            fx_ticker, start_12h, event_datetime, 10)
        results['HF_RV_12h_Before'] = rv_12h_before
        results['HF_LogReturn_12h_Before'] = logRet_12h_before
        end_3h = event_datetime + timedelta(hours=3)
        rv_3h_after, logRet_3h_after = self.calculate_realized_vol_intraday(
            fx_ticker, event_datetime, end_3h, 10)
        results['HF_RV_3h_After'] = rv_3h_after
        results['HF_LogReturn_3h_After'] = logRet_3h_after
        end_12h = event_datetime + timedelta(hours=12)
        rv_12h_after, logRet_12h_after = self.calculate_realized_vol_intraday(
            fx_ticker, event_datetime, end_12h, 10)
        results['HF_RV_12h_After'] = rv_12h_after
        results['HF_LogReturn_12h_After'] = logRet_12h_after
        return results
    # ------------- Overnight Vol Analysis -------------
    def calculate_baseline_vols(self, fx_ticker: str, event_date: datetime, 
                               lookback_days: int = 10) -> Dict[str, float]:
        if 'Curncy' in fx_ticker:
            fx_ticker = fx_ticker.replace(' Curncy', '')
        dayBefore_event_date = event_date - timedelta(days=1)
        start_date = event_date - timedelta(days=lookback_days)
        tickers = [f'{fx_ticker}VON', f'{fx_ticker}V1W']
        try:
            event_Vols = daily_GetTickers(tickers, start_date, event_date)
            event_Vols.index = pd.to_datetime(event_Vols.index)
            # Previous Daily ON Vol Data Average/STD (EXCLUDING OUTLIERS)
            on_vols_clean = event_Vols[f'{fx_ticker}VON'][event_Vols.index < dayBefore_event_date].copy()
            mean = on_vols_clean.mean()
            std = on_vols_clean.std()
            on_vols_clean = on_vols_clean[(on_vols_clean < mean + 3*std) & (on_vols_clean > mean - 3*std)]
            baseline_historical = on_vols_clean.mean()
            # Vol Surface Interpolation - (Baseline_ON = sqrt((1W_vol² × 7 - EventON_vol² × 1) / 6))
            vol_1w = event_Vols.loc[dayBefore_event_date, f'{fx_ticker}V1W'] # 1W Vol 5pm day before event   
            event_on_vol = event_Vols.loc[dayBefore_event_date, f'{fx_ticker}VON'] # ON Vol 5pm day before event
            var_1w = (vol_1w / 100) ** 2
            var_event_on = (event_on_vol / 100) ** 2
            days_in_week = 7
            var_baseline = (var_1w * 7 - var_event_on * 1) / (days_in_week - 1)
            if var_baseline > 0:
                baseline_interpolated = np.sqrt(var_baseline) * 100
            else:
                baseline_interpolated = baseline_historical
            # Weighted Combination - Historical and Implied ON-1W Interpolated
            hist_weight = 0.3
            interp_weight = 0.7
            baseline_weighted = hist_weight * baseline_historical + interp_weight * baseline_interpolated
            return {
                'baseline_historical': baseline_historical,
                'baseline_interpolated': baseline_interpolated,
                'baseline_weighted': baseline_weighted,
                'event_on_vol': event_on_vol}
        except Exception as e:
            print(f"Error calculating baseline vols for {fx_ticker}: {str(e)}")
            return {
                'baseline_historical': np.nan,
                'baseline_interpolated': np.nan,
                'baseline_weighted': np.nan,
                'event_on_vol': np.nan,
                'vol_1w': np.nan}
    def calculate_event_premium_metrics(self, fx_ticker: str, event_datetime: datetime,
                                    baseline_vol: float, event_on_vol: float,
                                    interval: int = 10) -> Dict[str, float]:
        if event_datetime.hour == 8 and event_datetime.minute == 30:
            pre_event_datetime = (event_datetime - timedelta(days=1)).replace(hour=17, minute=0, second=0)
            realized_end_datetime = event_datetime.replace(hour=10, minute=0, second=0)
        else:
            pre_event_datetime = (event_datetime - timedelta(days=1)).replace(hour=17, minute=0, second=0)
            realized_end_datetime = event_datetime.replace(hour=10, minute=0, second=0)
        try:
            event_realized, logRet = self.calculate_realized_vol_intraday(
                fx_ticker, pre_event_datetime, realized_end_datetime, interval)
            event_impact =  event_on_vol - event_realized  if pd.notna(event_realized) else np.nan
            # Pre-Event Vol Prem = Pre-Event IV - Baseline Hist/Imp ON IV -- How much Event ON IV from Hist/Non-Event ON Vols
            event_premium_implied = event_on_vol - baseline_vol 
            # Implied Vol Prem - Actaul Impact of event (Realized - Baseline) -- Realized IV Overchage - Expected Overcharge 
            premium_vs_realized = event_premium_implied - event_impact if pd.notna(event_impact) else np.nan
            # Premium Efficency = (Implied - Fwd Realized) / Implied × 100  -- X% of the vol level didn't realize 
            premium_efficiency = ((event_on_vol - event_realized) / event_on_vol * 100 if pd.notna(event_realized) and event_on_vol > 0 else np.nan) 
            # Vol Risk Ratio = (Event_ON_Vol - Realized) / Baseline  -- Per Unit of baseline vol, X units were over priced
            vol_risk_ratio = ((event_on_vol - event_realized) / baseline_vol if pd.notna(event_realized) and baseline_vol > 0 else np.nan)
            # Vol Seller PnL (received premium - paid realized) 
            vol_seller_pnl = event_on_vol - event_realized if pd.notna(event_realized) else np.nan
            return {
                'event_premium_implied': event_premium_implied,
                'event_realized_vol': event_realized,
                'event_impact': event_impact,
                'premium_vs_realized': premium_vs_realized,
                'premium_efficiency': premium_efficiency,
                'vol_risk_ratio': vol_risk_ratio,
                'vol_seller_pnl': vol_seller_pnl}
        except Exception as e:
            print(f"Error calculating event premium metrics for {fx_ticker}: {str(e)}")
            return {
                'event_premium_implied': event_premium_implied,
                'event_realized_vol': np.nan,
                'event_impact': np.nan,
                'premium_vs_realized': np.nan,
                'premium_efficiency': np.nan,
                'vol_risk_ratio': np.nan,
                'vol_seller_pnl': np.nan}
        
    # ========================= COMBINED DATA METHOD =========================
    def data_singleEvent_CCY_Tenors(self, fx_ticker: str, event_date: datetime,
                                    days_before: int = 10, 
                                    days_after: int = 10,
                                    include_forward_rv: bool = True,
                                    max_lookback_days: int = 15) -> pd.DataFrame:
        tenor = '1W'
        tenor_offset = self.parse_tenor_to_offset(tenor)
        tenor_days = self.parse_tenor_to_days(tenor)
        vol_start = event_date - timedelta(days=days_before * 2)
        if include_forward_rv:
            vol_end = event_date + timedelta(days=days_after + tenor_days + 60)
        spot_data = self.get_fx_around_event(fx_ticker, event_date)
        if spot_data.empty:
            print(f"No spot data available for {fx_ticker}")
            return pd.DataFrame()
        combined = spot_data.copy()
        iv_data = self.get_implied_vol(fx_ticker, vol_start, vol_end, tenor)
        if iv_data.empty:
            print(f"Warning: No IV data for {tenor}")
            return combined
        rv_data = self.get_realized_vol(fx_ticker, vol_start, vol_end, tenor)
        if rv_data.empty:
            print(f"Warning: No RV data for {tenor}")
            return combined
        iv_data.index = pd.to_datetime(iv_data.index)
        combined = combined.join(iv_data[[f'IV_{tenor}']], how='left')
        rv_data.index = pd.to_datetime(rv_data.index)
        combined = combined.join(rv_data[[f'RV_{tenor}']], how='left')
        combined[f'RV_current_{tenor}'] = combined[f'RV_{tenor}']
        if include_forward_rv:
            combined[f'Target_Date_{tenor}'] = combined.index + tenor_offset
            rv_series = rv_data[f'RV_{tenor}']
            combined[f'RV_forward_{tenor}'] = combined[f'Target_Date_{tenor}'].apply(
                lambda x: self.get_nearest_rv(x, rv_series, max_lookback_days))
            combined[f'Premium_{tenor}'] = combined[f'IV_{tenor}'] - combined[f'RV_forward_{tenor}']
            combined[f'IV_minus_current_RV_{tenor}'] = combined[f'IV_{tenor}'] - combined[f'RV_current_{tenor}']
            combined[f'IV_Overstated_{tenor}'] = (combined[f'Premium_{tenor}'] > 0).astype(float)
            combined.loc[combined[f'Premium_{tenor}'].isna(), f'IV_Overstated_{tenor}'] = np.nan
        else:
            if f'RV_{tenor}' in combined.columns:
                combined = combined.drop(columns=[f'RV_{tenor}'])
        combined['FX_Pair'] = fx_ticker
        return combined
    
    # ------------- MULTI-CURRENCY ANALYSIS -------------
    def data_singleEvent_CCYS_Tenors(self, fx_tickers: List[str], event_date: datetime, days_before: int = 10, 
                                    days_after: int = 10, include_forward_rv: bool = True,
                                    max_lookback_days: int = 15) -> Dict[str, pd.DataFrame]:
        results = {}
        for fx_ticker in fx_tickers:
            combined = self.data_singleEvent_CCY_Tenors(
                fx_ticker=fx_ticker,
                event_date=event_date,
                days_before=days_before,
                days_after=days_after,
                include_forward_rv=include_forward_rv,
                max_lookback_days=max_lookback_days)
            if not combined.empty:
                results[fx_ticker] = combined
            else:
                print(f"  Warning: No data for {fx_ticker}")
        return results
    
    def summary_singleEvent_CCYS_Tenors_with_fixed_time(self, multi_currency_data: Dict[str, pd.DataFrame], event_datetime: datetime,
                                                        event_row: pd.Series, include_premium: bool = True,
                                                        include_intraday_rv: bool = True, include_event_premium_analysis: bool = True,
                                                        baseline_lookback_days: int = 30) -> pd.DataFrame:
        tenor = '1W'
        summary_list = []
        for fx_pair, data in multi_currency_data.items():
            if data.empty:
                continue
            pre_event = data[data['DaysFromEvent'] < 0].sort_values('DaysFromEvent')
            pre_week = pre_event.tail(5)
            pre_1day = pre_event.tail(1)
            post_event = data[data['DaysFromEvent'] > 0].sort_values('DaysFromEvent')
            post_week = post_event.head(5)
            post_1day = post_event.head(1)
            post_2days = post_event.head(2)
            event_day = data[data['DaysFromEvent'] == 0]
            # ---- DAILY Spot Data ---- 
            summary = {
                'FX_Pair': fx_pair,
                'EventDate': data['EventDate'].iloc[0] if len(data) > 0 else None,
                'Daily_Event_Return_%': event_day['Daily_LogRet'].iloc[0] * 100 if len(event_day) > 0 and not event_day['Daily_LogRet'].isna().all() else np.nan,
                'Daily_t-1W_LogRet_%': pre_week['Daily_LogRet'].sum() * 100 if len(pre_week) > 0 else np.nan,
                'Daily_t+1_logRet_%': (np.log(post_1day['Spot'].iloc[0] / event_day['Spot'].iloc[0]) * 100 if len(event_day) > 0 and len(post_1day) > 0 else np.nan),
                'Daily_t+2_logRet_%': (np.log(post_2days['Spot'].iloc[-1] / event_day['Spot'].iloc[0]) * 100 if len(event_day) > 0 and len(post_2days) >= 2 else np.nan),
                'Daily_Outliers': [(row.name.strftime('%Y-%m-%d'), row['Daily_LogRet']*100) for idx, row in pre_event[pre_event['Abnormal_Flag'] == 1].iterrows()]}
            # ---- INTRADAY Spot Data + RV Calculations ----
            if include_intraday_rv:
                try:
                    intraday_vols = self.calculate_event_intraday_volsRet(fx_pair, event_datetime)
                    summary['HF_RV_12h_Before'] = intraday_vols.get('HF_RV_12h_Before', np.nan)
                    summary['HF_RV_3h_After'] = intraday_vols.get('HF_RV_3h_After', np.nan)
                    summary['HF_RV_12h_After'] = intraday_vols.get('HF_RV_12h_After', np.nan)
                    summary['HF_LogReturn_12h_Before'] = intraday_vols.get('HF_LogReturn_12h_Before', np.nan)
                    summary['HF_LogReturn_3h_After'] = intraday_vols.get('HF_LogReturn_3h_After', np.nan)
                    summary['HF_LogReturn_12h_After'] = intraday_vols.get('HF_LogReturn_12h_After', np.nan)
                except Exception as e:
                    print(f"  Warning: Could not calculate additional intraday data for {fx_pair}: {str(e)}")
                    summary['HF_RV_12h_Before'] = np.nan
                    summary['HF_RV_3h_After'] = np.nan
                    summary['HF_RV_12h_After'] = np.nan
                    summary['HF_LogReturn_12h_Before'] = np.nan
                    summary['HF_LogReturn_3h_After'] = np.nan
                    summary['HF_LogReturn_12h_After'] = np.nan
            summary['     '] = ''
            summary['  -----   '] = '-----'                              

            # ---- 1W VOL METRICS ----
            iv_col = f'IV_{tenor}'
            rv_current_col = f'RV_current_{tenor}'
            rv_forward_col = f'RV_forward_{tenor}'
            premium_col = f'Premium_{tenor}'
            if iv_col not in data.columns:
                summary_list.append(summary)
                continue
            pre_eventWeight = data[data['DaysFromEvent'] < -7].sort_values('DaysFromEvent')
            if len(pre_event) > 0:
                summary[f'Earliest_Day_{tenor}'] = pre_event['DaysFromEvent'].iloc[0]
                summary[f'Earliest_IV_{tenor}'] = pre_event[iv_col].iloc[0]
                if rv_current_col in data.columns:
                    summary[f'Earliest_RV_{tenor}'] = pre_event[rv_current_col].iloc[0]
                if premium_col in data.columns and include_premium:
                    summary[f'Earliest_Premium_{tenor}'] = pre_event[premium_col].iloc[0]
                summary[f'Avg_PreEventWeighted_IV_{tenor}'] = pre_eventWeight[iv_col].mean()
                if rv_current_col in data.columns:
                    summary[f'Avg_PreEventWeighted_RV_{tenor}'] = pre_eventWeight[rv_current_col].mean()
                if premium_col in data.columns and include_premium:
                    summary[f'Avg_PreEventWeighted_Premium_{tenor}'] = pre_eventWeight[premium_col].mean()
            summary['      '] = ' '

            day_7 = data[data['DaysFromEvent'] == -7]
            if len(day_7) > 0:
                summary[f'Day7_IV_{tenor}'] = day_7[iv_col].iloc[0]
                if rv_current_col in data.columns:
                    summary[f'Day7_RV_current_{tenor}'] = day_7[rv_current_col].iloc[0]
                if rv_forward_col in data.columns and include_premium:
                    summary[f'Day7_RV_forward_{tenor}'] = day_7[rv_forward_col].iloc[0]
                if premium_col in data.columns and include_premium:
                    summary[f'Day7_Premium_{tenor}'] = day_7[premium_col].iloc[0]
            summary['       '] = ' '
            
            if len(pre_eventWeight) > 0 and len(day_7) > 0:
                day_8 = pre_eventWeight.tail(1)
                day_8_iv_val = day_8[iv_col].iloc[0]
                day_7_iv_val = day_7[iv_col].iloc[0]
                if pd.notna(day_8_iv_val) and pd.notna(day_7_iv_val):
                    summary[f'IV_1d_EventRoll_{tenor}'] = day_7_iv_val - day_8_iv_val
                if premium_col in data.columns and include_premium:
                    day_8_prem_val = day_8[premium_col].iloc[0]
                    day_7_prem_val = day_7[premium_col].iloc[0]
                    if pd.notna(day_8_prem_val) and pd.notna(day_7_prem_val):
                        summary[f'Premium_1d_EventRoll_{tenor}'] = day_7_prem_val - day_8_prem_val
            summary['        '] = ' '
            
            if -1 in data['DaysFromEvent'].values:
                pre_eventday = data[data['DaysFromEvent'] == -1]
            else:
                pre_eventday = data[data['DaysFromEvent'] == -3]
            if len(pre_eventday) > 0:
                summary[f'PreEventDay_Num_{tenor}'] = pre_eventday['DaysFromEvent'].iloc[0]
                summary[f'PreEventDay_IV_{tenor}'] = pre_eventday[iv_col].iloc[0]
                if rv_current_col in data.columns:
                    summary[f'PreEventDay_RV_{tenor}'] = pre_eventday[rv_current_col].iloc[0]
                if premium_col in data.columns and include_premium:
                    summary[f'PreEventDay_Premium_{tenor}'] = pre_eventday[premium_col].iloc[0]
            summary['         '] = ' '
            
            if len(event_day) > 0 and not event_day[iv_col].isna().all():
                summary[f'EventDay_IV_{tenor}'] = event_day[iv_col].iloc[0]
                if rv_current_col in data.columns:
                    summary[f'EventDay_RV_{tenor}'] = event_day[rv_current_col].iloc[0]
                if premium_col in data.columns and include_premium:
                    summary[f'EventDay_Premium_{tenor}'] = event_day[premium_col].iloc[0]
                if len(pre_eventday) > 0:
                    pre_iv = pre_eventday[iv_col].iloc[0]
                    event_iv = event_day[iv_col].iloc[0]
                    if pd.notna(pre_iv) and pd.notna(event_iv):
                        iv_crush = pre_iv - event_iv
                        summary[f'IV_PostEvent_VolChange_{tenor}'] = iv_crush
            summary['          '] = ' '
            
            if len(post_event) > 0:
                summary[f'Latest_Day_{tenor}'] = post_event['DaysFromEvent'].iloc[-1]
                summary[f'Avg_PostEvent_IV_{tenor}'] = post_event[iv_col].mean()
                if rv_current_col in data.columns:
                    summary[f'Avg_PostEvent_RV_{tenor}'] = post_event[rv_current_col].mean()
                if premium_col in data.columns and include_premium:
                    summary[f'Avg_PostEvent_Premium_{tenor}'] = post_event[premium_col].mean()
                day_3_post = data[data['DaysFromEvent'] == 3]
                if len(day_3_post) > 0 and premium_col in data.columns and include_premium:
                    day_3_prem = day_3_post[premium_col].iloc[0]
                    summary[f'Day3_Premium_{tenor}'] = day_3_prem
                    if len(day_7) > 0:
                        day_7_prem = day_7[premium_col].iloc[0]
                        if pd.notna(day_7_prem) and pd.notna(day_3_prem):
                            summary[f'Premium_Collapse_{tenor}'] = day_7_prem - day_3_prem
            summary['           '] = ' '
            
            if len(day_7) > 0 and rv_forward_col in data.columns and include_premium:
                day_7_iv_val = day_7[iv_col].iloc[0]
                day_7_rv_fwd_val = day_7[rv_forward_col].iloc[0]
                if pd.notna(day_7_iv_val) and pd.notna(day_7_rv_fwd_val):
                    summary[f'VolSeller_PnL_1W_{tenor}'] = day_7_iv_val - day_7_rv_fwd_val
                    summary[f'VolBuyer_PnL_1W_{tenor}'] = day_7_rv_fwd_val - day_7_iv_val
            summary['            '] = ' '
            
            if len(day_7) > 0 and len(pre_eventWeight) > 0 and rv_current_col in data.columns:
                day_7_iv_val = day_7[iv_col].iloc[0]
                avg_pre_rv = pre_eventWeight[rv_current_col].mean()
                if pd.notna(day_7_iv_val) and pd.notna(avg_pre_rv) and avg_pre_rv > 0:
                    summary[f'Vol_Risk_Ratio_1W_{tenor}'] = day_7_iv_val / avg_pre_rv
            summary['------------ '] = '------------'
            summary['                      '] = ' '

            # ---- ON Event Vol Analysis ----
            if include_event_premium_analysis:
                event_date = pd.to_datetime(data['EventDate'].iloc[0])
                baseline_metrics = self.calculate_baseline_vols(
                    fx_pair, event_date, lookback_days=baseline_lookback_days)
                summary['Event_ON_Vol'] = baseline_metrics['event_on_vol']
                summary['                                  '] = ' '

                summary[f'Baseline_Hist{baseline_lookback_days}dAve_ON'] = baseline_metrics['baseline_historical']
                summary['Baseline_ONv1W_Calander_Interp_ON'] = baseline_metrics['baseline_interpolated']
                summary['Baseline_Weighted_ON_30hist/70cal'] = baseline_metrics['baseline_weighted']
                premium_metrics = self.calculate_event_premium_metrics(fx_pair, event_datetime,
                    baseline_metrics['baseline_weighted'], baseline_metrics['event_on_vol'],
                    interval=10)
                summary['EventON_vs_WeightedBaselineON_diff'] = premium_metrics['event_premium_implied']
                summary['             '] = ' '

                summary['EventRV_5pm_10am'] = premium_metrics['event_realized_vol']
                summary['Event_ONimplied_vs_ONrealized_diff'] = premium_metrics['event_impact']
                summary['                  '] = ' '

                summary['Event_RealizedVolPrem_vs_preEventVolPrem_diff'] = premium_metrics['premium_vs_realized']
                summary['Premium_Efficiency_%'] = premium_metrics['premium_efficiency']
                summary['                    '] = ' '

                summary['Vol_Risk_Ratio'] = premium_metrics['vol_risk_ratio']
                summary['VolSeller_PnL'] = premium_metrics['vol_seller_pnl']                
                summary['  '] = ''

            summary_list.append(summary)
        return pd.DataFrame(summary_list)
    
    def analyze_indicator_events(self, indicator_ticker: str, 
                                fx_tickers: List[str],
                                event_time: str = '08:30:00',
                                lookback_days: int = 100,
                                days_before: int = 5,
                                days_after: int = 5,
                                include_forward_rv: bool = True,
                                max_lookback_days: int = 15,
                                include_intraday_rv: bool = True,
                                include_event_premium_analysis: bool = True,
                                baseline_lookback_days: int = 30,
                                interval: int = 60,
                                num_events: Optional[int] = None) -> pd.DataFrame:
        
        event_data = self.get_pastData(indicator_ticker, lookback_days)
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
            print(f"Error parsing event_time '{event_time}': {str(e)}")
            print("Using default time 08:30:00")
            fixed_hour, fixed_minute, fixed_second = 8, 30, 0
        all_results = []
        for event_num, (idx, row) in enumerate(event_data.iterrows(), start=1):
            event_date = pd.to_datetime(row['ReleaseDate'])
            event_datetime = event_date.replace(
                hour=fixed_hour, 
                minute=fixed_minute - 10, 
                second=fixed_second)
            multi_ccy_data = self.data_singleEvent_CCYS_Tenors(
                fx_tickers=fx_tickers,
                event_date=event_date,
                days_before=days_before,
                days_after=days_after,
                include_forward_rv=include_forward_rv,
                max_lookback_days=max_lookback_days)
            summary = self.summary_singleEvent_CCYS_Tenors_with_fixed_time(
                multi_currency_data=multi_ccy_data,
                event_datetime=event_datetime,
                event_row=row,
                include_premium=include_forward_rv,
                include_intraday_rv=include_intraday_rv,
                include_event_premium_analysis=include_event_premium_analysis,
                baseline_lookback_days=baseline_lookback_days)
            if not summary.empty:
                summary['EventName'] = row.get('NAME', indicator_ticker)
                summary['Actual'] = row.get('Actual', np.nan)
                summary['SMed'] = row.get('SMed', np.nan)
                summary['Surprise'] = (row.get('Actual', np.nan) - row.get('SMed', np.nan) 
                                    if pd.notna(row.get('Actual')) and pd.notna(row.get('SMed')) 
                                    else np.nan)
                summary['ReleaseDateTime'] = event_datetime
                all_results.append(summary)
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            return combined
        else:
            print("No results generated")
            return pd.DataFrame()
        





# analyzer = FXEventAnalyzer(blp, pdblp_port=8194, pdblp_timeout=5000)


# results = analyzer.analyze_indicator_events(
#     indicator_ticker='CPI YOY Index',
#     fx_tickers=['EURUSD'],
#     event_time='08:30:00',
#     lookback_days=100,
#     days_before=10,
#     days_after=10,
#     include_forward_rv=True,
#     include_intraday_rv=True,
#     include_event_premium_analysis=True,  
#     baseline_lookback_days=30,  
#     num_events=1)


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.expand_frame_repr', False)


# print(results.T)