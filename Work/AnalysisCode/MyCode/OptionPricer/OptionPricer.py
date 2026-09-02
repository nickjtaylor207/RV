from xbbg import blp
import pdblp

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from scipy.interpolate import interp1d

from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt



# Get the OIS RiskFree Rate for CCY and tenor
def riskFreeRateCurve(ccy, days_till):
    # US OIS Rates
    if ccy == 'USD':
        ois_tickers = [
            'USSO1Z BGN Curncy', 'USSO2Z BGN Curncy', 'USSO3Z BGN Curncy',
            'USSOA BGN Curncy', 'USSOB BGN Curncy', 'USSOC BGN Curncy', 
            'USSOF BGN Curncy', 'USSO1 BGN Curncy', 'USSO1F BGN Curncy', 
            'USSO2 BGN Curncy']

    # EUR OIS Rates
    if ccy == 'EUR':
        ois_tickers = [
            'EESWE1Z BGN Curncy', 'EESWE2Z BGN Curncy', 'EESWE3Z BGN Curncy', 
            'EESWEA BGN Curncy', 'EESWEB BGN Curncy', 'EESWEC BGN Curncy', 
            'EESWEF BGN Curncy', 'EESWE1 BGN Curncy', 'EESWE1F BGN Curncy', 
            'EESWE2 BGN Curncy']

    # GBP OIS Rates
    if ccy == 'GBP':
        ois_tickers = [
            'BPSWS1Z BGN Curncy', 'BPSWS2Z BGN Curncy', 'BPSWS3Z BGN Curncy', 
            'BPSWSA BGN Curncy', 'BPSWSB BGN Curncy', 'BPSWSC BGN Curncy', 
            'BPSWSF BGN Curncy', 'BPSWS1 BGN Curncy', 'BPSWS1F BGN Curncy', 
            'BPSWS2 BGN Curncy']

    # JPY OIS Rates
    if ccy == 'JPY':
        ois_tickers = [
            'JYSO1Z BGN Curncy', 'JYSO2Z BGN Curncy', 'JYSO3Z BGN Curncy', # 3W
            'JYSOA BGN Curncy', 'JYSOB BGN Curncy', 'JYSOC BGN Curncy', # 3M
            'JYSOF BGN Curncy', 'JYSO1 BGN Curncy', 'JYSO1F BGN Curncy', # 18M
            'JYSO2 BGN Curncy']


    # CAD OIS Rates
    if ccy == 'CAD':
        ois_tickers = [
            'CDSO1Z BGN Curncy', 'CDSO2Z BGN Curncy', 'CDSO3Z BGN Curncy', # 3W
            'CDSOA BGN Curncy', 'CDSOB BGN Curncy', 'CDSOC BGN Curncy', # 3M
            'CDSOF BGN Curncy', 'CDSO1 BGN Curncy', 'CDSO1F BGN Curncy', # 18M
            'CDSO2 BGN Curncy']



    # CHF OIS Rates
    if ccy == 'CHF':
        ois_tickers = [
            'SFSNT1Z BGN Curncy', 'SFSNT2Z BGN Curncy', 'SFSNT3Z BGN Curncy', # 3W
            'SFSNTA BGN Curncy', 'SFSNTB BGN Curncy', 'SFSNTC BGN Curncy', # 3M
            'SFSNTF BGN Curncy', 'SFSNT1 BGN Curncy', 'SFSNT1F BGN Curncy', # 18M
            'SFSNT2 BGN Curncy']

    # AUD OIS Rates
    if ccy == 'AUD':
        ois_tickers = [
            'ADSO1Z BGN Curncy', 'ADSO2Z BGN Curncy', 'ADSO3Z BGN Curncy', # 3W
            'ADSOA BGN Curncy', 'ADSOB BGN Curncy', 'ADSOC BGN Curncy', # 3M
            'ADSOF BGN Curncy', 'ADSO1 BGN Curncy', 'ADSO1F BGN Curncy', # 18M
            'ADSO2 BGN Curncy']

    # NZD OIS Rates
    if ccy == 'NZD':
        ois_tickers = [
            'NDSO1Z BGN Curncy', 'NDSO2Z BGN Curncy', 'NDSO3Z BGN Curncy', # 3W
            'NDSOA BGN Curncy', 'NDSOB BGN Curncy', 'NDSOC BGN Curncy', # 3M
            'NDSOF BGN Curncy', 'NDSO1 BGN Curncy', 'NDSO1F BGN Curncy', # 18M
            'NDSO2 BGN Curncy']


    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    names = ['1W', '2W', '3W', '1M', '2M', '3M', '6M', '1Y', '18M', '2Y']

    # Gather data
    ois_data = blp.bdh(
        tickers=ois_tickers,
        flds='PX_LAST',
        start_date=start_date,
        end_date=end_date
    )

    # Clean dataframe
    ois_rates = ois_data.iloc[-1].reset_index()
    ois_rates['level_0'] = names
    ois_rates = ois_rates.drop(columns=['level_1'])
    ois_rates.columns = ['Tenor', 'Rate']
    ois_rates['Rate'] /= 100  # Convert percentages to decimals

    # Map tenors to times (in years)
    tenor_map = {'W': 7 / 365, 'M': 30 / 365, 'Y': 1}
    ois_rates['Time'] = ois_rates['Tenor'].str.extract(r'(\d+)([WMY])') \
        .apply(lambda x: int(x[0]) * tenor_map[x[1]], axis=1)

    # Interpolate the rate curve directly
    def interpolate_rate_curve(ois_curve):
        return interp1d(
            ois_curve['Time'], ois_curve['Rate'],
            kind='cubic', fill_value='extrapolate'
        )

    # Create the interpolation function
    interp_func = interpolate_rate_curve(ois_rates)
    
    # Convert days to years
    time_in_years = days_till / 365
    
    # Get the interpolated rate
    interpolated_rate = interp_func(time_in_years)
    
    return interpolated_rate








# ----------------------------------------------------------------------------------------------------------------------------
# --------------------Getting Implied Vol ------------------------------------------------------------------------------------


# Get 25D and 10D vol for desired CCY
# 25D Call | 25D Put | 10D Call | 10D Put
def getC_PVols(ccy):

    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()

    tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '1Y', '18M', '2Y']


    results = {}

    for tenor in tenors:
        ticker = [
            f'{ccy}V{tenor} Curncy',  # ATM Vol
            f'{ccy}25R{tenor} Curncy',  # 25 Delta Risk Reversal
            f'{ccy}25B{tenor} Curncy',  # 25 Delta Butterfly
            f'{ccy}10R{tenor} Curncy',  # 10 Delta Risk Reversal
            f'{ccy}10B{tenor} Curncy',  # 10 Delta Butterfly
            f'{ccy}15R{tenor} Curncy',  # 15 Delta Risk Reversal
            f'{ccy}15B{tenor} Curncy',  # 15 Delta Butterfly
            f'{ccy}35R{tenor} Curncy',  # 35 Delta Risk Reversal
            f'{ccy}35B{tenor} Curncy'   # 35 Delta Butterfly
        ]

        
        
        vol_data = con.ref(ticker, ["PX_LAST"])['value']
        
        # Extract values from the data
        ATM_Vol = vol_data[0]  # ATM Volatility
        R25 = vol_data[1]      # 25 Delta Risk Reversal
        B25 = vol_data[2]      # 25 Delta Butterfly
        R10 = vol_data[3]      # 10 Delta Risk Reversal
        B10 = vol_data[4]      # 10 Delta Butterfly
        R15 = vol_data[5]      # 15 Delta Risk Reversal
        B15 = vol_data[6]      # 15 Delta Butterfly
        R35 = vol_data[7]      # 35 Delta Risk Reversal
        B35 = vol_data[8]      # 35 Delta Butterfly
        
        # Calculate the 25D Call, 25D Put, 10D Call, and 10D Put
        Call_25D = ATM_Vol + 0.5 * R25 + B25
        Put_25D = ATM_Vol - 0.5 * R25 + B25
        Call_15D = ATM_Vol + 0.5 * R15 + B15
        Put_15D = ATM_Vol - 0.5 * R15 + B15
        Call_10D = ATM_Vol + 0.5 * R10 + B10
        Put_10D = ATM_Vol - 0.5 * R10 + B10
        Call_35D = ATM_Vol + 0.5 * R35 + B35
        Put_35D = ATM_Vol - 0.5 * R35 + B35


        # Store the results in the dictionary
        results[tenor] = {
            'ATM': ATM_Vol,
            '25D Call': Call_25D,
            '25D Put': Put_25D,
            '15D Call': Call_15D,
            '15D Put': Put_15D,
            '35D Call': Call_35D,
            '35D Put': Put_35D,
            '10D Call': Call_10D,
            '10D Put': Put_10D
        }

    # Convert the results dictionary into a DataFrame
    df_vols = pd.DataFrame.from_dict(results, orient='index')

    # Display the resulting DataFrame
    return df_vols




# Getting Implied Vol based on Tenor and Delta (In Call Delta --- 20D Call = 80D Put)
def getIV(ccy, tenor, delta):
    # Get Vols
    df_vols = getC_PVols(ccy)

    # Change tenor to decimal
    tenor_map = {'W': 7 / 365, 'M': 30 / 365, 'Y': 1}
    df_vols.index = df_vols.index.str.extract(r'(\d+)([WMY])') \
        .apply(lambda x: int(x[0]) * tenor_map[x[1]], axis=1)

    tenors = df_vols.index.values
    deltas = [0.10, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9]

    # Reorder vol_matrix columns to match the new delta order
    vol_matrix = np.array([
        df_vols['10D Call'].values, # Corresponds to 0.10
        df_vols['15D Call'].values, # Corresponds to 0.10
        df_vols['25D Call'].values, # Corresponds to 0.25
        df_vols['35D Call'].values, # Corresponds to 0.10
        df_vols['ATM'].values,       # Corresponds to 0.50
        df_vols['35D Put'].values, # Corresponds to 0.10
        df_vols['25D Put'].values,  # Corresponds to -0.25
        df_vols['15D Put'].values, # Corresponds to 0.10
        df_vols['10D Put'].values  # Corresponds to -0.10
    ]).T  


    interpolator = RegularGridInterpolator(
        (tenors, deltas),
        vol_matrix,
        bounds_error=False,
        fill_value=None
    )

    interpolated_vol = interpolator((tenor, delta))

    return interpolated_vol



























ccy = 'USDCHF'

tenor = 1/365  # 6 months
delta = 0.5  # 40P


print(getIV(ccy, tenor, delta))



# -----------------------------------------------------------------------------

# Plotting generated vol smile 

# # Define parameters
# ccy = 'EURUSD'
# tenor = 1 / 12  # 1M tenor
# deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # Deltas to evaluate

# # Calculate implied volatilities
# vols = [getIV(ccy, tenor, delta) for delta in deltas]

# # Plot the results
# plt.figure(figsize=(8, 6))
# plt.plot(deltas, vols, marker='o', linestyle='-', label=f'{tenor*12:.0f}M Tenor')

# # Add value labels to each point
# for delta, vol in zip(deltas, vols):
#     plt.text(delta, vol, f'{vol:.2f}', fontsize=9, ha='center', va='bottom')  # Adjust alignment

# # Add vertical line at ATM
# plt.axvline(0, color='black', linestyle='--', alpha=0.5)

# # Graph details
# plt.title(f'Implied Volatility for {ccy} at {tenor*12:.0f}M Tenor')
# plt.xlabel('Delta')
# plt.ylabel('Implied Volatility (%)')
# plt.legend()
# plt.grid()
# plt.show()


# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# #Getting riskFree rate from OIS Data

# ccy = 'NZD'
# days_till = 31

# print(riskFreeRateCurve(ccy, days_till))