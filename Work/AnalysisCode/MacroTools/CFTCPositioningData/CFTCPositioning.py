from xbbg import blp
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import numpy as np
from itertools import product


# ----------------------------------------------------------------------------------------------------------
# --------------------- Data Pulling --------------------------------------------------------------------

# ---------------ALL LEGACY -- COMERCIAL VS NON-COMERCIAL 

# 1 CCY Data Frame of all Institutional Vs Speculators CFTC Data
def Inst_Spec_futurePositioning(spot, years):


    # ----- Getting futures codes based on ccy ticker -----
    if spot == 'EURUSD': # Euro Dollar
        ccy_F = "IMMBE"
        ccy_F_O = 'IMMPF'

    if spot == 'USDJPY': # Dollar Yen
        ccy_F = "IMM5J"
        ccy_F_O = "IMMOJ"

    if spot == 'GBPUSD': # Pound Dollar
        ccy_F = "IMM5P"
        ccy_F_O = "IMMOP"

    if spot == 'USDCAD': # Dollar Cad
        ccy_F = "IMM3C"
        ccy_F_O = "IMMOC"

    if spot == 'AUDUSD': # Aussie Dollar
        ccy_F = "IMM6A"
        ccy_F_O = "IMMOA"

    if spot == 'NZDUSD': # Kiwi Dollar
        ccy_F = "IMM6Z"
        ccy_F_O = "IMMTZ"

    if spot == 'USDCHF': # Dollar Swiss
        ccy_F = "IMM4S"
        ccy_F_O = "IMMTS"

    if spot == 'EURGBP': # Euro Pound
        ccy_F = "CFF8E"
        ccy_F_O = "CFC8T"


    spot_tick = f"{spot} Curncy"


    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    # Catagory Codes 
    codes = [
        'TLL', 'TLS', 'TLN',  # Total Positions   (Long, Short, Net)
        'NCL', 'NCS', 'NCP', 'NCN',  # Speculators Positions   (Long, Short, Spreading, Net)
        'COL', 'COS', 'CON'  # Institutional Hedgers   (Long, Short, Net)
    ]


    column_names = [
        'tot_Long', 'tot_Short', 'tot_Net',
        'spec_Long', 'spec_Short', 'spec_Spreading', 'spec_Net',
        'inst_Long', 'inst_Short', 'inst_Net'
    ]


    ticker = [f"{base}{code}" for base in [ccy_F, ccy_F_O] for code in codes] # Generating combo of all tickers (All Futures THEN All Fut&Opt)
    ticker = [f"{code} Index" for code in ticker] # 


    # Pull data using blp.bdh()
    df_data = blp.bdh(
        tickers=ticker,
        start_date=start_date,
        end_date=end_date
    )

    df_data.columns = ['_'.join(col).strip() for col in df_data.columns.values] # Join column headers into 1


    # Re-structuring of column names in CME Data 
    tickers = [f"{base}{code} Index_Last_Price" for base in [ccy_F, ccy_F_O] for code in codes]
    columns_with_names = {
        ticker: f"{f'{spot}_F' if ccy_F in ticker else f'{spot}_F_O'}_{column_names[i % len(column_names)]}"
        for i, ticker in enumerate(tickers)}
    df_data.rename(columns=columns_with_names, inplace=True)

    # Gathering Spot Data
    df_spot = blp.bdh(
        tickers=spot_tick,
        start_date=start_date,
        end_date=end_date
    )

    # Re-structuring of column names in spot data
    df_spot.columns = ['_'.join(col).strip() for col in df_spot.columns.values]
    df_spot.rename(columns={f"{spot} Curncy_Last_Price": spot_tick}, inplace=True)

    # Merging Fut&Opt data with Spot data
    df_data[f'{spot_tick}'] = df_spot[f'{spot_tick}']
    df_data.insert(0, f'{spot_tick}', df_data.pop(f'{spot_tick}'))


    return df_data, df_spot










# ---------------ALL DISAGGREGATED -- (ASSET MANAGER, DEALER INTERMEDIARY, LEVERAGED FUNDS)

# 1 CCY - Data Frame of Traders (Levered and Asset Managers) in Financial Futures from CFTC Data
def Traders_futurePositioning(spot, years):


    # ----- Getting futures codes based on ccy ticker -----
    if spot == 'EURUSD': # Euro Dollar
        ccy_F = "TFF1E"
        ccy_F_O = 'TFC1E'

    if spot == 'USDJPY': # Dollar Yen
        ccy_F = "TFF1D"
        ccy_F_O = "TFC1D"

    if spot == 'GBPUSD': # Pound Dollar
        ccy_F = "TFF1C"
        ccy_F_O = "TFC1C"

    if spot == 'USDCAD': # Dollar Cad
        ccy_F = "TFF1A"
        ccy_F_O = "TFC1A"

    if spot == 'AUDUSD': # Aussie Dollar
        ccy_F = "TFF1F"
        ccy_F_O = "TFC1F"

    if spot == 'NZDUSD': # Kiwi Dollar
        ccy_F = "TFF1I"
        ccy_F_O = "TFC1I"

    if spot == 'USDCHF': # Dollar Swiss
        ccy_F = "TFF1B"
        ccy_F_O = "TFC1B"

    if spot == 'USD':
        ccy_F = 'TFF2N'
        ccy_F_O = 'TFC2N'

    if spot == 'EURGBP': # Euro Pound
        ccy_F = "TFF3E"
        ccy_F_O = "TFC3E"





    spot_tick = f"{spot} Curncy"


    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")



    # List of corresponding codes 
    #   (LONG, SHORT, SPREAD, NET)
    codes = [
        "AIL", "AIS", "AID", "AIN", 
        "DIL", "DIS", "DID", "DIN", 
        "LFL", "LFS", "LFD", "LFN", 
        "ORL", "ORS", "ORD", "ORN"
    ]




    # Catagory Codes 
    column_names = [
        "AM_Long", "AM_Short", "AM_Spreading", "AM_Net",        # Assent Managers 
        "Dealer_Long", "Dealer_Short", "Dealer_Spreading", "Dealer_Net",        # Dealer Intermediary
        "Lever_Long", "Lever_Short", "Lever_Spreading", "Lever_Net",        # Leveraged Funds
        "Other_Long", "Other_Short", "Other_Spreading", "Other_Net"     # Other reportables
    ]

    

    ticker = [f"{base}{code}" for base in [ccy_F, ccy_F_O] for code in codes] # Generating combo of all tickers (All Futures THEN All Fut&Opt)
    ticker = [f"{code} Index" for code in ticker] # 


    # Pull data using blp.bdh()
    df_data = blp.bdh(
        tickers=ticker,
        start_date=start_date,
        end_date=end_date
    )

    df_data.columns = ['_'.join(col).strip() for col in df_data.columns.values] # Join column headers into 1


    # Re-structuring of column names in CME Data 
    tickers = [f"{base}{code} Index_Last_Price" for base in [ccy_F, ccy_F_O] for code in codes]
    columns_with_names = {
        ticker: f"{f'{spot}_F' if ccy_F in ticker else f'{spot}_F_O'}_{column_names[i % len(column_names)]}"
        for i, ticker in enumerate(tickers)}
    df_data.rename(columns=columns_with_names, inplace=True)

    # Gathering Spot Data
    df_spot = blp.bdh(
        tickers=spot_tick,
        start_date=start_date,
        end_date=end_date
    )

    # Re-structuring of column names in spot data
    df_spot.columns = ['_'.join(col).strip() for col in df_spot.columns.values]
    df_spot.rename(columns={f"{spot} Curncy_Last_Price": spot_tick}, inplace=True)

    # Merging Fut&Opt data with Spot data
    df_data[f'{spot_tick}'] = df_spot[f'{spot_tick}']
    df_data.insert(0, f'{spot_tick}', df_data.pop(f'{spot_tick}'))


    return df_data, df_spot








# CCY List - Asset Managers (Long and Short), Leveraged Funds (Long and Short)
def Traders_Agg_futurePositioning(years):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    ccy_codes = [
        "TFF1E",  # EURUSD
        "TFF1D",  # JPYUSD
        "TFF1C",  # GBPUSD
        "TFF1A",  # CADUSD
        "TFF1F",  # AUDUSD
        "TFF1I",  # NZDUSD
        "TFF1B",  # USDCHF
        "TFF2V",  # BLRUSD
        "TFF1H"  # MXNUSD
    ]


    # List of corresponding codes 
    #   (LONG, SHORT)
    codes = [
        "AIL", "AIS",  # Asset Manager positioning 
        "LFL", "LFS",  # Leveraged Funds Positioning
        ]



    ccy_tickers = [
        'EURUSD',
        'JPYUSD',
        'GBPUSD',
        'CADUSD',
        'AUDUSD',
        'NZDUSD',
        'CHFUSD',
        'BRLUSD',
        'MXNUSD'
    ]


    # Catagory Codes 
    column_names = [
        "AM_Long", "AM_Short",         # Assent Managers 
        "Lever_Long", "Lever_Short",      # Leveraged Funds
    ]


    combined_codes = [f"{ticker}{code}" for ticker, code in product(ccy_codes, codes)]
    ticker = [f"{code} Index" for code in combined_codes]



    df_data = blp.bdh(
            tickers=ticker,
            start_date=start_date,
            end_date=end_date
        )

    df_data.columns = ['_'.join(col).strip() for col in df_data.columns.values] # Join column headers into 1

    # Renaming columns for clarity
    new_column_names = [f"{ccy}_{col}" for ccy in ccy_tickers for col in column_names] 
    df_data.columns = new_column_names

    return df_data





# ----------------------------------------------------------------------------------------------------------
# --------------------- Technical Analysis -----------------------------------------------------------------











# ----------------------------------------------------------------------------------------------------------
# --------------------- Visual Plotting --------------------------------------------------------------------

# ---------------ALL LEGACY -- COMERCIAL VS NON-COMERCIAL 

# Plot long/short over time with spot compare
def plotLongShort_spot(spot, years): 

    df_data, df_spot = Inst_Spec_futurePositioning(spot, years)

    df_data['F_tot_longVshort'] = df_data[f'{spot}_F_tot_Long'] / df_data[f'{spot}_F_tot_Short']


    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot spot on the left y-axis
    ax1.plot(df_spot.index, df_spot[f"{spot} Curncy"], color='blue', label= f"{spot} Curncy")
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f"{spot} Curncy", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a twin y-axis for F_tot_longVshort
    ax2 = ax1.twinx()
    ax2.plot(df_data.index, df_data['F_tot_longVshort'], color='red', label='Long/Short')
    ax2.set_ylabel('long Vs short Ratio', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    # Add a title and legend
    plt.title(f'{spot} Spot and Long/Short Ratio Over Time')
    fig.tight_layout()
    plt.grid(True)

    # Show the plot
    plt.show()

    return


# Plot a (A List of) variables of choice with spot
def plotVariableChoice_spot(spot, years, varChoices):

    df_data, df_spot = Inst_Spec_futurePositioning(spot, years)

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot spot on the left y-axis
    ax1.plot(df_spot.index, df_spot[f"{spot} Curncy"], color='blue', label=f"{spot} Curncy")
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f"{spot} Curncy", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a twin y-axis for the varChoices
    ax2 = ax1.twinx()

    # Loop through varChoices and plot each on the same secondary y-axis
    colors = ['red', 'green', 'orange', 'purple', 'brown']  # Add more colors if needed
    for i, varChoice in enumerate(varChoices):
        ax2.plot(df_data.index, df_data[varChoice], label=varChoice, color=colors[i % len(colors)])

    ax2.set_ylabel('Values', color='black')  # General label for the secondary y-axis
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')

    # Add a title and legend
    plt.title(f'{spot} Spot and {", ".join(varChoices)} Over Time')
    fig.tight_layout()
    plt.grid(True)

    # Show the plot
    plt.show()

    return




# ---------------ALL DISAGGREGATED -- (ASSET MANAGER, DEALER INTERMEDIARY, LEVERAGED FUNDS) 
# Plot Leveraged and Asset Manager 
def plotTraderPositionalBarChart(years):

    


    ccy_tickers = [
        'EURUSD',
        'JPYUSD',
        'GBPUSD',
        'CADUSD',
        'AUDUSD',
        'NZDUSD',
        'CHFUSD',
        'BRLUSD',
        'MXNUSD'
    ]


    df_data = Traders_Agg_futurePositioning(years)




    AM_LongShort = {}
    Lever_LongSHort = {}

    # Loop through each currency in ccy_tickers
    for ccy in ccy_tickers:
        # Calculate the ratio of AM_Long to AM_Short
        AM_LongShort[f"{ccy}_AM_LongShort"] = df_data[f"{ccy}_AM_Long"] / df_data[f"{ccy}_AM_Short"]
        Lever_LongSHort[f"{ccy}_AM_LongShort"] = df_data[f"{ccy}_Lever_Long"] / df_data[f"{ccy}_Lever_Short"]

    # Convert the dictionary into a DataFrame
    df_AM_LongShort = pd.DataFrame(AM_LongShort, index=df_data.index)
    df_Lever_LongShort = pd.DataFrame(Lever_LongSHort, index=df_data.index)





    am_recent_values = df_AM_LongShort.iloc[-1]
    am_second_recent_values = df_AM_LongShort.iloc[-2]

    lever_recent_values = df_Lever_LongShort.iloc[-1]
    lever_second_recent_values = df_Lever_LongShort.iloc[-2]




    x = np.arange(1, len(ccy_tickers) + 1)  # Start at 1 instead of 0
    bar_width = 0.4

    # Create the figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10),  gridspec_kw={"hspace": 0.2})

    # Define colors for the bars
    colors = {
        "recent": "#022a5b",  # Dark blue
        "second_recent": "#61c5dc"  # Light blue
    }

    # Plot for df_AM_LongShort
    ax1.bar(
        x - bar_width / 2, am_recent_values - 1, bar_width,  # Shift bars so 1 is the baseline
        label=f"{am_recent_values.name}", color=colors["recent"]
    )
    ax1.bar(
        x + bar_width / 2, am_second_recent_values - 1, bar_width,  # Shift bars so 1 is the baseline
        label=f"{am_second_recent_values.name}", color=colors["second_recent"]
    )
    ax1.axhline(0, color="black", linewidth=1, linestyle="--")  # Centerline at y=1
    ax1.set_ylabel("Long/Short Positioning", fontsize=12)
    ax1.set_title("Asset Managers", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.spines["top"].set_visible(False)  # Hide top border
    ax1.spines["right"].set_visible(False)  # Hide right border
    ax1.spines["bottom"].set_visible(False)  # Hide bottom border

    # Enable x-axis tick labels for ax1
    ax1.set_xticks(x)  # Adjust tick positions to start at 1
    ax1.set_xticklabels(ccy_tickers, fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", which="both", labelsize=10)  # Force x-labels to appear for ax1

    # Adjust y-tick labels to shift them by 1
    ax1.set_yticks(ax1.get_yticks())  # Keep the original ticks
    ax1.set_yticklabels([f"{y + 1:.1f}" for y in ax1.get_yticks()], fontsize=10)  # Shift labels by 1





    # Plot for df_Lever_LongShort
    ax2.bar(
        x - bar_width / 2, lever_recent_values - 1, bar_width,  # Shift bars so 1 is the baseline
        label=f"{lever_recent_values.name}", color=colors["recent"]
    )
    ax2.bar(
        x + bar_width / 2, lever_second_recent_values - 1, bar_width,  # Shift bars so 1 is the baseline
        label=f"{lever_second_recent_values.name}", color=colors["second_recent"]
    )
    ax2.axhline(0, color="black", linewidth=1, linestyle="--")  # Centerline at y=1
    ax2.set_ylabel("Long/Short Positioning", fontsize=12)
    ax2.set_title("Leveraged Funds", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Adjust y-tick labels to shift them by 1
    ax2.set_yticks(ax2.get_yticks())  # Keep the original ticks
    ax2.set_yticklabels([f"{y + 1:.1f}" for y in ax2.get_yticks()], fontsize=10)  # Shift labels by 1

    # Set the x-axis labels for ax2
    ax2.set_xticks(x)  # Adjust tick positions to start at 1
    ax2.set_xticklabels(ccy_tickers, fontsize=10, fontweight="bold")

    # General adjustments
    plt.suptitle("Long/Short Ratios of Futures Positions - CFTC", fontsize=18, fontweight="bold", y=0.96)

    plt.figtext(
        0.5, 0.01,  # X, Y position (relative to figure dimensions)
        "*Metrics calculated by long contracts / short contracts\n"
        "**Note CFTC data is provided in local ccy format so some ccy conventions flipped above\n"
        "***CFTC data is released every Friday, for previous Tuesday to Tuesday, so is not indicative of exact position today",
        wrap=True, ha="center", fontsize=10, color="gray", fontstyle="italic"
    )



    # Show the chart
    plt.show()

    return




# ---------------------------------------------------------------------------------------------

# years = 5

# spot = 'EURUSD'

# # plotLongShort_spot(spot, years)
years = 1
plotTraderPositionalBarChart(years)