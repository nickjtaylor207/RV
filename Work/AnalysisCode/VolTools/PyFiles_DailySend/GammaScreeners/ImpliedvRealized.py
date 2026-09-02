import pdblp
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


import matplotlib.pyplot as plt
from pandas.plotting import table
from PIL import Image












def gammaScreenerPresent(ccy_pairs): 
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    data_combined = []
    for ccy in ccy_pairs:
        spot_data = con.ref(f"{ccy} Curncy", ["PX_LAST"])
        spot_price = spot_data["value"].iloc[0] if not spot_data.empty else None
        iv_tickers = [
            f"{ccy}V1W Curncy", 
            f"{ccy}V2W Curncy", 
            f"{ccy}V1M Curncy"]
        # Get implied volatility data
        iv_data = con.ref(iv_tickers, ["PX_LAST"])
        iv_1w = iv_data["value"].iloc[0] if not iv_data.empty else None
        iv_2w = iv_data["value"].iloc[1] if not iv_data.empty else None
        iv_1m = iv_data["value"].iloc[2] if not iv_data.empty else None

        # Define tickers for realized volatilities (1W, 2W, 1M)
        rv_tickers = [
            f"{ccy}H1W Curncy",  # 1W realized vol
            f"{ccy}H2W Curncy",  # 2W realized vol
            f"{ccy}H1M Curncy",  # 1M realized vol
        ]
        # Get realized volatility data
        rv_data = con.ref(rv_tickers, ["PX_LAST"])
        rv_1w = rv_data["value"].iloc[0] if not rv_data.empty else None
        rv_2w = rv_data["value"].iloc[1] if not rv_data.empty else None
        rv_1m = rv_data["value"].iloc[2] if not rv_data.empty else None

        # Append combined data for this currency pair to the list
        data_combined.append([
            ccy, spot_price, iv_1w, iv_2w, iv_1m, rv_1w, rv_2w, rv_1m
        ])




    # Create a DataFrame from the combined data
    df_combined = pd.DataFrame(
        data_combined,
        columns=["Currency Pair", "Spot Rate", "1W_IV", "2W_IV", "1M_IV", "1W_RV", "2W_RV", "1M_RV"]
    )


    df_diff = pd.DataFrame({
        "Currency Pair": df_combined["Currency Pair"],
        "1W Realized": df_combined["1W_RV"],
        "1W Implied": df_combined["1W_IV"],
        "1W R - I": df_combined["1W_RV"] - df_combined["1W_IV"],
        "2W Realized": df_combined["2W_RV"],
        "2W Implied": df_combined["2W_IV"],
        "2W R - I": df_combined["2W_RV"] - df_combined["2W_IV"],
        "1M Realized": df_combined["1M_RV"],
        "1M Implied": df_combined["1M_IV"],
        "1M R - I": df_combined["1M_RV"] - df_combined["1M_IV"] 
    })





    # Set the Currency Pair as index for the heatmap
    df_diff.set_index('Currency Pair', inplace=True)

    df_diff_filtered = df_diff[["1W R - I", "2W R - I", "1M R - I"]]




    # Create a MultiIndex for the columns
    columns = pd.MultiIndex.from_tuples([
        ("1W", "Realized"), ("1W", "Implied"), ("1W", "R - I"),
        ("2W", "Realized"), ("2W", "Implied"), ("2W", "R - I"),
        ("1M", "Realized"), ("1M", "Implied"), ("1M", "R - I")
    ])

    # Reassign the columns with the MultiIndex
    df_diff.columns = columns





    # Normalize the color scale around zero
    norm = mcolors.TwoSlopeNorm(vmin=-max(abs(df_diff_filtered.min().min()), abs(df_diff_filtered.max().max())),
                                vcenter=0,
                                vmax=max(abs(df_diff_filtered.min().min()), abs(df_diff_filtered.max().max())))

    # Create a heatmap using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # Create a masked array for coloring only specific columns
    color_mask = np.zeros_like(df_diff, dtype=bool)
    for col in [("1W", "R - I"), ("2W", "R - I"), ("1M", "R - I")]:
        color_mask[:, df_diff.columns.get_loc(col)] = True

    # Separate data into color_data (for coloring) and annot_data (for annotations)
    color_data = df_diff.where(color_mask, np.nan)
    annot_data = df_diff.values

    # Plot the heatmap
    cmap = plt.cm.RdYlGn
    for i, row in enumerate(df_diff.index):
        for j, col in enumerate(df_diff.columns):
            value = annot_data[i, j]
            color_value = color_data.iloc[i, j]
            if not np.isnan(color_value):
                color = cmap(norm(color_value))  # Apply the TwoSlopeNorm normalization
            else:
                color = "white"

            # Draw colored rectangles
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(rect)

            # Add text annotations
            ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=9)

    # Set axis limits
    ax.set_xlim(0, len(df_diff.columns))
    ax.set_ylim(0, len(df_diff.index))

    # Adjust ticks for grouped headers
    ax.set_xticks(np.arange(len(df_diff.columns)) + 0.5)
    ax.set_yticks(np.arange(len(df_diff.index)) + 0.5)
    ax.set_xticklabels([col[1] for col in df_diff.columns], fontsize=10, rotation=0)
    ax.set_yticklabels(df_diff.index, fontsize=10)
    ax.xaxis.tick_top()

    # Add grouped headers
    for idx, label in enumerate(["1W", "2W", "1M"]):
        start = idx * 3  # Each group has 3 sub-columns
        end = start + 3
        ax.text((start + end - 1) / 2, -0.8, label, ha="center", va="center", fontsize=12, weight="bold")

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(color_data.values)
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("Volatility Difference", fontsize=10)



    # Add title and labels
    fig.suptitle(
        "DRAX High Frequency Gamma Monitor (30 Min HV Intervals)", 
        fontsize=20, 
        fontstyle="italic"
    )
    plt.ylabel("Currency Pair", fontsize=12)



    # Invert y-axis and adjust layout
    plt.gca().invert_yaxis()
    plt.show()

    return





# currency_pairs = [
#     'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK',
#     'USDMXN', 'USDBRL', 'USDCNH', 'EURGBP', 'EURCHF'
# ]



# gammaScreenerPresent(currency_pairs)
