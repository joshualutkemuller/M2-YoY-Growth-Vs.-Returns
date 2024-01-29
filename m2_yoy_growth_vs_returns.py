import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import pandas as pd
import pandas_datareader.data as web
from fredapi import Fred
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

##### User Constraints ####
ticker_names = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones Industrial Average',
    '^RUT': 'Russell 2000',
    '^NDX':"Nasdaq 100",
    '^RUA':'Russell 3000'}

# List of tickers you want to calculate HPR for
tickers = ['^GSPC','^RUT','^DJI','^RUA','^NDX']

# API Key from FRED
api_key = 'your_api_key'  # Your FRED API key

# Set annualization or not
annualize=True


def create_group_hpr_chart_horizontal(df, tickers, save_path_directory):
    # Drop rows where HPR is null for all investments
    df = df.dropna(subset=[f'{ticker}_HPR' for ticker in tickers], how='all')
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Number of groups
    n_groups = len(df)
    index = np.arange(n_groups)
    bar_height = 0.1  # Height of the bars
    opacity = 0.8  # Opacity of the bars
    
    # Initialize a variable to store the annotation text for average and median HPRs
    avg_median_annotation_text = ""

    # Initialize a list to keep track of legend handles
    legend_handles = []

    # Track if an annotation has been added for a group to prevent duplicates
    annotation_added_for_group = set()

    # Plot HPR for each ticker
    for i, ticker in enumerate(tickers):
        hpr_column = f'{ticker}_HPR'
        hpr_values = df[hpr_column].fillna(0)  # Fill NaN with 0 for plotting
        bars = ax.barh(index + i * bar_height, hpr_values, bar_height, alpha=opacity, label=ticker)
        
        # Add the first bar of each ticker to the legend_handles list
        legend_handles.append(bars[0])

        # Add value labels on each bar
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2%}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

        # Calculate average and median for the current ticker
        avg_hpr = np.nanmean(hpr_values)
        median_hpr = np.nanmedian(hpr_values)
        
        # Append the calculated average and median to the annotation text
        avg_median_annotation_text += f"{ticker} Avg: {avg_hpr:.2%}, Med: {median_hpr:.2%}\n"

    # Place the average and median HPR annotation in the top left corner
    ax.text(0, 1, avg_median_annotation_text.strip(),
            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    # Dynamically adjust annotation positions to ensure they stay within the chart
    for idx, row in df.iterrows():
        # Calculate the vertical position for the annotation
        # Ensure it stays within the bounds of the plot
        if idx not in annotation_added_for_group:
            vertical_position = max(min(idx + (len(tickers) - 1) * bar_height / 2, n_groups - 1), 0)
            
            annotation_text = f"Consecutive Months: {row['Consecutive_Months_of_Declining_YoY_Growth']}"
            # f"Months: {row['Consecutive_Months_of_Declining_YoY_Growth']}, Start YoY: {row['Start_YoY_Growth']/100:.2%}, End YoY: {row['End_YoY_Growth']/100:.2%}"
            ax.text(0, vertical_position, annotation_text,
                    ha='left', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))
            annotation_added_for_group.add(idx)

    # Adjust figure margins to provide more space for annotations if necessary
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    # Set y-axis labels to concatenated Start and End dates
    y_labels = [f"{row['Start_Date'].strftime('%Y-%m-%d')}\n{row['End_Date'].strftime('%Y-%m-%d')}" for _, row in df.iterrows()]
    ax.set_yticks(index + bar_height / 2 * (len(tickers) - 1))
    ax.set_yticklabels(y_labels)

    # Format the x-axis as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Final plot adjustments
    ax.set_ylabel('Consecutive Months of Declines (Periods)')

    if annualize:
        ax.set_xlabel('Holding Period Return (Annualized)')
        ax.set_title('HPR for Each Investment During Periods with Consecutive Months of Declining M2 YoY Growth (%) (Annualized)')
    else:
        ax.set_xlabel('Holding Period Return (HPR)')
        ax.set_title('HPR for Each Investment During Periods with Consecutive Months of Declining M2 YoY Growth (%)')
    
    # Create legend with ticker names
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [ticker_names.get(label, label) for label in labels])


    plt.tight_layout()
    print(save_path_directory)
    plt.savefig(os.path.join(save_path_directory,'Periods of Declining M2 YoY Growth and Holding Period Returns.png'))

def fetch_historical_prices(ticker, start_date, end_date):
    """Fetch historical closing prices for a ticker from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_hpr(row, ticker,annualize):
    """Calculate the Holding Period Return for a given row and ticker."""
    prices = fetch_historical_prices(ticker, row['Start_Date'].strftime('%Y-%m-%d'), row['End_Date'].strftime('%Y-%m-%d'))
    if annualize:
        if not prices.empty:
            # Calculate the raw Holding Period Return
            hpr = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

            # Calculate the number of days in the holding period
            holding_period_days = (row['End_Date'] - row['Start_Date']).days

            # Avoid division by zero for very short holding periods
            if holding_period_days > 0:
                # Calculate the annualized return
                annualized_return = (1 + hpr) ** (252 / holding_period_days) - 1
                return annualized_return
            else:
                return hpr  # Return the raw HPR if the holding period is less than a day
        else:
            return None


    else:
        if not prices.empty:
            return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        else:
            return None

def add_hpr_columns(df, tickers,annualize=False):
    """Add new columns to df with the HPR for each ticker."""
    for ticker in tickers:
        # Column name based on the ticker symbol
        column_name = f"{ticker}_HPR"
        # Calculate HPR for each row and ticker, then add it as a new column
        df[column_name] = df.apply(calculate_hpr, axis=1, ticker=ticker,annualize=annualize)
    return df

class FredDataFetcher:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.script_path = script_path = os.path.abspath(__file__)
        self.script_directory = os.path.dirname(script_path)

    def fetch_m2_data(self, series_id='M2SL'):
        """Fetch M2 Money Supply data using fredapi."""
        data = self.fred.get_series(series_id)
        df = pd.DataFrame(data, columns=['value'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df

    def calculate_yoy_change(self, df):
        """Calculate Year-over-Year change for the dataset."""
        df['YoY Change'] = df['value'].pct_change(periods=12) * 100  # Adjust periods based on data frequency
        return df

    def save_data(self, df, filename='M2_YoY_Data.csv'):
        """Save the dataframe to a specified directory."""
        directory = os.path.join(self.script_directory,'M2 YoY Data')
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
        return directory

    def run(self):
        m2_data = self.fetch_m2_data()
        m2_yoy_data = self.calculate_yoy_change(m2_data)
        save_path_directory = self.save_data(m2_yoy_data)
        return m2_yoy_data, save_path_directory


def find_top_negative_sequences(df, column='YoY Change'):
    df['Group'] = 0  # Initialize group identifier column
    current_group = 0
    in_negative_sequence = False

    # Iterate through the DataFrame to identify and label groups of consecutive negatives
    for i in range(len(df)):
        if df.iloc[i][column] < 0:  # Check for negative YoY Change
            if not in_negative_sequence:  # Start of a new negative sequence
                current_group += 1
                in_negative_sequence = True
            df.at[df.index[i], 'Group'] = current_group
        else:
            in_negative_sequence = False

    # Filter to include only negative sequences and exclude standalone negatives
    negative_sequences = df[df['Group'] > 0]
    group_counts = negative_sequences['Group'].value_counts()
    valid_groups = group_counts[group_counts > 1].index  # Groups with more than one member
    negative_sequences = negative_sequences[negative_sequences['Group'].isin(valid_groups)]

    # Aggregate information for each group
    group_info = negative_sequences.groupby('Group').agg(
        Count=('Group', 'size'),
        Start_Date=('Group', lambda x: x.index.min()),
        End_Date=('Group', lambda x: x.index.max()),
        First_YoY_Value=(column, 'first'),
        Last_YoY_Value=(column, 'last')
    )

    # Sort by the count of consecutive negatives and select the top 10
    top_groups = group_info.sort_values(by='Count', ascending=False).head(10)

    # Clean up the DataFrame by removing the 'Group' column
    df.drop(columns=['Group'], inplace=True)

    return top_groups.reset_index(drop=True)
def find_declining_sequences(df, column='YoY Change',top_groups = 10):
    df['Decline_Group'] = 0  # Initialize decline group identifier
    current_group = 0
    previous_value = df[column].iloc[0]

    # Identify declining sequences
    for i in range(1, len(df)):
        current_value = df[column].iloc[i]
        if current_value < previous_value:
            if current_group == 0 or df['Decline_Group'].iloc[i-1] != current_group:
                current_group += 1  # Start a new group
            df['Decline_Group'].iloc[i] = current_group
        previous_value = current_value

    # Filter to include only declining sequences
    declining_sequences = df[df['Decline_Group'] > 0]

    group_info = declining_sequences.groupby('Decline_Group').agg(
        Consecutive_Months_of_Declining_YoY_Growth=('Decline_Group', 'size'),
        Start_Date=('Decline_Group', lambda x: x.index.min()),  # Use the min date of the group
        End_Date=('Decline_Group', lambda x: x.index.max()),  # Use the max date of the group
        Start_YoY_Growth=(column, 'first'),
        End_YoY_Growth=(column, 'last')
    )
    # Sort by the length of the decline and select the top groups
    top_groups = group_info.sort_values(by='Consecutive_Months_of_Declining_YoY_Growth', ascending=False).head(top_groups)

    # Clean up the DataFrame by removing the 'Decline_Group' column
    df.drop(columns=['Decline_Group'], inplace=True)


    return top_groups.reset_index(drop=True)


if __name__ == "__main__":

    fred_data_fetcher = FredDataFetcher(api_key)
    m2_yoy_data, save_path_directory = fred_data_fetcher.run()

    # Assuming 'df' is your DataFrame with M2 Money Supply data and YoY changes calculated
    #top_negative_groups = find_top_negative_sequences(m2_yoy_data)
    top_declining_groups_df = find_declining_sequences(m2_yoy_data,top_groups=15)

    for col in [col for col in top_declining_groups_df.columns if 'Date' in col]:
        top_declining_groups_df[col] = pd.to_datetime(top_declining_groups_df[col])

    # Assuming group_info is your DataFrame with 'Start_Date' and 'End_Date'
    group_info_with_hrp = add_hpr_columns(top_declining_groups_df, tickers,)

    # Call Charting Functions
    create_group_hpr_chart_horizontal(group_info_with_hrp,tickers,save_path_directory)



