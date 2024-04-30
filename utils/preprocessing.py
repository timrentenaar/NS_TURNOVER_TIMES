import pandas as pd
import numpy as np

def set_types(df):
    df["DRIVER_CHANGE"] = df["DRIVER_CHANGE"].astype(str)
    df['TRAINNUMBER'] = df['TRAINNUMBER'].astype(str)
    df['PREVIOUS_TRAINNUMBER'] = df['PREVIOUS_TRAINNUMBER'].astype(str)
    df["PLAN_DATETIME"] = pd.to_datetime(df['PLAN_DATETIME'])
    df["REALIZED_DATETIME"] = pd.to_datetime(df['REALIZED_DATETIME'])
    df["DEPARTURE_SIGNAL_SHOWS_SAFE"] = pd.to_datetime(df['DEPARTURE_SIGNAL_SHOWS_SAFE'])

    return df

def calc_turnover_end(df):
    # Create dataframe with rows where there is a turnover and previous trains are valid
    df_turnover = df.loc[(df["TURNOVER_INDICATOR"] == 1)]
    valid_trainnumber = set(df['TRAINNUMBER'])  # Valid train numbers
    filtered_df = df_turnover[df_turnover['PREVIOUS_TRAINNUMBER'].isin(valid_trainnumber)]

    # Filter DataFrame to include only arrival records
    arrival_df = df[df['ACTIVITYTYPE'].isin(['A', 'K_A'])]
    plan_latest_arrivals = arrival_df.groupby(['TRAFFIC_DATE', 'TRAINNUMBER', 'STATION'])['PLAN_DATETIME'].max()
    real_latest_arrivals = arrival_df.groupby(['TRAFFIC_DATE', 'TRAINNUMBER', 'STATION'])['REALIZED_DATETIME'].max()

    # Calc planned turnover time per row
    plan_turnover_time = filtered_df.apply(
        lambda row: row['PLAN_DATETIME'] - plan_latest_arrivals.get((row['TRAFFIC_DATE'], row['PREVIOUS_TRAINNUMBER'], row['STATION']), pd.NaT),
        axis=1
    ).dt.total_seconds()

    # Calc realized turnover time per row
    real_turnover_time = filtered_df.apply(
        lambda row: row['REALIZED_DATETIME'] - real_latest_arrivals.get((row['TRAFFIC_DATE'], row['PREVIOUS_TRAINNUMBER'], row['STATION']), pd.NaT),
        axis=1
    ).dt.total_seconds()

    # Set new turnover time columns
    filtered_df["PLAN_TURNOVER_TIME"] = plan_turnover_time
    filtered_df["REALIZED_TURNOVER_TIME"] = real_turnover_time

    return filtered_df

def calc_turnover_middle(df):
    # Arrivals of previous rows
    plan_previous_row_arrival = df['PLAN_DATETIME'].shift(1)
    real_previous_row_arrival = df['REALIZED_DATETIME'].shift(1)

    # Create dataframe with rows where there is a turnover and no previous trains are indicated
    df_turnover = df.loc[(df["TURNOVER_INDICATOR"] == 1)]
    df_nan = df_turnover[df_turnover['PREVIOUS_TRAINNUMBER'] == 'nan']

    # Calculate turnover times
    plan_turnover_time = (df_nan['PLAN_DATETIME'] - plan_previous_row_arrival.loc[df_nan.index]).dt.total_seconds()
    real_turnover_time = (df_nan['REALIZED_DATETIME'] - real_previous_row_arrival.loc[df_nan.index]).dt.total_seconds()
    
    # Set new turnover time columns
    df_nan["PLAN_TURNOVER_TIME"] = plan_turnover_time
    df_nan["REALIZED_TURNOVER_TIME"] = real_turnover_time

    return df_nan

def calc_turnover(df):
    return pd.concat([calc_turnover_middle(df), calc_turnover_end(df)])

def calc_needed_turnover(df):
    df["NEEDED_PLAN_TURNOVER_TIME"] = df["PLAN_TURNOVER_TIME"] + 60 - df["DELAY"]
    df["NEEDED_REALIZED_TURNOVER_TIME"] = df["REALIZED_TURNOVER_TIME"] + 60 - df["DELAY"]

    return df

# split data into big and small delays (e.g. +/- 5 min)
def df_subsets_big_small(df, delay = 300):
  df_big_delay = df[df['DELAY'] > delay]
  df_small_delay = df[df['DELAY'] <= delay]

  return df_big_delay, df_small_delay

# add column max 'MAX_DEPARTURE_TIME", which is max of 'REALIZED_DATETIME' and 'DEPARTURE_SIGNAL_SHOWS_SAFE'
def add_max_departure_time(df):
  df['MAX_DEPARTURE_TIME'] = df[['REALIZED_DATETIME', 'DEPARTURE_SIGNAL_SHOWS_SAFE']].max(axis=1)
  return df

# remove unnecessary rows with not relevant data or small amount of data
def remove_unnecessary_rows(df):
  # remove from ACTIVITY_TYPE == ('R_A, R_V, R'), because few occurences & not relevant
  df = df[~df['ACTIVITYTYPE'].isin(['R_A', 'R_V', 'R'])]

  # remove from TRAINSERIE_DIRECTION == ('NB, VB, VST'), because few occurences & not relevant
  df = df[~df['TRAINSERIE_DIRECTION'].isin(['NB', 'VB', 'VST'])]

  # remove from ROLLINGSTOCK_TYPE == ('MS'), because not relevant
  df = df[~df['ROLLINGSTOCK_TYPE'].isin(['MS'])]

  return df
