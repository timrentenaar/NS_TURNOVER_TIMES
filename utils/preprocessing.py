import pandas as pd
import numpy as np
import datetime
import math


# change types in df
def set_types(df):
    df["DRIVER_CHANGE"] = pd.to_numeric(df["DRIVER_CHANGE"], errors="coerce").astype(
        "Int64"
    )
    df["TRAINNUMBER"] = df["TRAINNUMBER"].astype(str)
    df["PREVIOUS_TRAINNUMBER"] = df["PREVIOUS_TRAINNUMBER"].astype(str)
    df["PLAN_DATETIME"] = pd.to_datetime(df["PLAN_DATETIME"])
    df["REALIZED_DATETIME"] = pd.to_datetime(df["REALIZED_DATETIME"])
    df["DEPARTURE_SIGNAL_SHOWS_SAFE"] = pd.to_datetime(
        df["DEPARTURE_SIGNAL_SHOWS_SAFE"]
    )

    return df


# sort df by 'TRAFFIC_DATE', 'TRAINNUMBER' and 'PLAN_DATETIME'
def sort(df):
    df = df.sort_values(
        by=["TRAFFIC_DATE", "TRAINNUMBER", "PLAN_DATETIME"], ignore_index=True
    )

    return df


# add column 'CUM_DISTANCE_M' which is the cumulative distance per train
def cum_distance(df):
    df["CUM_DISTANCE_M"] = df.groupby(["TRAFFIC_DATE", "TRAINNUMBER"])[
        "DISTANCE_M"
    ].cumsum()

    return df


# calculate turnover time at the end of the traject
def calc_turnover_end(df):
    # Create dataframe with rows where there is a turnover and previous trains are valid
    df_turnover = df.loc[(df["TURNOVER_INDICATOR"] == 1)]
    valid_trainnumber = set(df["TRAINNUMBER"])  # Valid train numbers
    filtered_df = df_turnover[
        df_turnover["PREVIOUS_TRAINNUMBER"].isin(valid_trainnumber)
    ]

    # Filter DataFrame to include only arrival records
    arrival_df = df[df["ACTIVITYTYPE"].isin(["A", "K_A"])]
    plan_latest_arrivals = arrival_df.groupby(
        ["TRAFFIC_DATE", "TRAINNUMBER", "STATION"]
    )["PLAN_DATETIME"].max()
    real_latest_arrivals = arrival_df.groupby(
        ["TRAFFIC_DATE", "TRAINNUMBER", "STATION"]
    )["REALIZED_DATETIME"].max()

    # Calc planned turnover time per row
    plan_turnover_time = filtered_df.apply(
        lambda row: row["PLAN_DATETIME"]
        - plan_latest_arrivals.get(
            (row["TRAFFIC_DATE"], row["PREVIOUS_TRAINNUMBER"], row["STATION"]), pd.NaT
        ),
        axis=1,
    ).dt.total_seconds()

    # Calc realized turnover time per row
    real_turnover_time = filtered_df.apply(
        lambda row: row["REALIZED_DATETIME"]
        - real_latest_arrivals.get(
            (row["TRAFFIC_DATE"], row["PREVIOUS_TRAINNUMBER"], row["STATION"]), pd.NaT
        ),
        axis=1,
    ).dt.total_seconds()

    # Set new turnover time columns
    filtered_df["PLAN_TURNOVER_TIME"] = plan_turnover_time
    filtered_df["REALIZED_TURNOVER_TIME"] = real_turnover_time

    # Create a mapping of previous train's cumulative distance
    previous_train_distances = df.groupby(["TRAFFIC_DATE", "TRAINNUMBER", "STATION"])[
        "CUM_DISTANCE_M"
    ].max()

    # Set the cumulative distance column for the filtered DataFrame
    filtered_df["CUM_DISTANCE_M"] = filtered_df.apply(
        lambda row: previous_train_distances.get(
            (row["TRAFFIC_DATE"], row["PREVIOUS_TRAINNUMBER"], row["STATION"]), pd.NA
        ),
        axis=1,
    )

    return filtered_df


# calculate turnover time for stations in the middle of the traject
def calc_turnover_middle(df):
    # Arrivals of previous rows
    plan_previous_row_arrival = df["PLAN_DATETIME"].shift(1)
    real_previous_row_arrival = df["REALIZED_DATETIME"].shift(1)

    # Create dataframe with rows where there is a turnover and no previous trains are indicated
    df_turnover = df.loc[(df["TURNOVER_INDICATOR"] == 1)]
    df_nan = df_turnover[df_turnover["PREVIOUS_TRAINNUMBER"] == "nan"]

    # Calculate turnover times
    plan_turnover_time = (
        df_nan["PLAN_DATETIME"] - plan_previous_row_arrival.loc[df_nan.index]
    ).dt.total_seconds()
    real_turnover_time = (
        df_nan["REALIZED_DATETIME"] - real_previous_row_arrival.loc[df_nan.index]
    ).dt.total_seconds()

    # Set new turnover time columns
    df_nan["PLAN_TURNOVER_TIME"] = plan_turnover_time
    df_nan["REALIZED_TURNOVER_TIME"] = real_turnover_time

    return df_nan


# add turnover time to df
def calc_turnover(df):
    return pd.concat([calc_turnover_middle(df), calc_turnover_end(df)])


# calculate the needed turnover time for each train
def calc_needed_turnover(df):
    df["NEEDED_PLAN_TURNOVER_TIME"] = df["PLAN_TURNOVER_TIME"] + 60 - df["DELAY"]
    df["NEEDED_REALIZED_TURNOVER_TIME"] = (
        df["REALIZED_TURNOVER_TIME"] + 60 - df["DELAY"]
    )

    return df


# split data into big and small delays (e.g. +/- 5 min)
def df_subsets_big_small(df, delay=300):
    df_big_delay = df[df["DELAY"] > delay]
    df_small_delay = df[df["DELAY"] <= delay]

    return df_big_delay, df_small_delay


# add column max 'MAX_DEPARTURE_TIME", which is max of 'REALIZED_DATETIME' and 'DEPARTURE_SIGNAL_SHOWS_SAFE'
def add_max_departure_time(df):
    df["MAX_DEPARTURE_TIME"] = df[
        ["REALIZED_DATETIME", "DEPARTURE_SIGNAL_SHOWS_SAFE"]
    ].max(axis=1)
    return df


# remove unnecessary rows with not relevant data or small amount of data
def remove_unnecessary_rows(df):
    # remove from ACTIVITY_TYPE == ('R_A, R_V, R'), because few occurences & not relevant
    df = df[~df["ACTIVITYTYPE"].isin(["R_A", "R_V", "R"])]

    # remove from TRAINSERIE_DIRECTION == ('NB, VB, VST'), because few occurences & not relevant
    df = df[~df["TRAINSERIE_DIRECTION"].isin(["NB", "VB", "VST"])]

    # remove from ROLLINGSTOCK_TYPE == ('MS'), because not relevant
    df = df[~df["ROLLINGSTOCK_TYPE"].isin(["MS"])]

    return df


# Calulate how long each train had to wait on a green signal, then remove trains that had to wait more than 0 seconds
def calculate_signal_safe_delay(df):
    df["PLAN_SIGNAL_SAFE_DELAY"] = (
        df["DEPARTURE_SIGNAL_SHOWS_SAFE"] - df["PLAN_DATETIME"]
    ).dt.total_seconds()
    df["REALIZED_SIGNAL_SAFE_DELAY"] = (
        df["DEPARTURE_SIGNAL_SHOWS_SAFE"] - df["REALIZED_DATETIME"]
    ).dt.total_seconds()
    df = df.loc[df["PLAN_SIGNAL_SAFE_DELAY"] <= 0]

    return df


# Filter out rows which have extreme values of PLAN_TURNOVER_TIME and DELAY
def filter_outliers(df):
    df = df[(df["PLAN_TURNOVER_TIME"] >= 30) & (df["PLAN_TURNOVER_TIME"] <= 6000)]
    df = df[(df["DELAY"] >= -600) & (df["DELAY"] <= 10000)]

    return df


# WE CAN CHANGE THIS LATER IF WE WANT MORE INFOMRATION ABOUT THE COMBINING/SPLITTING PROCESS
# Transform combine and split into binary columns
def categorise_combine_spilt(df):
    df["COMBINE"] = df["COMBINE"].notna().astype(int)
    df["SPLIT"] = df["SPLIT"].notna().astype(int)

    return df


# add category, if train is travelling in daluren, not-daluren
def determine_daluren(df):
    # daluren is dependant on day and time of day
    temp = df.copy()
    temp["DAY_IN_WEEK"] = temp["PLAN_DATETIME"].dt.dayofweek
    temp["24-TIME"] = pd.to_datetime(
        temp["PLAN_DATETIME"], format="%I%M%p"
    ).dt.strftime("%H:%M")

    # set type of 24-TIME to datetime
    temp["24-TIME"] = pd.to_datetime(temp["24-TIME"])

    # define daluren times
    time_0900 = datetime.time(9, 00, 0)
    time_1600 = datetime.time(16, 00, 0)
    time_1830 = datetime.time(18, 30, 0)
    time_0630 = datetime.time(6, 30, 0)
    time_0400 = datetime.time(4, 00, 0)

    # determine whether time is in daluren or not
    temp["DALUREN"] = temp.apply(
        lambda x: (
            True
            if
            # maandag t/m vrijdag van 9.00 tot 16.00 uur
            (
                x["24-TIME"].time() >= time_0900
                and x["24-TIME"].time() <= time_1600
                and x["DAY_IN_WEEK"] in [0, 1, 2, 3, 4]
            )
            # maandag t/m vrijdag van 18.30 tot 6.30 uur
            or (
                (x["24-TIME"].time() >= time_1830 or x["24-TIME"].time() <= time_0630)
                and x["DAY_IN_WEEK"] in [0, 1, 2, 3, 4]
            )
            # vrijdagavond na 18.30 uur
            or (x["24-TIME"].time() >= time_1830 and x["DAY_IN_WEEK"] == 4)
            # zaterdag en zondag
            or (x["DAY_IN_WEEK"] in [5, 6])
            # maandagochtend voor 4.00 uur
            or (x["24-TIME"].time() < time_0400 and x["DAY_IN_WEEK"] == 0)
            else False
        ),
        axis=1,
    )

    # remove 24-Time and days-in-week columns
    temp.drop(columns=["24-TIME", "DAY_IN_WEEK"], inplace=True)

    return temp


# add columns for day of week and hour of day
def days_and_hours(df):
    df["DAY_OF_WEEK"] = df["PLAN_DATETIME"].dt.dayofweek
    df["HOUR"] = df["PLAN_DATETIME"].dt.hour

    return df


# Encodes variables that are cyclical (such as hours and days) into sin and cos components of a unit circle
def cyclical_encoder(df, column):
    max_val = df[column].max()

    df[column + "_sin"] = df[column].apply(
        lambda x: math.sin(2 * math.pi * x / (max_val + 1))
    )
    df[column + "_cos"] = df[column].apply(
        lambda x: math.cos(2 * math.pi * x / (max_val + 1))
    )

    return df


# add category for difference in turnover time
def add_cat_diff_turnover_time(df):
    working_df = df.copy()
    # calculate difference in needed_plan_turnover_time and needed_realized_turnover_time
    working_df["DIFF_TURNOVER_TIME"] = (
        working_df["PLAN_TURNOVER_TIME"] - working_df["REALIZED_TURNOVER_TIME"]
    )

    # add category variable for DIFF_TURNOVER_TIME, depending on if it is positive, negative or zero
    # 4 categories
    diff_turnover_bins = [-np.inf, -60, 60, 180, np.inf]
    diff_turnover_labels = ["too early", "perfect", "too late", "past 3-minute mark"]
    working_df["DIFF_TURNOVER_TIME_CAT"] = pd.cut(
        working_df["DIFF_TURNOVER_TIME"],
        bins=diff_turnover_bins,
        labels=diff_turnover_labels,
    )

    return working_df


def remove_past_3_min(df):
    return df[df["DIFF_TURNOVER_TIME"] <= 180]
