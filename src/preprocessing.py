import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import StandardScaler , MinMaxScaler


# -----------------------------
# Haversine distance
# -----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# -----------------------------
# Datetime extraction
# -----------------------------
def extract_datetime_features(df, datetime_col="pickup_datetime"):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["hour"] = df[datetime_col].dt.hour
    df["weekday"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month
    df["is_weekend"] = (df["weekday"] > 4).astype(int)
    df["is_peak_hour"] = df["hour"].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)

    def time_of_day(hour):
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    df["time_of_day"] = df["hour"].apply(time_of_day)
    return df


# -----------------------------
# Feature interactions
# -----------------------------
def add_feature_interactions(df):
    # Distance and speed
    df["distance_per_passenger"] = df["haversine_distance"] / df["passenger_count"]
    df["speed_per_passenger"] = df["speed_km_h"] / df["passenger_count"]
    df["distance_weekday"] = df["haversine_distance"] * df["weekday"]
    df["distance_hour"] = df["haversine_distance"] * df["hour"]
    df["hour_weekday"] = df["hour"] * df["weekday"]

    # Time-of-day interactions
    df["distance_time_of_day"] = df["haversine_distance"] * df["time_of_day"].astype("category").cat.codes
    df["passenger_count_time_of_day"] = df["passenger_count"] * df["time_of_day"].astype("category").cat.codes

    return df


# -----------------------------
# Outlier handling
# -----------------------------
def handle_outliers(df):
    # Passenger count
    df = df[df['passenger_count'] > 0]

    # Trip duration
    df = df[df["trip_duration"] > 60]
    df = df[df["trip_duration"] < df["trip_duration"].quantile(0.99)]

    # Distance and speed
    df = df[df["haversine_distance"] > 0]
    df = df[df["haversine_distance"] < df["haversine_distance"].quantile(0.99)]
    df = df[df["speed_km_h"] > 1]
    df = df[df["speed_km_h"] < df["speed_km_h"].quantile(0.99)]

    return df


# -----------------------------
# Encoding
# -----------------------------
def encode_categorical(df):
    # Binary encoding
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y": 1, "N": 0})
    df["vendor_id"] = df["vendor_id"].map({1: 0, 2: 1})

    # One-hot encoding
    df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)
    return df


# -----------------------------
# Scaling numeric features
# -----------------------------
def scale_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# -----------------------------
# Main preprocessing function
# -----------------------------
def preprocess(df):
    # Datetime
    df = extract_datetime_features(df)

    # Haversine distance & speed
    df["haversine_distance"] = df.apply(lambda row: haversine_distance(
        row["pickup_latitude"], row["pickup_longitude"],
        row["dropoff_latitude"], row["dropoff_longitude"]
    ), axis=1)
    df["speed_km_h"] = df["haversine_distance"] / (df["trip_duration"] / 3600)

    # Feature interactions
    df = add_feature_interactions(df)

    # Outliers
    df = handle_outliers(df)

    # Encoding
    df = encode_categorical(df)

    # Log-transform target
    df["trip_duration_transformed"] = np.log1p(df["trip_duration"])

    #drop non-numeric and unused data
    df.drop(columns=["pickup_datetime", "trip_duration", "id"], inplace=True, errors="ignore")

    return df
