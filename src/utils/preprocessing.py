import pandas as pd

def clean_daily_data(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip().str.lower()

    # FIX YOUR DATASET
    df.rename(columns={
        "date": "Date",
        "total_kwh": "Units"
    }, inplace=True)

    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
    df.dropna(subset=["Units"], inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def clean_appliance_data(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip().str.lower()

    df.rename(columns={
        "appliance": "Appliance",
        "kwh": "Units"
    }, inplace=True)

    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
    df.dropna(subset=["Units"], inplace=True)

    return df