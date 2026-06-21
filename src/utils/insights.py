import pandas as pd


def generate_insights(df):
    insights = []

    mean = df["Units"].mean()
    std = df["Units"].std()

    if std < 5:
        insights.append("Usage is stable")
    else:
        insights.append("Usage varies significantly")

    if df["Units"].max() > mean * 1.5:
        insights.append("High usage spikes detected")
    else:
        insights.append("No major spikes detected")

    return insights


def detect_anomalies(df):
    df = df.copy()

    mean = df["Units"].mean()
    std = df["Units"].std()

    df["Anomaly"] = df["Units"].apply(
        lambda x: "Spike" if x > mean + 2*std else ("Drop" if x < mean - 2*std else "Normal")
    )

    return df