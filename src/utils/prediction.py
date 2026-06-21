import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression

MODEL_PATH = "src/models/model.pkl"


def train_model(df):
    df = df.copy()
    df["Day"] = range(len(df))

    X = df[["Day"]]
    y = df["Units"]

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs("src/models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except:
            return None
    return None


def predict_future(model, df, days=7, price=10):
    future_days = np.arange(len(df), len(df)+days).reshape(-1,1)

    pred_units = model.predict(future_days)
    pred_cost = pred_units * price

    return pred_units, pred_cost