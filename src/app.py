import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pickle
from gtts import gTTS
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Smart Electricity Dashboard", layout="wide")

# =============================
# TITLE
# =============================
st.markdown("""
<h1 style='text-align: center; color: #00C9A7;'>
⚡ Smart Electricity Dashboard
</h1>
""", unsafe_allow_html=True)

# =============================
# CSS
# =============================
st.markdown("""
<style>
.metric-box {
    background: linear-gradient(135deg,#1f4037,#99f2c8);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# CONFIG
# =============================
DATA_FOLDER = "src/data"
MODEL_PATH = "src/model/model.pkl"
os.makedirs("src/model", exist_ok=True)

# =============================
# LANGUAGE
# =============================
lang = st.sidebar.selectbox("🌍 Language", ["English", "Hindi", "Telugu"])
st.sidebar.markdown("### 🤖 Model Control Panel")

TEXT = {
    "English": "Total units consumed is {} and total cost is {} rupees",
    "Hindi": "कुल खपत {} यूनिट है और लागत {} रुपये है",
    "Telugu": "మొత్తం వినియోగం {} యూనిట్లు మరియు ఖర్చు {} రూపాయలు"
}

# =============================
# VOICE
# =============================
def speak_gtts(text, lang):
    lang_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}[lang]
    tts = gTTS(text=text, lang=lang_code)
    tts.save("voice.mp3")
    audio_file = open("voice.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# =============================
# LOAD DATA
# =============================
# =============================
# LOAD DATA (DOCKER SAFE FIX)
# =============================

datasets = os.listdir(DATA_FOLDER)

# Separate folders and files
dataset_options = []
for d in datasets:
    full_path = os.path.join(DATA_FOLDER, d)
    if os.path.isdir(full_path):
        dataset_options.append(d)

# If no folders → fallback CSV mode
if len(dataset_options) == 0:
    st.warning("⚠️ No dataset folders found. Using CSV mode.")

    csv_files = [f for f in datasets if f.endswith(".csv")]

    selected_file = st.sidebar.selectbox("📂 CSV File", csv_files)
    file_path = os.path.join(DATA_FOLDER, selected_file)

    df = pd.read_csv(file_path)

    # Auto handling
    if "appliance" in df.columns:
        app_df = df
        daily_df = df.copy()
        daily_df["Date"] = pd.date_range(start="2024-01-01", periods=len(df))
        daily_df.rename(columns={"kwh": "Units"}, inplace=True)
    else:
        st.error("❌ CSV format not supported")
        st.stop()

else:
    selected = st.sidebar.selectbox("📂 Dataset", dataset_options)
    folder = os.path.join(DATA_FOLDER, selected)

    app_df = pd.read_csv(os.path.join(folder, "appliance_usage.csv"))
    daily_df = pd.read_csv(os.path.join(folder, "daily_aggregated.csv"))
    
# =============================
# CLEAN + FIX COLUMNS (IMPORTANT)
# =============================

app_df.columns = app_df.columns.str.lower()
daily_df.columns = daily_df.columns.str.lower()

# Fix app_df
if "kwh" in app_df.columns:
    app_df.rename(columns={"kwh": "Units"}, inplace=True)
elif "units" in app_df.columns:
    app_df.rename(columns={"units": "Units"}, inplace=True)

if "appliance" in app_df.columns:
    app_df.rename(columns={"appliance": "Appliance"}, inplace=True)

# Fix daily_df
if "total_kwh" in daily_df.columns:
    daily_df.rename(columns={"total_kwh": "Units"}, inplace=True)
elif "kwh" in daily_df.columns:
    daily_df.rename(columns={"kwh": "Units"}, inplace=True)
elif "units" in daily_df.columns:
    daily_df.rename(columns={"units": "Units"}, inplace=True)

if "date" in daily_df.columns:
    daily_df.rename(columns={"date": "Date"}, inplace=True)

# Convert date
if "Date" in daily_df.columns:
    daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors="coerce")

# FINAL SAFETY CHECK
if "Units" not in daily_df.columns:
    st.error("❌ 'Units' column missing. Check your dataset.")
    st.write("Columns available:", daily_df.columns)
    st.stop()
# =============================
# KPI
# =============================
price = st.sidebar.number_input("₹ per unit", value=10.0)

total_units = daily_df["Units"].sum()
total_cost = total_units * price

# =============================
# MODEL FUNCTIONS
# =============================
def save_model(model, df):
    version = 1
    old = load_model()

    if old and "version" in old:
        version = old["version"] + 1

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "data_size": len(df),
            "timestamp": str(pd.Timestamp.now()),
            "version": version
        }, f)

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)

                # Old format fix
                if isinstance(data, dict):
                    return data
                else:
                    return {
                        "model": data,
                        "data_size": "Old",
                        "timestamp": "Old",
                        "version": 1
                    }
        except:
            return None
    return None

def train(df):
    df = df.copy()
    df["Day"] = np.arange(len(df))

    model = LinearRegression()
    model.fit(df[["Day"]], df["Units"])
    return model

# =============================
# NAVBAR
# =============================
menu = option_menu(
    None,
    ["Dashboard", "Prediction", "Deep Analysis"],
    icons=["house", "graph-up", "search"],
    orientation="horizontal"
)

# =============================
# DASHBOARD
# =============================
if menu == "Dashboard":

    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='metric-box'>⚡ Units<br>{round(total_units,2)}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'>💰 Cost<br>₹ {round(total_cost,2)}</div>", unsafe_allow_html=True)

    app_group = app_df.groupby("Appliance")["Units"].sum().reset_index()

    fig = px.bar(app_group, x="Appliance", y="Units", color="Appliance", title="Appliance Usage")
    st.plotly_chart(fig, width='stretch')

    # Drilldown
    selected_app = st.selectbox("🔍 Select Appliance", app_group["Appliance"])
    st.dataframe(app_df[app_df["Appliance"] == selected_app])

    # INSIGHTS
    st.markdown("### 🧠 Easy Insights")

    top = app_group.sort_values("Units", ascending=False).iloc[0]
    low = app_group.sort_values("Units", ascending=True).iloc[0]

    st.info(f"""
    🔥 Highest Usage: {top['Appliance']} ({round(top['Units'],2)} units)  
    💡 Lowest Usage: {low['Appliance']} ({round(low['Units'],2)} units)  

    👉 Suggestion: Reduce usage of {top['Appliance']} to save electricity.
    """)

# =============================
# PREDICTION
# =============================
elif menu == "Prediction":

    model_data = load_model()

    if model_data:
        model = model_data["model"]
        st.sidebar.success("✅ Model Loaded")
        st.sidebar.write(f"📊 Data: {model_data['data_size']}")
        st.sidebar.write(f"🧠 Version: v{model_data['version']}")
    else:
        model = train(daily_df)
        st.sidebar.warning("⚠️ No model found → trained new")

    # Auto retrain
    if model_data and model_data["data_size"] != len(daily_df):
        st.sidebar.warning("⚠️ New Data → Auto Retrain")
        model = train(daily_df)
        save_model(model, daily_df)

    if st.sidebar.button("💾 Save Model"):
        save_model(model, daily_df)
        st.sidebar.success("Model Saved")

    # Prediction
    future = pd.DataFrame({"Day": np.arange(len(daily_df), len(daily_df)+30)})
    pred = model.predict(future)

    pred_df = pd.DataFrame({"Day": future["Day"], "Predicted Units": pred})
    st.dataframe(pred_df)

    fig = px.line(pred_df, x="Day", y="Predicted Units", title="Future Prediction")
    st.plotly_chart(fig, width='stretch')

    # Accuracy
    daily_df["Day"] = np.arange(len(daily_df))
    score = r2_score(daily_df["Units"], model.predict(daily_df[["Day"]]))

    st.success(f"📊 Model Accuracy (R²): {round(score,3)}")

    # Monthly insight
    total_future = pred.sum()

    st.warning(f"""
    ⚡ Next 30 days usage: {round(total_future,2)} units  
    💰 Estimated cost: ₹ {round(total_future*price,2)}  

    👉 Reduce high-usage appliances to save money
    """)

# =============================
# ANALYSIS
# =============================
elif menu == "Deep Analysis":

    mean = daily_df["Units"].mean()
    std = daily_df["Units"].std()

    daily_df["Anomaly"] = daily_df["Units"].apply(
        lambda x: "Spike" if x > mean + 2*std else "Normal"
    )

    fig = px.scatter(daily_df, x="Date", y="Units", color="Anomaly")
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Insights")
    st.write(f"Average: {round(mean,2)}")
    st.write(f"Max: {daily_df['Units'].max()}")
    st.write(f"Spikes: {sum(daily_df['Anomaly']=='Spike')}")

# =============================
# VOICE
# =============================
st.markdown("### 🎤 Voice Assistant")

if st.button("🔊 Speak"):
    text = TEXT[lang].format(round(total_units), round(total_cost))
    st.success(text)
    speak_gtts(text, lang)

# =============================
# DONE
# =============================
st.success("🚀 FINAL SMART AI DASHBOARD READY")