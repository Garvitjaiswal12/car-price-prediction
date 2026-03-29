import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- Load Model, Scaler & Columns ---------------- #
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")   # ✅ VERY IMPORTANT

# ---------------- Page Config ---------------- #
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------- Custom Dark Theme CSS ---------------- #
st.markdown("""
<style>
    body { background-color: #0E1117; color: white; }
    .main { background-color: #0E1117; }

    .title-box {
        background: linear-gradient(90deg, #141E30, #243B55);
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        color: white;
        font-size: 34px;
        font-weight: bold;
        box-shadow: 0px 0px 25px rgba(36, 59, 85, 0.6);
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #D1D5DB;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    .card {
        background-color: #161B22;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 0px 18px rgba(0,0,0,0.55);
        margin-top: 15px;
    }

    .result-card {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        padding: 20px;
        border-radius: 18px;
        color: black;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: #9CA3AF;
    }

    .stButton button {
        width: 100%;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        border: none;
    }

    .stButton button:hover {
        background: linear-gradient(90deg, #dd2476, #ff512f);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ---------------- #
st.markdown('<div class="title-box">Welcome to Car Price Prediction model</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict Car Price instantly using Machine Learning 🚀</div>', unsafe_allow_html=True)

# ---------------- Sidebar ---------------- #
st.sidebar.title("⚙️ Project Info")
st.sidebar.write("**Model:** Ridge Regression")
st.sidebar.write("**Scaler:** StandardScaler")
st.sidebar.write("**Total Features Used in Training:** 28")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Created by Garvit Jaiswal**")

# ---------------- UI Input Features ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Enter Car Details (Important Features)")

col1, col2, col3 = st.columns(3)

with col1:
    symboling = st.number_input("symboling", min_value=-3, max_value=3, value=0, step=1)
    wheelbase = st.number_input("wheelbase", value=95.0)
    carlength = st.number_input("carlength", value=170.0)
    carwidth = st.number_input("carwidth", value=65.0)
    curbweight = st.number_input("curbweight", value=2000.0)

with col2:
    enginesize = st.number_input("enginesize", value=120.0)
    horsepower = st.number_input("horsepower", value=100.0)
    peakrpm = st.number_input("peakrpm", value=5000.0)
    citympg = st.number_input("citympg", value=20.0)

with col3:
    carbody = st.selectbox("carbody", ["convertible", "hardtop", "hatchback", "sedan", "wagon"])
    drivewheel = st.selectbox("drivewheel", ["fwd", "rwd", "4wd"])
    enginelocation = st.selectbox("enginelocation", ["front", "rear"])
    enginetype = st.selectbox("enginetype", ["dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"])
    cylindernumber = st.selectbox("cylindernumber", ["two", "three", "four", "five", "six", "eight", "twelve"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction Section ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)

if st.button("🚗 Predict Car Price"):

    # ✅ STEP 1: Create raw input (NO manual encoding)
    input_dict = {
        'symboling': symboling,
        'wheelbase': wheelbase,
        'carlength': carlength,
        'carwidth': carwidth,
        'curbweight': curbweight,
        'enginesize': enginesize,
        'horsepower': horsepower,
        'peakrpm': peakrpm,
        'citympg': citympg,
        'carbody': carbody,
        'drivewheel': drivewheel,
        'enginelocation': enginelocation,
        'enginetype': enginetype,
        'cylindernumber': cylindernumber
    }

    # ✅ STEP 2: Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # ✅ STEP 3: Apply SAME encoding as training
    input_df = pd.get_dummies(input_df)

    # ✅ STEP 4: Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # ✅ STEP 5: Scale
    scaled_data = scaler.transform(input_df)

    # ✅ STEP 6: Predict
    prediction = model.predict(scaled_data)[0]

    st.markdown(f"""
        <div class="result-card">
            ✅ Predicted Car Price: ₹{prediction:,.2f}
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ---------------- #
st.markdown("""
<div class="footer">
    🚀 Car Price Prediction Model | Dark UI Streamlit App <br>
    👨‍💻 Created by <b>Garvit Jaiswal</b>
</div>
""", unsafe_allow_html=True)