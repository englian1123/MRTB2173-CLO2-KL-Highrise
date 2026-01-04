import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import time

# --- NEW IMPORT ---
from log_utils import log_prediction_feedback

# Set page config
st.set_page_config(page_title="KL High-Rise Price Predictor", layout="wide")

# Paths based on your structure
script_dir = os.path.dirname(__file__) 
csv_file_path = os.path.join(script_dir, '..', 'data', 'KLHighRise.csv')
modelv1_file_path = os.path.join(script_dir, '..', 'model', 'model_v1.pkl')
modelv2_file_path = os.path.join(script_dir, '..', 'model', 'model_v2.pkl')

@st.cache_data
def get_mukim_scheme_mapping():
    try:
        df = pd.read_csv(csv_file_path)
        mapping = df.groupby('Mukim')['SchemeName'].unique().apply(lambda x: sorted(list(x))).to_dict()
        return mapping
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}

@st.cache_resource
def load_models():
    v1 = joblib.load(modelv1_file_path)
    v2 = joblib.load(modelv2_file_path)
    return v1, v2

if 'history' not in st.session_state:
    st.session_state.history = []
if 'pred_ready' not in st.session_state:
    st.session_state.pred_ready = False

mukim_scheme_map = get_mukim_scheme_mapping()
try:
    model_v1, model_v2 = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Property Details")
if mukim_scheme_map:
    selected_mukim = st.sidebar.selectbox("Select Mukim", sorted(list(mukim_scheme_map.keys())))
    selected_scheme = st.sidebar.selectbox("Select Scheme Name", mukim_scheme_map.get(selected_mukim, []))
else:
    selected_mukim = st.sidebar.selectbox("Select Mukim", ['Kuala Lumpur'])
    selected_scheme = st.sidebar.text_input("Scheme Name")

tenure = st.sidebar.selectbox("Tenure", ["Freehold", "Leasehold"])
area = st.sidebar.number_input("Parcel Area (sq.m)", min_value=1.0, value=100.0)
level = st.sidebar.number_input("Unit Level", min_value=-2, value=1)
year = st.sidebar.number_input("Year", min_value=2000, value=datetime.date.today().year)
month = st.sidebar.slider("Month", 1, 12, datetime.date.today().month)

# --- PREDICTION ---
if st.sidebar.button("Predict Price"):
    input_v1 = pd.DataFrame({'Mukim': [selected_mukim], 'Tenure': [tenure], 'ParcelArea': [area], 'UnitLevel_Cleaned': [level], 'Year': [year], 'Month': [month]})
    input_v2 = pd.DataFrame({'SchemeName': [selected_scheme], 'Mukim': [selected_mukim], 'Tenure': [tenure], 'ParcelArea': [area], 'UnitLevel_Cleaned': [level], 'Year': [year], 'Month': [month]})
    
    start_v1 = time.time()
    pred_v1 = model_v1.predict(input_v1)[0]
    lat_v1 = (time.time() - start_v1) * 1000.0

    start_v2 = time.time()
    pred_v2 = model_v2.predict(input_v2)[0]
    lat_v2 = (time.time() - start_v2) * 1000.0
    
    st.session_state.history.append({
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Mukim': selected_mukim, 'SchemeName': selected_scheme,
        'Prediction_V1': pred_v1, 'Latency_V1_ms': lat_v1,
        'Prediction_V2': pred_v2, 'Latency_V2_ms': lat_v2,
        'Rating_V1': None, 'Rating_V2': None, 'Comments': None
    })
    st.session_state.pred_ready = True

# --- UI ---
st.title("🏙️ KL High-Rise Price Predictor")
if st.session_state.pred_ready:
    latest = st.session_state.history[-1]
    c1, c2 = st.columns(2)
    c1.metric("Model V1 (Baseline)", f"RM {latest['Prediction_V1']:,.2f}", help=f"Latency: {latest['Latency_V1_ms']:.2f}ms")
    c2.metric("Model V2 (Specific)", f"RM {latest['Prediction_V2']:,.2f}", help=f"Latency: {latest['Latency_V2_ms']:.2f}ms")

    st.divider()
    st.subheader("Model Feedback")
    f1, f2 = st.columns(2)
    r1 = f1.select_slider("Rate V1 Accuracy", options=[1, 2, 3, 4, 5], value=3, key="v1_r")
    r2 = f2.select_slider("Rate V2 Accuracy", options=[1, 2, 3, 4, 5], value=3, key="v2_r")
    comm = st.text_area("Comments")
    
    if st.button("Submit Feedback"):
        # Update session history
        st.session_state.history[-1].update({'Rating_V1': r1, 'Rating_V2': r2, 'Comments': comm})
        
        # --- REFACTORED CALL ---
        log_prediction_feedback(st.session_state.history[-1])
        
        st.success("Feedback & Latency metrics logged to prediction_logs.csv!")

# --- DISPLAY LOGS ---
if st.session_state.history:
    st.divider()
    st.subheader("Session History")
    st.dataframe(pd.DataFrame(st.session_state.history).sort_index(ascending=False), use_container_width=True)