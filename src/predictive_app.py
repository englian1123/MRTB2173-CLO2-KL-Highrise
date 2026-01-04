import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import time

# Set page config
st.set_page_config(page_title="KL High-Rise Price Predictor", layout="wide")

# Get the directory where the script is located
script_dir = os.path.dirname(__file__) 

# Construct the paths based on desired structure
csv_file_path = os.path.join(script_dir, '..', 'data', 'KLHighRise.csv')
modelv1_file_path = os.path.join(script_dir, '..', 'model', 'model_v1.pkl')
modelv2_file_path = os.path.join(script_dir, '..', 'model', 'model_v2.pkl')

# Construct the path to the log file
log_file_path = os.path.join(script_dir, '..', 'log', 'prediction_permodel_logs.csv')

# Load data for the cascading dropdowns
@st.cache_data
def get_mukim_scheme_mapping():
    try:
        df = pd.read_csv(csv_file_path)
        mapping = df.groupby('Mukim')['SchemeName'].unique().apply(lambda x: sorted(list(x))).to_dict()
        return mapping
    except Exception as e:
        st.error(f"Error loading 'KLHighRise.csv': {e}")
        return {}

# Load the pre-trained models
@st.cache_resource
def load_models():
    v1 = joblib.load(modelv1_file_path)
    v2 = joblib.load(modelv2_file_path)
    return v1, v2

# Initialize session state for tracking prediction history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pred_ready' not in st.session_state:
    st.session_state.pred_ready = False

# Load mapping and models
mukim_scheme_map = get_mukim_scheme_mapping()
try:
    model_v1, model_v2 = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Property Details")

if mukim_scheme_map:
    mukim_options = sorted(list(mukim_scheme_map.keys()))
    selected_mukim = st.sidebar.selectbox("Select Mukim", mukim_options)
    scheme_options = mukim_scheme_map.get(selected_mukim, [])
    selected_scheme = st.sidebar.selectbox("Select Scheme Name", scheme_options)
else:
    selected_mukim = st.sidebar.selectbox("Select Mukim", ['Kuala Lumpur Town Centre', 'Mukim Batu', 'Mukim Cheras'])
    selected_scheme = st.sidebar.text_input("Scheme Name", value="FERNLEA COURT")

tenure = st.sidebar.selectbox("Tenure", ["Freehold", "Leasehold"])
parcel_area = st.sidebar.number_input("Parcel Area (sq.m)", min_value=1.0, value=100.0)
unit_level = st.sidebar.number_input("Unit Level (0 for Ground)", min_value=-2, value=1)

today = datetime.date.today()
year = st.sidebar.number_input("Transaction Year", min_value=2000, max_value=2030, value=today.year)
month = st.sidebar.slider("Transaction Month", 1, 12, today.month)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Price"):
    # Data for Model V1
    input_v1 = pd.DataFrame({
        'Mukim': [selected_mukim], 'Tenure': [tenure], 'ParcelArea': [parcel_area],
        'UnitLevel_Cleaned': [unit_level], 'Year': [year], 'Month': [month]
    })
    
    # Data for Model V2
    input_v2 = pd.DataFrame({
        'SchemeName': [selected_scheme], 'Mukim': [selected_mukim], 'Tenure': [tenure],
        'ParcelArea': [parcel_area], 'UnitLevel_Cleaned': [unit_level], 'Year': [year], 'Month': [month]
    })
    
    # Generate predictions with Latency measurement
    start_v1 = time.time()
    pred_v1 = model_v1.predict(input_v1)[0]
    latency_v1 = (time.time() - start_v1) * 1000.0

    start_v2 = time.time()
    pred_v2 = model_v2.predict(input_v2)[0]
    latency_v2 = (time.time() - start_v2) * 1000.0
    
    # Store in session state
    result = {
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Mukim': selected_mukim,
        'SchemeName': selected_scheme,
        'Prediction_V1': pred_v1,
        'Latency_V1_ms': latency_v1,
        'Prediction_V2': pred_v2,
        'Latency_V2_ms': latency_v2,
        'Rating_V1': None,
        'Rating_V2': None,
        'Comments': None
    }
    st.session_state.history.append(result)
    st.session_state.pred_ready = True

# --- MAIN DASHBOARD DISPLAY ---
st.title("🏙️ KL High-Rise Residential Price Predictor")

if st.session_state.pred_ready:
    latest = st.session_state.history[-1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model V1 (Baseline)")
        st.metric("Estimated Price", f"RM {latest['Prediction_V1']:,.2f}")
        st.caption(f"Latency: {latest['Latency_V1_ms']:.2f} ms")
        
    with col2:
        st.subheader("Model V2 (Project Specific)")
        st.metric("Estimated Price", f"RM {latest['Prediction_V2']:,.2f}", 
                  delta=f"{latest['Prediction_V2'] - latest['Prediction_V1']:,.2f}")
        st.caption(f"Latency: {latest['Latency_V2_ms']:.2f} ms")

    # --- PER-MODEL FEEDBACK SECTION ---
    st.divider()
    st.subheader("Detailed Model Feedback")
    
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        v1_rating = st.select_slider("Rate Model V1 Accuracy", options=[1, 2, 3, 4, 5], value=3, key="v1_rate")
    with f_col2:
        v2_rating = st.select_slider("Rate Model V2 Accuracy", options=[1, 2, 3, 4, 5], value=3, key="v2_rate")
    
    feedback_comments = st.text_area("Additional Comments (optional)")
    
    if st.button("Submit Feedback"):
        # Update session state with ratings
        st.session_state.history[-1]['Rating_V1'] = v1_rating
        st.session_state.history[-1]['Rating_V2'] = v2_rating
        st.session_state.history[-1]['Comments'] = feedback_comments
        
        # Log to file
        log_df = pd.DataFrame([st.session_state.history[-1]])
        file_exists = os.path.isfile(log_file_path)
        log_df.to_csv(log_file_path, mode='a', index=False, header=not file_exists)
        st.success("Feedback & Latency data logged successfully!")

# --- HISTORY LOGS ---
st.divider()
st.subheader("Prediction & Performance History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.sort_index(ascending=False), use_container_width=True)
    
    if os.path.exists(log_file_path):
        with open(log_file_path, 'rb') as f:
            st.download_button('📥 Download Full Logs', f, file_name='prediction_permodel_logs.csv')
else:
    st.info("Run a prediction to see latency and model performance logs.")