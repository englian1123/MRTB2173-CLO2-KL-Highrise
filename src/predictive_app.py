import streamlit as st
import pandas as pd
import joblib
import datetime
import os

# Set page config
st.set_page_config(page_title="KL High-Rise Price Predictor", layout="wide")

# Get the directory where the script is located
script_dir = os.path.dirname(__file__) 

# Construct the path to the csv

modelv1_file_path = os.path.join(script_dir, '..', 'model', 'model_v1.pkl')
modelv2_file_path = os.path.join(script_dir, '..', 'model', 'model_v2.pkl')

# Construct the path to the log file
log_file_path = os.path.join(script_dir, '..', 'log', 'prediction_logs.csv')

# Load models
@st.cache_resource
def load_models():
    # Model V1: Basic features
    v1 = joblib.load(modelv1_file_path)
    # Model V2: Includes SchemeName with Target Encoding
    v2 = joblib.load(modelv2_file_path)
    return v1, v2

try:
    model_v1, model_v2 = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure the .joblib files are in the same directory.")
    st.stop()

# Initialize session state for tracking prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR INPUTS ---
st.sidebar.header("Property Details")

mukim_options = [
    'Kuala Lumpur Town Centre', 'Mukim Batu', 'Mukim Cheras',
    'Mukim Kuala Lumpur', 'Mukim Petaling', 'Mukim Ampang',
    'Mukim Setapak', 'Mukim Ulu Kelang'
]

mukim = st.sidebar.selectbox("Select Mukim", mukim_options)
tenure = st.sidebar.selectbox("Tenure", ["Freehold", "Leasehold"])
scheme_name = st.sidebar.text_input("Scheme Name (e.g. FERNLEA COURT)", value="FERNLEA COURT")
parcel_area = st.sidebar.number_input("Parcel Area (sq.m)", min_value=1.0, value=100.0)
unit_level = st.sidebar.number_input("Unit Level (0 for Ground)", min_value=-2, value=1)

# Date inputs for market timing
today = datetime.date.today()
year = st.sidebar.number_input("Transaction Year", min_value=2000, max_value=2030, value=today.year)
month = st.sidebar.slider("Transaction Month", 1, 12, today.month)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Price"):
    # Prepare data for Model V1 (Basic)
    input_v1 = pd.DataFrame({
        'Mukim': [mukim],
        'Tenure': [tenure],
        'ParcelArea': [parcel_area],
        'UnitLevel_Cleaned': [unit_level],
        'Year': [year],
        'Month': [month]
    })
    
    # Prepare data for Model V2 (Advanced with SchemeName)
    input_v2 = pd.DataFrame({
        'SchemeName': [scheme_name],
        'Mukim': [mukim],
        'Tenure': [tenure],
        'ParcelArea': [parcel_area],
        'UnitLevel_Cleaned': [unit_level],
        'Year': [year],
        'Month': [month]
    })
    
    # Generate predictions
    pred_v1 = model_v1.predict(input_v1)[0]
    pred_v2 = model_v2.predict(input_v2)[0]
    
    # Store results in session state
    result = {
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Mukim': mukim,
        'SchemeName': scheme_name,
        'Area': parcel_area,
        'Level': unit_level,
        'Prediction_V1': pred_v1,
        'Prediction_V2': pred_v2,
        'Feedback': None,
        'Rating': None
    }
    st.session_state.history.append(result)

# --- MAIN DASHBOARD DISPLAY ---
st.title("🏙️ KL High-Rise Residential Price Predictor")
st.markdown("Compare predictions between the **Basic Model (v1)** and the **Scheme-Aware Model (v2)**.")

if st.session_state.history:
    latest = st.session_state.history[-1]
    
    # Show side-by-side metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model V1 (General)")
        st.metric("Estimated Price", f"RM {latest['Prediction_V1']:,.2f}")
        st.caption("Based on Location, Area, and Level.")
        
    with col2:
        st.subheader("Model V2 (Project Specific)")
        st.metric("Estimated Price", f"RM {latest['Prediction_V2']:,.2f}", 
                  delta=f"{latest['Prediction_V2'] - latest['Prediction_V1']:,.2f}")
        st.caption("Includes Specific Scheme/Building valuation.")

    # Feedback Section
    st.divider()
    st.subheader("User Feedback")
    with st.expander("Is this prediction accurate? Leave feedback"):
        feedback_text = st.text_area("Comments")
        rating = st.select_slider("Rate accuracy (1=Poor, 5=Excellent)", options=[1, 2, 3, 4, 5], value=3)
        
        if st.button("Submit Feedback"):
            # Update the record in session state
            st.session_state.history[-1]['Feedback'] = feedback_text
            st.session_state.history[-1]['Rating'] = rating
            
            # Append to persistent CSV log
            log_df = pd.DataFrame([st.session_state.history[-1]])
            file_exists = os.path.isfile(log_file_path)
            log_df.to_csv(log_file_path, mode='a', index=False, header=not file_exists)
            
            st.success("Thank you! Feedback logged to 'prediction_logs.csv'.")

# --- LOGS TABLE ---
st.divider()
st.subheader("Prediction History & Logs")
if st.session_state.history:
    # Display dataframe of current session history
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.sort_index(ascending=False), use_container_width=True)
    
    # Download button for the persistent log file
    if os.path.exists(log_file_path):
        with open(log_file_path, 'rb') as f:
            st.download_button('📥 Download All-Time Logs (CSV)', f, file_name='prediction_logs.csv')
else:
    st.info("Adjust the parameters in the sidebar and click 'Predict Price' to begin.")