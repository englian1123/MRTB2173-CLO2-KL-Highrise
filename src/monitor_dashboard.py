import os
import pandas as pd
import streamlit as st
from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring", layout="wide")

st.title("🏙️ KL High-Rise: Model Monitoring Dashboard")

# Static Training Benchmarks
TRAINING_METRICS = {
    "v1_baseline": {"R2": 0.727, "MAE": 278145.56},
    "v2_project_specific": {"R2": 0.879, "MAE": 175333.45}
}

@st.cache_data(ttl=30)
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp")

logs = load_logs()

if logs.empty:
    st.warning("No logs found. Please run the predictor app first.")
    st.stop()

# Sidebar
st.sidebar.header("Training Benchmarks")
for m, vals in TRAINING_METRICS.items():
    st.sidebar.markdown(f"**{m}**")
    st.sidebar.code(f"R²: {vals['R2']}\nMAE: RM{vals['MAE']:,.0f}")

st.sidebar.divider()
selected_model = st.sidebar.selectbox("Filter by Model", ["All"] + list(logs["model_version"].unique()))
filtered = logs if selected_model == "All" else logs[logs["model_version"] == selected_model]

# KPIs
st.subheader("Live Performance KPIs")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Predictions", len(filtered))
kpi2.metric("Avg User Rating", f"{filtered['feedback_score'].mean():.2f} / 5.0")
kpi3.metric("Avg Latency", f"{filtered['latency_ms'].mean():.1f} ms")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Accuracy Comparison", "💬 Recent Feedback", "📄 Raw Logs"])

with tab1:
    st.subheader("Training vs. Live Metrics")
    summary = logs.groupby("model_version").agg({"feedback_score": "mean", "latency_ms": "mean"}).rename(columns={"feedback_score": "Live User Score", "latency_ms": "Latency (ms)"})
    summary['Training R²'] = [TRAINING_METRICS[m]['R2'] for m in summary.index]
    summary['Training MAE'] = [TRAINING_METRICS[m]['MAE'] for m in summary.index]
    st.dataframe(summary.style.highlight_max(subset=['Live User Score', 'Training R²'], color='lightgreen'))
    
    st.subheader("R² Benchmark Chart")
    r2_df = pd.DataFrame({"Model": ["V1", "V2"], "R2": [0.727, 0.879]}).set_index("Model")
    st.bar_chart(r2_df)

with tab2:
    st.subheader("User Comments")
    comments = logs.dropna(subset=["feedback_text"]).sort_values("timestamp", ascending=False)
    if comments.empty:
        st.info("No text feedback yet.")
    else:
        for _, row in comments.head(10).iterrows():
            st.write(f"**{row['timestamp'].strftime('%Y-%m-%d %H:%M')}** | {row['model_version']} | Rating: {row['feedback_score']}⭐")
            st.info(row["feedback_text"])

with tab3:
    st.dataframe(filtered, use_container_width=True)