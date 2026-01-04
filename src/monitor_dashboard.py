import os
import pandas as pd
import streamlit as st
from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring & Feedback", layout="wide")

st.title("🏙️ KL High-Rise: Model Monitoring Dashboard")

@st.cache_data(ttl=60) # Refresh cache every minute
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp")

logs = load_logs()

if logs.empty:
    st.warning("No monitoring logs found yet. Run the prediction app and submit feedback first.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
models = ["All"] + sorted(logs["model_version"].unique().tolist())
selected_model = st.sidebar.selectbox("Model version", models)

filtered = logs if selected_model == "All" else logs[logs["model_version"] == selected_model]

# --- KEY METRICS ---
st.subheader("Performance Overviews")
col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", len(filtered))

# Avg Feedback
avg_fb = filtered["feedback_score"].mean()
col2.metric("Avg User Rating", f"{avg_fb:.2f} / 5.0" if not pd.isna(avg_fb) else "N/A")

# Avg Latency
avg_lat = filtered["latency_ms"].mean()
col3.metric("Avg Latency", f"{avg_lat:.1f} ms" if not pd.isna(avg_lat) else "N/A")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "💬 Feedback Analysis", "📄 Raw Logs"])

with tab1:
    st.subheader("V1 vs V2 Comparison")
    summary = logs.groupby("model_version").agg({
        "feedback_score": "mean",
        "latency_ms": "mean",
        "prediction": "count"
    }).rename(columns={
        "feedback_score": "Avg Score",
        "latency_ms": "Avg Latency (ms)",
        "prediction": "Total Calls"
    })
    st.table(summary.style.highlight_max(subset=['Avg Score'], color='lightgreen'))

    # Latency Trend
    st.subheader("Latency Trend over Time")
    st.line_chart(logs.pivot(index='timestamp', columns='model_version', values='latency_ms'))

with tab2:
    st.subheader("User Feedback Scores")
    fb_dist = logs.groupby("model_version")["feedback_score"].mean()
    st.bar_chart(fb_dist)

    st.subheader("Recent Qualitative Feedback")
    comments = logs.dropna(subset=["feedback_text"])
    comments = comments[comments["feedback_text"].str.strip() != ""]
    comments = comments.sort_values("timestamp", ascending=False).head(10)

    if comments.empty:
        st.info("No text comments recorded yet.")
    else:
        for _, row in comments.iterrows():
            with st.container():
                st.write(f"**{row['timestamp'].strftime('%Y-%m-%d %H:%M')}** | {row['model_version']}")
                st.caption(f"Rating: {row['feedback_score']} ⭐")
                st.info(row["feedback_text"])

with tab3:
    st.subheader("Full Prediction Logbook")
    st.dataframe(filtered, use_container_width=True)