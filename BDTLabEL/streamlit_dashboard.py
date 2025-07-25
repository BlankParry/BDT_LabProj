"""
Streamlit Energy Monitoring Dashboard for German Smart Meter Data
Modern, dynamic, and interactive dashboard with real-time trend visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime, timedelta
import os

# --- CONFIG ---
DATA_PATH = "data/processed/processed_data.parquet"
MONITORING_STATS_PATH = "output/mapreduce_monitoring_analysis.json"
ANOMALY_RESULTS_PATH = "output/timeseries_anomaly_results.json"

HOUSEHOLDS = ["residential3", "residential4", "residential6"]

st.set_page_config(
    page_title="Smart Monitoring Dashboard for Power Consumption",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Smart Monitoring Dashboard for Power Consumption")
st.markdown("""
A modern, interactive dashboard for real-time monitoring, anomaly detection, and trend analysis of smart meter data.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Controls")
st.sidebar.markdown("""
- **Select Household:** Choose which household's data to analyze.
- **Appliance selection:** Choose which real appliances to visualize and compare.
- **Time range:** Filter data and anomalies by recent period.
""")
household = st.sidebar.selectbox("Select Household", HOUSEHOLDS)
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["All", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
    index=0
)

# --- FILE CHECKS & DATA FRESHNESS ---
def file_status(path, must_exist=True):
    if not os.path.exists(path):
        return False, None
    mtime = os.path.getmtime(path)
    return True, datetime.fromtimestamp(mtime)

# Check all required files
required_files = {
    'Processed Data': DATA_PATH,
    'Monitoring Statistics': MONITORING_STATS_PATH,
    'Anomaly Results': ANOMALY_RESULTS_PATH
}

st.sidebar.markdown('---')
st.sidebar.header('Data Status')
all_files_ok = True
for label, path in required_files.items():
    exists, mtime = file_status(path)
    if not exists:
        st.sidebar.error(f"❌ {label} missing")
        all_files_ok = False
    else:
        st.sidebar.success(f"✅ {label} (updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

if not all_files_ok:
    st.warning("Some required files are missing. Please run the full pipeline in main.py before using the dashboard.")

# --- DATA LOADING ---
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception:
        st.error("Processed data not found. Please run the data pipeline first.")
        return None
    return df

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def filter_time(df, time_range):
    if time_range == "All":
        return df
    now = df["timestamp"].max()
    if time_range == "Last 24 Hours":
        start = now - pd.Timedelta(hours=24)
    elif time_range == "Last 7 Days":
        start = now - pd.Timedelta(days=7)
    elif time_range == "Last 30 Days":
        start = now - pd.Timedelta(days=30)
    else:
        start = df["timestamp"].min()
    return df[df["timestamp"] >= start]

# --- MAIN DATA ---
df = load_data() if all_files_ok else None
if df is not None:
    # Filter by time range
    df = filter_time(df, time_range)

    # --- Dynamic Appliance Extraction ---
    def get_appliance_columns(df, household):
        prefix = f"DE_KN_{household}_"
        exclude = ['grid_export', 'grid_import', 'pv']
        return [col for col in df.columns if col.startswith(prefix) and not any(x in col for x in exclude)]

    real_appliances = get_appliance_columns(df, household)
    st.sidebar.markdown(f"**Appliances for {household}:**")
    st.sidebar.write(real_appliances)
    selected_appliances = st.sidebar.multiselect(
        "Select appliances to visualize (for breakdown)", real_appliances, default=real_appliances[:3]
    )

    # --- Appliance Consumption Table ---
    st.subheader(f"Appliance Consumption Table for {household}")
    st.dataframe(df[selected_appliances + ["timestamp"]].set_index("timestamp").tail(48))

    # --- Pie/Donut Chart ---
    st.subheader(f"Appliance Contribution Pie Chart ({household})")
    pie_data = df[selected_appliances].sum().reset_index()
    pie_data.columns = ["Appliance", "Total Consumption"]
    fig_pie = px.pie(pie_data, names="Appliance", values="Total Consumption", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Bar Chart ---
    st.subheader(f"Average Consumption per Appliance ({household})")
    bar_data = df[selected_appliances].mean().reset_index()
    bar_data.columns = ["Appliance", "Average Consumption"]
    fig_bar = px.bar(bar_data, x="Appliance", y="Average Consumption", color="Appliance",
                    color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Box Plot ---
    st.subheader(f"Consumption Distribution (Box Plot) for {household}")
    box_data = df[selected_appliances].melt(var_name="Appliance", value_name="Consumption")
    fig_box = px.box(box_data, x="Appliance", y="Consumption", color="Appliance",
                    color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_box, use_container_width=True)

    # --- Calendar Heatmap (Daily Total) ---
    st.subheader(f"Calendar Heatmap of Daily Total Consumption ({household})")
    df["date"] = df["timestamp"].dt.date
    cal_data = df.groupby("date")[selected_appliances].sum()
    cal_data["total"] = cal_data.sum(axis=1)
    cal_data = cal_data.reset_index()
    cal_data["date"] = pd.to_datetime(cal_data["date"])
    cal_data["dow"] = cal_data["date"].dt.dayofweek
    cal_data["week"] = cal_data["date"].dt.isocalendar().week
    fig_cal = px.density_heatmap(cal_data, x="week", y="dow", z="total",
                                labels={"z": "Total Consumption", "week": "Week", "dow": "Day of Week"},
                                color_continuous_scale="Viridis")
    fig_cal.update_yaxes(
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # --- Correlation Matrix ---
    st.subheader(f"Correlation Matrix of Appliances ({household})")
    corr = df[selected_appliances].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- HEATMAP ---
    st.subheader("Consumption Intensity Heatmap (Hour vs Day)")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.day_name()
    heatmap_data = df.pivot_table(index="dayofweek", columns="hour", values=selected_appliances, aggfunc="sum")
    if isinstance(heatmap_data.columns, pd.MultiIndex):
        # Collapse the MultiIndex by summing across appliances (new pandas style)
        heatmap_data = heatmap_data.T.groupby(level=1).sum().T
    heatmap_data = heatmap_data.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig2 = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Total Consumption (W)"),
        aspect="auto",
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- ANOMALY LOADING ---
    anomaly_json = load_json(ANOMALY_RESULTS_PATH)
    anomaly_indices = []
    anomaly_times = []
    if anomaly_json and household in anomaly_json:
        anomaly_times = anomaly_json[household].get("anomaly_timestamps", [])
        # Ensure df["timestamp"] is UTC and datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        anomaly_times_dt = pd.to_datetime(anomaly_times, utc=True)
        # Only keep anomalies in the filtered time range
        filtered_anomaly_times = set(anomaly_times_dt) & set(df["timestamp"])
        anomaly_indices = df[df["timestamp"].isin(filtered_anomaly_times)].index.tolist()
        anomaly_times = [str(ts) for ts in filtered_anomaly_times]

    # --- TREND LINE CHART ---
    st.subheader(f"Consumption Trend for {household.title()} (sum of selected appliances)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[selected_appliances].sum(axis=1),
        mode="lines", name="Raw Consumption",
        line=dict(color="#1976D2", width=2), opacity=0.7
    ))
    # Anomaly overlay
    if anomaly_indices:
        fig.add_trace(go.Scatter(
            x=df.loc[anomaly_indices, "timestamp"],
            y=df.loc[anomaly_indices, selected_appliances].sum(axis=1),
            mode="markers", name="Anomalies",
            marker=dict(size=12, color="#E53935", symbol="x"),
            hoverinfo="x+y",
            showlegend=True
        ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Total Consumption (W)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        transition=dict(duration=500, easing="cubic-in-out")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # --- REAL-TIME SIMULATION ---
    st.info("This dashboard simulates real-time updates. Click 'Update Now' to refresh data.")
    if st.button("Update Now", type="primary"):
        st.experimental_rerun()

    # --- EXPORT OPTIONS ---
    st.markdown("---")
    st.download_button(
        label="Export Data as CSV",
        data=df.to_csv(index=False),
        file_name=f"{household}_appliance_data.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Export Trend Chart as PNG",
        data=fig.to_image(format="png"),
        file_name=f"{household}_trend.png",
        mime="image/png"
    )
else:
    st.warning("No data available. Please ensure the data pipeline has been run and processed data is present.")

# --- FOOTER ---
st.markdown("""
---
*Powered by Streamlit · Interactive, real-time energy analytics for smart meters.*
""")
