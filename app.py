# app.py - DataSync Dashboard using synthetic CSV
# Make sure datasync_survey_synthetic.csv is in the same folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="DataSync Analytics")

@st.cache_data
def load_data():
    df = pd.read_csv("datasync_survey_synthetic.csv")
    # Create a few numeric helper columns
    df["Error_Score"] = df["Manual_Error_Freq"].map({"Low":1, "Medium":2, "High":3})
    df["Insights_Score"] = df["RealTime_Insights"].map({"Available":3, "Limited":2, "None":1})
    df["Issue_Flag"] = (df["Sys_Integration_Issue"] == "Yes").astype(int)
    return df

df = load_data()

st.title("🛠️ DataSync Analytics – E‑commerce Integration Survey")

# Sidebar filters
st.sidebar.header("Filters")
region_filter = st.sidebar.multiselect(
    "Region", options=sorted(df["Region"].unique()), default=list(df["Region"].unique())
)
size_filter = st.sidebar.multiselect(
    "Company Size", options=sorted(df["Company_Size"].unique()), default=list(df["Company_Size"].unique())
)

df_f = df[df["Region"].isin(region_filter) & df["Company_Size"].isin(size_filter)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Share with Issues", f"{df_f['Issue_Flag'].mean():.0%}")
col2.metric("Avg Sync Latency", f"{df_f['Avg_Sync_Latency_min'].mean():.1f} min")
col3.metric("Avg Maint. Cost", f"${df_f['Maintenance_Cost_USD'].mean():,.0f}")
col4.metric("Avg Budget Willing", f"${df_f['Budget_Willing_USD'].mean():,.0f}")

# Chart 1: Issues vs No Issues – latency & cost
st.subheader("Impact of Disconnected Systems")
issues_group = df_f.groupby("Sys_Integration_Issue").agg({
    "Avg_Sync_Latency_min":"mean",
    "Maintenance_Cost_USD":"mean",
    "Budget_Willing_USD":"mean"
}).reset_index()

fig1 = px.bar(
    issues_group,
    x="Sys_Integration_Issue",
    y=["Avg_Sync_Latency_min", "Maintenance_Cost_USD"],
    barmode="group",
    title="Latency & Maintenance Cost – Issues vs No Issues"
)
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Error vs Budget (colored by Interest)
st.subheader("Error Levels vs Budget & Interest")
fig2 = px.scatter(
    df_f,
    x="Error_Score",
    y="Budget_Willing_USD",
    color="Interest_In_DataSync",
    size="Monthly_Orders",
    hover_data=["Company_Size", "Tech_Comfort"],
    title="Higher Errors → Higher Budget & Interest for DataSync"
)
st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Heatmap of key metrics
st.subheader("Correlation Heatmap – Key Metrics")
corr = df_f[[
    "Error_Score",
    "Avg_Sync_Latency_min",
    "Insight_Delay_hours",
    "Maintenance_Cost_USD",
    "Budget_Willing_USD"
]].corr()

fig3 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation between Errors, Latency, Cost & Budget"
)
st.plotly_chart(fig3, use_container_width=True)

# Table: Top target accounts (high issues + high budget)
st.subheader("Top Target Accounts for DataSync")
targets = df_f[df_f["Sys_Integration_Issue"] == "Yes"].copy()
targets = targets.sort_values("Budget_Willing_USD", ascending=False).head(10)
st.dataframe(
    targets[[
        "Respondent_ID", "Company_Size", "Region", "Tech_Comfort",
        "Manual_Error_Freq", "Avg_Sync_Latency_min",
        "Maintenance_Cost_USD", "Budget_Willing_USD"
    ]],
    use_container_width=True
)

st.caption("DataSync Analytics – Synthetic survey data for PBL")
