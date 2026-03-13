import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="DataSync Analytics")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("datasync_survey_synthetic.csv")
    except Exception:
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'Respondent_ID': range(1, n+1),
            'Company_Size': np.random.choice(['Small','Medium','Large'], n),
            'Region': np.random.choice(['UAE','GCC','India','Other'], n),
            'Tech_Comfort': np.random.choice(['Low','Medium','High'], n),
            'Sys_Integration_Issue': np.random.choice(['Yes','No'], n, p=[0.65,0.35]),
            'Manual_Error_Freq': np.random.choice(['Low','Medium','High'], n),
            'RealTime_Insights': np.random.choice(['Available','Limited','None'], n),
            'Maintenance_Cost_USD': np.random.lognormal(9.5, 0.4, n).round(0),
            'Avg_Sync_Latency_min': np.random.exponential(4, n).round(1),
            'Interest_In_DataSync': np.random.choice(['Yes','No','Maybe'], n),
            'Budget_Willing_USD': np.random.lognormal(8.5, 0.5, n).round(0)
        })
    return df

df = load_data()

st.title("⚡ DataSync Analytics — Predictive Dashboard")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("With Integration Issues", f"{(df['Sys_Integration_Issue']=='Yes').mean():.0%}")
c2.metric("Avg Maintenance Cost", f"${df['Maintenance_Cost_USD'].mean():,.0f}")
c3.metric("Avg Budget Willing", f"${df['Budget_Willing_USD'].mean():,.0f}")
c4.metric("Avg Latency", f"{df['Avg_Sync_Latency_min'].mean():.1f} min")

tab1, tab2, tab3 = st.tabs(["Overview", "Predictive", "Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        agg = df.groupby('Sys_Integration_Issue')['Maintenance_Cost_USD'].mean().reset_index()
        fig1 = px.bar(agg, x='Sys_Integration_Issue', y='Maintenance_Cost_USD',
                      title="Maintenance Cost: Issues vs No Issues", color='Sys_Integration_Issue')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x='Interest_In_DataSync', color='Tech_Comfort',
                            title="Interest in DataSync by Tech Comfort")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        # Simulated 30-day forecast (trend line)
        days = list(range(1, 31))
        actual = [df['Maintenance_Cost_USD'].mean() * (1 + 0.02*np.sin(i)) for i in days[:20]]
        forecast = [df['Maintenance_Cost_USD'].mean() * (0.95 - 0.01*i) for i in range(1, 12)]
        forecast_df = pd.DataFrame({
            'Day': days[:20] + list(range(21, 32)),
            'Value': actual + forecast,
            'Type': ['Actual']*20 + ['Forecast']*11
        })
        fig3 = px.line(forecast_df, x='Day', y='Value', color='Type',
                       title="30-Day Maintenance Cost Forecast")
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        corr = df[['Maintenance_Cost_USD','Avg_Sync_Latency_min','Budget_Willing_USD']].corr()
        fig4 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                         title="Correlation Heatmap")
        st.plotly_chart(fig4, use_container_width=True)

    # Anomaly detection (simple stats)
    mean_cost = df['Maintenance_Cost_USD'].mean()
    std_cost = df['Maintenance_Cost_USD'].std()
    anomalies = df[df['Maintenance_Cost_USD'] > mean_cost + 2*std_cost]
    st.warning(f"🚨 {len(anomalies)} anomalies detected (cost > 2σ above mean)")
    st.dataframe(anomalies[['Respondent_ID','Company_Size','Maintenance_Cost_USD','Avg_Sync_Latency_min']].head(5))

with tab3:
    st.dataframe(df, use_container_width=True)

st.caption("DataSync Analytics | McKinsey | Dubai E-com | PBL Submission")
