import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("datasync_survey_synthetic.csv")
    except:
        # Fallback synthetic data
        np.random.seed(42)
        df = pd.DataFrame({
            'Sys_Integration_Issue': np.random.choice(['Yes','No'], 300, p=[0.65,0.35]),
            'Manual_Error_Freq': np.random.choice(['Low','Medium','High'], 300),
            'Maintenance_Cost_USD': np.random.lognormal(9.5, 0.4, 300).round(0),
            'Budget_Willing_USD': np.random.lognormal(8.5, 0.5, 300).round(0),
            'Interest_In_DataSync': np.random.choice(['Yes','Maybe','No'], 300)
        })
    return df

df = load_data()

st.title("🛠️ DataSync Analytics Dashboard")
st.markdown("**E-commerce Integration Pain Points**")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Systems with Issues", f"{(df['Sys_Integration_Issue']=='Yes').mean():.0%}")
col2.metric("Avg Maintenance Cost", f"${df['Maintenance_Cost_USD'].mean():,.0f}")
col3.metric("Avg Client Budget", f"${df['Budget_Willing_USD'].mean():,.0f}")

# Charts
col1, col2 = st.columns(2)

with col1:
    # Issues vs Budget
    agg = df.groupby('Sys_Integration_Issue')['Budget_Willing_USD'].mean().reset_index()
    fig1 = px.bar(agg, x='Sys_Integration_Issue', y='Budget_Willing_USD',
                  title="Budget Willingness: Issues vs No Issues")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Interest by Error Level
    agg2 = df.groupby('Manual_Error_Freq')['Interest_In_DataSync'].value_counts().unstack(fill_value=0)
    fig2 = px.bar(agg2, title="Interest by Error Frequency")
    st.plotly_chart(fig2, use_container_width=True)

# Data table
st.subheader("Raw Survey Data (sample)")
st.dataframe(df.head(20))

st.success("✅ Dashboard Live! Use for PBL submission.")
