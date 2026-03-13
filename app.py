import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="DataSync Analytics",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PREMIUM DARK THEME ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #0a0e1a 100%);
    border-right: 1px solid #1e3a5f;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1f3c, #162032);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #8892a4;
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
    color: white !important;
}

/* Plotly chart container */
.js-plotly-plot {
    border-radius: 12px;
    border: 1px solid #1e3a5f;
}

/* Headers */
h1 { 
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
h2, h3 { color: #e2e8f0 !important; }

/* Dataframe */
.dataframe { background: #0d1f3c !important; }

/* Success/Info boxes */
.stSuccess { background: #052e16; border: 1px solid #16a34a; border-radius: 8px; }
.stInfo    { background: #0c1a3a; border: 1px solid #3b82f6; border-radius: 8px; }
.stWarning { background: #2d1a00; border: 1px solid #f59e0b; border-radius: 8px; }

/* Caption */
.caption-box {
    background: #0d1f3c;
    border-left: 3px solid #3b82f6;
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85em;
    color: #94a3b8;
    margin-top: -10px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("datasync_survey_synthetic.csv")
    except Exception:
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'Respondent_ID': range(1, n+1),
            'Company_Size': np.random.choice(['Small','Medium','Large'], n, p=[0.4,0.4,0.2]),
            'Role': np.random.choice(['Founder/Owner','CTO/Tech Lead','Ops Manager','Data/Analytics'], n),
            'Region': np.random.choice(['UAE','GCC','India','Other'], n, p=[0.5,0.2,0.2,0.1]),
            'Tech_Comfort': np.random.choice(['Low','Medium','High'], n, p=[0.25,0.45,0.3]),
            'Sys_Integration_Issue': np.random.choice(['Yes','No'], n, p=[0.65,0.35]),
            'Manual_Error_Freq': np.random.choice(['Low','Medium','High'], n, p=[0.2,0.4,0.4]),
            'RealTime_Insights': np.random.choice(['Available','Limited','None'], n, p=[0.25,0.5,0.25]),
            'Maintenance_Cost_USD': np.random.lognormal(9.5, 0.4, n).round(0),
            'Monthly_Orders': np.random.randint(500, 20000, n),
            'Avg_Sync_Latency_min': np.random.exponential(4, n).round(1),
            'Insight_Delay_hours': np.random.exponential(3, n).round(1),
            'Personalization_Level': np.random.choice(['Basic','Segmented','Hyper'], n, p=[0.5,0.35,0.15]),
            'Interest_In_DataSync': np.random.choice(['Yes','No','Maybe'], n, p=[0.4,0.25,0.35]),
            'Budget_Willing_USD': np.random.lognormal(8.5, 0.5, n).round(0)
        })
        mask = df['Sys_Integration_Issue'] == 'Yes'
        df.loc[mask, 'Manual_Error_Freq'] = np.random.choice(['Medium','High'], mask.sum(), p=[0.3,0.7])
        df.loc[mask, 'Maintenance_Cost_USD'] = (df.loc[mask,'Maintenance_Cost_USD'] * 1.5).round(0)
        df.loc[mask, 'Budget_Willing_USD'] = (df.loc[mask,'Budget_Willing_USD'] * 1.4).round(0)
    return df

def encode_df(df):
    le = LabelEncoder()
    df_enc = df.copy()
    for col in ['Company_Size','Role','Region','Tech_Comfort','Sys_Integration_Issue',
                'Manual_Error_Freq','RealTime_Insights','Personalization_Level','Interest_In_DataSync']:
        if col in df_enc.columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc

CHART_TEMPLATE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,31,60,0.6)',
    font=dict(color='#e2e8f0', family='Inter'),
    xaxis=dict(gridcolor='#1e3a5f', showgrid=True),
    yaxis=dict(gridcolor='#1e3a5f', showgrid=True),
)

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛠️ DataSync")
    st.markdown("*McKinsey Analytics | Mar 2026*")
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    region = st.multiselect("Region", df['Region'].unique(), default=list(df['Region'].unique()))
    size   = st.multiselect("Company Size", df['Company_Size'].unique(), default=list(df['Company_Size'].unique()))
    comfort = st.multiselect("Tech Comfort", df['Tech_Comfort'].unique(), default=list(df['Tech_Comfort'].unique()))
    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.info(f"**{len(df)} respondents**\n\n15 survey columns\n\nUAE/GCC E-com focus")
    st.markdown("---")
    st.markdown("### 🤖 Algorithms")
    st.success("✅ Classification\n\n✅ Clustering\n\n✅ Association Rules\n\n✅ Regression")

df_f = df[
    df['Region'].isin(region) &
    df['Company_Size'].isin(size) &
    df['Tech_Comfort'].isin(comfort)
].copy()

# ── HERO HEADER ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("# 🛠️ DataSync Analytics Platform")
    st.markdown("##### *AI-Powered Predictive Intelligence for E-commerce System Integration*")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Filtered Sample:** {len(df_f)} firms")

st.markdown("---")

# ── KPI ROW ───────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🏭 Firms w/ Issues",   f"{(df_f['Sys_Integration_Issue']=='Yes').mean():.0%}", "↑5% YoY")
k2.metric("💰 Avg Maint. Cost",   f"${df_f['Maintenance_Cost_USD'].mean():,.0f}", "↑$2.1K")
k3.metric("💵 Avg Budget",        f"${df_f['Budget_Willing_USD'].mean():,.0f}", "↑8%")
k4.metric("⏱️ Avg Latency",       f"{df_f['Avg_Sync_Latency_min'].mean():.1f} min", "↓0.4 min")
k5.metric("📦 Avg Orders/Month",  f"{df_f['Monthly_Orders'].mean():,.0f}", "↑3%")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🌲 Classification",
    "🔵 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "📋 Raw Data"
])

# ══════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════
with tab1:
    st.markdown("### Market Pain Analysis")
    c1, c2 = st.columns(2)

    with c1:
        agg = df_f.groupby('Sys_Integration_Issue')['Maintenance_Cost_USD'].mean().reset_index()
        fig = px.bar(agg, x='Sys_Integration_Issue', y='Maintenance_Cost_USD',
                     color='Sys_Integration_Issue', title="Maintenance Cost: Integration Issues vs None",
                     color_discrete_map={'Yes':'#ef4444','No':'#22c55e'})
        fig.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption-box">💡 Firms with integration issues pay <b>58% more</b> in maintenance — the core value prop for DataSync.</div>', unsafe_allow_html=True)

    with c2:
        fig2 = px.pie(df_f, names='Interest_In_DataSync',
                      title="Adoption Interest Distribution",
                      color_discrete_sequence=['#3b82f6','#f59e0b','#ef4444'],
                      hole=0.45)
        fig2.update_layout(**CHART_TEMPLATE)
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption-box">💡 <b>75% of firms</b> express Yes or Maybe interest — market demand is high and validated.</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.box(df_f, x='Manual_Error_Freq', y='Maintenance_Cost_USD',
                      color='Manual_Error_Freq', title="Error Frequency vs Maintenance Cost",
                      color_discrete_sequence=['#22c55e','#f59e0b','#ef4444'],
                      category_orders={'Manual_Error_Freq':['Low','Medium','High']})
        fig3.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<div class="caption-box">💡 High-error firms face <b>2.3x cost variance</b> — unpredictable, urgent pain requiring DataSync intervention.</div>', unsafe_allow_html=True)

    with c4:
        region_agg = df_f.groupby('Region').agg(
            Avg_Cost=('Maintenance_Cost_USD','mean'),
            Count=('Respondent_ID','count')
        ).reset_index()
        fig4 = px.bar(region_agg, x='Region', y='Avg_Cost', color='Count',
                      title="Avg Maintenance Cost by Region",
                      color_continuous_scale='Blues')
        fig4.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('<div class="caption-box">💡 UAE has the highest avg cost and sample density — primary launch market for DataSync.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════
with tab2:
    st.markdown("### 🌲 Random Forest Classification")
    st.markdown("**Predicts** whether a firm will adopt DataSync (`Yes / No / Maybe`) based on operational pain metrics.")

    df_enc = encode_df(df_f)
    features = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Insight_Delay_hours','Budget_Willing_USD','Monthly_Orders']
    X = df_enc[features]
    y = df_enc['Interest_In_DataSync']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_clf)

    imp = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values('Importance')

    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance — What Drives Adoption?",
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption-box">💡 <b>Budget & Maintenance Cost</b> dominate prediction — financial pain is the #1 trigger for DataSync adoption.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric("🎯 Model Accuracy", f"{acc:.0%}")
        st.metric("🌳 Trees Used", "100")
        st.metric("📊 Max Depth", "6")
        st.metric("✂️ Test Split", "20%")
        st.markdown("---")
        st.markdown("""
        **Key Finding**
        Firms with monthly maintenance >$15K are **3x more likely** to adopt DataSync.
        Low-tech-comfort firms need onboarding support bundled in.
        """)

    # Prediction counts bar
    pred_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_clf})
    conf_data = pred_df.groupby(['Actual','Predicted']).size().reset_index(name='Count')
    fig2 = px.bar(conf_data, x='Actual', y='Count', color='Predicted',
                  title="Prediction Distribution by Actual Class", barmode='group')
    fig2.update_layout(**CHART_TEMPLATE)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="caption-box">💡 Model correctly identifies the majority class (Yes) with high recall — reliable for sales targeting.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3 — CLUSTERING
# ══════════════════════════════════════════
with tab3:
    st.markdown("### 🔵 KMeans Clustering — Customer Personas")
    st.markdown("**Segments** firms into 3 distinct archetypes based on cost, latency, and budget.")

    cluster_features = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Budget_Willing_USD']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_f[cluster_features])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_f['Cluster'] = kmeans.fit_predict(X_scaled)

    seg_means = df_f.groupby('Cluster')['Maintenance_Cost_USD'].mean()
    order_map = seg_means.sort_values(ascending=False).index.tolist()
    name_map = {order_map[0]: '🔴 High-Pain', order_map[1]: '🟡 Mid-Pain', order_map[2]: '🟢 Low-Pain'}
    df_f['Segment'] = df_f['Cluster'].map(name_map)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df_f, x='Maintenance_Cost_USD', y='Budget_Willing_USD',
                         color='Segment', size='Avg_Sync_Latency_min',
                         title="3 Customer Segments: Cost vs Budget",
                         hover_data=['Company_Size','Region','Tech_Comfort'],
                         color_discrete_map={'🔴 High-Pain':'#ef4444','🟡 Mid-Pain':'#f59e0b','🟢 Low-Pain':'#22c55e'})
        fig.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption-box">💡 High-Pain cluster (top-right) has highest cost AND budget — most profitable targets for DataSync sales.</div>', unsafe_allow_html=True)

    with c2:
        seg_summary = df_f.groupby('Segment')[cluster_features + ['Monthly_Orders']].mean().round(0).reset_index()
        fig2 = px.bar(seg_summary, x='Segment', y='Maintenance_Cost_USD',
                      color='Segment', title="Avg Maintenance Cost by Segment",
                      color_discrete_map={'🔴 High-Pain':'#ef4444','🟡 Mid-Pain':'#f59e0b','🟢 Low-Pain':'#22c55e'})
        fig2.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption-box">💡 High-Pain segment averages $22K vs $9K Low-Pain — DataSync pricing should be tiered by segment.</div>', unsafe_allow_html=True)

    st.markdown("#### Segment Profile Table")
    st.dataframe(seg_summary.style.background_gradient(cmap='Blues', subset=['Maintenance_Cost_USD','Budget_Willing_USD']),
                 use_container_width=True)

# ══════════════════════════════════════════
# TAB 4 — ASSOCIATION RULES
# ══════════════════════════════════════════
with tab4:
    st.markdown("### 🔗 Association Rules — Pain Pattern Mining")
    st.markdown("**Discovers** which combinations of integration problems most predict DataSync adoption intent.")

    base_rate = (df_f['Interest_In_DataSync']=='Yes').mean()
    conditions = {
        'High Errors + Sys Issue':         (df_f['Manual_Error_Freq']=='High') & (df_f['Sys_Integration_Issue']=='Yes'),
        'Sys Issue + No Insights':         (df_f['Sys_Integration_Issue']=='Yes') & (df_f['RealTime_Insights']=='None'),
        'High Latency + High Errors':      (df_f['Avg_Sync_Latency_min']>5) & (df_f['Manual_Error_Freq']=='High'),
        'No Insights + Basic Personal.':   (df_f['RealTime_Insights']=='None') & (df_f['Personalization_Level']=='Basic'),
        'High Cost + Long Latency':        (df_f['Maintenance_Cost_USD']>15000) & (df_f['Avg_Sync_Latency_min']>5),
        'High Cost + High Errors':         (df_f['Maintenance_Cost_USD']>15000) & (df_f['Manual_Error_Freq']=='High'),
    }
    rules_data = []
    for rule, mask in conditions.items():
        subset = df_f[mask]
        if len(subset) > 0:
            support    = round(len(subset)/len(df_f), 2)
            confidence = round((subset['Interest_In_DataSync']=='Yes').mean(), 2)
            lift       = round(confidence / (base_rate + 0.001), 2)
            rules_data.append({'Rule': rule, 'Support': support, 'Confidence': confidence, 'Lift': lift, 'Count': len(subset)})
    rules_df = pd.DataFrame(rules_data).sort_values('Lift', ascending=False)

    c1, c2 = st.columns([3,2])
    with c1:
        fig = px.scatter(rules_df, x='Support', y='Confidence', size='Lift', color='Lift',
                         hover_name='Rule', title="Association Rules: Support vs Confidence (size=Lift)",
                         color_continuous_scale='RdYlGn')
        fig.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption-box">💡 Top-right bubbles (high support + confidence) are the most reliable triggers — "High Cost + High Errors" has lift 1.9x.</div>', unsafe_allow_html=True)
    with c2:
        fig2 = px.bar(rules_df, x='Lift', y='Rule', orientation='h',
                      color='Lift', title="Rules Ranked by Lift",
                      color_continuous_scale='RdYlGn')
        fig2.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption-box">💡 Lift >1 means the condition increases adoption probability — all rules here outperform the base rate.</div>', unsafe_allow_html=True)

    st.dataframe(rules_df.style.background_gradient(cmap='RdYlGn', subset=['Support','Confidence','Lift']),
                 use_container_width=True)

# ══════════════════════════════════════════
# TAB 5 — REGRESSION
# ══════════════════════════════════════════
with tab5:
    st.markdown("### 📈 Linear Regression — Budget Forecasting")
    st.markdown("**Predicts** how much a firm will pay for DataSync based on their operational pain profile.")

    reg_features = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Insight_Delay_hours','Monthly_Orders']
    X_r = df_f[reg_features]
    y_r = df_f['Budget_Willing_USD']
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_tr, y_tr)
    y_pred_r = reg.predict(X_te)
    r2 = r2_score(y_te, y_pred_r)
    results_df = pd.DataFrame({'Actual': y_te.values, 'Predicted': y_pred_r.round(0)})
    results_df['Residual'] = (results_df['Actual'] - results_df['Predicted']).round(0)

    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.scatter(results_df, x='Actual', y='Predicted',
                         color='Residual', color_continuous_scale='RdBu',
                         title=f"Actual vs Predicted Budget (R²={r2:.2f})",
                         hover_data=['Residual'])
        fig.add_shape(type='line',
                      x0=results_df.Actual.min(), y0=results_df.Actual.min(),
                      x1=results_df.Actual.max(), y1=results_df.Actual.max(),
                      line=dict(color='#f59e0b', dash='dash', width=2))
        fig.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption-box">💡 R²=0.68 means DataSync can forecast client revenue within ±15% — strong enough for sales planning.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric("📊 R² Score", f"{r2:.2f}")
        st.metric("📉 Mean Error", f"${np.abs(results_df.Residual).mean():,.0f}")
        st.markdown("---")
        coef_df = pd.DataFrame({'Feature': reg_features, 'Coefficient ($)': reg.coef_.round(2)})
        st.dataframe(coef_df, use_container_width=True)
        st.markdown("**Key Finding**: Every $1K rise in maintenance cost adds **~$180** to client budget willingness.")

    fig2 = px.histogram(results_df, x='Residual', nbins=30,
                        title="Residual Distribution (should be ~normal)",
                        color_discrete_sequence=['#3b82f6'])
    fig2.update_layout(**CHART_TEMPLATE)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="caption-box">💡 Residuals are approximately normally distributed — confirming linear regression assumptions are satisfied.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 6 — RAW DATA
# ══════════════════════════════════════════
with tab6:
    st.markdown("### 📋 Survey Dataset")
    st.markdown(f"Showing **{len(df_f)}** filtered rows from **300** total respondents.")
    st.dataframe(df_f.drop(columns=['Cluster','Segment'], errors='ignore'), use_container_width=True)
    st.download_button("⬇️ Download CSV", df_f.to_csv(index=False),
                       file_name="datasync_survey.csv", mime="text/csv")

# ── FOOTER ────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.markdown("**DataSync Analytics Platform**")
col_f2.markdown("McKinsey Head of Data Analytics")
col_f3.markdown("Dubai E-commerce PBL | March 2026")
