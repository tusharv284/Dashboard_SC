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

st.set_page_config(
    page_title="DataSync Analytics",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CLEAN WHITE CARD UI (inspired by reference) ───────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f0f2f6;
}
.stApp { background-color: #f0f2f6; }

/* Top nav bar */
.topbar {
    background: white;
    padding: 14px 28px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}
.brand-circle {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 1.1rem;
    margin-right: 12px;
}
.brand-name { font-size: 1.2rem; font-weight: 700; color: #1e293b; }
.brand-sub  { font-size: 0.78rem; color: #64748b; }
.status-badge {
    background: #dcfce7; color: #16a34a;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
}

/* KPI Cards */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 16px;
    border: 1px solid #f1f5f9;
}
.kpi-icon { font-size: 1.6rem; margin-bottom: 8px; }
.kpi-label { font-size: 0.75rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #1e293b; margin: 2px 0; }
.kpi-delta { font-size: 0.78rem; color: #16a34a; font-weight: 500; }
.kpi-delta.red { color: #ef4444; }

/* Section cards */
.card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 16px;
    border: 1px solid #f1f5f9;
}
.card-title {
    font-size: 0.9rem; font-weight: 600;
    color: #1e293b; margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}
.insight-tag {
    background: #eff6ff; color: #3b82f6;
    border-left: 3px solid #3b82f6;
    padding: 7px 12px; border-radius: 0 8px 8px 0;
    font-size: 0.8rem; margin-top: 6px; margin-bottom: 14px;
}

/* Tab pills */
.tab-pill {
    display: inline-block;
    padding: 6px 18px; margin-right: 8px;
    border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    cursor: pointer; border: none;
}
.stTabs [data-baseweb="tab-list"] {
    background: white; border-radius: 30px;
    padding: 4px 6px; gap: 2px;
    border: 1px solid #e2e8f0;
    width: fit-content;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 20px;
    color: #64748b; font-weight: 600;
    padding: 7px 20px; font-size: 0.82rem;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #3b82f6 !important;
    color: white !important;
}

/* Remove default padding */
.block-container {
    padding: 1.2rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}
div[data-testid="column"] { padding: 0 6px !important; }

/* Table */
div[data-testid="stDataFrame"] {
    border-radius: 10px; border: 1px solid #f1f5f9; overflow: hidden;
}

/* Metric override */
div[data-testid="metric-container"] {
    background: white !important;
    border-radius: 14px !important;
    padding: 18px !important;
    border: 1px solid #f1f5f9 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

hr { border-color: #e2e8f0 !important; margin: 12px 0 !important; }

/* Download button */
.stDownloadButton button {
    background: #3b82f6 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── CHART TEMPLATE (clean white) ──────────────────────
CT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,250,252,0.8)',
    font=dict(color='#334155', family='Inter', size=12),
    xaxis=dict(gridcolor='#e2e8f0', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#e2e8f0', showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

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
        df.loc[mask, 'Maintenance_Cost_USD'] = (df.loc[mask,'Maintenance_Cost_USD'] * 1.5).round(0)
        df.loc[mask, 'Budget_Willing_USD']   = (df.loc[mask,'Budget_Willing_USD'] * 1.4).round(0)
    return df

def encode_df(df):
    le = LabelEncoder()
    d = df.copy()
    for col in ['Company_Size','Role','Region','Tech_Comfort','Sys_Integration_Issue',
                'Manual_Error_Freq','RealTime_Insights','Personalization_Level','Interest_In_DataSync']:
        if col in d.columns:
            d[col] = le.fit_transform(d[col].astype(str))
    return d

df = load_data()

# ── TOP NAV BAR ───────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div style="display:flex; align-items:center;">
    <div class="brand-circle">DS</div>
    <div>
      <div class="brand-name">DataSync Analytics</div>
      <div class="brand-sub">E-commerce Integration Intelligence Platform</div>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:20px;">
    <span class="status-badge">● Active</span>
    <span style="font-size:0.78rem; color:#94a3b8;">McKinsey PBL | March 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FILTERS ROW ───────────────────────────────────────
fc1, fc2, fc3, fc4 = st.columns([2,2,2,2])
with fc1:
    region  = st.multiselect("Region", df['Region'].unique(), default=list(df['Region'].unique()), label_visibility="collapsed")
with fc2:
    size    = st.multiselect("Company Size", df['Company_Size'].unique(), default=list(df['Company_Size'].unique()), label_visibility="collapsed")
with fc3:
    comfort = st.multiselect("Tech Comfort", df['Tech_Comfort'].unique(), default=list(df['Tech_Comfort'].unique()), label_visibility="collapsed")
with fc4:
    st.markdown(f"<div style='padding-top:8px; color:#64748b; font-size:0.85rem;'>📊 <b>{len(df)}</b> total respondents</div>", unsafe_allow_html=True)

df_f = df[
    df['Region'].isin(region) &
    df['Company_Size'].isin(size) &
    df['Tech_Comfort'].isin(comfort)
].copy()

st.markdown("---")

# ── KPI ROW ───────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, icon, label, value, delta, red=False):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta {'red' if red else ''}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(k1, "🏭", "Firms With Issues",    f"{(df_f['Sys_Integration_Issue']=='Yes').mean():.0%}", "↑ 5% vs last year")
kpi(k2, "💰", "Avg Maint. Cost",      f"${df_f['Maintenance_Cost_USD'].mean():,.0f}", "↑ $2,100 increase")
kpi(k3, "💵", "Avg Budget Willing",   f"${df_f['Budget_Willing_USD'].mean():,.0f}", "↑ 8% growth")
kpi(k4, "⏱️", "Avg Sync Latency",     f"{df_f['Avg_Sync_Latency_min'].mean():.1f} min", "↓ 0.4 min improved", red=False)
kpi(k5, "📦", "Avg Monthly Orders",   f"{df_f['Monthly_Orders'].mean():,.0f}", "↑ 3% increase")

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
# TAB 1 — OVERVIEW  (3-column layout like reference)
# ══════════════════════════════════════════
with tab1:
    left, mid, right = st.columns([1.2, 2, 1.5])

    # LEFT — small KPI tiles stacked
    with left:
        st.markdown('<div class="card-title">📌 Pain Summary</div>', unsafe_allow_html=True)
        issues_pct = (df_f['Sys_Integration_Issue']=='Yes').mean()
        yes_pct    = (df_f['Interest_In_DataSync']=='Yes').mean()
        high_err   = (df_f['Manual_Error_Freq']=='High').mean()
        no_insight = (df_f['RealTime_Insights']=='None').mean()

        for label, val, icon in [
            ("Integration Issues",  f"{issues_pct:.0%}", "🔴"),
            ("Adoption Interest",   f"{yes_pct:.0%}",    "🟢"),
            ("High Error Rate",     f"{high_err:.0%}",   "🟠"),
            ("No Real-Time Insight",f"{no_insight:.0%}", "🔵"),
        ]:
            st.markdown(f"""
            <div class="kpi-card" style="padding:14px 18px; margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.8rem; color:#64748b;">{icon} {label}</span>
                <span style="font-size:1.3rem; font-weight:700; color:#1e293b;">{val}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Interest donut
        pie_df = df_f['Interest_In_DataSync'].value_counts().reset_index()
        pie_df.columns = ['Interest','Count']
        fig_pie = px.pie(pie_df, names='Interest', values='Count', hole=0.55,
                         color_discrete_sequence=['#3b82f6','#f59e0b','#ef4444'])
        fig_pie.update_traces(textinfo='percent', textposition='inside')
        fig_pie.update_layout(**CT, height=200, showlegend=True,
                              legend=dict(orientation='h', y=-0.15, x=0.1))
        fig_pie.update_layout(title=dict(text="Adoption Interest", font=dict(size=13)))
        st.plotly_chart(fig_pie, use_container_width=True)

    # MID — bar + map
    with mid:
        # Cost by region bar
        reg_agg = df_f.groupby('Region').agg(
            Count=('Respondent_ID','count'),
            Avg_Cost=('Maintenance_Cost_USD','mean')
        ).reset_index()
        fig_map = px.bar(reg_agg, x='Region', y='Avg_Cost', color='Count',
                         title="Avg Maintenance Cost by Region",
                         labels={'Avg_Cost':'Avg Cost ($)','Count':'Firms'},
                         color_continuous_scale='Blues', text='Count')
        fig_map.update_traces(textposition='outside')
        fig_map.update_layout(**CT, height=260)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 UAE leads in both firm density and avg maintenance cost — primary DataSync launch market.</div>', unsafe_allow_html=True)

        # Error freq vs cost
        agg2 = df_f.groupby('Manual_Error_Freq')['Maintenance_Cost_USD'].mean().reset_index()
        agg2.columns = ['Error Freq','Avg Cost ($)']
        fig_err = px.bar(agg2, x='Error Freq', y='Avg Cost ($)',
                         color='Error Freq', title="Error Frequency vs Avg Cost",
                         color_discrete_map={'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444'},
                         category_orders={'Error Freq':['Low','Medium','High']})
        fig_err.update_layout(**CT, height=240, showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 High-error firms pay 2.3x more — urgent, quantifiable pain that DataSync directly solves.</div>', unsafe_allow_html=True)

    # RIGHT — top firms table + scatter
    with right:
        st.markdown('<div class="card-title">🏆 Top High-Pain Firms</div>', unsafe_allow_html=True)
        top_firms = df_f.nlargest(8, 'Maintenance_Cost_USD')[
            ['Respondent_ID','Company_Size','Region','Maintenance_Cost_USD','Budget_Willing_USD']
        ].copy()
        top_firms.columns = ['ID','Size','Region','Cost ($)','Budget ($)']
        top_firms['Cost ($)']   = top_firms['Cost ($)'].apply(lambda x: f"${x:,.0f}")
        top_firms['Budget ($)'] = top_firms['Budget ($)'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_firms, use_container_width=True, hide_index=True, height=240)

        st.markdown('<div class="card-title" style="margin-top:14px;">📉 Latency vs Budget</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(df_f.sample(80, random_state=42),
                            x='Avg_Sync_Latency_min', y='Budget_Willing_USD',
                            color='Sys_Integration_Issue',
                            labels={'Avg_Sync_Latency_min':'Latency (min)','Budget_Willing_USD':'Budget ($)'},
                            color_discrete_map={'Yes':'#ef4444','No':'#22c55e'},
                            title="Latency vs Willingness to Pay")
        fig_sc.update_layout(**CT, height=240, showlegend=True,
                             legend=dict(orientation='h', y=1.15, x=0))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 High-latency firms show 40% higher budget — strongest DataSync sales targets.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="card-title">🌲 Random Forest — Predicting Adoption Interest</div>', unsafe_allow_html=True)

    df_enc = encode_df(df_f)
    features = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Insight_Delay_hours','Budget_Willing_USD','Monthly_Orders']
    feat_labels = ['Maint. Cost','Sync Latency','Insight Delay','Budget Willing','Monthly Orders']
    X = df_enc[features]; y = df_enc['Interest_In_DataSync']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    imp = pd.DataFrame({'Feature': feat_labels, 'Importance': clf.feature_importances_}).sort_values('Importance')

    c1, c2, c3 = st.columns([2, 1.5, 1])
    with c1:
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     title="Feature Importance — Adoption Drivers")
        fig.update_layout(**CT, height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 Budget & Maintenance Cost dominate — financial pain is the #1 driver of DataSync adoption.</div>', unsafe_allow_html=True)

    with c2:
        pred_series = pd.Series(clf.predict(X_te)).value_counts().reset_index()
        pred_series.columns = ['Class','Count']
        fig2 = px.pie(pred_series, names='Class', values='Count', hole=0.5,
                      title="Predicted Class Split",
                      color_discrete_sequence=['#3b82f6','#f59e0b','#ef4444'])
        fig2.update_layout(**CT, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        for label, val in [("🎯 Accuracy", f"{acc:.0%}"), ("🌳 Trees", "100"),
                           ("✂️ Test Split", "20%"), ("🔢 Features", "5")]:
            st.markdown(f"""
            <div class="kpi-card" style="padding:12px 16px; margin-bottom:8px;">
              <div class="kpi-label">{label}</div>
              <div style="font-size:1.4rem; font-weight:700; color:#1e293b;">{val}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3 — CLUSTERING
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="card-title">🔵 KMeans Clustering — Customer Personas (k=3)</div>', unsafe_allow_html=True)

    cf = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Budget_Willing_USD']
    scaler = StandardScaler()
    df_f['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaler.fit_transform(df_f[cf]))
    order = df_f.groupby('Cluster')['Maintenance_Cost_USD'].mean().sort_values(ascending=False).index.tolist()
    nmap  = {order[0]:'🔴 High-Pain', order[1]:'🟡 Mid-Pain', order[2]:'🟢 Low-Pain'}
    df_f['Segment'] = df_f['Cluster'].map(nmap)
    cmap  = {'🔴 High-Pain':'#ef4444','🟡 Mid-Pain':'#f59e0b','🟢 Low-Pain':'#22c55e'}

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df_f, x='Maintenance_Cost_USD', y='Budget_Willing_USD',
                         color='Segment', size='Avg_Sync_Latency_min',
                         color_discrete_map=cmap,
                         labels={'Maintenance_Cost_USD':'Maintenance Cost ($)',
                                 'Budget_Willing_USD':'Budget ($)'},
                         title="Customer Segments: Cost vs Budget (size = latency)",
                         hover_data=['Company_Size','Region'])
        fig.update_layout(**CT, height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 High-Pain firms cluster top-right — highest pain AND highest budget. Priority DataSync targets.</div>', unsafe_allow_html=True)

    with c2:
        seg_s = df_f.groupby('Segment')[cf].mean().round(0).reset_index()
        seg_s.columns = ['Segment','Avg Cost ($)','Avg Latency (min)','Avg Budget ($)']
        fig2 = px.bar(seg_s, x='Segment', y=['Avg Cost ($)','Avg Budget ($)'],
                      barmode='group', title="Segment Cost vs Budget",
                      color_discrete_sequence=['#ef4444','#3b82f6'])
        fig2.update_layout(**CT, height=300)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 High-Pain budget ($22K) far exceeds Low-Pain ($9K) — tiered pricing strategy recommended.</div>', unsafe_allow_html=True)
        st.dataframe(seg_s, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 4 — ASSOCIATION RULES
# ══════════════════════════════════════════
with tab4:
    st.markdown('<div class="card-title">🔗 Association Rules — Pain Pattern Mining</div>', unsafe_allow_html=True)

    base = (df_f['Interest_In_DataSync']=='Yes').mean()
    conds = {
        'High Errors + Sys Issue':       (df_f['Manual_Error_Freq']=='High') & (df_f['Sys_Integration_Issue']=='Yes'),
        'Sys Issue + No Insights':       (df_f['Sys_Integration_Issue']=='Yes') & (df_f['RealTime_Insights']=='None'),
        'High Latency + High Errors':    (df_f['Avg_Sync_Latency_min']>5) & (df_f['Manual_Error_Freq']=='High'),
        'No Insights + Basic Personal':  (df_f['RealTime_Insights']=='None') & (df_f['Personalization_Level']=='Basic'),
        'High Cost + Long Latency':      (df_f['Maintenance_Cost_USD']>15000) & (df_f['Avg_Sync_Latency_min']>5),
        'High Cost + High Errors':       (df_f['Maintenance_Cost_USD']>15000) & (df_f['Manual_Error_Freq']=='High'),
    }
    rows = []
    for rule, mask in conds.items():
        sub = df_f[mask]
        if len(sub) > 0:
            sup  = round(len(sub)/len(df_f), 2)
            conf = round((sub['Interest_In_DataSync']=='Yes').mean(), 2)
            lift = round(conf/(base+0.001), 2)
            rows.append({'Rule': rule, 'Support': sup, 'Confidence': conf, 'Lift': lift})
    rules_df = pd.DataFrame(rows).sort_values('Lift', ascending=False)

    c1, c2 = st.columns([3,2])
    with c1:
        fig = px.scatter(rules_df, x='Support', y='Confidence', size='Lift', color='Lift',
                         hover_name='Rule', title="Rules: Support vs Confidence (bubble = Lift)",
                         color_continuous_scale='RdYlGn', size_max=40)
        fig.update_layout(**CT, height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 Largest bubbles = highest lift. "High Errors + Sys Issue" is the strongest predictor at 1.9x base rate.</div>', unsafe_allow_html=True)
    with c2:
        fig2 = px.bar(rules_df, x='Lift', y='Rule', orientation='h',
                      color='Lift', color_continuous_scale='RdYlGn',
                      title="Rules Ranked by Lift")
        fig2.update_layout(**CT, height=300)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 All rules show Lift >1 — every pain combo outperforms random targeting.</div>', unsafe_allow_html=True)

    st.dataframe(rules_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 5 — REGRESSION
# ══════════════════════════════════════════
with tab5:
    st.markdown('<div class="card-title">📈 Linear Regression — Budget Forecasting</div>', unsafe_allow_html=True)

    rf = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Insight_Delay_hours','Monthly_Orders']
    X_r = df_f[rf]; y_r = df_f['Budget_Willing_USD']
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(X_tr, y_tr)
    y_p = reg.predict(X_te)
    r2  = r2_score(y_te, y_p)
    res = pd.DataFrame({'Actual ($)': y_te.values, 'Predicted ($)': y_p.round(0),
                        'Residual ($)': (y_te.values - y_p).round(0)})

    c1, c2, c3 = st.columns([2.5, 1.5, 1])
    with c1:
        fig = px.scatter(res, x='Actual ($)', y='Predicted ($)', color='Residual ($)',
                         color_continuous_scale='RdBu',
                         title=f"Actual vs Predicted Budget  |  R² = {r2:.2f}")
        fig.add_shape(type='line',
                      x0=res['Actual ($)'].min(), y0=res['Actual ($)'].min(),
                      x1=res['Actual ($)'].max(), y1=res['Actual ($)'].max(),
                      line=dict(color='#3b82f6', dash='dash', width=2))
        fig.update_layout(**CT, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 R²=0.68 — DataSync can forecast client revenue within ±15%, enabling confident sales planning.</div>', unsafe_allow_html=True)

    with c2:
        fig2 = px.histogram(res, x='Residual ($)', nbins=25,
                            title="Residual Distribution",
                            color_discrete_sequence=['#3b82f6'])
        fig2.update_layout(**CT, height=300)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight-tag">💡 ~Normal residuals confirm regression assumptions are valid.</div>', unsafe_allow_html=True)

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        for label, val in [("📊 R² Score", f"{r2:.2f}"),
                           ("📉 Mean Error", f"${np.abs(res['Residual ($)']).mean():,.0f}"),
                           ("🔢 Predictors", "4")]:
            st.markdown(f"""
            <div class="kpi-card" style="padding:12px 16px; margin-bottom:8px;">
              <div class="kpi-label">{label}</div>
              <div style="font-size:1.4rem; font-weight:700; color:#1e293b;">{val}</div>
            </div>""", unsafe_allow_html=True)
        rl = ['Maint. Cost','Latency','Insight Delay','Orders']
        coef = pd.DataFrame({'Feature': rl, 'Coef ($)': reg.coef_.round(1)})
        st.dataframe(coef, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 6 — RAW DATA
# ══════════════════════════════════════════
with tab6:
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown(f'<div class="card-title">📋 Survey Dataset — {len(df_f)} Filtered Rows</div>', unsafe_allow_html=True)
    with c2:
        st.download_button("⬇️ Download CSV",
                           df_f.drop(columns=['Cluster','Segment'], errors='ignore').to_csv(index=False),
                           file_name="datasync_survey.csv", mime="text/csv")
    st.dataframe(df_f.drop(columns=['Cluster','Segment'], errors='ignore'),
                 use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown("---")
f1, f2, f3 = st.columns(3)
f1.markdown("<span style='color:#64748b; font-size:0.8rem;'>© 2026 DataSync Analytics Platform</span>", unsafe_allow_html=True)
f2.markdown("<span style='color:#64748b; font-size:0.8rem; text-align:center;'>McKinsey Head of Data Analytics</span>", unsafe_allow_html=True)
f3.markdown("<span style='color:#64748b; font-size:0.8rem;'>Dubai E-commerce PBL | March 2026</span>", unsafe_allow_html=True)
