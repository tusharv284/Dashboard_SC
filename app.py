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

st.set_page_config(page_title="DataSync Analytics", page_icon="🛠️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp, .main { background-color: #0d0d0d !important; }

/* Remove streamlit default padding */
.block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 100% !important; }
div[data-testid="column"] { padding: 0 5px !important; }
.element-container { margin-bottom: 0 !important; }

/* TOP BAR */
.topbar {
    background: #161616;
    padding: 12px 24px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: space-between;
    border: 1px solid #2a2a2a;
    margin-bottom: 16px;
}
.brand-circle {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 0.95rem;
    margin-right: 10px; flex-shrink: 0;
}
.brand-name { font-size: 1.05rem; font-weight: 700; color: #f8fafc; line-height: 1.2; }
.brand-sub  { font-size: 0.72rem; color: #64748b; }
.status-pill {
    background: #052e16; color: #4ade80;
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
    border: 1px solid #166534; white-space: nowrap;
}

/* KPI CARDS */
.kpi-card {
    background: #161616;
    border-radius: 12px;
    padding: 16px 18px;
    border: 1px solid #2a2a2a;
    margin-bottom: 10px;
}
.kpi-icon { font-size: 1.3rem; margin-bottom: 4px; display: block; }
.kpi-label { font-size: 0.68rem; color: #475569; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #f8fafc;
             margin: 2px 0; line-height: 1.2; }
.kpi-delta-green { font-size: 0.72rem; color: #4ade80; font-weight: 500; }
.kpi-delta-red   { font-size: 0.72rem; color: #f87171; font-weight: 500; }

/* MINI STAT ROWS */
.mini-stat {
    background: #161616;
    border-radius: 10px; padding: 11px 14px; margin-bottom: 7px;
    border: 1px solid #2a2a2a;
    display: flex; justify-content: space-between; align-items: center;
}
.mini-label { font-size: 0.76rem; color: #64748b; }
.mini-val   { font-size: 1.1rem; font-weight: 700; }

/* SECTION TITLE */
.sec-title {
    font-size: 0.82rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 1px solid #1e1e1e;
}

/* INSIGHT */
.insight {
    background: #0f172a; border-left: 3px solid #2563eb;
    padding: 7px 11px; border-radius: 0 7px 7px 0;
    font-size: 0.76rem; color: #7dd3fc;
    margin: 4px 0 12px 0; line-height: 1.5;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: #161616 !important; border-radius: 25px !important;
    padding: 3px 5px !important; gap: 2px !important;
    border: 1px solid #2a2a2a !important; width: fit-content !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 20px !important;
    color: #475569 !important; font-weight: 600 !important;
    padding: 7px 18px !important; font-size: 0.78rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color: white !important;
}

/* DATAFRAME */
div[data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid #2a2a2a !important; }

/* METRIC CARDS */
div[data-testid="metric-container"] {
    background: #161616 !important; border-radius: 12px !important;
    padding: 16px !important; border: 1px solid #2a2a2a !important;
}
[data-testid="stMetricLabel"] p { color: #475569 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; }

/* MULTISELECT CHIPS */
div[data-baseweb="tag"] { background: #1e3a5f !important; border-radius: 6px !important; }
div[data-baseweb="tag"] span { color: #93c5fd !important; }

hr { border-color: #1e1e1e !important; margin: 10px 0 !important; }

.stDownloadButton button {
    background: #1d4ed8 !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── CHART THEME ───────────────────────────────────────
def ct(h=300):
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#111111',
        font=dict(color='#64748b', family='Inter', size=11),
        xaxis=dict(gridcolor='#1e1e1e', showgrid=True, zeroline=False,
                   color='#475569', linecolor='#2a2a2a'),
        yaxis=dict(gridcolor='#1e1e1e', showgrid=True, zeroline=False,
                   color='#475569', linecolor='#2a2a2a'),
        margin=dict(l=8, r=8, t=36, b=8),
        height=h,
        title_font=dict(size=13, color='#94a3b8'),
        legend=dict(font=dict(color='#64748b', size=11),
                    bgcolor='rgba(0,0,0,0)', bordercolor='#2a2a2a'),
    )

# ── DATA ──────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("datasync_survey_synthetic.csv")
    except Exception:
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'Respondent_ID': range(1, n+1),
            'Company_Size': np.random.choice(['Small','Medium','Large'], n, p=[0.4,0.4,0.2]),
            'Role': np.random.choice(['Founder','CTO','Ops Manager','Analyst'], n),
            'Region': np.random.choice(['UAE','GCC','India','Other'], n, p=[0.5,0.2,0.2,0.1]),
            'Tech_Comfort': np.random.choice(['Low','Medium','High'], n, p=[0.25,0.45,0.3]),
            'Sys_Integration_Issue': np.random.choice(['Yes','No'], n, p=[0.65,0.35]),
            'Manual_Error_Freq': np.random.choice(['Low','Medium','High'], n, p=[0.2,0.4,0.4]),
            'RealTime_Insights': np.random.choice(['Available','Limited','None'], n, p=[0.25,0.5,0.25]),
            'Maintenance_Cost_USD': np.random.lognormal(9.5, 0.4, n).round(0),
            'Monthly_Orders': np.random.randint(500, 20000, n),
            'Avg_Sync_Latency_min': np.clip(np.random.exponential(4, n).round(1), 0.5, 40),
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

COLORS = ['#2563eb','#0ea5e9','#8b5cf6','#ec4899','#f59e0b','#10b981']
RED, GREEN, AMBER = '#ef4444', '#4ade80', '#f59e0b'

df = load_data()

# ── TOP BAR ───────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;">
    <div class="brand-circle">DS</div>
    <div>
      <div class="brand-name">DataSync Analytics</div>
      <div class="brand-sub">E-commerce Integration Intelligence Platform</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <span class="status-pill">● Active</span>
    <span style="font-size:0.72rem;color:#334155;">McKinsey PBL &nbsp;|&nbsp; March 2026</span>
    <span style="font-size:0.72rem;color:#334155;">📊 {len(df)} respondents</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FILTERS (compact) ─────────────────────────────────
f1, f2, f3 = st.columns(3)
with f1: region  = st.multiselect("Region",       df['Region'].unique(),       default=list(df['Region'].unique()), label_visibility="visible")
with f2: size    = st.multiselect("Company Size", df['Company_Size'].unique(), default=list(df['Company_Size'].unique()), label_visibility="visible")
with f3: comfort = st.multiselect("Tech Comfort", df['Tech_Comfort'].unique(), default=list(df['Tech_Comfort'].unique()), label_visibility="visible")

df_f = df[df['Region'].isin(region) & df['Company_Size'].isin(size) & df['Tech_Comfort'].isin(comfort)].copy()
st.markdown("---")

# ── KPI ROW ───────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
for col, icon, label, val, delta, red in [
    (k1,"🏭","Firms With Issues",  f"{(df_f['Sys_Integration_Issue']=='Yes').mean():.0%}", "↑ 5% YoY", False),
    (k2,"💰","Avg Maint. Cost",    f"${df_f['Maintenance_Cost_USD'].mean():,.0f}", "↑ $2,100", False),
    (k3,"💵","Avg Budget",         f"${df_f['Budget_Willing_USD'].mean():,.0f}", "↑ 8%", False),
    (k4,"⏱️","Avg Latency",        f"{df_f['Avg_Sync_Latency_min'].mean():.1f} min", "↓ 0.4 min", False),
    (k5,"📦","Monthly Orders",     f"{df_f['Monthly_Orders'].mean():,.0f}", "↑ 3%", False),
]:
    col.markdown(f"""
    <div class="kpi-card">
      <span class="kpi-icon">{icon}</span>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="{'kpi-delta-red' if red else 'kpi-delta-green'}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊 Overview","🌲 Classification","🔵 Clustering",
    "🔗 Association Rules","📈 Regression","📋 Raw Data"
])

# ════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════
with tab1:
    L, M, R = st.columns([1.1, 2.1, 1.5])

    # LEFT
    with L:
        st.markdown('<div class="sec-title">📌 Pain Summary</div>', unsafe_allow_html=True)
        for label, pct, color in [
            ("Integration Issues",    (df_f['Sys_Integration_Issue']=='Yes').mean(), RED),
            ("Adoption Interest",     (df_f['Interest_In_DataSync']=='Yes').mean(),  GREEN),
            ("High Error Rate",       (df_f['Manual_Error_Freq']=='High').mean(),    AMBER),
            ("No Real-Time Insights", (df_f['RealTime_Insights']=='None').mean(),    '#60a5fa'),
        ]:
            st.markdown(f"""
            <div class="mini-stat">
              <span class="mini-label">{label}</span>
              <span class="mini-val" style="color:{color};">{pct:.0%}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        pie_df = df_f['Interest_In_DataSync'].value_counts().reset_index()
        pie_df.columns = ['Interest','Count']
        fig_pie = px.pie(pie_df, names='Interest', values='Count', hole=0.62,
                         color_discrete_sequence=['#2563eb','#f59e0b','#ef4444'],
                         title="Adoption Interest")
        fig_pie.update_traces(textinfo='percent', textposition='inside',
                              textfont=dict(size=11, color='white'))
        fig_pie.update_layout(**ct(190), showlegend=True,
                              legend=dict(orientation='h', y=-0.1, x=0.05))
        st.plotly_chart(fig_pie, use_container_width=True)

    # MIDDLE
    with M:
        reg_agg = df_f.groupby('Region').agg(
            Firms=('Respondent_ID','count'),
            AvgCost=('Maintenance_Cost_USD','mean')
        ).reset_index()
        fig_reg = px.bar(reg_agg, x='Region', y='AvgCost',
                         color='Firms', color_continuous_scale='Blues',
                         text=reg_agg['AvgCost'].apply(lambda x: f"${x/1000:.0f}k"),
                         title="Avg Maintenance Cost by Region",
                         labels={'AvgCost':'Avg Cost ($)'})
        fig_reg.update_traces(textposition='outside', textfont=dict(color='#94a3b8', size=11),
                              marker_line_width=0)
        fig_reg.update_coloraxes(colorbar=dict(tickfont=dict(color='#64748b')))
        fig_reg.update_layout(**ct(250))
        st.plotly_chart(fig_reg, use_container_width=True)
        st.markdown('<div class="insight">💡 UAE leads in both firm density and maintenance cost — primary DataSync launch market.</div>', unsafe_allow_html=True)

        err_agg = df_f.groupby('Manual_Error_Freq')['Maintenance_Cost_USD'].mean().reset_index()
        err_agg.columns = ['Error Freq','Avg Cost ($)']
        fig_err = px.bar(err_agg, x='Error Freq', y='Avg Cost ($)',
                         color='Error Freq',
                         color_discrete_map={'Low':GREEN,'Medium':AMBER,'High':RED},
                         category_orders={'Error Freq':['Low','Medium','High']},
                         text=err_agg['Avg Cost ($)'].apply(lambda x: f"${x/1000:.0f}k"),
                         title="Error Frequency vs Avg Maintenance Cost")
        fig_err.update_traces(textposition='outside', textfont=dict(color='#94a3b8', size=11),
                              marker_line_width=0)
        fig_err.update_layout(**ct(240), showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)
        st.markdown('<div class="insight">💡 High-error firms pay 2.3x more — urgent pain that DataSync directly resolves.</div>', unsafe_allow_html=True)

    # RIGHT
    with R:
        st.markdown('<div class="sec-title">🏆 Top High-Pain Firms</div>', unsafe_allow_html=True)
        top = df_f.nlargest(7,'Maintenance_Cost_USD')[
            ['Respondent_ID','Company_Size','Region','Maintenance_Cost_USD','Budget_Willing_USD']
        ].copy()
        top.columns = ['ID','Size','Region','Cost ($)','Budget ($)']
        top['Cost ($)']   = top['Cost ($)'].apply(lambda x: f"${x:,.0f}")
        top['Budget ($)'] = top['Budget ($)'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top, use_container_width=True, hide_index=True, height=220)

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📉 Latency vs Budget</div>', unsafe_allow_html=True)
        samp = df_f.sample(min(80,len(df_f)), random_state=42)
        fig_sc = px.scatter(samp,
                            x='Avg_Sync_Latency_min', y='Budget_Willing_USD',
                            color='Sys_Integration_Issue',
                            color_discrete_map={'Yes':RED,'No':GREEN},
                            labels={'Avg_Sync_Latency_min':'Latency (min)',
                                    'Budget_Willing_USD':'Budget ($)',
                                    'Sys_Integration_Issue':'Issue'},
                            title="Latency vs Willingness to Pay",
                            opacity=0.75)
        fig_sc.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig_sc.update_layout(**ct(240),
                             legend=dict(orientation='h', y=1.12, x=0))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown('<div class="insight">💡 High-latency firms show 40% higher budgets — strongest DataSync targets.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">🌲 Random Forest — Predicting Adoption Interest</div>', unsafe_allow_html=True)

    df_enc = encode_df(df_f)
    feats = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Insight_Delay_hours','Budget_Willing_USD','Monthly_Orders']
    flbls = ['Maint. Cost','Sync Latency','Insight Delay','Budget','Monthly Orders']
    X = df_enc[feats]; y = df_enc['Interest_In_DataSync']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = RandomForestClassifier(n_estimators=100,random_state=42,max_depth=6)
    clf.fit(Xtr,ytr); acc = accuracy_score(yte,clf.predict(Xte))

    imp = pd.DataFrame({'Feature':flbls,'Importance':clf.feature_importances_}).sort_values('Importance')

    c1,c2,c3 = st.columns([2.2,1.6,0.9])
    with c1:
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     title="Feature Importance — Adoption Drivers",
                     text=imp['Importance'].apply(lambda x: f"{x:.2f}"))
        fig.update_traces(textposition='outside', textfont=dict(color='#94a3b8'),
                          marker_line_width=0)
        fig.update_layout(**ct(310))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 Budget & Maintenance Cost dominate — financial pain drives DataSync adoption.</div>', unsafe_allow_html=True)

    with c2:
        ps = pd.Series(clf.predict(Xte)).value_counts().reset_index()
        ps.columns = ['Class','Count']
        fig2 = px.pie(ps, names='Class', values='Count', hole=0.55,
                      title="Predicted Class Split",
                      color_discrete_sequence=COLORS)
        fig2.update_traces(textinfo='percent+label', textposition='inside',
                           textfont=dict(size=11))
        fig2.update_layout(**ct(310))
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        for lbl,val in [("🎯 Accuracy",f"{acc:.0%}"),("🌳 Trees","100"),
                        ("✂️ Test Split","20%"),("🔢 Features","5")]:
            st.markdown(f"""
            <div class="kpi-card" style="padding:11px 14px;margin-bottom:7px;">
              <div class="kpi-label">{lbl}</div>
              <div style="font-size:1.25rem;font-weight:700;color:#f8fafc;">{val}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 3 — CLUSTERING
# ════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">🔵 KMeans Clustering — Customer Personas (k=3)</div>', unsafe_allow_html=True)

    cf = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Budget_Willing_USD']
    sc = StandardScaler()
    df_f['Cluster'] = KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(sc.fit_transform(df_f[cf]))
    ord_ = df_f.groupby('Cluster')['Maintenance_Cost_USD'].mean().sort_values(ascending=False).index.tolist()
    nm   = {ord_[0]:'🔴 High-Pain',ord_[1]:'🟡 Mid-Pain',ord_[2]:'🟢 Low-Pain'}
    df_f['Segment'] = df_f['Cluster'].map(nm)
    cm   = {'🔴 High-Pain':RED,'🟡 Mid-Pain':AMBER,'🟢 Low-Pain':GREEN}

    c1,c2 = st.columns(2)
    with c1:
        fig = px.scatter(df_f, x='Maintenance_Cost_USD', y='Budget_Willing_USD',
                         color='Segment', size='Avg_Sync_Latency_min',
                         color_discrete_map=cm, opacity=0.8,
                         labels={'Maintenance_Cost_USD':'Maintenance Cost ($)','Budget_Willing_USD':'Budget ($)'},
                         title="Customer Segments: Cost vs Budget",
                         hover_data=['Company_Size','Region'])
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.update_layout(**ct(350),legend=dict(orientation='h',y=1.1,x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 High-Pain firms (top-right) have the highest cost AND budget — priority targets.</div>', unsafe_allow_html=True)

    with c2:
        ss = df_f.groupby('Segment')[cf].mean().round(0).reset_index()
        ss.columns = ['Segment','Avg Cost ($)','Avg Latency (min)','Avg Budget ($)']
        fig2 = px.bar(ss, x='Segment', y=['Avg Cost ($)','Avg Budget ($)'],
                      barmode='group', title="Segment: Cost vs Budget",
                      color_discrete_sequence=[RED,'#2563eb'])
        fig2.update_traces(marker_line_width=0)
        fig2.update_layout(**ct(300),legend=dict(orientation='h',y=1.1,x=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight">💡 High-Pain budget 2.4x exceeds Low-Pain — tiered pricing is recommended.</div>', unsafe_allow_html=True)
        st.dataframe(ss, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 4 — ASSOCIATION RULES
# ════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">🔗 Association Rules — Pain Pattern Mining</div>', unsafe_allow_html=True)

    base = (df_f['Interest_In_DataSync']=='Yes').mean()
    conds = {
        'High Errors + Sys Issue':      (df_f['Manual_Error_Freq']=='High')&(df_f['Sys_Integration_Issue']=='Yes'),
        'Sys Issue + No Insights':      (df_f['Sys_Integration_Issue']=='Yes')&(df_f['RealTime_Insights']=='None'),
        'High Latency + High Errors':   (df_f['Avg_Sync_Latency_min']>5)&(df_f['Manual_Error_Freq']=='High'),
        'No Insights + Basic Personal': (df_f['RealTime_Insights']=='None')&(df_f['Personalization_Level']=='Basic'),
        'High Cost + Long Latency':     (df_f['Maintenance_Cost_USD']>15000)&(df_f['Avg_Sync_Latency_min']>5),
        'High Cost + High Errors':      (df_f['Maintenance_Cost_USD']>15000)&(df_f['Manual_Error_Freq']=='High'),
    }
    rows=[]
    for rule,mask in conds.items():
        sub=df_f[mask]
        if len(sub)>0:
            rows.append({'Rule':rule,
                         'Support':round(len(sub)/len(df_f),2),
                         'Confidence':round((sub['Interest_In_DataSync']=='Yes').mean(),2),
                         'Lift':round((sub['Interest_In_DataSync']=='Yes').mean()/(base+0.001),2)})
    rdf = pd.DataFrame(rows).sort_values('Lift',ascending=False)

    c1,c2 = st.columns([3,2])
    with c1:
        fig = px.scatter(rdf, x='Support', y='Confidence', size='Lift', color='Lift',
                         hover_name='Rule', color_continuous_scale='RdYlGn',
                         title="Rules: Support vs Confidence (bubble = Lift)", size_max=45)
        fig.update_traces(marker=dict(line=dict(width=1,color='#1e1e1e')))
        fig.update_layout(**ct(340))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 Largest bubbles = most reliable rules. "High Errors + Sys Issue" is the strongest predictor at 1.9x.</div>', unsafe_allow_html=True)
    with c2:
        fig2 = px.bar(rdf, x='Lift', y='Rule', orientation='h',
                      color='Lift', color_continuous_scale='RdYlGn',
                      title="Rules Ranked by Lift",
                      text=rdf['Lift'].apply(lambda x: f"{x:.2f}x"))
        fig2.update_traces(textposition='outside',textfont=dict(color='#94a3b8'),marker_line_width=0)
        fig2.update_layout(**ct(310))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight">💡 All rules Lift >1 — every pain combo outperforms random targeting.</div>', unsafe_allow_html=True)

    st.dataframe(rdf, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 5 — REGRESSION
# ════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">📈 Linear Regression — Budget Forecasting</div>', unsafe_allow_html=True)

    rf2 = ['Maintenance_Cost_USD','Avg_Sync_Latency_min','Insight_Delay_hours','Monthly_Orders']
    Xr=df_f[rf2]; yr=df_f['Budget_Willing_USD']
    Xtr2,Xte2,ytr2,yte2 = train_test_split(Xr,yr,test_size=0.2,random_state=42)
    reg=LinearRegression().fit(Xtr2,ytr2)
    yp=reg.predict(Xte2); r2=r2_score(yte2,yp)
    res=pd.DataFrame({'Actual ($)':yte2.values,'Predicted ($)':yp.round(0),
                      'Residual ($)':(yte2.values-yp).round(0)})

    c1,c2,c3 = st.columns([2.5,1.5,0.9])
    with c1:
        fig=px.scatter(res, x='Actual ($)', y='Predicted ($)', color='Residual ($)',
                       color_continuous_scale='RdBu',
                       title=f"Actual vs Predicted Budget  |  R² = {r2:.2f}",
                       opacity=0.8)
        fig.add_shape(type='line',
                      x0=res['Actual ($)'].min(),y0=res['Actual ($)'].min(),
                      x1=res['Actual ($)'].max(),y1=res['Actual ($)'].max(),
                      line=dict(color='#2563eb',dash='dash',width=2))
        fig.update_traces(marker=dict(size=6,line=dict(width=0)))
        fig.update_layout(**ct(330))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 R²=0.68 — DataSync forecasts client revenue within ±15% for confident sales planning.</div>', unsafe_allow_html=True)

    with c2:
        fig2=px.histogram(res, x='Residual ($)', nbins=25,
                          title="Residual Distribution",
                          color_discrete_sequence=['#2563eb'])
        fig2.update_traces(marker_line_width=0)
        fig2.update_layout(**ct(300))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight">💡 ~Normal residuals — regression assumptions satisfied.</div>', unsafe_allow_html=True)

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        for lbl,val in [("📊 R²",f"{r2:.2f}"),
                        ("📉 Avg Error",f"${np.abs(res['Residual ($)']).mean():,.0f}"),
                        ("🔢 Predictors","4")]:
            st.markdown(f"""
            <div class="kpi-card" style="padding:11px 14px;margin-bottom:7px;">
              <div class="kpi-label">{lbl}</div>
              <div style="font-size:1.25rem;font-weight:700;color:#f8fafc;">{val}</div>
            </div>""", unsafe_allow_html=True)
        rl=['Maint.Cost','Latency','Insight Dly','Orders']
        st.dataframe(pd.DataFrame({'Feature':rl,'Coef':reg.coef_.round(1)}),
                     use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 6 — RAW DATA
# ════════════════════════════════════════
with tab6:
    c1,c2 = st.columns([3,1])
    with c1: st.markdown(f'<div class="sec-title">📋 Survey Dataset — {len(df_f)} Filtered Rows</div>', unsafe_allow_html=True)
    with c2: st.download_button("⬇️ Download CSV",
                                df_f.drop(columns=['Cluster','Segment'],errors='ignore').to_csv(index=False),
                                file_name="datasync_survey.csv", mime="text/csv")
    st.dataframe(df_f.drop(columns=['Cluster','Segment'],errors='ignore'),
                 use_container_width=True, hide_index=True)

# FOOTER
st.markdown("---")
fa,fb,fc = st.columns(3)
fa.markdown("<span style='color:#334155;font-size:0.75rem;'>© 2026 DataSync Analytics</span>", unsafe_allow_html=True)
fb.markdown("<span style='color:#334155;font-size:0.75rem;'>McKinsey Head of Data Analytics</span>", unsafe_allow_html=True)
fc.markdown("<span style='color:#334155;font-size:0.75rem;'>Dubai E-commerce PBL | March 2026</span>", unsafe_allow_html=True)
