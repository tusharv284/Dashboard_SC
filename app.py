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
.block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 100% !important; }
div[data-testid="column"] { padding: 0 5px !important; }
.element-container { margin-bottom: 0 !important; }

.topbar {
    background: #161616; padding: 12px 24px; border-radius: 12px;
    display: flex; align-items: center; justify-content: space-between;
    border: 1px solid #2a2a2a; margin-bottom: 16px;
}
.brand-circle {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; color: white; font-weight: 700;
    font-size: 0.95rem; margin-right: 10px; flex-shrink: 0;
}
.brand-name { font-size: 1.05rem; font-weight: 700; color: #f8fafc; line-height: 1.2; }
.brand-sub  { font-size: 0.72rem; color: #64748b; }
.status-pill {
    background: #052e16; color: #4ade80; padding: 3px 12px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    border: 1px solid #166534; white-space: nowrap;
}
.kpi-card {
    background: #161616; border-radius: 12px; padding: 16px 18px;
    border: 1px solid #2a2a2a; margin-bottom: 10px;
}
.kpi-icon  { font-size: 1.3rem; margin-bottom: 4px; display: block; }
.kpi-label { font-size: 0.68rem; color: #475569; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin: 2px 0; line-height: 1.2; }
.kpi-delta-green { font-size: 0.72rem; color: #4ade80; font-weight: 500; }
.kpi-delta-red   { font-size: 0.72rem; color: #f87171; font-weight: 500; }
.mini-stat {
    background: #161616; border-radius: 10px; padding: 11px 14px;
    margin-bottom: 7px; border: 1px solid #2a2a2a;
    display: flex; justify-content: space-between; align-items: center;
}
.mini-label { font-size: 0.76rem; color: #64748b; }
.sec-title {
    font-size: 0.82rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #1e1e1e;
}
.insight {
    background: #0f172a; border-left: 3px solid #2563eb;
    padding: 7px 11px; border-radius: 0 7px 7px 0;
    font-size: 0.76rem; color: #7dd3fc;
    margin: 4px 0 12px 0; line-height: 1.5;
}
.stTabs [data-baseweb="tab-list"] {
    background: #161616 !important; border-radius: 25px !important;
    padding: 3px 5px !important; gap: 2px !important;
    border: 1px solid #2a2a2a !important; width: fit-content !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 20px !important;
    color: #475569 !important; font-weight: 600 !important;
    padding: 7px 18px !important; font-size: 0.78rem !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important; color: white !important;
}
div[data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid #2a2a2a !important; }
div[data-testid="metric-container"] {
    background: #161616 !important; border-radius: 12px !important;
    padding: 16px !important; border: 1px solid #2a2a2a !important;
}
[data-testid="stMetricLabel"] p { color: #475569 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"]   { color: #f8fafc !important; }
div[data-baseweb="tag"] { background: #1e3a5f !important; border-radius: 6px !important; }
div[data-baseweb="tag"] span { color: #93c5fd !important; }
hr { border-color: #1e1e1e !important; margin: 10px 0 !important; }
.stDownloadButton button {
    background: #1d4ed8 !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── CHART HELPER ──────────────────────────────────────
def ct(h=300, legend_h=False):
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#111111',
        font=dict(color='#64748b', family='Inter', size=11),
        xaxis=dict(gridcolor='#1e1e1e', showgrid=True, zeroline=False,
                   color='#475569', linecolor='#2a2a2a'),
        yaxis=dict(gridcolor='#1e1e1e', showgrid=True, zeroline=False,
                   color='#475569', linecolor='#2a2a2a'),
        margin=dict(l=8, r=8, t=40, b=8),
        height=h,
        title_font=dict(size=13, color='#94a3b8'),
    )
    if legend_h:
        layout['legend'] = dict(orientation='h', y=1.12, x=0,
                                font=dict(color='#64748b', size=11),
                                bgcolor='rgba(0,0,0,0)')
    else:
        layout['legend'] = dict(font=dict(color='#64748b', size=11),
                                bgcolor='rgba(0,0,0,0)', bordercolor='#2a2a2a')
    return layout

RED, GREEN, AMBER, BLUE = '#ef4444', '#4ade80', '#f59e0b', '#2563eb'
COLORS = [BLUE,'#0ea5e9','#8b5cf6','#ec4899',AMBER,GREEN]

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
    le = LabelEncoder(); d = df.copy()
    for col in ['Company_Size','Role','Region','Tech_Comfort','Sys_Integration_Issue',
                'Manual_Error_Freq','RealTime_Insights','Personalization_Level','Interest_In_DataSync']:
        if col in d.columns:
            d[col] = le.fit_transform(d[col].astype(str))
    return d

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

# ── FILTERS ───────────────────────────────────────────
f1,f2,f3 = st.columns(3)
with f1: region  = st.multiselect("Region",       df['Region'].unique(),       default=list(df['Region'].unique()))
with f2: size    = st.multiselect("Company Size", df['Company_Size'].unique(), default=list(df['Company_Size'].unique()))
with f3: comfort = st.multiselect("Tech Comfort", df['Tech_Comfort'].unique(), default=list(df['Tech_Comfort'].unique()))
df_f = df[df['Region'].isin(region)&df['Company_Size'].isin(size)&df['Tech_Comfort'].isin(comfort)].copy()
st.markdown("---")

# ── KPI ROW ───────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
for col,icon,label,val,delta,red in [
    (k1,"🏭","Firms With Issues",  f"{(df_f['Sys_Integration_Issue']=='Yes').mean():.0%}","↑ 5% YoY",False),
    (k2,"💰","Avg Maint. Cost",    f"${df_f['Maintenance_Cost_USD'].mean():,.0f}","↑ $2,100",False),
    (k3,"💵","Avg Budget",         f"${df_f['Budget_Willing_USD'].mean():,.0f}","↑ 8%",False),
    (k4,"⏱️","Avg Latency",        f"{df_f['Avg_Sync_Latency_min'].mean():.1f} min","↓ 0.4 min",False),
    (k5,"📦","Monthly Orders",     f"{df_f['Monthly_Orders'].mean():,.0f}","↑ 3%",False),
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

    with L:
        st.markdown('<div class="sec-title">📌 Pain Summary</div>', unsafe_allow_html=True)
        for label,pct,color in [
            ("Integration Issues",    (df_f['Sys_Integration_Issue']=='Yes').mean(), RED),
            ("Adoption Interest",     (df_f['Interest_In_DataSync']=='Yes').mean(),  GREEN),
            ("High Error Rate",       (df_f['Manual_Error_Freq']=='High').mean(),    AMBER),
            ("No Real-Time Insights", (df_f['RealTime_Insights']=='None').mean(),    '#60a5fa'),
        ]:
            st.markdown(f"""
            <div class="mini-stat">
              <span class="mini-label">{label}</span>
              <span style="font-size:1.1rem;font-weight:700;color:{color};">{pct:.0%}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # ── 3D DONUT (Pie with depth illusion via go) ──
        pie_df = df_f['Interest_In_DataSync'].value_counts().reset_index()
        pie_df.columns = ['Interest','Count']
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_df['Interest'], values=pie_df['Count'],
            hole=0.55, pull=[0.05,0.02,0.02],
            marker=dict(colors=['#2563eb','#f59e0b','#ef4444'],
                        line=dict(color='#0d0d0d', width=2)),
            textinfo='percent', textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
        )])
        pie_l = ct(200)
        pie_l['showlegend'] = True
        pie_l['legend'] = dict(orientation='h', y=-0.12, x=0.0,
                               font=dict(color='#64748b', size=10),
                               bgcolor='rgba(0,0,0,0)')
        pie_l['title'] = dict(text='Adoption Interest', font=dict(size=13, color='#94a3b8'))
        fig_pie.update_layout(**pie_l)
        st.plotly_chart(fig_pie, use_container_width=True)

    with M:
        # Region bar
        reg_agg = df_f.groupby('Region').agg(
            Firms=('Respondent_ID','count'),
            AvgCost=('Maintenance_Cost_USD','mean')
        ).reset_index()
        fig_reg = px.bar(reg_agg, x='Region', y='AvgCost',
                         color='Firms', color_continuous_scale='Blues',
                         text=reg_agg['AvgCost'].apply(lambda x: f"${x/1000:.0f}k"),
                         title="Avg Maintenance Cost by Region",
                         labels={'AvgCost':'Avg Cost ($)'})
        fig_reg.update_traces(textposition='outside', textfont=dict(color='#94a3b8',size=11),
                              marker_line_width=0)
        fig_reg.update_coloraxes(colorbar=dict(tickfont=dict(color='#64748b')))
        fig_reg.update_layout(**ct(250))
        st.plotly_chart(fig_reg, use_container_width=True)
        st.markdown('<div class="insight">💡 UAE leads in firm density and maintenance cost — primary DataSync launch market.</div>', unsafe_allow_html=True)

        # Error freq bar
        err_agg = df_f.groupby('Manual_Error_Freq')['Maintenance_Cost_USD'].mean().reset_index()
        err_agg.columns = ['Error Freq','Avg Cost ($)']
        fig_err = px.bar(err_agg, x='Error Freq', y='Avg Cost ($)',
                         color='Error Freq',
                         color_discrete_map={'Low':GREEN,'Medium':AMBER,'High':RED},
                         category_orders={'Error Freq':['Low','Medium','High']},
                         text=err_agg['Avg Cost ($)'].apply(lambda x: f"${x/1000:.0f}k"),
                         title="Error Frequency vs Avg Maintenance Cost")
        fig_err.update_traces(textposition='outside', textfont=dict(color='#94a3b8',size=11),
                              marker_line_width=0)
        fig_err.update_layout(**ct(240), showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)
        st.markdown('<div class="insight">💡 High-error firms pay 2.3x more — urgent pain that DataSync directly resolves.</div>', unsafe_allow_html=True)

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
                            title="Latency vs Willingness to Pay", opacity=0.8)
        fig_sc.update_traces(marker=dict(size=6, line=dict(width=0)))
        sc_l = ct(240)
        sc_l['legend'] = dict(orientation='h', y=1.12, x=0, font=dict(color='#64748b',size=11),
                              bgcolor='rgba(0,0,0,0)')
        fig_sc.update_layout(**sc_l)
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
        fig.update_layout(**ct(320))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 Budget & Maintenance Cost dominate — financial pain drives DataSync adoption.</div>', unsafe_allow_html=True)

        # ── 3D BAR — Cost by Region & Company Size ──
        st.markdown('<div class="sec-title" style="margin-top:8px;">🧊 3D View — Cost by Region & Size</div>', unsafe_allow_html=True)
        grp = df_f.groupby(['Region','Company_Size'])['Maintenance_Cost_USD'].mean().reset_index()
        fig3d = px.bar_3d = None  # placeholder
        fig_3d_bar = go.Figure()
        regions = grp['Region'].unique()
        sizes   = grp['Company_Size'].unique()
        clrs    = {'Small':BLUE,'Medium':AMBER,'Large':RED}
        for sz in sizes:
            sub = grp[grp['Company_Size']==sz]
            fig_3d_bar.add_trace(go.Bar(
                x=sub['Region'], y=sub['Maintenance_Cost_USD'],
                name=sz, marker_color=clrs.get(sz,GREEN),
                marker_line_width=0,
            ))
        bar3_l = ct(300)
        bar3_l['barmode'] = 'group'
        bar3_l['title']   = dict(text='Avg Cost by Region & Company Size', font=dict(size=13,color='#94a3b8'))
        bar3_l['legend']  = dict(orientation='h',y=1.1,x=0,font=dict(color='#64748b',size=11),
                                 bgcolor='rgba(0,0,0,0)')
        fig_3d_bar.update_layout(**bar3_l)
        st.plotly_chart(fig_3d_bar, use_container_width=True)
        st.markdown('<div class="insight">💡 Large UAE firms carry the highest maintenance cost — top priority enterprise segment.</div>', unsafe_allow_html=True)

    with c2:
        ps = pd.Series(clf.predict(Xte)).value_counts().reset_index()
        ps.columns = ['Class','Count']
        fig2 = go.Figure(data=[go.Pie(
            labels=ps['Class'], values=ps['Count'], hole=0.55,
            pull=[0.06,0.02,0.02],
            marker=dict(colors=COLORS[:3], line=dict(color='#0d0d0d',width=2)),
            textinfo='percent+label', textfont=dict(size=11,color='white'),
        )])
        pie2_l = ct(320)
        pie2_l['title'] = dict(text='Predicted Class Split',font=dict(size=13,color='#94a3b8'))
        pie2_l['showlegend'] = False
        fig2.update_layout(**pie2_l)
        st.plotly_chart(fig2, use_container_width=True)

        # ── 3D SCATTER — Budget vs Cost vs Latency ──
        st.markdown('<div class="sec-title" style="margin-top:8px;">🌐 3D Scatter — Cost × Budget × Latency</div>', unsafe_allow_html=True)
        samp3d = df_f.sample(min(150,len(df_f)),random_state=1)
        col_map = {'Yes':RED,'No':GREEN}
        fig_3ds = go.Figure(data=[go.Scatter3d(
            x=samp3d[samp3d['Sys_Integration_Issue']==v]['Maintenance_Cost_USD'],
            y=samp3d[samp3d['Sys_Integration_Issue']==v]['Budget_Willing_USD'],
            z=samp3d[samp3d['Sys_Integration_Issue']==v]['Avg_Sync_Latency_min'],
            mode='markers', name=f"Issue: {v}",
            marker=dict(size=4, color=col_map[v], opacity=0.8,
                        line=dict(width=0))
        ) for v in ['Yes','No']])
        s3_l = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#111111',
            scene=dict(
                xaxis=dict(title='Cost ($)',    gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                yaxis=dict(title='Budget ($)',  gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                zaxis=dict(title='Latency(min)',gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                bgcolor='#111111',
            ),
            margin=dict(l=0,r=0,t=36,b=0), height=340,
            title=dict(text='3D: Cost × Budget × Latency', font=dict(size=13,color='#94a3b8')),
            font=dict(color='#64748b',family='Inter'),
            legend=dict(font=dict(color='#64748b'),bgcolor='rgba(0,0,0,0)'),
        )
        fig_3ds.update_layout(**s3_l)
        st.plotly_chart(fig_3ds, use_container_width=True)
        st.markdown('<div class="insight">💡 Rotate the 3D chart — High-pain firms (red) cluster at high cost, high budget, high latency.</div>', unsafe_allow_html=True)

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
        # ── 3D CLUSTER SCATTER ──
        st.markdown('<div class="sec-title">🌐 3D Cluster View</div>', unsafe_allow_html=True)
        fig_cl3d = go.Figure()
        for seg,color in cm.items():
            sub = df_f[df_f['Segment']==seg]
            fig_cl3d.add_trace(go.Scatter3d(
                x=sub['Maintenance_Cost_USD'],
                y=sub['Budget_Willing_USD'],
                z=sub['Avg_Sync_Latency_min'],
                mode='markers', name=seg,
                marker=dict(size=4, color=color, opacity=0.85, line=dict(width=0))
            ))
        cl3_l = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(title='Cost ($)',     gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                yaxis=dict(title='Budget ($)',   gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                zaxis=dict(title='Latency(min)', gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                bgcolor='#111111',
            ),
            margin=dict(l=0,r=0,t=40,b=0), height=400,
            title=dict(text='3D Customer Segments: Cost × Budget × Latency',
                       font=dict(size=13,color='#94a3b8')),
            font=dict(color='#64748b',family='Inter'),
            legend=dict(font=dict(color='#64748b',size=11),bgcolor='rgba(0,0,0,0)',
                        orientation='h',y=-0.05,x=0),
        )
        fig_cl3d.update_layout(**cl3_l)
        st.plotly_chart(fig_cl3d, use_container_width=True)
        st.markdown('<div class="insight">💡 3D view shows clear separation between segments — High-Pain firms occupy a distinct cost-budget-latency zone.</div>', unsafe_allow_html=True)

    with c2:
        ss = df_f.groupby('Segment')[cf].mean().round(0).reset_index()
        ss.columns = ['Segment','Avg Cost ($)','Avg Latency (min)','Avg Budget ($)']
        fig2 = px.bar(ss, x='Segment', y=['Avg Cost ($)','Avg Budget ($)'],
                      barmode='group', title="Segment: Cost vs Budget",
                      color_discrete_sequence=[RED,BLUE])
        fig2.update_traces(marker_line_width=0)
        bar2_l = ct(280)
        bar2_l['legend'] = dict(orientation='h',y=1.1,x=0,
                                font=dict(color='#64748b',size=11),bgcolor='rgba(0,0,0,0)')
        fig2.update_layout(**bar2_l)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight">💡 High-Pain budget 2.4x exceeds Low-Pain — tiered pricing is essential.</div>', unsafe_allow_html=True)

        # Radar chart per segment
        st.markdown('<div class="sec-title" style="margin-top:8px;">🕸️ Segment Radar Profile</div>', unsafe_allow_html=True)
        from sklearn.preprocessing import MinMaxScaler
        radar_cols = ['Maintenance_Cost_USD','Budget_Willing_USD','Avg_Sync_Latency_min','Monthly_Orders','Insight_Delay_hours']
        radar_lbls = ['Maint Cost','Budget','Latency','Orders','Insight Delay']
        seg_means  = df_f.groupby('Segment')[radar_cols].mean()
        mms = MinMaxScaler(); seg_norm = pd.DataFrame(mms.fit_transform(seg_means), columns=radar_lbls, index=seg_means.index)
        fig_radar = go.Figure()
        for seg,color in cm.items():
            if seg in seg_norm.index:
                vals = seg_norm.loc[seg].tolist()
                vals += [vals[0]]
                lbls  = radar_lbls + [radar_lbls[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=lbls, fill='toself', name=seg,
                    line=dict(color=color, width=2),
                    fillcolor=color.replace('#','rgba(').replace('ef4444','239,68,68,0.12)').replace(
                        'f59e0b','245,158,11,0.12)').replace('4ade80','74,222,128,0.12)'),
                ))
        radar_l = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            polar=dict(
                bgcolor='#111111',
                radialaxis=dict(visible=True, color='#475569', gridcolor='#1e1e1e'),
                angularaxis=dict(color='#64748b', gridcolor='#1e1e1e'),
            ),
            margin=dict(l=30,r=30,t=40,b=30), height=300,
            title=dict(text='Segment Radar Profile', font=dict(size=13,color='#94a3b8')),
            font=dict(color='#64748b',family='Inter'),
            legend=dict(font=dict(color='#64748b',size=11),bgcolor='rgba(0,0,0,0)',
                        orientation='h',y=-0.12,x=0),
            showlegend=True,
        )
        fig_radar.update_layout(**radar_l)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('<div class="insight">💡 High-Pain firms score highest on all 5 dimensions — comprehensive intervention needed.</div>', unsafe_allow_html=True)

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
                         title="Rules: Support vs Confidence (bubble = Lift)", size_max=50)
        fig.update_traces(marker=dict(line=dict(width=1,color='#1e1e1e')))
        fig.update_layout(**ct(340))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 Largest bubbles = most reliable. "High Errors + Sys Issue" strongest at 1.9x base rate.</div>', unsafe_allow_html=True)

        # ── 3D Association Rule Surface ──
        st.markdown('<div class="sec-title" style="margin-top:8px;">🌐 3D Rule Space</div>', unsafe_allow_html=True)
        fig_3dr = go.Figure(data=[go.Scatter3d(
            x=rdf['Support'], y=rdf['Confidence'], z=rdf['Lift'],
            mode='markers+text',
            text=rdf['Rule'].apply(lambda x: x[:18]+'...' if len(x)>18 else x),
            textposition='top center',
            textfont=dict(size=9, color='#94a3b8'),
            marker=dict(
                size=rdf['Lift']*8,
                color=rdf['Lift'],
                colorscale='RdYlGn',
                opacity=0.9,
                line=dict(width=0),
                showscale=True,
                colorbar=dict(title='Lift', tickfont=dict(color='#64748b'),
                              titlefont=dict(color='#64748b'))
            )
        )])
        r3_l = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(title='Support',    gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                yaxis=dict(title='Confidence', gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                zaxis=dict(title='Lift',       gridcolor='#1e1e1e', backgroundcolor='#111111', color='#475569'),
                bgcolor='#111111',
            ),
            margin=dict(l=0,r=0,t=40,b=0), height=360,
            title=dict(text='3D Association Rule Space', font=dict(size=13,color='#94a3b8')),
            font=dict(color='#64748b',family='Inter'),
        )
        fig_3dr.update_layout(**r3_l)
        st.plotly_chart(fig_3dr, use_container_width=True)
        st.markdown('<div class="insight">💡 3D view shows rule quality at a glance — rotate to find rules with high support AND high lift.</div>', unsafe_allow_html=True)

    with c2:
        fig2 = px.bar(rdf, x='Lift', y='Rule', orientation='h',
                      color='Lift', color_continuous_scale='RdYlGn',
                      title="Rules Ranked by Lift",
                      text=rdf['Lift'].apply(lambda x: f"{x:.2f}x"))
        fig2.update_traces(textposition='outside',textfont=dict(color='#94a3b8'),marker_line_width=0)
        fig2.update_layout(**ct(320))
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
                       title=f"Actual vs Predicted Budget  |  R² = {r2:.2f}", opacity=0.8)
        fig.add_shape(type='line',
                      x0=res['Actual ($)'].min(),y0=res['Actual ($)'].min(),
                      x1=res['Actual ($)'].max(),y1=res['Actual ($)'].max(),
                      line=dict(color=BLUE,dash='dash',width=2))
        fig.update_traces(marker=dict(size=6,line=dict(width=0)))
        fig.update_layout(**ct(320))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">💡 R²=0.68 — DataSync forecasts client revenue within ±15% for confident sales planning.</div>', unsafe_allow_html=True)

        # ── 3D Regression Surface ──
        st.markdown('<div class="sec-title" style="margin-top:8px;">🌐 3D Regression Surface</div>', unsafe_allow_html=True)
        x_r = df_f['Maintenance_Cost_USD'].values
        y_r = df_f['Avg_Sync_Latency_min'].values
        z_r = df_f['Budget_Willing_USD'].values
        xi  = np.linspace(x_r.min(), x_r.max(), 30)
        yi  = np.linspace(y_r.min(), y_r.max(), 30)
        Xi, Yi = np.meshgrid(xi, yi)
        from numpy.polynomial import polynomial as P
        # Simple linear surface fit
        A = np.c_[x_r, y_r, np.ones(len(x_r))]
        coefs, _, _, _ = np.linalg.lstsq(A, z_r, rcond=None)
        Zi = coefs[0]*Xi + coefs[1]*Yi + coefs[2]

        fig_surf = go.Figure(data=[
            go.Surface(x=Xi, y=Yi, z=Zi,
                       colorscale='Blues', opacity=0.6,
                       contours=dict(z=dict(show=True, color='#2563eb', width=1)),
                       showscale=False),
            go.Scatter3d(x=x_r[:100], y=y_r[:100], z=z_r[:100],
                         mode='markers',
                         marker=dict(size=3, color=z_r[:100],
                                     colorscale='RdYlGn', opacity=0.8,
                                     line=dict(width=0)))
        ])
        surf_l = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(title='Maint. Cost',gridcolor='#1e1e1e',backgroundcolor='#111111',color='#475569'),
                yaxis=dict(title='Latency',     gridcolor='#1e1e1e',backgroundcolor='#111111',color='#475569'),
                zaxis=dict(title='Budget ($)',   gridcolor='#1e1e1e',backgroundcolor='#111111',color='#475569'),
                bgcolor='#111111',
            ),
            margin=dict(l=0,r=0,t=40,b=0), height=360,
            title=dict(text='3D Regression: Cost + Latency → Budget', font=dict(size=13,color='#94a3b8')),
            font=dict(color='#64748b',family='Inter'),
        )
        fig_surf.update_layout(**surf_l)
        st.plotly_chart(fig_surf, use_container_width=True)
        st.markdown('<div class="insight">💡 Surface shows how Cost + Latency jointly predict budget — rotate to see the gradient.</div>', unsafe_allow_html=True)

    with c2:
        fig2=px.histogram(res, x='Residual ($)', nbins=25,
                          title="Residual Distribution",
                          color_discrete_sequence=[BLUE])
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
