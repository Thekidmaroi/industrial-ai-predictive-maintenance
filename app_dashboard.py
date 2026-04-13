import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

from src.features.engineering import load_skab, add_rolling_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
 page_title="Industrial AI — Eranove",
 page_icon="",
 layout="wide",
 initial_sidebar_state="expanded"
)

# ── CSS custom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
 .main { background-color: #0e1117; }
 .block-container { padding-top: 1rem; padding-bottom: 1rem; }

 .kpi-card {
 background: linear-gradient(135deg, #1a1f2e, #16213e);
 border: 1px solid #2d3561;
 border-radius: 16px;
 padding: 20px 24px;
 text-align: center;
 box-shadow: 0 4px 24px rgba(0,0,0,0.4);
 }
 .kpi-label {
 font-size: 12px;
 color: #8b95a1;
 text-transform: uppercase;
 letter-spacing: 1.5px;
 margin-bottom: 6px;
 }
 .kpi-value {
 font-size: 36px;
 font-weight: 700;
 background: linear-gradient(90deg, #4fc3f7, #7c4dff);
 -webkit-background-clip: text;
 -webkit-text-fill-color: transparent;
 line-height: 1.1;
 }
 .kpi-sub {
 font-size: 11px;
 color: #5c6370;
 margin-top: 4px;
 }

 .health-card {
 background: linear-gradient(135deg, #1a1f2e, #16213e);
 border-radius: 16px;
 padding: 24px;
 border: 1px solid #2d3561;
 text-align: center;
 }
 .health-score {
 font-size: 72px;
 font-weight: 800;
 line-height: 1;
 }

 .alert-critical {
 background: linear-gradient(135deg, #2d1515, #3d1a1a);
 border: 1px solid #e53935;
 border-left: 4px solid #e53935;
 border-radius: 8px;
 padding: 12px 16px;
 margin: 6px 0;
 color: #ef9a9a;
 font-size: 13px;
 }
 .alert-warning {
 background: linear-gradient(135deg, #2d2415, #3d2e1a);
 border: 1px solid #fb8c00;
 border-left: 4px solid #fb8c00;
 border-radius: 8px;
 padding: 12px 16px;
 margin: 6px 0;
 color: #ffcc80;
 font-size: 13px;
 }
 .alert-ok {
 background: linear-gradient(135deg, #152d1e, #1a3d26);
 border: 1px solid #43a047;
 border-left: 4px solid #43a047;
 border-radius: 8px;
 padding: 12px 16px;
 margin: 6px 0;
 color: #a5d6a7;
 font-size: 13px;
 }

 .section-title {
 font-size: 18px;
 font-weight: 600;
 color: #e0e6ed;
 margin: 24px 0 12px 0;
 padding-bottom: 8px;
 border-bottom: 1px solid #2d3561;
 }

 div[data-testid="stSidebar"] {
 background: linear-gradient(180deg, #0d1117, #161b22);
 border-right: 1px solid #21262d;
 }
 div[data-testid="stSidebar"] .stMarkdown p {
 color: #8b949e;
 font-size: 12px;
 }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
 cfg = yaml.safe_load(f)

SENSOR_COLS = cfg['data']['sensor_cols']
COLORS = {
 'primary' : '#4fc3f7',
 'secondary': '#7c4dff',
 'danger' : '#e53935',
 'warning' : '#fb8c00',
 'success' : '#43a047',
 'normal' : '#4fc3f7',
 'anomaly' : '#e53935',
 'bg' : '#0e1117',
 'card' : '#1a1f2e',
 'border' : '#2d3561',
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
 st.markdown("## Industrial AI")
 st.markdown("**Predictive Maintenance**")
 st.markdown("---")

 st.markdown("#### ️ Données")
 source_filter = st.multiselect(
 "Sources SCADA",
 options=["valve1", "valve2", "other"],
 default=["valve1", "valve2", "other"],
 label_visibility="collapsed"
 )

 st.markdown("#### ️ Paramètres modèle")
 contamination = st.slider("Contamination estimée", 10, 60, 35, 5,
 format="%d%%") / 100
 window = st.slider("Fenêtre glissante (s)", 10, 60, 30, 5)

 st.markdown("#### Seuils alertes")
 threshold_critical = st.slider("Score critique", 50, 80, 65, 5, format="%d")
 threshold_warning = st.slider("Score warning", 30, 60, 40, 5, format="%d")

 st.markdown("---")
 if st.button(" Lancer l'analyse", use_container_width=True, type="primary"):
        st.session_state["analysis_done"] = True

 st.markdown("---")
 st.markdown("**Stack**")
 st.markdown("Python · Scikit-learn · Streamlit · Plotly")
 st.caption("Prototype ERANOVE — 2025")

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
 st.markdown("# Industrial AI Dashboard")
 st.markdown("Détection d'anomalies en temps réel sur infrastructure énergétique")
with col_h2:
 st.markdown("<br>", unsafe_allow_html=True)
 status_placeholder = st.empty()

if not st.session_state.get("analysis_done", False):
 status_placeholder.info("En attente d'analyse")
 st.markdown("---")
 st.info(" Configure les paramètres et clique sur **Lancer l'analyse**")
 st.stop()

status_placeholder.warning("⏳ Analyse en cours...")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(folders, win):
 df = load_skab(cfg['data']['path'], list(folders), SENSOR_COLS)
 df_work = df[SENSOR_COLS + ['anomaly']].reset_index(drop=True)
 df_feat = add_rolling_features(
 df_work, SENSOR_COLS,
 cfg['features']['discriminant_sensors'], window=win
 )
 labels = df_work['anomaly'].astype(int).values[win-1:win-1+len(df_feat)]
 return df_work, df_feat, labels

df_work, df_feat, labels = load_and_process(tuple(source_filter), window)

# ── Model ─────────────────────────────────────────────────────────────────────
X = df_feat.values
y = labels[:len(X)]
X_norm = X[y == 0]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
Xn_sc = scaler.transform(X_norm)

iso = IsolationForest(n_estimators=200, contamination=contamination,
 random_state=42, n_jobs=-1)
iso.fit(Xn_sc)
scores_raw = -iso.score_samples(X_sc)

prec_arr, rec_arr, thr_arr = precision_recall_curve(y, scores_raw)
f1_arr = 2*(prec_arr*rec_arr)/(prec_arr+rec_arr+1e-8)
best_thr = thr_arr[f1_arr.argmax()]
preds = (scores_raw >= best_thr).astype(int)

# Normaliser le score anomalie 0-100
scores_norm = ((scores_raw - scores_raw.min()) /
 (scores_raw.max() - scores_raw.min()) * 100)

# Score de santé = inverse du score anomalie sur fenêtre récente
recent_n = min(500, len(scores_norm))
health_score = int(100 - scores_norm[-recent_n:].mean())
health_score = max(0, min(100, health_score))

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
k1, k2, k3, k4, k5 = st.columns(5)

metrics = [
 (k1, "ROC-AUC", f"{roc_auc_score(y, scores_raw):.3f}", "Discrimination globale"),
 (k2, "Recall", f"{recall_score(y, preds):.3f}", "Anomalies détectées"),
 (k3, "Précision", f"{precision_score(y, preds):.3f}", "Fiabilité alertes"),
 (k4, "F1-Score", f"{f1_score(y, preds):.3f}", "Équilibre P/R"),
 (k5, "Anomalies détectées",f"{preds.sum():,}", f"sur {len(preds):,} points"),
]
for col, label, value, sub in metrics:
 col.markdown(f"""
 <div class="kpi-card">
 <div class="kpi-label">{label}</div>
 <div class="kpi-value">{value}</div>
 <div class="kpi-sub">{sub}</div>
 </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Score de santé + Alertes ──────────────────────────────────────────────────
st.markdown('<div class="section-title"> Score de santé équipement</div>',
 unsafe_allow_html=True)

col_health, col_alerts = st.columns([1, 2])

with col_health:
 if health_score >= threshold_critical:
        hcolor, hemoji = "#43a047", ""
 elif health_score >= threshold_warning:
        hcolor, hemoji = "#fb8c00", ""
 else:
        hcolor, hemoji = "#e53935", ""

 st.markdown(f"""
 <div class="health-card">
 <div class="kpi-label">Score de santé global</div>
 <div class="health-score" style="color:{hcolor}">{health_score}</div>
 <div style="font-size:32px; margin: 4px 0">{hemoji}</div>
 <div style="font-size:13px; color:#8b95a1">Score /100 — {recent_n} dernières mesures</div>
 </div>""", unsafe_allow_html=True)

with col_alerts:
 st.markdown("**Alertes actives**")
 recent_preds = preds[-recent_n:]
 anomaly_rate = recent_preds.mean() * 100

 if anomaly_rate > 40:
     st.markdown(f'<div class="alert-critical"> <b>CRITIQUE</b> — Taux d\'anomalie élevé : {anomaly_rate:.1f}% sur les dernières mesures. Inspection immédiate recommandée.</div>', unsafe_allow_html=True)
 elif anomaly_rate > 20:
     st.markdown(f'<div class="alert-warning">️ <b>ATTENTION</b> — Taux d\'anomalie : {anomaly_rate:.1f}%. Surveillance renforcée conseillée.</div>', unsafe_allow_html=True)
 else:
     st.markdown(f'<div class="alert-ok"> <b>NORMAL</b> — Taux d\'anomalie : {anomaly_rate:.1f}%. Système opérationnel.</div>', unsafe_allow_html=True)

 # Capteurs les plus perturbés
 st.markdown("**Capteurs les plus impactés**")
 df_anom = df_feat[SENSOR_COLS][preds==1]
 df_norm = df_feat[SENSOR_COLS][preds==0]
 if len(df_anom) > 0:
        delta = ((df_anom.mean() - df_norm.mean()) / (df_norm.mean().abs() + 1e-6) * 100)
 top3 = delta.abs().nlargest(3)
 for sensor, val in top3.items():
        direction = "↑" if delta[sensor] > 0 else "↓"
 css = "alert-warning" if abs(val) < 30 else "alert-critical"
 st.markdown(f'<div class="{css}">{direction} <b>{sensor}</b> — écart : {val:+.1f}%</div>',
 unsafe_allow_html=True)

# ── Score d'anomalie temporel ─────────────────────────────────────────────────
st.markdown('<div class="section-title"> Score d\'anomalie — vue temporelle</div>',
 unsafe_allow_html=True)

idx = np.arange(len(y))

fig_main = make_subplots(
 rows=2, cols=1, shared_xaxes=True,
 row_heights=[0.65, 0.35],
 vertical_spacing=0.06
)

# Zones anomalies réelles
anom_starts = np.where(np.diff(np.concatenate([[0], y, [0]])) == 1)[0]
anom_ends = np.where(np.diff(np.concatenate([[0], y, [0]])) == -1)[0]
for s, e in zip(anom_starts, anom_ends):
 fig_main.add_vrect(x0=s, x1=e, fillcolor="rgba(229,57,53,0.12)",
 line_width=0, row=1, col=1)

fig_main.add_trace(go.Scatter(
 x=idx, y=scores_norm, mode='lines',
 line=dict(color=COLORS['primary'], width=0.8),
 name='Score anomalie', fill='tozeroy',
 fillcolor='rgba(79,195,247,0.08)'
), row=1, col=1)

fig_main.add_hline(
 y=(best_thr - scores_raw.min()) / (scores_raw.max() - scores_raw.min()) * 100,
 line_dash="dash", line_color=COLORS['warning'],
 line_width=1.5, row=1, col=1,
 annotation_text=f"Seuil détection",
 annotation_font_color=COLORS['warning']
)

# TP / FP / FN
tp = (preds==1) & (y==1)
fp = (preds==1) & (y==0)
fn = (preds==0) & (y==1)

for mask, color, name in [
 (tp, 'rgba(67,160,71,0.7)', 'TP — Correct'),
 (fp, 'rgba(251,140,0,0.5)', 'FP — Fausse alarme'),
 (fn, 'rgba(229,57,53,0.7)', 'FN — Manquée'),
]:
 fig_main.add_trace(go.Scatter(
 x=idx[mask], y=np.ones(mask.sum()),
 mode='markers', marker=dict(color=color, size=3, symbol='square'),
 name=name
 ), row=2, col=1)

fig_main.update_layout(
 height=500,
 paper_bgcolor=COLORS['bg'],
 plot_bgcolor='rgba(26,31,46,0.8)',
 font=dict(color='#e0e6ed', size=11),
 legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
 margin=dict(l=60, r=20, t=20, b=40),
 hovermode='x unified'
)
fig_main.update_xaxes(gridcolor='#1e2538', zerolinecolor='#1e2538')
fig_main.update_yaxes(gridcolor='#1e2538', zerolinecolor='#1e2538')
fig_main.update_yaxes(title_text="Score (0-100)", row=1, col=1)
fig_main.update_yaxes(title_text="Détection", row=2, col=1)
fig_main.update_xaxes(title_text="Index temporel", row=2, col=1)

st.plotly_chart(fig_main, use_container_width=True)

# ── Score de santé évolution ──────────────────────────────────────────────────
st.markdown('<div class="section-title"> Évolution du score de santé</div>',
 unsafe_allow_html=True)

roll_w = 300
health_ts = pd.Series(100 - scores_norm).rolling(roll_w).mean().fillna(method='bfill')

fig_health = go.Figure()
fig_health.add_trace(go.Scatter(
 x=idx, y=health_ts,
 mode='lines', line=dict(color=COLORS['success'], width=1.5),
 fill='tozeroy', fillcolor='rgba(67,160,71,0.08)',
 name='Score santé'
))
fig_health.add_hline(y=threshold_critical, line_dash="dash",
 line_color=COLORS['warning'], line_width=1,
 annotation_text="Seuil alerte",
 annotation_font_color=COLORS['warning'])
fig_health.add_hline(y=threshold_warning, line_dash="dash",
 line_color=COLORS['danger'], line_width=1,
 annotation_text="Seuil critique",
 annotation_font_color=COLORS['danger'])
fig_health.update_layout(
 height=300,
 paper_bgcolor=COLORS['bg'],
 plot_bgcolor='rgba(26,31,46,0.8)',
 font=dict(color='#e0e6ed', size=11),
 margin=dict(l=60, r=20, t=20, b=40),
 yaxis=dict(range=[0, 100], title="Score santé (%)"),
 xaxis=dict(title="Index temporel", gridcolor='#1e2538'),
 yaxis_gridcolor='#1e2538',
 showlegend=False
)
st.plotly_chart(fig_health, use_container_width=True)

# ── Distribution capteurs ─────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Distribution des capteurs — Normal vs Anomalie</div>',
 unsafe_allow_html=True)

df_plot = df_feat[SENSOR_COLS].copy()
df_plot['y'] = y
cols_per_row = 4
rows_sensor = (len(SENSOR_COLS) + cols_per_row - 1) // cols_per_row

fig_dist = make_subplots(
 rows=rows_sensor, cols=cols_per_row,
 subplot_titles=SENSOR_COLS,
 vertical_spacing=0.12, horizontal_spacing=0.06
)

for i, col in enumerate(SENSOR_COLS):
 r = i // cols_per_row + 1
 c = i % cols_per_row + 1
 normal_data = df_plot[df_plot['y']==0][col].dropna()
 anom_data = df_plot[df_plot['y']==1][col].dropna()

 fig_dist.add_trace(go.Histogram(
 x=normal_data, nbinsx=40, histnorm='probability density',
 marker_color=COLORS['normal'], opacity=0.65,
 name='Normal', showlegend=(i==0)
 ), row=r, col=c)
 fig_dist.add_trace(go.Histogram(
 x=anom_data, nbinsx=40, histnorm='probability density',
 marker_color=COLORS['anomaly'], opacity=0.65,
 name='Anomalie', showlegend=(i==0)
 ), row=r, col=c)

fig_dist.update_layout(
 height=420, barmode='overlay',
 paper_bgcolor=COLORS['bg'],
 plot_bgcolor='rgba(26,31,46,0.8)',
 font=dict(color='#e0e6ed', size=10),
 margin=dict(l=40, r=20, t=40, b=20),
 legend=dict(bgcolor='rgba(0,0,0,0)')
)
fig_dist.update_xaxes(gridcolor='#1e2538')
fig_dist.update_yaxes(gridcolor='#1e2538')
st.plotly_chart(fig_dist, use_container_width=True)

# ── Heatmap corrélation ───────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Corrélations capteurs — Normal vs Anomalie</div>',
 unsafe_allow_html=True)

col_c1, col_c2 = st.columns(2)
for col_ui, mask, title in [
 (col_c1, y==0, "Mode normal"),
 (col_c2, y==1, "Mode anomalie")
]:
 corr = df_feat[SENSOR_COLS][mask].corr().round(2)
 fig_corr = go.Figure(go.Heatmap(
 z=corr.values,
 x=[s.replace('RMS','').replace('Volume Flow Rate','VFR')
 .replace('Accelerometer','Acc') for s in SENSOR_COLS],
 y=[s.replace('RMS','').replace('Volume Flow Rate','VFR')
 .replace('Accelerometer','Acc') for s in SENSOR_COLS],
 colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
 text=corr.values.round(2),
 texttemplate="%{text}",
 textfont=dict(size=9),
 colorbar=dict(thickness=12, len=0.8)
 ))
 fig_corr.update_layout(
 title=dict(text=title, font=dict(size=13, color='#e0e6ed')),
 height=350,
 paper_bgcolor=COLORS['bg'],
 plot_bgcolor=COLORS['bg'],
 font=dict(color='#e0e6ed', size=10),
 margin=dict(l=80, r=20, t=50, b=80)
 )
 col_ui.plotly_chart(fig_corr, use_container_width=True)

# ── Precision-Recall curve ────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Courbe Precision-Recall</div>',
 unsafe_allow_html=True)

col_pr, col_roc = st.columns(2)

with col_pr:
 fig_pr = go.Figure()
 fig_pr.add_trace(go.Scatter(
 x=rec_arr, y=prec_arr, mode='lines',
 line=dict(color=COLORS['primary'], width=2), name='PR Curve'
 ))
 best_rec = rec_arr[f1_arr[:-1].argmax()]
 best_pre = prec_arr[f1_arr[:-1].argmax()]
 fig_pr.add_trace(go.Scatter(
 x=[best_rec], y=[best_pre], mode='markers',
 marker=dict(color=COLORS['danger'], size=12, symbol='star'),
 name=f'Optimal F1={f1_arr.max():.2f}'
 ))
 fig_pr.update_layout(
 title="Courbe Precision-Recall",
 height=300, paper_bgcolor=COLORS['bg'],
 plot_bgcolor='rgba(26,31,46,0.8)',
 font=dict(color='#e0e6ed', size=11),
 xaxis=dict(title="Recall", gridcolor='#1e2538', range=[0,1]),
 yaxis=dict(title="Precision", gridcolor='#1e2538', range=[0,1]),
 margin=dict(l=60, r=20, t=40, b=40),
 legend=dict(bgcolor='rgba(0,0,0,0)')
 )
 st.plotly_chart(fig_pr, use_container_width=True)

with col_roc:
 fig_dist2 = go.Figure()
 fig_dist2.add_trace(go.Histogram(
 x=scores_norm[y==0], histnorm='probability density',
 nbinsx=60, marker_color=COLORS['normal'],
 opacity=0.7, name='Normal'
 ))
 fig_dist2.add_trace(go.Histogram(
 x=scores_norm[y==1], histnorm='probability density',
 nbinsx=60, marker_color=COLORS['anomaly'],
 opacity=0.7, name='Anomalie'
 ))
 norm_thr = (best_thr - scores_raw.min()) / (scores_raw.max() - scores_raw.min()) * 100
 fig_dist2.add_vline(x=norm_thr, line_dash="dash",
 line_color=COLORS['warning'], line_width=2,
 annotation_text="Seuil",
 annotation_font_color=COLORS['warning'])
 fig_dist2.update_layout(
 title="Distribution des scores d'anomalie",
 height=300, barmode='overlay',
 paper_bgcolor=COLORS['bg'],
 plot_bgcolor='rgba(26,31,46,0.8)',
 font=dict(color='#e0e6ed', size=11),
 xaxis=dict(title="Score (0-100)", gridcolor='#1e2538'),
 yaxis=dict(title="Densité", gridcolor='#1e2538'),
 margin=dict(l=60, r=20, t=40, b=40),
 legend=dict(bgcolor='rgba(0,0,0,0)')
 )
 st.plotly_chart(fig_dist2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
 "<div style='text-align:center; color:#5c6370; font-size:12px'>"
 " Industrial AI Predictive Maintenance — Prototype ERANOVE | "
 "Python · Scikit-learn · TensorFlow · Streamlit · Plotly"
 "</div>",
 unsafe_allow_html=True
)

status_placeholder.success(" Analyse terminée")
