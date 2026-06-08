import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc, precision_recall_curve
)
from sklearn.datasets import make_classification
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI | منصة التنبؤ التعليمي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Orbitron:wght@700;900&display=swap');

  :root {
    --primary: #00d4ff;
    --secondary: #7b2ff7;
    --accent: #ff6b35;
    --success: #00e676;
    --danger: #ff1744;
    --bg-dark: #050714;
    --bg-card: #0d1117;
    --bg-card2: #161b27;
    --text: #e8eaf6;
    --text-dim: #90a4ae;
    --border: rgba(0, 212, 255, 0.2);
  }

  html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text);
  }

  .stApp {
    background: radial-gradient(ellipse at 20% 20%, rgba(123,47,247,0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(0,212,255,0.10) 0%, transparent 50%),
                var(--bg-dark);
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1117 100%);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

  /* ── Hero Header ── */
  .hero-header {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
  }
  .hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 900;
    background: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
  }
  .hero-subtitle {
    font-size: 1.2rem;
    color: var(--text-dim);
    font-weight: 300;
  }
  .hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--secondary), var(--primary));
    color: white;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.5rem;
  }
  .hero-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), var(--secondary), transparent);
    margin: 2rem auto;
    max-width: 600px;
    border: none;
  }

  /* ── Cards ── */
  .glass-card {
    background: linear-gradient(135deg, rgba(13,17,23,0.9), rgba(22,27,39,0.8));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
  }
  .glass-card:hover { border-color: rgba(0,212,255,0.5); }

  .metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,47,247,0.08));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.3s;
  }
  .metric-card:hover { transform: translateY(-3px); border-color: var(--primary); }
  .metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
  }
  .metric-label { font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }

  /* ── Prediction Result ── */
  .result-pass {
    background: linear-gradient(135deg, rgba(0,230,118,0.15), rgba(0,230,118,0.05));
    border: 2px solid var(--success);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
  }
  .result-fail {
    background: linear-gradient(135deg, rgba(255,23,68,0.15), rgba(255,23,68,0.05));
    border: 2px solid var(--danger);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
  }
  @keyframes pulse-green {
    0%,100% { box-shadow: 0 0 20px rgba(0,230,118,0.3); }
    50%      { box-shadow: 0 0 40px rgba(0,230,118,0.6); }
  }
  @keyframes pulse-red {
    0%,100% { box-shadow: 0 0 20px rgba(255,23,68,0.3); }
    50%      { box-shadow: 0 0 40px rgba(255,23,68,0.6); }
  }
  .result-icon { font-size: 4rem; margin-bottom: 0.5rem; }
  .result-text { font-family: 'Orbitron', sans-serif; font-size: 2rem; font-weight: 900; }
  .result-pass .result-text { color: var(--success); }
  .result-fail .result-text { color: var(--danger); }
  .result-prob { font-size: 1rem; color: var(--text-dim); margin-top: 0.5rem; }

  /* ── Section titles ── */
  .section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--primary);
    border-left: 4px solid var(--secondary);
    padding-left: 12px;
    margin-bottom: 1rem;
  }

  /* ── Sidebar labels ── */
  .sidebar-section {
    background: rgba(0,212,255,0.05);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  .sidebar-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
  }

  /* ── Streamlit overrides ── */
  .stSlider [data-testid="stThumbValue"] { color: var(--primary) !important; }
  .stButton>button {
    background: linear-gradient(135deg, var(--secondary), var(--primary));
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Cairo', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: opacity 0.2s, transform 0.2s;
  }
  .stButton>button:hover { opacity: 0.9; transform: translateY(-2px); }

  .stTabs [data-baseweb="tab"] {
    color: var(--text-dim);
    font-family: 'Cairo', sans-serif;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] { color: var(--primary) !important; }

  div[data-testid="stMetricValue"] { color: var(--primary); font-family: 'Orbitron', sans-serif; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-dark); }
  ::-webkit-scrollbar-thumb { background: var(--secondary); border-radius: 3px; }

  /* ── Plotly chart bg ── */
  .js-plotly-plot .plotly { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in {
    "model": None,
    "scaler": None,
    "trained": False,
    "X_test": None,
    "y_test": None,
    "X_train": None,
    "y_train": None,
    "df": None,
    "history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=1500, seed=42):
    np.random.seed(seed)
    study_hours   = np.random.uniform(0, 12, n)
    attendance    = np.random.uniform(50, 100, n)
    prev_grade    = np.random.uniform(30, 100, n)
    age           = np.random.randint(17, 30, n).astype(float)
    sleep_hours   = np.random.uniform(4, 10, n)
    assignments   = np.random.uniform(0, 100, n)
    participation = np.random.uniform(0, 100, n)
    motivation    = np.random.uniform(0, 10, n)

    score = (
        0.30 * study_hours / 12 +
        0.20 * (attendance - 50) / 50 +
        0.25 * prev_grade / 100 +
        0.10 * (sleep_hours - 4) / 6 +
        0.08 * assignments / 100 +
        0.07 * participation / 100 +
        np.random.normal(0, 0.05, n)
    )
    passed = (score >= 0.5).astype(int)

    return pd.DataFrame({
        "ساعات_الدراسة":    np.round(study_hours, 2),
        "نسبة_الحضور":     np.round(attendance, 1),
        "درجة_الاختبار_السابق": np.round(prev_grade, 1),
        "العمر":            age,
        "ساعات_النوم":     np.round(sleep_hours, 2),
        "درجة_الواجبات":   np.round(assignments, 1),
        "المشاركة_الصفية": np.round(participation, 1),
        "مستوى_الدافعية":  np.round(motivation, 2),
        "النتيجة":          passed
    })

# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(df, layers, activation, solver, alpha, max_iter, lr):
    feature_cols = [c for c in df.columns if c != "النتيجة"]
    X = df[feature_cols].values
    y = df["النتيجة"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=lr,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    model.fit(X_train_s, y_train)
    return model, scaler, X_train_s, X_test_s, y_train, y_test

# ─────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.8)",
    font=dict(family="Cairo", color="#e8eaf6"),
    margin=dict(l=40, r=40, t=50, b=40)
)

def plot_confusion(cm):
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=[[0,"#0d1117"],[0.5,"#7b2ff7"],[1,"#00d4ff"]],
        labels=dict(x="التنبؤ", y="الفعلي"),
        x=["راسب","ناجح"], y=["راسب","ناجح"]
    )
    fig.update_layout(
        title="مصفوفة الارتباك",
        **DARK_LAYOUT,
        coloraxis_showscale=False
    )
    fig.update_traces(textfont_size=18)
    return fig

def plot_roc(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.1)',
        line=dict(color='#00d4ff', width=2),
        name=f'AUC = {auc_val:.3f}'
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        line=dict(color='gray', dash='dash', width=1),
        name='Random', showlegend=True
    ))
    fig.update_layout(
        title="منحنى ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **DARK_LAYOUT
    )
    return fig

def plot_loss(model):
    if not hasattr(model, 'loss_curve_'):
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=model.loss_curve_,
        line=dict(color='#00d4ff', width=2),
        name='Training Loss',
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.05)'
    ))
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        fig.add_trace(go.Scatter(
            y=model.validation_scores_,
            line=dict(color='#ff6b35', width=2),
            name='Validation Score'
        ))
    fig.update_layout(
        title="منحنى التعلم",
        xaxis_title="Iteration",
        yaxis_title="Loss / Score",
        **DARK_LAYOUT
    )
    return fig

def plot_feature_importance(model, feature_names):
    weights = np.abs(model.coefs_[0]).mean(axis=1)
    weights = weights / weights.sum() * 100
    df_imp = pd.DataFrame({
        "الميزة": feature_names,
        "الأهمية": weights
    }).sort_values("الأهمية", ascending=True)
    
    colors = [f"rgba(0,212,255,{0.4 + 0.6*v/weights.max()})" for v in df_imp["الأهمية"]]
    fig = go.Figure(go.Bar(
        x=df_imp["الأهمية"],
        y=df_imp["الميزة"],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
    ))
    fig.update_layout(
        title="أهمية المتغيرات",
        xaxis_title="الأهمية (%)",
        **DARK_LAYOUT
    )
    return fig

def plot_distribution(df):
    fig = make_subplots(rows=2, cols=4,
        subplot_titles=[c for c in df.columns if c != "النتيجة"])
    features = [c for c in df.columns if c != "النتيجة"]
    colors_pass = "rgba(0,230,118,0.7)"
    colors_fail = "rgba(255,23,68,0.7)"
    for i, feat in enumerate(features):
        row, col = divmod(i, 4)
        for label, color, name in [(1, colors_pass, "ناجح"), (0, colors_fail, "راسب")]:
            fig.add_trace(go.Histogram(
                x=df[df["النتيجة"]==label][feat],
                marker_color=color,
                name=name, showlegend=(i==0),
                opacity=0.75, nbinsx=25
            ), row=row+1, col=col+1)
    fig.update_layout(
        height=500, barmode='overlay',
        title="توزيع المتغيرات (ناجح vs راسب)",
        **DARK_LAYOUT
    )
    return fig

def plot_scatter_matrix(df):
    cols = ["ساعات_الدراسة","نسبة_الحضور","درجة_الاختبار_السابق","النتيجة"]
    color_map = {0: "#ff1744", 1: "#00e676"}
    df_plot = df[cols].copy()
    df_plot["النتيجة_نص"] = df_plot["النتيجة"].map({0:"راسب",1:"ناجح"})
    fig = px.scatter_matrix(
        df_plot,
        dimensions=cols[:-1],
        color="النتيجة_نص",
        color_discrete_map={"ناجح":"#00e676","راسب":"#ff1744"},
        opacity=0.5
    )
    fig.update_layout(
        title="مصفوفة الارتباط بين المتغيرات",
        **DARK_LAYOUT,
        height=450
    )
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
      <div style='font-family:Orbitron; font-size:1.4rem; font-weight:900;
                  background:linear-gradient(135deg,#00d4ff,#7b2ff7);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        EduPredict AI
      </div>
      <div style='color:#90a4ae; font-size:0.75rem; margin-top:4px;'>منصة التنبؤ التعليمي</div>
    </div>
    <hr style='border-color:rgba(0,212,255,0.2);'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">⚙️ إعدادات الشبكة العصبية</div>', unsafe_allow_html=True)

    n_layers = st.selectbox("عدد الطبقات المخفية", [1, 2, 3, 4], index=1)
    
    layer_sizes = []
    for i in range(n_layers):
        size = st.slider(f"حجم الطبقة {i+1}", 16, 256, [64,128,64,32][i], 8)
        layer_sizes.append(size)
    
    activation = st.selectbox("دالة التفعيل", ["relu","tanh","logistic","identity"], index=0,
                               format_func=lambda x: {"relu":"ReLU ⚡","tanh":"Tanh 〰️","logistic":"Sigmoid 📈","identity":"Linear ➖"}[x])
    solver = st.selectbox("المحسِّن (Optimizer)", ["adam","sgd","lbfgs"], index=0,
                           format_func=lambda x: {"adam":"Adam 🚀","sgd":"SGD 📉","lbfgs":"L-BFGS 🎯"}[x])
    alpha = st.select_slider("معامل L2 (Alpha)", [0.0001,0.001,0.01,0.1], value=0.001)
    lr    = st.select_slider("معدل التعلم", [0.0001,0.001,0.01,0.1], value=0.001)
    max_iter = st.slider("الحد الأقصى للتكرارات", 100, 1000, 300, 50)
    n_samples = st.slider("حجم مجموعة البيانات", 500, 3000, 1500, 100)

    st.markdown("<hr style='border-color:rgba(0,212,255,0.2);'>", unsafe_allow_html=True)
    train_btn = st.button("🚀 تدريب النموذج", use_container_width=True)

    if st.session_state.trained:
        st.markdown("""
        <div style='background:rgba(0,230,118,0.1); border:1px solid #00e676;
                    border-radius:10px; padding:0.7rem; text-align:center; margin-top:1rem;'>
          <span style='color:#00e676; font-weight:700;'>✅ النموذج جاهز</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TRAIN ON BUTTON CLICK
# ─────────────────────────────────────────────
if train_btn:
    df = generate_dataset(n=n_samples)
    st.session_state.df = df
    
    with st.spinner("⚡ جاري تدريب الشبكة العصبية ..."):
        progress = st.progress(0)
        for p in range(0, 80, 10):
            time.sleep(0.05)
            progress.progress(p)
        
        model, scaler, X_train, X_test, y_train, y_test = train_model(
            df, tuple(layer_sizes), activation, solver, alpha, max_iter, lr
        )
        progress.progress(100)

    st.session_state.update({
        "model":   model,
        "scaler":  scaler,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "trained": True,
    })
    st.toast("✅ تم تدريب النموذج بنجاح!", icon="🎓")

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🎓 EduPredict AI</div>
  <div class="hero-subtitle">منصة التنبؤ الذكي لنتائج الطلاب | Powered by ANN & MLP</div>
  <div class="hero-badge">Deep Learning • Neural Networks • Real-time Prediction</div>
  <hr class="hero-divider">
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 التنبؤ الفوري",
    "📊 أداء النموذج",
    "📈 تحليل البيانات",
    "🧠 معمارية الشبكة",
    "📋 تاريخ التنبؤات"
])

# ══════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
    if not st.session_state.trained:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
          <div style="font-size:4rem;">🤖</div>
          <div style="font-size:1.5rem; color:#00d4ff; font-weight:700; margin:1rem 0;">
            النموذج لم يُدرَّب بعد
          </div>
          <div style="color:#90a4ae;">
            اضغط على زر <strong style="color:#7b2ff7;">تدريب النموذج</strong> في الشريط الجانبي للبدء
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">✍️ أدخل بيانات الطالب</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            study_hours   = st.slider("📚 ساعات الدراسة اليومية", 0.0, 12.0, 6.0, 0.5)
            attendance    = st.slider("🏫 نسبة الحضور (%)", 50.0, 100.0, 80.0, 1.0)
            prev_grade    = st.slider("📝 درجة الاختبار السابق", 30.0, 100.0, 65.0, 1.0)
            age           = st.slider("🎂 العمر (سنة)", 17, 30, 20)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            sleep_hours   = st.slider("😴 ساعات النوم", 4.0, 10.0, 7.0, 0.5)
            assignments   = st.slider("📓 درجة الواجبات (%)", 0.0, 100.0, 75.0, 1.0)
            participation = st.slider("🙋 المشاركة الصفية (%)", 0.0, 100.0, 70.0, 1.0)
            motivation    = st.slider("🔥 مستوى الدافعية (1-10)", 0.0, 10.0, 7.0, 0.5)
            st.markdown('</div>', unsafe_allow_html=True)
        
        predict_btn = st.button("🔮 تنبأ بالنتيجة الآن!", use_container_width=True)
        
        if predict_btn:
            features = np.array([[study_hours, attendance, prev_grade, age,
                                   sleep_hours, assignments, participation, motivation]])
            features_s = st.session_state.scaler.transform(features)
            prediction = st.session_state.model.predict(features_s)[0]
            proba      = st.session_state.model.predict_proba(features_s)[0]
            confidence = proba[prediction] * 100
            pass_prob  = proba[1] * 100

            # Store in history
            st.session_state.history.append({
                "ساعات الدراسة": study_hours,
                "الحضور%": attendance,
                "الدرجة السابقة": prev_grade,
                "العمر": age,
                "النتيجة": "✅ ناجح" if prediction == 1 else "❌ راسب",
                "نسبة الثقة%": f"{confidence:.1f}%"
            })

            col_r1, col_r2, col_r3 = st.columns([2,1,1])
            with col_r1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-pass">
                      <div class="result-icon">🎓</div>
                      <div class="result-text">ناجح</div>
                      <div class="result-prob">احتمالية النجاح: {pass_prob:.1f}%</div>
                      <div style="color:#00e676; font-size:0.9rem; margin-top:0.5rem;">
                        ثقة النموذج: {confidence:.1f}%
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fail">
                      <div class="result-icon">⚠️</div>
                      <div class="result-text">راسب</div>
                      <div class="result-prob">احتمالية النجاح: {pass_prob:.1f}%</div>
                      <div style="color:#ff1744; font-size:0.9rem; margin-top:0.5rem;">
                        ثقة النموذج: {confidence:.1f}%
                      </div>
                    </div>""", unsafe_allow_html=True)
            
            with col_r2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pass_prob,
                    title={"text": "احتمالية النجاح%", "font": {"family":"Cairo","size":13}},
                    gauge={
                        "axis": {"range":[0,100], "tickcolor":"#90a4ae"},
                        "bar": {"color": "#00e676" if pass_prob >= 50 else "#ff1744"},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "rgba(0,212,255,0.3)",
                        "steps": [
                            {"range":[0,50], "color":"rgba(255,23,68,0.1)"},
                            {"range":[50,100],"color":"rgba(0,230,118,0.1)"}
                        ],
                        "threshold": {"line":{"color":"#00d4ff","width":3},"value":50}
                    },
                    number={"suffix":"%","font":{"family":"Orbitron","size":24,"color":"#00d4ff"}}
                ))
                fig_gauge.update_layout(
                    height=220,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8eaf6", family="Cairo"),
                    margin=dict(l=20,r=20,t=50,b=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_r3:
                fig_bar = go.Figure(go.Bar(
                    x=["راسب","ناجح"],
                    y=[proba[0]*100, proba[1]*100],
                    marker_color=["rgba(255,23,68,0.8)","rgba(0,230,118,0.8)"],
                    text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    textposition='auto'
                ))
                fig_bar.update_layout(
                    title="توزيع الاحتمالات",
                    height=220,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13,17,23,0.8)",
                    font=dict(color="#e8eaf6", family="Cairo"),
                    margin=dict(l=10,r=10,t=40,b=10),
                    yaxis=dict(range=[0,100])
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Advice
            st.markdown("<div class='section-title'>💡 توصيات للطالب</div>", unsafe_allow_html=True)
            advice_cols = st.columns(4)
            advices = []
            if study_hours < 4:  advices.append(("📚", "زد ساعات الدراسة", "حاول الوصول لـ 6+ ساعات يومياً"))
            if attendance < 75:  advices.append(("🏫", "حسّن نسبة الحضور", "الحضور المنتظم يرفع الدرجات كثيراً"))
            if sleep_hours < 6:  advices.append(("😴", "نم بشكل كافٍ", "7-8 ساعات نوم تحسّن التركيز"))
            if motivation < 5:   advices.append(("🔥", "ارفع دافعيتك", "ضع أهدافاً واضحة قصيرة المدى"))
            if assignments < 60: advices.append(("📓", "أكمل واجباتك", "الواجبات تعزز الفهم والدرجات"))
            if not advices:
                advices = [("⭐","أداء ممتاز!","استمر في هذا المستوى الرائع")]

            for i, (icon, title, desc) in enumerate(advices[:4]):
                with advice_cols[i % 4]:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                      <div style="font-size:2rem;">{icon}</div>
                      <div style="color:#00d4ff; font-weight:700; font-size:0.9rem;">{title}</div>
                      <div style="color:#90a4ae; font-size:0.78rem; margin-top:4px;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    if not st.session_state.trained:
        st.info("⚠️ قم بتدريب النموذج أولاً من الشريط الجانبي")
    else:
        model  = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred)

        # Metrics row
        m1,m2,m3,m4,m5 = st.columns(5)
        metrics_data = [
            (m1, f"{acc*100:.1f}%", "الدقة الكلية", "Accuracy"),
            (m2, f"{report['1']['precision']*100:.1f}%", "الدقة الإيجابية", "Precision"),
            (m3, f"{report['1']['recall']*100:.1f}%", "الاستدعاء", "Recall"),
            (m4, f"{report['1']['f1-score']*100:.1f}%", "F1 Score", "F-Measure"),
            (m5, f"{model.n_iter_}",  "عدد التكرارات", "Iterations"),
        ]
        for col, val, ar_label, en_label in metrics_data:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{val}</div>
                  <div class="metric-label">{ar_label}</div>
                  <div style="color:#7b2ff7; font-size:0.7rem;">{en_label}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_confusion(cm), use_container_width=True)
        with c2: st.plotly_chart(plot_roc(model, X_test, y_test), use_container_width=True)
        
        c3, c4 = st.columns(2)
        with c3:
            loss_fig = plot_loss(model)
            if loss_fig: st.plotly_chart(loss_fig, use_container_width=True)
        with c4:
            feature_names = [c for c in st.session_state.df.columns if c != "النتيجة"]
            st.plotly_chart(plot_feature_importance(model, feature_names), use_container_width=True)

        # Full classification report
        st.markdown("<div class='section-title'>📋 تقرير التصنيف الكامل</div>", unsafe_allow_html=True)
        df_report = pd.DataFrame(report).T.round(3)
        st.dataframe(df_report.style.background_gradient(cmap='Blues'), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — DATA ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    df = st.session_state.df if st.session_state.df is not None else generate_dataset()
    if st.session_state.df is None:
        st.session_state.df = df

    st.markdown("<div class='section-title'>📊 نظرة عامة على البيانات</div>", unsafe_allow_html=True)
    
    c1,c2,c3,c4 = st.columns(4)
    pass_rate = df["النتيجة"].mean() * 100
    for col, val, label in [
        (c1, len(df), "إجمالي الطلاب"),
        (c2, f"{pass_rate:.1f}%", "نسبة النجاح"),
        (c3, f"{(100-pass_rate):.1f}%", "نسبة الرسوب"),
        (c4, df.shape[1]-1, "عدد المتغيرات"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_distribution(df), use_container_width=True)
    st.plotly_chart(plot_scatter_matrix(df), use_container_width=True)

    st.markdown("<div class='section-title'>🔗 مصفوفة الارتباط</div>", unsafe_allow_html=True)
    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=[[0,"#ff1744"],[0.5,"#0d1117"],[1,"#00d4ff"]],
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(title="Correlation Matrix", **DARK_LAYOUT, height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<div class='section-title'>📄 عينة من البيانات</div>", unsafe_allow_html=True)
    n_show = st.slider("عدد الصفوف", 5, 50, 10)
    st.dataframe(df.head(n_show), use_container_width=True)
    
    csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button("⬇️ تحميل مجموعة البيانات", csv, "student_data.csv", "text/csv")

# ══════════════════════════════════════════════
#  TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>🧠 معمارية الشبكة العصبية (MLP)</div>", unsafe_allow_html=True)
    
    if st.session_state.trained:
        model = st.session_state.model
        layers_info = [8] + list(model.hidden_layer_sizes) + [1]
        
        # Visualize network layers
        fig_arch = go.Figure()
        max_nodes = max(layers_info)
        layer_names = (
            ["Input\n(8 features)"] +
            [f"Hidden {i+1}\n({s} nodes)" for i, s in enumerate(model.hidden_layer_sizes)] +
            ["Output\n(Pass/Fail)"]
        )
        
        for l_idx, (n_nodes, l_name) in enumerate(zip(layers_info, layer_names)):
            x_pos = l_idx
            display_nodes = min(n_nodes, 10)
            y_positions = np.linspace(-display_nodes/2, display_nodes/2, display_nodes)
            
            for y_pos in y_positions:
                color = "#00d4ff" if l_idx == 0 else ("#7b2ff7" if l_idx < len(layers_info)-1 else "#ff6b35")
                fig_arch.add_trace(go.Scatter(
                    x=[x_pos], y=[y_pos],
                    mode="markers",
                    marker=dict(size=20, color=color, line=dict(color="white", width=1)),
                    hoverinfo="skip", showlegend=False
                ))
            
            if l_idx < len(layers_info)-1:
                next_nodes = min(layers_info[l_idx+1], 10)
                next_y = np.linspace(-next_nodes/2, next_nodes/2, next_nodes)
                for y1 in y_positions[:3]:
                    for y2 in next_y[:3]:
                        fig_arch.add_trace(go.Scatter(
                            x=[l_idx, l_idx+1], y=[y1, y2],
                            mode="lines",
                            line=dict(color="rgba(0,212,255,0.1)", width=0.5),
                            hoverinfo="skip", showlegend=False
                        ))
            
            fig_arch.add_annotation(
                x=l_idx, y=-display_nodes/2 - 1.2,
                text=l_name.replace("\n","<br>"),
                showarrow=False,
                font=dict(color="#90a4ae", size=11, family="Cairo"),
                align="center"
            )
        
        fig_arch.update_layout(
            title="بنية الشبكة العصبية متعددة الطبقات (MLP)",
            **DARK_LAYOUT,
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_arch, use_container_width=True)
        
        # Architecture summary
        st.markdown("<div class='section-title'>📐 تفاصيل البنية</div>", unsafe_allow_html=True)
        arch_cols = st.columns(len(layers_info))
        colors_list = ["#00d4ff"] + ["#7b2ff7"]*len(model.hidden_layer_sizes) + ["#ff6b35"]
        icons_list  = ["🔵"] + ["🟣"]*len(model.hidden_layer_sizes) + ["🟠"]
        names_list  = ["طبقة الإدخال"] + [f"طبقة مخفية {i+1}" for i in range(len(model.hidden_layer_sizes))] + ["طبقة الإخراج"]
        
        for col, n, name, color, icon in zip(arch_cols, layers_info, names_list, colors_list, icons_list):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:1.5rem;">{icon}</div>
                  <div class="metric-value" style="color:{color}; font-size:1.5rem;">{n}</div>
                  <div class="metric-label">{name}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        info_c1, info_c2 = st.columns(2)
        with info_c1:
            st.markdown(f"""
            <div class="glass-card">
              <div class="section-title">⚙️ إعدادات النموذج</div>
              <table style="width:100%; color:#e8eaf6; font-size:0.9rem;">
                <tr><td style="color:#90a4ae;">دالة التفعيل</td><td style="color:#00d4ff;">{model.activation}</td></tr>
                <tr><td style="color:#90a4ae;">المحسّن</td><td style="color:#00d4ff;">{model.solver}</td></tr>
                <tr><td style="color:#90a4ae;">معامل Alpha</td><td style="color:#00d4ff;">{model.alpha}</td></tr>
                <tr><td style="color:#90a4ae;">عدد التكرارات الفعلية</td><td style="color:#00d4ff;">{model.n_iter_}</td></tr>
                <tr><td style="color:#90a4ae;">معدل التعلم</td><td style="color:#00d4ff;">{model.learning_rate_init}</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)
        
        with info_c2:
            total_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
            st.markdown(f"""
            <div class="glass-card">
              <div class="section-title">📊 إحصاءات النموذج</div>
              <table style="width:100%; color:#e8eaf6; font-size:0.9rem;">
                <tr><td style="color:#90a4ae;">إجمالي المعاملات</td><td style="color:#7b2ff7;">{total_params:,}</td></tr>
                <tr><td style="color:#90a4ae;">عدد الطبقات الكلية</td><td style="color:#7b2ff7;">{len(layers_info)}</td></tr>
                <tr><td style="color:#90a4ae;">خسارة التدريب النهائية</td><td style="color:#7b2ff7;">{model.loss_:.4f}</td></tr>
                <tr><td style="color:#90a4ae;">إيقاف مبكر</td><td style="color:#7b2ff7;">✅ مفعّل</td></tr>
                <tr><td style="color:#90a4ae;">نوع النموذج</td><td style="color:#7b2ff7;">MLPClassifier</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("⚠️ قم بتدريب النموذج أولاً لعرض معمارية الشبكة")

# ══════════════════════════════════════════════
#  TAB 5 — PREDICTION HISTORY
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>📋 سجل التنبؤات</div>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
          <div style="font-size:3rem;">📭</div>
          <div style="color:#90a4ae; margin-top:1rem;">لا توجد تنبؤات بعد</div>
          <div style="color:#7b2ff7; font-size:0.85rem;">انتقل لتبويب التنبؤ الفوري وادخل بيانات الطالب</div>
        </div>""", unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        
        pass_count = sum(1 for h in st.session_state.history if "ناجح" in h["النتيجة"])
        fail_count = len(st.session_state.history) - pass_count
        
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{len(st.session_state.history)}</div>
              <div class="metric-label">إجمالي التنبؤات</div>
            </div>""", unsafe_allow_html=True)
        with h2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#00e676;">{pass_count}</div>
              <div class="metric-label">ناجح</div>
            </div>""", unsafe_allow_html=True)
        with h3:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#ff1744;">{fail_count}</div>
              <div class="metric-label">راسب</div>
            </div>""", unsafe_allow_html=True)
        
        fig_hist = go.Figure(go.Pie(
            values=[pass_count, fail_count],
            labels=["ناجح","راسب"],
            marker_colors=["#00e676","#ff1744"],
            hole=0.5
        ))
        fig_hist.update_layout(
            title="توزيع نتائج التنبؤات",
            **DARK_LAYOUT, height=350
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        if st.button("🗑️ مسح السجل"):
            st.session_state.history = []
            st.rerun()

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(0,212,255,0.15); margin-top:3rem;'>
<div style='text-align:center; color:#546e7a; font-size:0.8rem; padding:1rem;'>
  🎓 EduPredict AI • Powered by <span style='color:#7b2ff7;'>ANN & MLP</span> • Built with
  <span style='color:#00d4ff;'>Streamlit + Scikit-learn + Plotly</span>
</div>
""", unsafe_allow_html=True)
