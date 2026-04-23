"""
=============================================================================
POLYMER Tg PREDICTION — Full Streamlit Deployment
Mirrors the complete Jupyter Notebook (230107041.ipynb) step-by-step

Student Name : Kshitij Verma
Roll No      : 230107041

HOW TO RUN:
  pip install streamlit rdkit scikit-learn pandas numpy matplotlib seaborn joblib
  streamlit run 230107041_app.py

FOLDER STRUCTURE:
  230107041_app.py
  JCIM_sup_bigsmiles.csv
  outputs/
      best_model_rf.pkl
      imputer.pkl
      scaler.pkl
      pca.pkl
      feature_names.pkl
      top_features.pkl
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Polymer Tg ML Pipeline | 230107041",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1b2a; }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

  .page-header {
    background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 100%);
    padding: 1.4rem 2rem; border-radius: 10px; margin-bottom: 1.5rem;
  }
  .page-header h1 { color: white; margin: 0; font-size: 1.85rem; }
  .page-header p  { color: #cce4f7; margin: 0.3rem 0 0; font-size: 0.95rem; }

  .metric-card {
    background: #f0f6ff; border-left: 5px solid #2e75b6;
    border-radius: 8px; padding: 0.8rem 1.2rem;
  }
  .metric-card .lbl { font-size: 0.78rem; color: #666; font-weight: 700; text-transform: uppercase; }
  .metric-card .val { font-size: 1.7rem; font-weight: 900; color: #1f4e79; }

  .result-box {
    background: linear-gradient(135deg, #e8f4f8, #f0f9ff);
    border-left: 6px solid #1f77b4;
    padding: 1.4rem 1.8rem; border-radius: 8px; margin-top: 1rem;
  }
  .result-tg  { font-size: 3.2rem; font-weight: 900; color: #1f4e79; line-height: 1; }
  .result-cat { font-size: 1.2rem; font-weight: 700; margin-top: 0.4rem; }

  .step-badge {
    background: #2e75b6; color: white; padding: 0.2rem 0.8rem;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
    display: inline-block; margin-bottom: 0.6rem;
  }
  .insight-box {
    background: #f8fffe; border-left: 5px solid #1abc9c;
    padding: 0.8rem 1.2rem; border-radius: 6px; margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================
def clean_smiles(s):
    return s.replace("*", "[H]")

@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv("JCIM_sup_bigsmiles.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = ["SMILES", "BigSMILES", "Tg_C"]
    return df

@st.cache_data(show_spinner=False)
def build_features(_df):
    records, failed = [], []
    for idx, row in _df.iterrows():
        mol = Chem.MolFromSmiles(clean_smiles(row["SMILES"]))
        if mol is None:
            failed.append(idx); continue
        desc = {}
        for name, func in Descriptors._descList:
            try:    desc[name] = func(mol)
            except: desc[name] = np.nan
        desc["Tg_C"] = row["Tg_C"]; desc["SMILES"] = row["SMILES"]
        records.append(desc)
    return pd.DataFrame(records), len(failed)

@st.cache_data(show_spinner=False)
def preprocess(_feat_df):
    X_raw = _feat_df.drop(columns=["Tg_C", "SMILES"])
    y     = _feat_df["Tg_C"].values
    imputer  = SimpleImputer(strategy="median")
    X_imp    = imputer.fit_transform(X_raw)
    X_imp_df = pd.DataFrame(X_imp, columns=X_raw.columns)
    zero_var = X_imp_df.var()[X_imp_df.var() == 0].index.tolist()
    X_filt   = X_imp_df.drop(columns=zero_var)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_filt)
    return X_scaled, y, X_filt.columns.tolist(), imputer, scaler, len(zero_var)

@st.cache_data(show_spinner=False)
def run_pca(_X):
    pf = PCA(random_state=42); pf.fit(_X)
    cv = np.cumsum(pf.explained_variance_ratio_)
    n  = int(np.argmax(cv >= 0.95)) + 1
    p  = PCA(n_components=n, random_state=42)
    return p, p.fit_transform(_X), pf.explained_variance_ratio_, cv, n

@st.cache_data(show_spinner=False)
def run_feat_sel(_X, _y, _names):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(_X, _y)
    imp  = pd.Series(rf.feature_importances_, index=_names).sort_values(ascending=False)
    top  = imp.head(30).index.tolist()
    Xsel = pd.DataFrame(_X, columns=_names)[top].values
    return imp, top, Xsel

@st.cache_data(show_spinner=False)
def train_models(_Xo, _Xp, _Xs, _y):
    def ev(m, Xtr, Xte, ytr, yte):
        m.fit(Xtr, ytr); yp = m.predict(Xte)
        return dict(RMSE=round(float(np.sqrt(mean_squared_error(yte,yp))),3),
                    MAE=round(float(mean_absolute_error(yte,yp)),3),
                    R2=round(float(r2_score(yte,yp)),4), y_pred=yp)

    Xtr_o,Xte_o,ytr,yte = train_test_split(_Xo,_y,test_size=0.2,random_state=42)
    Xtr_p,Xte_p,_,_     = train_test_split(_Xp,_y,test_size=0.2,random_state=42)
    Xtr_s,Xte_s,_,_     = train_test_split(_Xs,_y,test_size=0.2,random_state=42)

    mdefs = {"Ridge":Ridge(alpha=1.0), "Decision Tree":DecisionTreeRegressor(random_state=42),
             "Random Forest":RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1),
             "SVR":SVR(kernel="rbf",C=10,epsilon=0.1)}

    res = {}
    for fsn,(Xtr,Xte) in [("Original",(Xtr_o,Xte_o)),("PCA",(Xtr_p,Xte_p)),("RF-Sel",(Xtr_s,Xte_s))]:
        for mn,md in mdefs.items():
            res[f"{mn}|{fsn}"] = ev(md, Xtr, Xte, ytr, yte)
    return res, ytr, yte, Xtr_o, Xte_o

@st.cache_data(show_spinner=False)
def tune_rf(_Xtr, _ytr, _Xte, _yte):
    gs = GridSearchCV(RandomForestRegressor(random_state=42,n_jobs=-1),
                      {"n_estimators":[100,200],"max_depth":[None,10,20],
                       "min_samples_split":[2,5],"max_features":["sqrt","log2"]},
                      cv=5, scoring="r2", n_jobs=-1)
    gs.fit(_Xtr,_ytr)
    yp = gs.best_estimator_.predict(_Xte)
    return gs.best_estimator_, gs.best_params_, gs.best_score_, yp

@st.cache_data(show_spinner=False)
def tune_svr(_Xtr, _ytr, _Xte, _yte):
    gs = GridSearchCV(SVR(),
                      {"C":[1,10,50],"epsilon":[0.1,0.5,1.0],"kernel":["rbf","linear"]},
                      cv=5, scoring="r2", n_jobs=-1)
    gs.fit(_Xtr,_ytr)
    yp = gs.best_estimator_.predict(_Xte)
    return gs.best_estimator_, gs.best_params_, gs.best_score_, yp

def single_desc(smiles):
    mol = Chem.MolFromSmiles(clean_smiles(smiles))
    if mol is None: return None
    d = {}
    for name,func in Descriptors._descList:
        try:    d[name]=func(mol)
        except: d[name]=np.nan
    return d


# =============================================================================
# LOAD ALL DATA (cached)
# =============================================================================
with st.spinner("🔄 Loading dataset and computing descriptors (first run ~60 s)..."):
    df = load_dataset()
    feat_df, n_failed = build_features(df)
    X_scaled, y, feature_names, imputer, scaler, n_zero_var = preprocess(feat_df)
    pca, X_pca, ev_ratio, cumvar, N_PCA = run_pca(X_scaled)
    importances, top_features, X_sel    = run_feat_sel(X_scaled, y, feature_names)
    results, y_train, y_test, X_train, X_test = train_models(X_scaled, X_pca, X_sel, y)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🧪 Polymer Tg ML")
    st.markdown("**Kshitij Verma**  \n**Roll No:** 230107041")
    st.markdown("---")
    page = st.radio("📋 Navigate to Step:", [
        "🏠  Overview",
        "📂  Step 2 — Dataset",
        "⚙️   Step 3 — Preprocessing",
        "📊  Step 4 — EDA",
        "📉  Step 5 — Dim. Reduction",
        "✂️   Step 6 — Train-Test Split",
        "🤖  Step 7 — Train Models",
        "🔧  Step 8 — Hyperparameter Tuning",
        "📈  Step 9 — Evaluation",
        "🏆  Step 10 — Model Comparison",
        "✅  Step 11 — Best Model",
        "💾  Step 12 — Save & Deploy",
        "🔮  Live Prediction",
    ])
    st.markdown("---")
    st.markdown("**Pipeline Ready ✅**")
    st.markdown(f"- {len(feat_df)} molecules processed  \n"
                f"- {len(feature_names)} features  \n"
                f"- {N_PCA} PCA components  \n"
                f"- Best R² = **0.611**")


# =============================================================================
#  ██████  ██    ██ ███████ ██████  ██    ██ ██ ███████ ██     ██
# ██    ██ ██    ██ ██      ██   ██ ██    ██ ██ ██      ██     ██
# ██    ██ ██    ██ █████   ██████  ██    ██ ██ █████   ██  █  ██
# ██    ██  ██  ██  ██      ██   ██  ██  ██  ██ ██      ██ ███ ██
#  ██████    ████   ███████ ██   ██   ████   ██ ███████  ███ ███
# =============================================================================

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown("""<div class="page-header">
      <h1>🧪 Polymer Glass Transition Temperature — ML Pipeline</h1>
      <p>End-to-end deployment of Jupyter Notebook 230107041.ipynb · Kshitij Verma · Roll No: 230107041</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
This Streamlit app is a **fully interactive deployment of the Jupyter Notebook**.
Every step from the notebook is replicated here — use the sidebar to navigate.

| Step | Notebook Cell | This App Page |
|------|--------------|---------------|
| 2 | Load dataset | 📂 Step 2 — Dataset |
| 3 | Descriptor generation & preprocessing | ⚙️ Step 3 — Preprocessing |
| 4 | EDA — plots & statistics | 📊 Step 4 — EDA |
| 5 | PCA + RF feature selection | 📉 Step 5 — Dim. Reduction |
| 6 | Train-test split | ✂️ Step 6 — Split |
| 7 | Train 4 models × 3 feature sets | 🤖 Step 7 — Train Models |
| 8 | GridSearchCV tuning | 🔧 Step 8 — Tuning |
| 9 | Residual analysis & evaluation | 📈 Step 9 — Evaluation |
| 10 | Full model comparison chart | 🏆 Step 10 — Comparison |
| 11 | Best model selection | ✅ Step 11 — Best Model |
| 12 | Save artifacts & deployment | 💾 Step 12 — Deploy |
| — | Real-time prediction | 🔮 Live Prediction |
        """)
    with col_r:
        for label, val in [("Total Polymers","662"),("RDKit Descriptors","217"),
                           ("Models Trained","12"),("Best R²","0.611"),
                           ("Best RMSE","76.7 °C"),("Best Model","Random Forest")]:
            st.markdown(f"""<div class="metric-card" style="margin-bottom:0.6rem">
              <div class="lbl">{label}</div><div class="val">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset Quick Preview")
    st.dataframe(df.head(8), use_container_width=True)


# ── STEP 2 ────────────────────────────────────────────────────────────────────
elif "Step 2" in page:
    st.markdown('<div class="step-badge">Step 2</div>', unsafe_allow_html=True)
    st.title("📂 Load Dataset")

    st.code('df = pd.read_csv("JCIM_sup_bigsmiles.csv")\ndf.columns = ["SMILES", "BigSMILES", "Tg_C"]',
            language="python")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",    str(df.shape[0]))
    c2.metric("Columns", str(df.shape[1]))
    c3.metric("Tg Min",  f"{df.Tg_C.min():.1f} °C")
    c4.metric("Tg Max",  f"{df.Tg_C.max():.1f} °C")

    st.markdown("#### Full Dataset Table")
    st.dataframe(df, use_container_width=True, height=380)

    st.markdown("#### Target Statistics")
    st.dataframe(df["Tg_C"].describe().rename("Tg (°C)").to_frame().T.round(3),
                 use_container_width=True)

    st.markdown("#### Sample SMILES")
    for i, row in df.head(5).iterrows():
        with st.expander(f"Polymer #{i+1}  |  Tg = {row.Tg_C:.2f} °C"):
            st.markdown(f"**SMILES:** `{row.SMILES}`")
            st.markdown(f"**BigSMILES:** `{row.BigSMILES[:120]}...`")


# ── STEP 3 ────────────────────────────────────────────────────────────────────
elif "Step 3" in page:
    st.markdown('<div class="step-badge">Step 3</div>', unsafe_allow_html=True)
    st.title("⚙️ Data Preprocessing")

    for title, body in {
        "1️⃣  SMILES Cleaning": "Replace `*` chain-end wildcards with `[H]` so RDKit can parse each repeat unit as a closed molecule.",
        "2️⃣  Descriptor Generation": "Compute all **217 RDKit 2D molecular descriptors** — topology, electronics, surface area, LogP, BCUT, VSA…",
        "3️⃣  Median Imputation": "Fill NaN values (partial-charge & BCUT descriptors in edge-case molecules) with **column median**.",
        f"4️⃣  Variance Filtering (removed {n_zero_var})": "Drop constant features — they carry zero predictive information.",
        "5️⃣  StandardScaler": "Scale all features to **mean=0, std=1** — required for SVR and Ridge, harmless for trees.",
    }.items():
        with st.expander(title, expanded=True):
            st.markdown(body)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Valid molecules",   str(len(feat_df)))
    c2.metric("Failed parsing",    str(n_failed))
    c3.metric("Raw descriptors",   "217")
    c4.metric("Final features",    str(len(feature_names)))

    st.markdown("#### Code — Descriptor Generation")
    st.code("""def compute_rdkit_descriptors(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str.replace("*","[H]"))
    if mol is None: return None
    desc = {}
    for name, func in Descriptors._descList:
        try:    desc[name] = func(mol)
        except: desc[name] = np.nan
    return desc""", language="python")

    st.markdown("#### Sample of Scaled Feature Matrix (first 5 rows, first 8 features)")
    st.dataframe(pd.DataFrame(X_scaled[:5], columns=feature_names)[feature_names[:8]].round(3),
                 use_container_width=True)


# ── STEP 4 — EDA ──────────────────────────────────────────────────────────────
elif "Step 4" in page:
    st.markdown('<div class="step-badge">Step 4</div>', unsafe_allow_html=True)
    st.title("📊 Exploratory Data Analysis")

    # Tg distribution
    st.markdown("### Tg Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(y, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(y.mean(),     color="red",    linestyle="--", lw=2, label=f"Mean   = {y.mean():.1f} °C")
    axes[0].axvline(np.median(y), color="orange", linestyle="--", lw=2, label=f"Median = {np.median(y):.1f} °C")
    axes[0].set_xlabel("Tg (°C)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Tg (°C)"); axes[0].legend()
    axes[1].boxplot(y, patch_artist=True, notch=True,
                    boxprops=dict(facecolor="lightyellow", color="navy"),
                    medianprops=dict(color="red", linewidth=2.5),
                    flierprops=dict(marker="o", markersize=4, alpha=0.4))
    axes[1].set_ylabel("Tg (°C)"); axes[1].set_title("Boxplot — Outlier Detection")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Summary stats
    st.markdown("### Summary Statistics")
    st.dataframe(pd.Series(y, name="Tg (°C)").describe().to_frame().T.round(3),
                 use_container_width=True)

    # Correlation chart
    st.markdown("### Top Descriptor Correlations with Tg")
    n_top = st.slider("Show top N descriptors:", 5, 40, 20)
    cdf   = pd.DataFrame(X_scaled, columns=feature_names); cdf["Tg_C"] = y
    corrs = cdf.corr()["Tg_C"].drop("Tg_C").abs().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, max(4, int(n_top*0.38))))
    corrs.head(n_top).sort_values().plot(kind="barh", ax=ax2, color="teal", edgecolor="white")
    ax2.set_xlabel("Absolute Pearson Correlation with Tg")
    ax2.set_title(f"Top {n_top} Descriptors Correlated with Tg")
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Insights
    st.markdown("### 💡 Key Insights")
    for ins in [
        f"Tg spans **{y.min():.0f} °C to {y.max():.0f} °C** — extremely wide range, reflecting diverse polymer families",
        "Distribution is **right-skewed** (mean > median) — more low/mid-Tg polymers in the corpus",
        "**FractionCSP3** and ring-count descriptors show the strongest linear correlation with Tg",
        "High FractionCSP3 = flexible/saturated backbone → **low Tg** | Low FractionCSP3 = aromatic/rigid → **high Tg**",
        f"**{n_failed} molecules** failed RDKit parsing (0.3%) and were excluded from training",
    ]:
        st.markdown(f'<div class="insight-box">🔍 {ins}</div>', unsafe_allow_html=True)


# ── STEP 5 — DIM REDUCTION ────────────────────────────────────────────────────
elif "Step 5" in page:
    st.markdown('<div class="step-badge">Step 5</div>', unsafe_allow_html=True)
    st.title("📉 Dimensionality Reduction")

    tab1, tab2 = st.tabs(["5A — Principal Component Analysis (PCA)",
                           "5B — RF Feature Selection"])

    with tab1:
        c1,c2,c3 = st.columns(3)
        c1.metric("Original features",       str(len(feature_names)))
        c2.metric("PCA components (95% var)", str(N_PCA))
        c3.metric("Variance retained",        f"{pca.explained_variance_ratio_.sum()*100:.2f}%")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].bar(range(1,31), ev_ratio[:30]*100, color="steelblue", edgecolor="white")
        axes[0].set_xlabel("PC"); axes[0].set_ylabel("Explained Variance (%)")
        axes[0].set_title("Scree Plot — Top 30 PCs")

        axes[1].plot(range(1,len(cumvar)+1), cumvar*100, "b-", lw=2)
        axes[1].fill_between(range(1,len(cumvar)+1), cumvar*100, alpha=0.12, color="blue")
        for thr, col, lab in [(0.90,"orange","90%"), (0.95,"red","95%")]:
            n = int(np.argmax(cumvar>=thr))+1
            axes[1].axhline(thr*100, color=col, linestyle="--", lw=1.5, label=f"{lab} → {n} PCs")
        axes[1].axvline(N_PCA, color="green", linestyle="--", lw=1.5, label=f"Selected: {N_PCA}")
        axes[1].set_xlabel("n components"); axes[1].set_ylabel("Cumulative Variance (%)")
        axes[1].set_title("Cumulative Variance Explained"); axes[1].legend(fontsize=9)
        axes[1].set_xlim(1, 150)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(f"**95% threshold chosen:** reduces {len(feature_names)} → {N_PCA} features "
                f"({(1-N_PCA/len(feature_names))*100:.0f}% reduction) while retaining dominant variance.")

    with tab2:
        n_show = st.slider("Show top N features:", 5, 40, 20, key="fs_slider")
        fig2, ax2 = plt.subplots(figsize=(10, max(4, int(n_show*0.4))))
        importances.head(n_show).sort_values().plot(
            kind="barh", ax=ax2, color="darkorange", edgecolor="white")
        ax2.set_xlabel("Importance (MDI)"); ax2.set_title(f"Top {n_show} Features by RF Importance")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("#### Top 30 Selected Features")
        imp_df = importances.head(30).reset_index()
        imp_df.columns = ["Descriptor","Importance"]; imp_df["Importance"] = imp_df["Importance"].round(5)
        st.dataframe(imp_df, use_container_width=True, height=300)
        st.success("**`FractionCSP3`** accounts for ~40% of total importance — "
                   "directly encodes backbone flexibility, the primary physical driver of Tg.")


# ── STEP 6 ────────────────────────────────────────────────────────────────────
elif "Step 6" in page:
    st.markdown('<div class="step-badge">Step 6</div>', unsafe_allow_html=True)
    st.title("✂️ Train-Test Split (80 / 20)")

    c1,c2,c3 = st.columns(3)
    c1.metric("Train samples", f"{len(y_train)}  ({len(y_train)/len(y)*100:.0f}%)")
    c2.metric("Test samples",  f"{len(y_test)}   ({len(y_test)/len(y)*100:.0f}%)")
    c3.metric("random_state", "42")

    st.code("""X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)""", language="python")

    # Distribution comparison
    idx_all = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx_all, test_size=0.2, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y[idx_tr], bins=35, alpha=0.65, color="steelblue",  edgecolor="white", label=f"Train (n={len(idx_tr)})")
    ax.hist(y[idx_te], bins=35, alpha=0.65, color="darkorange", edgecolor="white", label=f"Test  (n={len(idx_te)})")
    ax.set_xlabel("Tg (°C)"); ax.set_ylabel("Count")
    ax.set_title("Tg Distribution: Train vs Test"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.info("Both sets cover the full Tg range — a random 80/20 split is valid for this i.i.d. tabular dataset. "
            "The same random_state=42 is applied to all three feature representations.")


# ── STEP 7 — TRAIN MODELS ─────────────────────────────────────────────────────
elif "Step 7" in page:
    st.markdown('<div class="step-badge">Step 7</div>', unsafe_allow_html=True)
    st.title("🤖 Train Multiple Models")

    st.markdown("4 models × 3 feature sets = **12 combinations**")

    rows = []
    for key, r in results.items():
        mn, fs = key.split("|")
        rows.append({"Model":mn,"Feature Set":fs,"RMSE (°C)":r["RMSE"],"MAE (°C)":r["MAE"],"R²":r["R2"]})
    comp_df = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)

    fs_sel = st.multiselect("Filter Feature Set:", ["Original","PCA","RF-Sel"],
                             default=["Original","PCA","RF-Sel"])
    filt = comp_df[comp_df["Feature Set"].isin(fs_sel)]
    st.dataframe(
        filt.style.background_gradient(subset=["R²"],       cmap="RdYlGn")
                  .background_gradient(subset=["RMSE (°C)"],cmap="RdYlGn_r"),
        use_container_width=True
    )

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    p1 = comp_df.pivot_table(index="Model",columns="Feature Set",values="R²")
    p2 = comp_df.pivot_table(index="Model",columns="Feature Set",values="RMSE (°C)")
    p1.plot(kind="bar",ax=axes[0],edgecolor="white",alpha=0.87,width=0.75)
    axes[0].set_title("R² by Model & Feature Set"); axes[0].set_ylabel("R²")
    axes[0].set_xticklabels(axes[0].get_xticklabels(),rotation=22,ha="right")
    p2.plot(kind="bar",ax=axes[1],edgecolor="white",alpha=0.87,width=0.75,colormap="Set2")
    axes[1].set_title("RMSE by Model & Feature Set"); axes[1].set_ylabel("RMSE (°C)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(),rotation=22,ha="right")
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ── STEP 8 — HYPERPARAMETER TUNING ───────────────────────────────────────────
elif "Step 8" in page:
    st.markdown('<div class="step-badge">Step 8</div>', unsafe_allow_html=True)
    st.title("🔧 Hyperparameter Tuning (GridSearchCV)")

    st.info("5-fold cross-validation on training set. Click button to run live tuning (~2 min).")

    if st.button("▶  Run Live Hyperparameter Tuning", type="primary"):
        with st.spinner("Tuning Random Forest (GridSearchCV, 5-fold)..."):
            rf_best, rf_params, rf_cv, rf_yp = tune_rf(X_train, y_train, X_test, y_test)
        with st.spinner("Tuning SVR (GridSearchCV, 5-fold)..."):
            svr_best, svr_params, svr_cv, svr_yp = tune_svr(X_train, y_train, X_test, y_test)
        st.session_state["rf_tuned"]  = (rf_best, rf_params,  rf_cv,  rf_yp)
        st.session_state["svr_tuned"] = (svr_best,svr_params, svr_cv, svr_yp)

    if "rf_tuned" in st.session_state:
        _, rf_params, rf_cv, rf_yp    = st.session_state["rf_tuned"]
        _, svr_params, svr_cv, svr_yp = st.session_state["svr_tuned"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🌲 Tuned Random Forest")
            st.json(rf_params)
            st.metric("CV R²",   f"{rf_cv:.4f}")
            st.metric("Test R²", f"{r2_score(y_test,rf_yp):.4f}")
            st.metric("RMSE",    f"{np.sqrt(mean_squared_error(y_test,rf_yp)):.2f} °C")
        with col2:
            st.markdown("#### 🔷 Tuned SVR")
            st.json(svr_params)
            st.metric("CV R²",   f"{svr_cv:.4f}")
            st.metric("Test R²", f"{r2_score(y_test,svr_yp):.4f}")
            st.metric("RMSE",    f"{np.sqrt(mean_squared_error(y_test,svr_yp)):.2f} °C")
    else:
        st.markdown("#### Pre-computed Results (from Notebook)")
        st.markdown("""
| Model | Best Params | CV R² | Test R² | Test RMSE |
|---|---|---|---|---|
| **RF (Tuned)** | max_depth=10, n_est=200, max_features=sqrt | 0.600 | **0.611** | **76.7 °C** |
| SVR (Tuned) | kernel=linear, C=1, epsilon=1.0 | 0.550 | 0.493 | 87.6 °C |
""")

    st.markdown("#### Parameter Grid Used")
    st.markdown("""
| Model | Parameter | Values |
|---|---|---|
| Random Forest | n_estimators | 100, 200 |
| Random Forest | max_depth | None, 10, 20 |
| Random Forest | min_samples_split | 2, 5 |
| Random Forest | max_features | sqrt, log2 |
| SVR | C | 1, 10, 50 |
| SVR | epsilon | 0.1, 0.5, 1.0 |
| SVR | kernel | rbf, linear |
""")


# ── STEP 9 — EVALUATION ───────────────────────────────────────────────────────
elif "Step 9" in page:
    st.markdown('<div class="step-badge">Step 9</div>', unsafe_allow_html=True)
    st.title("📈 Model Evaluation — Best Model")

    rf_res    = results["Random Forest|Original"]
    y_pred    = rf_res["y_pred"]
    residuals = y_test - y_pred

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²",    f"{rf_res['R2']:.4f}")
    c2.metric("RMSE",  f"{rf_res['RMSE']} °C")
    c3.metric("MAE",   f"{rf_res['MAE']} °C")
    c4.metric("MAPE",  f"{mean_absolute_percentage_error(y_test,y_pred)*100:.1f}%")

    fig, axes = plt.subplots(1,3,figsize=(16,5))
    lo,hi = min(y_test.min(),y_pred.min()), max(y_test.max(),y_pred.max())
    axes[0].scatter(y_test,y_pred,alpha=0.55,color="steelblue",edgecolor="white",s=45)
    axes[0].plot([lo,hi],[lo,hi],"r--",lw=2,label="Perfect fit")
    axes[0].set_xlabel("Actual Tg (°C)"); axes[0].set_ylabel("Predicted Tg (°C)")
    axes[0].set_title(f"Predicted vs Actual  R²={rf_res['R2']:.4f}"); axes[0].legend()

    axes[1].scatter(y_pred,residuals,alpha=0.55,color="darkorange",edgecolor="white",s=45)
    axes[1].axhline(0,color="red",linestyle="--",lw=1.5)
    axes[1].axhline( 2*residuals.std(),color="gray",linestyle=":",lw=1.2)
    axes[1].axhline(-2*residuals.std(),color="gray",linestyle=":",lw=1.2)
    axes[1].set_xlabel("Predicted Tg (°C)"); axes[1].set_ylabel("Residual (°C)")
    axes[1].set_title("Residuals vs Predicted")

    axes[2].hist(residuals,bins=30,color="mediumseagreen",edgecolor="white",alpha=0.9)
    axes[2].axvline(0,color="red",linestyle="--",lw=1.5)
    axes[2].axvline( residuals.std(),color="gray",linestyle=":",label=f"±σ={residuals.std():.1f}")
    axes[2].axvline(-residuals.std(),color="gray",linestyle=":")
    axes[2].set_xlabel("Residual (°C)"); axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution"); axes[2].legend()

    plt.suptitle("Random Forest — Full Evaluation", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Residual Statistics")
    st.dataframe(pd.Series(residuals,name="Residuals (°C)").describe().round(3).to_frame().T,
                 use_container_width=True)

    st.markdown("#### Worst Predictions (top 10 by absolute error)")
    err_df = pd.DataFrame({"Actual Tg":y_test.round(2),"Predicted Tg":y_pred.round(2),
                            "Abs Error":np.abs(residuals).round(2)}
                          ).sort_values("Abs Error",ascending=False).head(10).reset_index(drop=True)
    st.dataframe(err_df, use_container_width=True)


# ── STEP 10 — COMPARISON ──────────────────────────────────────────────────────
elif "Step 10" in page:
    st.markdown('<div class="step-badge">Step 10</div>', unsafe_allow_html=True)
    st.title("🏆 Full Model Comparison")

    rows = []
    for key,r in results.items():
        mn,fs = key.split("|")
        rows.append({"Model":mn,"Feature Set":fs,"RMSE (°C)":r["RMSE"],"MAE (°C)":r["MAE"],"R²":r["R2"]})
    rows += [
        {"Model":"RF (Tuned)","Feature Set":"Original","RMSE (°C)":76.7,"MAE (°C)":59.6,"R²":0.611},
        {"Model":"SVR (Tuned)","Feature Set":"Original","RMSE (°C)":87.6,"MAE (°C)":67.4,"R²":0.493},
    ]
    comp = pd.DataFrame(rows).sort_values("R²",ascending=False).reset_index(drop=True)
    comp.index += 1

    st.dataframe(
        comp.style.background_gradient(subset=["R²"],       cmap="RdYlGn")
                  .background_gradient(subset=["RMSE (°C)"],cmap="RdYlGn_r")
                  .highlight_max(subset=["R²"],  color="#b6f7c1")
                  .highlight_min(subset=["RMSE (°C)"], color="#b6f7c1"),
        use_container_width=True
    )

    col1,col2 = st.columns(2)
    for ax_idx, (metric, cmap, label) in enumerate([("R²","RdYlGn","R²"),("RMSE (°C)","RdYlGn_r","RMSE (°C)")]):
        fig,ax = plt.subplots(figsize=(7,5))
        pivot = comp.pivot_table(index="Model",columns="Feature Set",values=metric)
        pivot.plot(kind="bar",ax=ax,edgecolor="white",alpha=0.87,width=0.75,
                   colormap="viridis" if ax_idx==0 else "Set2")
        ax.set_title(f"{label} by Model & Feature Set"); ax.set_ylabel(label)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=22,ha="right")
        plt.tight_layout()
        (col1 if ax_idx==0 else col2).pyplot(fig); plt.close()


# ── STEP 11 — BEST MODEL ──────────────────────────────────────────────────────
elif "Step 11" in page:
    st.markdown('<div class="step-badge">Step 11</div>', unsafe_allow_html=True)
    st.title("✅ Best Model — Selection & Justification")

    st.success("🏆 **Best Model: Tuned Random Forest Regressor** | Feature Set: Original (200 descriptors)")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²","0.6114"); c2.metric("RMSE","76.7 °C")
    c3.metric("MAE","59.6 °C"); c4.metric("Features","200")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Strengths")
        for s in ["Captures **nonlinear** structure–Tg relationships",
                  "Ensemble averaging reduces overfitting vs single tree",
                  "Robust to wide Tg range (600 °C spread)",
                  "Feature importance = **chemical interpretability**",
                  "Handles 200 correlated features natively"]:
            st.markdown(f"- {s}")
    with col2:
        st.markdown("#### ⚠️ Limitations")
        for s in ["Repeat-unit only — ignores chain length, tacticity",
                  "Cannot extrapolate beyond training chemical space",
                  "2D descriptors miss 3D conformational effects",
                  "Heavier than Ridge at inference time"]:
            st.markdown(f"- {s}")

    st.markdown("---")
    st.info("""**Why PCA did NOT improve RF performance:**
PCA maximises captured variance, not Tg predictive relevance.
Some high-variance descriptor directions are chemical noise.
RF itself handles correlated high-dimensional inputs natively — PCA
compression removed subtlety that the RF already handled well.

RF-selected features outperformed PCA **for Ridge Regression**:
Ridge + PCA → R² = −0.034 | Ridge + RF-selected → R² = 0.563 (+80% gain)""")

    st.warning("""**Why NOT time-series methods (ARIMA / LSTM)?**
The integer index column is an arbitrary enumeration — not a time axis.
Tg values show no autocorrelation across the index.
This is a tabular i.i.d. regression problem — temporal methods are scientifically invalid here.""")


# ── STEP 12 — DEPLOY ─────────────────────────────────────────────────────────
elif "Step 12" in page:
    st.markdown('<div class="step-badge">Step 12</div>', unsafe_allow_html=True)
    st.title("💾 Save Artifacts & Deployment")

    st.markdown("### Artifacts Saved with `joblib`")
    for fname,desc in {"best_model_rf.pkl":"Tuned Random Forest Regressor",
                        "imputer.pkl":       "MedianImputer (fit on train)",
                        "scaler.pkl":        "StandardScaler (fit on train)",
                        "pca.pkl":           "PCA object (67 components)",
                        "feature_names.pkl": "200 post-filter feature names",
                        "top_features.pkl":  "Top 30 RF-selected features"}.items():
        st.markdown(f"- `outputs/{fname}` — {desc}")

    st.code("""import joblib
joblib.dump(rf_best,       "outputs/best_model_rf.pkl")
joblib.dump(imputer,       "outputs/imputer.pkl")
joblib.dump(scaler,        "outputs/scaler.pkl")
joblib.dump(pca,           "outputs/pca.pkl")
joblib.dump(feature_names, "outputs/feature_names.pkl")
joblib.dump(top_features,  "outputs/top_features.pkl")""", language="python")

    st.markdown("---")
    st.markdown("### 🚀 Deployment Instructions")
    tab1,tab2,tab3 = st.tabs(["Local","Streamlit Cloud","Folder Structure"])
    with tab1:
        st.code("pip install streamlit rdkit scikit-learn pandas numpy matplotlib seaborn joblib\nstreamlit run 230107041_app.py", language="bash")
    with tab2:
        st.markdown("**requirements.txt:**")
        st.code("streamlit\nrdkit\nscikit-learn\npandas\nnumpy\nmatplotlib\nseaborn\njoblib")
        st.markdown("Push to GitHub → connect repo on [share.streamlit.io](https://share.streamlit.io)")
    with tab3:
        st.code("""project/
├── 230107041_app.py          ← This Streamlit app
├── 230107041.ipynb           ← Jupyter notebook
├── 230107041.py              ← Plain Python pipeline
├── JCIM_sup_bigsmiles.csv    ← Dataset
└── outputs/
    ├── best_model_rf.pkl
    ├── imputer.pkl
    ├── scaler.pkl
    ├── pca.pkl
    ├── feature_names.pkl
    └── top_features.pkl""")


# ── LIVE PREDICTION ───────────────────────────────────────────────────────────
elif "Live Prediction" in page:
    st.title("🔮 Live Polymer Tg Prediction")
    st.markdown("Enter any polymer repeat-unit SMILES (`*` = chain-end wildcards) for instant Tg prediction.")

    try:
        m_pred = joblib.load("outputs/best_model_rf.pkl")
        i_pred = joblib.load("outputs/imputer.pkl")
        s_pred = joblib.load("outputs/scaler.pkl")
        f_pred = joblib.load("outputs/feature_names.pkl")
        art_ok = True
    except:
        art_ok = False
        st.error("Saved model not found. Run the pipeline (Step 12) first.")

    EXAMPLES = {
        "Nylon-like polycarbonate": "*OC(=O)NCCNC(=O)OCC*",
        "Polythioether (low Tg)":   "*SCCCCC*",
        "Aliphatic polyester":      "*OC(=O)CCCCC*",
        "Aromatic polyurethane":    "*c1ccc(NC(=O)OCC*)cc1*",
        "Custom (type below)":      "",
    }

    col1, col2 = st.columns([3,1])
    with col1:
        ex = st.selectbox("Load example:", list(EXAMPLES.keys()))
        smiles_in = st.text_area("Polymer SMILES:", value=EXAMPLES[ex], height=80,
                                 placeholder="e.g. *OC(=O)NCCNC(=O)OCC*")

    if smiles_in.strip():
        mol_v = Chem.MolFromSmiles(clean_smiles(smiles_in.strip()))
        st.success("✅ Valid SMILES") if mol_v else st.error("❌ Invalid SMILES")

    if st.button("🔬 Predict Tg", type="primary", disabled=not art_ok):
        if not smiles_in.strip():
            st.warning("Enter a SMILES string first.")
        else:
            with st.spinner("Computing molecular descriptors..."):
                desc = single_desc(smiles_in.strip())
            if desc is None:
                st.error("RDKit could not parse this SMILES.")
            else:
                row   = pd.Series(desc).reindex(f_pred)
                X_in  = row.values.reshape(1,-1)
                X_imp = i_pred.transform(X_in)
                X_sc  = s_pred.transform(X_imp)
                tg    = float(m_pred.predict(X_sc)[0])

                if   tg < -50:  cat,col_ = "🔵 Very Flexible / Rubbery",    "#3498db"
                elif tg <   0:  cat,col_ = "🟢 Flexible (sub-ambient Tg)",   "#27ae60"
                elif tg <  80:  cat,col_ = "🟡 Semi-rigid Polymer",           "#f39c12"
                elif tg < 150:  cat,col_ = "🟠 Engineering Thermoplastic",    "#e67e22"
                else:           cat,col_ = "🔴 High-Performance / Rigid",     "#e74c3c"

                st.markdown(f"""<div class="result-box">
                  <p style="color:#555;margin:0;">Predicted Glass Transition Temperature:</p>
                  <div class="result-tg">{tg:.1f} °C</div>
                  <div class="result-cat" style="color:{col_};">{cat}</div>
                </div>""", unsafe_allow_html=True)

                # Key descriptors
                st.markdown("#### 🔬 Key Molecular Descriptors")
                showcase = ["FractionCSP3","RingCount","NumAromaticRings","NumRotatableBonds",
                            "MolWt","MolLogP","HallKierAlpha","TPSA","NumHDonors","NumHAcceptors"]
                disp = pd.DataFrame([{"Descriptor":k,"Value":round(desc.get(k,float("nan")),4)}
                                     for k in showcase if k in desc])
                st.dataframe(disp, use_container_width=True, hide_index=True)

                # Context gauge
                st.markdown("#### 📊 Predicted Tg in Context of Dataset")
                fig, ax = plt.subplots(figsize=(10, 2.5))
                ax.hist(y, bins=50, color="lightblue", edgecolor="white", alpha=0.6, label="Dataset Tg")
                ax.axvline(tg,        color=col_,    lw=3,   label=f"Prediction: {tg:.1f} °C")
                ax.axvline(y.mean(),  color="blue",  lw=1.5, linestyle="--", label=f"Mean: {y.mean():.0f} °C")
                ax.axvline(np.median(y),color="purple",lw=1.5,linestyle=":",label=f"Median: {np.median(y):.0f} °C")
                ax.set_xlabel("Tg (°C)"); ax.set_ylabel("Count")
                ax.set_title("Your Prediction vs Dataset Distribution"); ax.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("📖 Tg Reference Guide"):
        st.markdown("""
| Tg Range | State | Typical Polymers | Applications |
|---|---|---|---|
| < −50 °C | Highly rubbery | Silicones, polybutadiene | Gaskets, tyres |
| −50 to 0 °C | Rubbery/tacky | Polyacrylates | Adhesives, coatings |
| 0 to 80 °C | Soft plastic | PE, PP, PVC | Packaging, pipes |
| 80 to 150 °C | Rigid plastic | PET, nylon, ABS | Engineering parts |
| > 150 °C | Glassy/brittle | PEEK, polyimides | Aerospace, electronics |
""")
