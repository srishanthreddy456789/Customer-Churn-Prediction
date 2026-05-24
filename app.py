"""
app.py  –  Customer Churn Prediction · Streamlit Application
=============================================================
A professional, interactive web app for predicting telecom customer churn.

Features:
  • Single-customer prediction via sidebar inputs
  • Batch prediction via CSV file upload
  • Model insights: feature importance & metrics
  • EDA charts embedded in the app
  • Deployment-ready for Streamlit Cloud

Author: Srishanth Reddy
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════
#  GLOBAL STYLE  (dark glassmorphism theme)
# ════════════════════════════════════════════════════
STYLE = """
<style>
/* ── Google Font ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Hero header ─────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 2rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-header p {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
}

/* ── Metric cards ────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: transform 0.25s, box-shadow 0.25s;
    backdrop-filter: blur(8px);
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.55);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Result box ──────────────────────────────────── */
.result-churn {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.1));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
}
.result-safe {
    background: linear-gradient(135deg, rgba(52,211,153,0.2), rgba(16,185,129,0.1));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
}
.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.result-prob {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
}

/* ── Section title ───────────────────────────────── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #a78bfa;
    border-bottom: 2px solid rgba(167,139,250,0.3);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}

/* ── Dataframe ───────────────────────────────────── */
[data-testid="stDataFrameContainer"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Tab bar ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #4338ca);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.4);
}

/* ── Progress bar ────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    border-radius: 99px;
}

/* ── Info boxes ──────────────────────────────────── */
.stAlert {
    border-radius: 12px;
}

/* ── Footer ──────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: rgba(255,255,255,0.3);
    font-size: 0.8rem;
    margin-top: 3rem;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
#  CONSTANTS / PATHS
# ════════════════════════════════════════════════════
MODEL_PATH   = os.path.join("models", "churn_model.pkl")
CSV_PATH     = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

NUMERIC_FEATURES     = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "SeniorCitizen",
]

# ════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the trained pipeline from disk.
    If the model file doesn't exist (e.g. first run on Streamlit Cloud),
    automatically train and save it from the CSV dataset.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # ── Auto-train on first launch ──────────────────
    if not os.path.exists(CSV_PATH):
        return None  # Dataset missing — nothing we can do

    # Read data directly (avoid calling st.cache_data inside st.cache_resource)
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.utils import shuffle
    from xgboost import XGBClassifier

    with st.spinner("🚀 First launch: auto-training XGBoost model (~60 sec). Please wait..."):
        df = pd.read_csv(CSV_PATH)
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        X = df.drop("Churn", axis=1)
        y = LabelEncoder().fit_transform(df["Churn"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Balance via random under-sampling
        df_tr = X_train.copy(); df_tr["Churn"] = y_train
        majority = df_tr[df_tr.Churn == 0]
        minority = df_tr[df_tr.Churn == 1]
        df_bal   = shuffle(
            pd.concat([majority.sample(n=len(minority), random_state=42), minority]),
            random_state=42
        )
        X_bal = df_bal.drop("Churn", axis=1)
        y_bal = df_bal["Churn"]

        num_feats = [f for f in NUMERIC_FEATURES if f in X.columns]
        cat_feats = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("scaler", StandardScaler())]), num_feats),
            ("cat", Pipeline([("ohe",   OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
        ], remainder="passthrough")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   XGBClassifier(
                random_state=42, eval_metric="logloss",
                n_estimators=200, max_depth=5,
                learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            )),
        ])
        pipeline.fit(X_bal, y_bal)

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

    return pipeline


@st.cache_data(show_spinner="📂 Loading dataset...")
def load_data():
    """Load and lightly clean the raw CSV."""
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def train_and_save_model(df):
    """
    Train a full sklearn Pipeline (preprocessor + XGBoost) on the dataset,
    save it to disk, and return the trained pipeline.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.utils import shuffle
    from xgboost import XGBClassifier

    X = df.drop("Churn", axis=1)
    y = LabelEncoder().fit_transform(df["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance training set with random under-sampling
    df_tr = X_train.copy(); df_tr["Churn"] = y_train
    majority = df_tr[df_tr.Churn == 0]
    minority = df_tr[df_tr.Churn == 1]
    majority_down = majority.sample(n=len(minority), random_state=42)
    df_bal = shuffle(pd.concat([majority_down, minority]), random_state=42)
    X_bal = df_bal.drop("Churn", axis=1)
    y_bal = df_bal["Churn"]

    num_feats = [f for f in NUMERIC_FEATURES if f in X.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_feats),
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
    ], remainder="passthrough")

    xgb = XGBClassifier(
        random_state=42, eval_metric="logloss",
        n_estimators=200, max_depth=5,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", xgb)])
    pipeline.fit(X_bal, y_bal)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # Compute metrics for display
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_proba),
        "report":   classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True),
        "y_test":   y_test,
        "y_proba":  y_proba,
        "y_pred":   y_pred,
    }
    return pipeline, metrics


def dark_fig(w=8, h=4):
    """Return a matplotlib figure with a dark background."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#1a1730")
    # Use RGBA tuples (0-1 range) — matplotlib does NOT accept CSS rgba() strings
    spine_color = (1.0, 1.0, 1.0, 0.1)
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
    ax.tick_params(colors="#8888aa", labelsize=9)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("white")
    return fig, ax


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ════════════════════════════════════════════════════
#  SIDEBAR  – User inputs for single prediction
# ════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar with all input controls and return a dict of values."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center; padding:1rem 0;">'
            '<span style="font-size:2.5rem">📡</span>'
            '<h2 style="color:#a78bfa; margin:0.3rem 0 0.1rem">ChurnSense</h2>'
            '<p style="color:rgba(255,255,255,0.45); font-size:0.82rem">by Srishanth Reddy</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("### 👤 Customer Profile")

        # Demographics
        with st.expander("🧑 Demographics", expanded=True):
            gender         = st.selectbox("Gender", ["Male", "Female"], key="gender")
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", key="senior")
            partner        = st.selectbox("Has Partner?", ["Yes", "No"], key="partner")
            dependents     = st.selectbox("Has Dependents?", ["Yes", "No"], key="deps")

        # Services
        with st.expander("📶 Services", expanded=True):
            phone_service   = st.selectbox("Phone Service",     ["Yes", "No"], key="phone")
            multiple_lines  = st.selectbox("Multiple Lines",    ["Yes", "No", "No phone service"], key="mlines")
            internet_svc    = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"], key="inet")
            online_security = st.selectbox("Online Security",   ["Yes", "No", "No internet service"], key="osec")
            online_backup   = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"], key="obkp")
            device_protect  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="dprot")
            tech_support    = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"], key="tsup")
            streaming_tv    = st.selectbox("Streaming TV",      ["Yes", "No", "No internet service"], key="stv")
            streaming_movies= st.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"], key="smov")

        # Account
        with st.expander("💳 Account", expanded=True):
            contract         = st.selectbox("Contract Type",    ["Month-to-month", "One year", "Two year"], key="contract")
            paperless        = st.selectbox("Paperless Billing",["Yes", "No"], key="paper")
            payment_method   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ], key="pay")
            tenure           = st.slider("Tenure (months)", 0, 72, 24, key="tenure")
            monthly_charges  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5, key="mcharge")
            total_charges    = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges), step=10.0, key="tcharge")

        st.divider()
        predict_btn = st.button("🔮 Predict Churn", key="predict_btn")

    return {
        "gender":          gender,
        "SeniorCitizen":   senior_citizen,
        "Partner":         partner,
        "Dependents":      dependents,
        "tenure":          tenure,
        "PhoneService":    phone_service,
        "MultipleLines":   multiple_lines,
        "InternetService": internet_svc,
        "OnlineSecurity":  online_security,
        "OnlineBackup":    online_backup,
        "DeviceProtection":device_protect,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract":        contract,
        "PaperlessBilling":paperless,
        "PaymentMethod":   payment_method,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges,
    }, predict_btn


# ════════════════════════════════════════════════════
#  TAB 1 – Single Prediction
# ════════════════════════════════════════════════════

def tab_predict(model, user_inputs, predict_btn):
    st.markdown('<div class="hero-header"><h1>📡 Customer Churn Predictor</h1>'
                '<p>AI-powered churn risk analysis for telecom customers</p></div>',
                unsafe_allow_html=True)

    if model is None:
        st.error("❌ Dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` not found. "
                 "Please make sure the CSV file is in the project root directory.")
        return

    if predict_btn:
        input_df = pd.DataFrame([user_inputs])

        # ── Predict ──
        prob        = model.predict_proba(input_df)[0][1]
        prediction  = model.predict(input_df)[0]
        risk_pct    = prob * 100

        # ── Result banner ──
        if prediction == 1:
            st.markdown(
                f'<div class="result-churn">'
                f'<div class="result-title">🚨 High Churn Risk Detected</div>'
                f'<div class="result-prob">Churn Probability: <strong>{risk_pct:.1f}%</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-safe">'
                f'<div class="result-title">✅ Customer Likely to Stay</div>'
                f'<div class="result-prob">Churn Probability: <strong>{risk_pct:.1f}%</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Probability gauge ──
        st.markdown("#### Churn Risk Meter")
        col_gauge, col_info = st.columns([3, 1])
        with col_gauge:
            st.progress(int(risk_pct))
        with col_info:
            color = "🔴" if risk_pct > 70 else "🟡" if risk_pct > 40 else "🟢"
            st.markdown(
                f"**{color} {risk_pct:.1f}%**\n\n"
                f"{'Critical risk' if risk_pct>70 else 'Moderate risk' if risk_pct>40 else 'Low risk'}"
            )

        # ── Key risk factors ──
        st.markdown("---")
        st.markdown("#### 🔍 Key Risk Factors (from customer profile)")
        risk_factors = []
        if user_inputs["Contract"] == "Month-to-month":
            risk_factors.append(("📋 Month-to-month contract", "2× higher churn rate vs. annual contracts"))
        if user_inputs["InternetService"] == "Fiber optic":
            risk_factors.append(("🌐 Fiber optic internet", "Higher monthly cost → increased price sensitivity"))
        if user_inputs["tenure"] < 12:
            risk_factors.append(("⏱️ Short tenure", f"{user_inputs['tenure']} months — new customers churn more often"))
        if user_inputs["OnlineSecurity"] in ("No", "No internet service"):
            risk_factors.append(("🔒 No online security", "Customers without security add-ons churn more"))
        if user_inputs["TechSupport"] in ("No", "No internet service"):
            risk_factors.append(("🛠️ No tech support", "Lack of support increases frustration & churn"))
        if user_inputs["MonthlyCharges"] > 80:
            risk_factors.append(("💸 High monthly charges", f"${user_inputs['MonthlyCharges']:.0f}/mo — above average"))

        if risk_factors:
            for factor, detail in risk_factors:
                st.markdown(f"- **{factor}** — {detail}")
        else:
            st.success("✅ No major risk factors identified in this customer's profile.")

        # ── Input summary ──
        with st.expander("📋 Full Input Summary"):
            st.dataframe(pd.DataFrame([user_inputs]).T.rename(columns={0: "Value"}), use_container_width=True)

    else:
        # ── Placeholder state ──
        st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn** to get results.")
        cols = st.columns(3)
        tiles = [
            ("🤖", "XGBoost Model", "Trained on 7,000+ customers"),
            ("📊", "AUC ≈ 0.85", "Strong discriminative power"),
            ("⚡", "Real-time", "Instant risk scoring"),
        ]
        for col, (icon, title, sub) in zip(cols, tiles):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div style="font-size:2rem">{icon}</div>'
                    f'<div class="metric-value" style="font-size:1.1rem">{title}</div>'
                    f'<div class="metric-label">{sub}</div></div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════
#  TAB 2 – Batch Prediction (CSV Upload)
# ════════════════════════════════════════════════════

def tab_batch(model):
    st.markdown("## 📤 Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV with the same columns as the training dataset (without `customerID` and `Churn`).")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")

    if uploaded is not None and model is not None:
        df_up = pd.read_csv(uploaded)

        # Remove columns that shouldn't be present in new data
        for drop_col in ["customerID", "Churn"]:
            if drop_col in df_up.columns:
                df_up = df_up.drop(drop_col, axis=1)

        st.markdown(f"**Uploaded:** `{uploaded.name}` — **{len(df_up):,} rows**")
        st.dataframe(df_up.head(10), use_container_width=True)

        if st.button("▶️ Run Batch Prediction", key="run_batch"):
            with st.spinner("Running predictions..."):
                proba   = model.predict_proba(df_up)[:, 1]
                pred    = model.predict(df_up)
                df_out  = df_up.copy()
                df_out["Churn_Probability"]  = (proba * 100).round(2)
                df_out["Predicted_Churn"]    = ["Yes" if p == 1 else "No" for p in pred]
                df_out["Risk_Level"]         = pd.cut(
                    proba,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["🟢 Low", "🟡 Medium", "🔴 High"],
                )

            st.success(f"✅ Predictions complete for {len(df_out):,} customers.")

            # Summary metrics
            total      = len(df_out)
            churn_cnt  = (pred == 1).sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Customers",  f"{total:,}")
            c2.metric("Predicted Churn",  f"{churn_cnt:,}")
            c3.metric("Churn Rate",       f"{churn_cnt/total*100:.1f}%")
            c4.metric("Avg Probability",  f"{proba.mean()*100:.1f}%")

            # Results table
            st.dataframe(df_out[["Churn_Probability","Predicted_Churn","Risk_Level"]+df_up.columns.tolist()].head(50), use_container_width=True)

            # Download button
            csv_bytes = df_to_csv_bytes(df_out)
            st.download_button("⬇️ Download Predictions CSV", csv_bytes, "churn_predictions.csv", "text/csv")

            # Risk distribution chart
            st.markdown("#### Risk Distribution")
            fig, ax = dark_fig(8, 3)
            risk_counts = df_out["Risk_Level"].value_counts().sort_index()
            colors_bar  = ["#34d399", "#fbbf24", "#f87171"]
            bars = ax.bar(risk_counts.index, risk_counts.values, color=colors_bar, edgecolor="none", width=0.4)
            for bar, val in zip(bars, risk_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val),
                        ha="center", color="white", fontsize=10, fontweight="bold")
            ax.set_ylabel("Customer Count", color="#aaaacc")
            ax.set_title("Customers by Churn Risk Level", color="white", fontweight="bold")
            ax.set_facecolor("#1a1730")
            st.pyplot(fig)
            plt.close(fig)

    elif model is None:
        st.error("❌ Dataset not found. Please make sure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project root.")
    else:
        # Show sample download
        st.info("💡 Need a sample CSV to test? Use the button below to download a sample from the training dataset.")
        df_main = load_data()
        if df_main is not None:
            sample = df_main.drop("Churn", axis=1).head(20)
            st.download_button("⬇️ Download Sample CSV", df_to_csv_bytes(sample), "sample_input.csv", "text/csv")


# ════════════════════════════════════════════════════
#  TAB 3 – EDA Visualizations
# ════════════════════════════════════════════════════

def tab_eda(df):
    st.markdown("## 📊 Exploratory Data Analysis")
    if df is None:
        st.warning("Dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` not found.")
        return

    # ── Overview metrics ──
    total     = len(df)
    churned   = (df["Churn"] == "Yes").sum()
    no_churn  = total - churned
    churn_pct = churned / total * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{total:,}", "Total Customers"),
        (c2, f"{churned:,}", "Churned"),
        (c3, f"{no_churn:,}", "Retained"),
        (c4, f"{churn_pct:.1f}%", "Churn Rate"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Churn distribution | Contract vs Churn ──
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-title">Churn Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        colors  = ["#34d399", "#f87171"]
        counts  = df["Churn"].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            autopct="%1.1f%%", colors=colors,
            startangle=90, wedgeprops=dict(width=0.6),
        )
        for t in texts + autotexts:
            t.set_color("white")
        ax.set_title("Churn vs No Churn", color="white", fontweight="bold")
        st.pyplot(fig); plt.close(fig)

    with r1c2:
        st.markdown('<div class="section-title">Contract Type vs Churn</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        contract_churn = df.groupby(["Contract","Churn"]).size().unstack(fill_value=0)
        contract_churn.plot(
            kind="bar", ax=ax, color=["#34d399","#f87171"],
            edgecolor="none", width=0.6,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_title("Churn Rate by Contract Type", color="white", fontweight="bold")
        ax.legend(["No Churn","Churn"], facecolor="#1a1730", labelcolor="white")
        st.pyplot(fig); plt.close(fig)

    # ── Row 2: Monthly charges | Tenure distribution ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-title">Monthly Charges vs Churn</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        churn_no  = df[df["Churn"]=="No"]["MonthlyCharges"]
        churn_yes = df[df["Churn"]=="Yes"]["MonthlyCharges"]
        ax.hist(churn_no,  bins=30, alpha=0.7, color="#34d399", label="No Churn",  edgecolor="none")
        ax.hist(churn_yes, bins=30, alpha=0.7, color="#f87171", label="Churn",     edgecolor="none")
        ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Count")
        ax.set_title("Monthly Charges Distribution by Churn", color="white", fontweight="bold")
        ax.legend(facecolor="#1a1730", labelcolor="white")
        st.pyplot(fig); plt.close(fig)

    with r2c2:
        st.markdown('<div class="section-title">Tenure Distribution by Churn</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        tenure_no  = df[df["Churn"]=="No"]["tenure"]
        tenure_yes = df[df["Churn"]=="Yes"]["tenure"]
        ax.hist(tenure_no,  bins=30, alpha=0.7, color="#60a5fa", label="No Churn",  edgecolor="none")
        ax.hist(tenure_yes, bins=30, alpha=0.7, color="#fbbf24", label="Churn",     edgecolor="none")
        ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Count")
        ax.set_title("Tenure Distribution by Churn", color="white", fontweight="bold")
        ax.legend(facecolor="#1a1730", labelcolor="white")
        st.pyplot(fig); plt.close(fig)

    # ── Row 3: Internet service | Payment method ──
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="section-title">Internet Service vs Churn</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        inet_churn = df.groupby(["InternetService","Churn"]).size().unstack(fill_value=0)
        inet_churn.plot(kind="barh", ax=ax, color=["#34d399","#f87171"], edgecolor="none", width=0.6)
        ax.set_title("Churn by Internet Service Type", color="white", fontweight="bold")
        ax.legend(["No Churn","Churn"], facecolor="#1a1730", labelcolor="white")
        st.pyplot(fig); plt.close(fig)

    with r3c2:
        st.markdown('<div class="section-title">Churn Rate by Payment Method</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        pay_rate = (
            df.groupby("PaymentMethod")["Churn"]
            .apply(lambda x: (x=="Yes").mean() * 100)
            .sort_values(ascending=True)
        )
        bars = ax.barh(pay_rate.index, pay_rate.values,
                       color="#a78bfa", edgecolor="none", height=0.5)
        for bar, val in zip(bars, pay_rate.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", color="white", fontsize=9)
        ax.set_xlabel("Churn Rate (%)"); ax.set_xlim(0, 60)
        ax.set_title("Churn Rate by Payment Method", color="white", fontweight="bold")
        st.pyplot(fig); plt.close(fig)

    # ── Correlation heatmap ──
    st.markdown('<div class="section-title">Numeric Feature Correlations</div>', unsafe_allow_html=True)
    numeric_df = df[["tenure","MonthlyCharges","TotalCharges"]].copy()
    numeric_df["Churn_Num"] = (df["Churn"] == "Yes").astype(int)
    corr = numeric_df.corr()
    fig, ax = dark_fig(7, 4)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, linecolor="#0f0c29",
        annot_kws={"size": 11, "color": "white"},
        ax=ax, cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Pearson Correlation Matrix", color="white", fontweight="bold")
    st.pyplot(fig); plt.close(fig)

    # ── Raw data sample ──
    with st.expander("🗃️ View Raw Dataset Sample (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)


# ════════════════════════════════════════════════════
#  TAB 4 – Model Insights
# ════════════════════════════════════════════════════

def tab_insights(model, df):
    st.markdown("## 🧠 Model Insights & Performance")

    if model is None:
        st.error("❌ Dataset not found — model could not be auto-trained. "
                 "Please make sure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project root.")
        return
    if df is None:
        st.warning("Dataset not found.")
        return

    # ── Recompute test metrics ──
    X = df.drop("Churn", axis=1)
    y = LabelEncoder().fit_transform(df["Churn"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, target_names=["No Churn","Churn"], output_dict=True)

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{acc*100:.1f}%",        "Accuracy"),
        (c2, f"{auc:.3f}",             "ROC-AUC"),
        (c3, f"{report['Churn']['precision']*100:.1f}%", "Churn Precision"),
        (c4, f"{report['Churn']['recall']*100:.1f}%",    "Churn Recall"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    # ── ROC Curve ──
    with r1:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = dark_fig(6, 4)
        ax.plot(fpr, tpr, color="#a78bfa", lw=2, label=f"XGBoost (AUC = {auc:.3f})")
        ax.plot([0,1],[0,1],"r--", lw=1.5, label="Random Chance (AUC = 0.50)")
        ax.fill_between(fpr, tpr, alpha=0.15, color="#a78bfa")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Curve", color="white", fontweight="bold")
        ax.legend(facecolor="#1a1730", labelcolor="white", fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        st.pyplot(fig); plt.close(fig)

    # ── Confusion Matrix ──
    with r2:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm   = confusion_matrix(y_test, y_pred)
        fig, ax = dark_fig(5, 4)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"],
            linewidths=0.5, linecolor="#0f0c29",
            annot_kws={"size": 14, "weight": "bold"},
            ax=ax,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", color="white", fontweight="bold")
        st.pyplot(fig); plt.close(fig)

    # ── Feature Importance ──
    st.markdown('<div class="section-title">Top Feature Importances (XGBoost)</div>', unsafe_allow_html=True)
    try:
        xgb_step = model.named_steps["classifier"]
        prep_step = model.named_steps["preprocessor"]

        # Recover feature names after preprocessing
        feat_names = []
        for name, trans, cols in prep_step.transformers_:
            if name == "num":
                feat_names.extend(cols)
            elif name == "cat":
                ohe = trans.named_steps.get("ohe") or trans.named_steps.get("onehot")
                cats = ohe.get_feature_names_out(cols)
                feat_names.extend(cats)
            else:
                feat_names.extend(cols if isinstance(cols, list) else [cols])

        importances = xgb_step.feature_importances_
        n_show      = min(len(feat_names), len(importances), 20)
        top_idx     = np.argsort(importances)[-n_show:][::-1]

        top_feats  = [feat_names[i] for i in top_idx]
        top_vals   = importances[top_idx]

        fig, ax = dark_fig(10, 5)
        bar_colors = plt.cm.plasma(np.linspace(0.3, 0.9, n_show))
        ax.barh(range(n_show), top_vals[::-1], color=bar_colors[::-1], edgecolor="none", height=0.65)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 20 Feature Importances", color="white", fontweight="bold")
        ax.axvline(x=np.mean(top_vals), color="#60a5fa", linestyle="--", alpha=0.7, label="Mean")
        ax.legend(facecolor="#1a1730", labelcolor="white", fontsize=9)
        st.pyplot(fig); plt.close(fig)
    except Exception as e:
        st.warning(f"Could not extract feature importances: {e}")

    # ── Classification Report ──
    with st.expander("📋 Full Classification Report"):
        cr_df = pd.DataFrame(report).T.round(3)
        st.dataframe(cr_df, use_container_width=True)

    # ── Model metadata ──
    with st.expander("🔧 Model Pipeline Details"):
        st.code(str(model), language="text")


# ════════════════════════════════════════════════════
#  TAB 5 – Model Training (in-app training)
# ════════════════════════════════════════════════════

def tab_train():
    st.markdown("## 🔧 Model Training")
    st.markdown(
        "Click **Train Model** to train XGBoost on the dataset and save it to `models/churn_model.pkl`. "
        "This only needs to be done once."
    )

    df = load_data()
    if df is None:
        st.error(f"❌ Dataset `{CSV_PATH}` not found. Make sure it is in the project root directory.")
        return

    st.info(f"📦 Dataset found: **{len(df):,} rows × {df.shape[1]} columns**")

    if st.button("🚀 Train Model Now", key="train_now"):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        status_text.markdown("⚙️ Preprocessing data...")
        progress_bar.progress(20)

        with st.spinner("Training XGBoost (this may take 30–60 seconds)..."):
            pipeline, metrics = train_and_save_model(df)

        progress_bar.progress(100)
        status_text.markdown("✅ Training complete!")

        st.success(f"🎉 Model trained and saved to `{MODEL_PATH}`")
        st.markdown("#### 📊 Training Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
        c2.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")
        c3.metric("Training Records", f"{len(df):,}")

        st.markdown(
            "> ✨ **Refresh the app** (press F5 or R) to load the newly trained model."
        )

    # ── Check if model already exists ──
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        import datetime
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"✅ Model file exists: `{MODEL_PATH}` (last saved: {mtime_str})")
    else:
        st.warning("⚠️ No model file found yet. Click **Train Model Now** above.")


# ════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════

def main():
    model = load_model()
    df    = load_data()

    # Sidebar inputs + predict button
    user_inputs, predict_btn = render_sidebar()

    # Navigation tabs
    tabs = st.tabs([
        "🔮 Predict",
        "📤 Batch Prediction",
        "📊 EDA",
        "🧠 Model Insights",
        "🔧 Train Model",
    ])

    with tabs[0]:
        tab_predict(model, user_inputs, predict_btn)

    with tabs[1]:
        tab_batch(model)

    with tabs[2]:
        tab_eda(df)

    with tabs[3]:
        tab_insights(model, df)

    with tabs[4]:
        tab_train()

    # Footer
    st.markdown(
        '<div class="footer">Customer Churn Predictor · Built with ❤️ using Streamlit & XGBoost · '
        '<a href="https://github.com/srishanthreddy456789/Customer-Churn-Prediction" '
        'style="color:#a78bfa" target="_blank">GitHub</a></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
