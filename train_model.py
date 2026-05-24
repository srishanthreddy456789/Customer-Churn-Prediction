"""
train_model.py
==============
Run this script once to train the XGBoost churn model and save it.
The saved model is then loaded by app.py for real-time predictions.

Usage:
    python train_model.py

Output:
    models/churn_model.pkl      ← trained XGBoost pipeline
    models/feature_names.pkl    ← list of original feature column names
"""

import os
import pandas as pd
import numpy as np
import joblib
import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# 1. Load & Clean Data
# ──────────────────────────────────────────────
CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Drop non-informative customer ID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Fix TotalCharges (has spaces → coerce to NaN → fill 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print(f"✅ Data loaded. Shape: {df.shape}")

# ──────────────────────────────────────────────
# 2. Define Features & Target
# ──────────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encode target: Yes→1, No→0
le = LabelEncoder()
y = le.fit_transform(y)
print(f"✅ Target encoded. Churn=1: {y.sum()}, No Churn=0: {(y==0).sum()}")

# Save original column names for the Streamlit app
original_feature_names = X.columns.tolist()
joblib.dump(original_feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

# ──────────────────────────────────────────────
# 3. Identify Feature Types
# ──────────────────────────────────────────────
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# All object/category columns + SeniorCitizen (it's 0/1 int but categorical)
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
if "SeniorCitizen" not in categorical_features and "SeniorCitizen" in X.columns:
    categorical_features.append("SeniorCitizen")
categorical_features = [c for c in categorical_features if c not in numeric_features]

print(f"🔢 Numeric features  : {numeric_features}")
print(f"🔤 Categorical features: {categorical_features}")

# ──────────────────────────────────────────────
# 4. Train / Test Split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🔀 Train: {X_train.shape}, Test: {X_test.shape}")

# ──────────────────────────────────────────────
# 5. Balance Training Set (Random Under-Sampling)
# ──────────────────────────────────────────────
print("⚖️  Balancing training set via random under-sampling...")
df_train = X_train.copy()
df_train["Churn"] = y_train

df_majority = df_train[df_train["Churn"] == 0]
df_minority = df_train[df_train["Churn"] == 1]
df_majority_down = df_majority.sample(n=len(df_minority), random_state=42)
df_balanced = shuffle(pd.concat([df_majority_down, df_minority]), random_state=42)

X_train_bal = df_balanced.drop("Churn", axis=1)
y_train_bal = df_balanced["Churn"]
print(f"✅ Balanced training set: {df_balanced['Churn'].value_counts().to_dict()}")

# ──────────────────────────────────────────────
# 6. Build Preprocessing Pipeline
# ──────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)

# ──────────────────────────────────────────────
# 7. Build Full ML Pipeline (Preprocessing + XGBoost)
# ──────────────────────────────────────────────
xgb_clf = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
)

full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb_clf)])

print("🚀 Training XGBoost model...")
full_pipeline.fit(X_train_bal, y_train_bal)
print("✅ Model trained!")

# ──────────────────────────────────────────────
# 8. Evaluate on Test Set
# ──────────────────────────────────────────────
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n📊 Test Accuracy : {acc:.4f}")
print(f"📊 Test ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ──────────────────────────────────────────────
# 9. Save the full pipeline with joblib
# ──────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
joblib.dump(full_pipeline, model_path)
print(f"\n💾 Model saved to: {model_path}")
print("🎉 Done! You can now run: streamlit run app.py")
