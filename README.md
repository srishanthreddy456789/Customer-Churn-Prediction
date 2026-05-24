# 📡 Customer Churn Prediction — Streamlit Web App

> **Predict telecom customer churn in real-time with an AI-powered dashboard built on XGBoost.**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-006600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## 🖼️ App Screenshots

| Predict Tab | EDA Tab | Model Insights |
|-------------|---------|----------------|
| Single-customer risk scoring | Interactive charts & distributions | ROC curve, confusion matrix, feature importance |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Single Prediction** | Enter customer details in the sidebar → instant churn probability |
| 📤 **Batch Prediction** | Upload a CSV → predict churn for thousands of customers at once |
| 📊 **EDA Dashboard** | Interactive visualizations: churn distribution, contract analysis, tenure & charges |
| 🧠 **Model Insights** | ROC-AUC curve, confusion matrix, top-20 feature importances |
| 🔧 **In-App Training** | Train the XGBoost model directly from the browser with one click |
| 🎨 **Premium Dark UI** | Glassmorphism design with smooth gradients & micro-animations |
| ⬇️ **Download Results** | Export batch predictions as CSV |

---

## 🚀 Quickstart (Local)

### 1 — Clone the repo
```bash
git clone https://github.com/srishanthreddy456789/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2 — Create & activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Train the model (one-time)

**Option A — From command line:**
```bash
python train_model.py
```

**Option B — From the browser:**
Run the app first, then go to the **🔧 Train Model** tab and click "Train Model Now".

### 5 — Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1 — Push your repo to GitHub
```bash
git add .
git commit -m "feat: add Streamlit app"
git push origin main
```

> **Important:** If the `models/` folder is not committed (it's in `.gitignore`), the app will automatically offer in-browser training on first load via the **🔧 Train Model** tab. This is the recommended approach for Streamlit Cloud.

### Step 2 — Create a Streamlit Cloud account
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account

### Step 3 — Deploy
1. Click **"New app"**
2. Select your repository: `srishanthreddy456789/Customer-Churn-Prediction`
3. Set the **main file path** to: `app.py`
4. Set **Python version** to `3.10` or `3.11`
5. Click **"Deploy!"**

Streamlit Cloud will automatically install `requirements.txt` and launch your app. 🚀

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                              ← 🌟 Main Streamlit application
├── train_model.py                      ← Training script (CLI alternative)
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Dataset (IBM Telco Churn)
├── customer_churn_project.ipynb        ← Original research notebook
│
├── models/
│   └── churn_model.pkl                 ← Saved XGBoost pipeline (generated)
│
├── .streamlit/
│   └── config.toml                     ← Dark theme configuration
│
├── ChurnRate_vs_ContractType.png       ← EDA viz
├── Churn_vs_MonthlyCharges.png         ← EDA viz
├── Customer_Tenure_Distribution_by_Churn.png
├── ROC-AUC_Curves_Model_Comparison.png
├── balanced_dataset.png
└── unbalanced_dataset.png
```

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | XGBoost Classifier |
| **Preprocessing** | StandardScaler (numeric) + OneHotEncoder (categorical) |
| **Balancing** | Random Under-Sampling |
| **Test ROC-AUC** | ~0.85 |
| **Test Accuracy** | ~80% |
| **Features** | 19 customer attributes |
| **Training records** | ~7,000 (after balancing) |

### Features Used

| Type | Features |
|------|----------|
| **Numeric** | tenure, MonthlyCharges, TotalCharges |
| **Categorical** | gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, SeniorCitizen |

---

## 📊 Model Pipeline

```
Raw CSV
  ↓
Data Cleaning (drop customerID, fix TotalCharges)
  ↓
Train/Test Split (80/20, stratified)
  ↓
Random Under-Sampling (balance training set)
  ↓
ColumnTransformer
  ├── StandardScaler   → numeric features
  └── OneHotEncoder    → categorical features
  ↓
XGBClassifier (n_estimators=200, max_depth=5, lr=0.1)
  ↓
models/churn_model.pkl  (joblib)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Web Framework** | Streamlit 1.32+ |
| **ML** | XGBoost 2.0, Scikit-learn 1.3+ |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Dataset** | IBM Telco Customer Churn |

---

## 🧑‍💻 Git Commands for Contributors

```bash
# Clone
git clone https://github.com/srishanthreddy456789/Customer-Churn-Prediction.git

# Create a feature branch
git checkout -b feature/your-feature-name

# Stage all changes
git add .

# Commit
git commit -m "feat: describe your change here"

# Push
git push origin feature/your-feature-name

# Open a pull request on GitHub
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Srishanth Reddy**  
GitHub: [@srishanthreddy456789](https://github.com/srishanthreddy456789)

---

*Made with ❤️ and ☕ — powered by Streamlit & XGBoost*
