# Customer Churn Prediction

A machine learning project to predict customer churn using structured customer data and deliver actionable churn insights. The model is trained on 12k+ customer records and achieves strong performance using XGBoost with feature engineering and hyperparameter tuning.

## 🔥 Highlights
- Built churn prediction model on **12k+ customer records**
- Achieved **AUC ~0.85** using **XGBoost**
- Compared multiple ML models: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning using **GridSearchCV**
- Explained churn drivers using **Feature Importance / SHAP**
- Deployment-ready with **Docker + AWS (EC2/S3)**

---

## Tech Stack
- **Language:** Python  
- **ML:** Scikit-learn, XGBoost  
- **EDA:** Pandas, NumPy, Matplotlib, Seaborn  
- **Explainability:** SHAP  
- **Deployment:** Docker, AWS EC2 / S3  

---

## Project Workflow
1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (LR / RF / XGBoost)
5. Hyperparameter tuning using GridSearchCV
6. Model Evaluation using ROC-AUC, Precision, Recall, F1-score
7. Insights: churn drivers using SHAP + feature importance
8. Deployment using Docker + AWS

---

## Model Performance
| Model | Metric |
|------|--------|
| XGBoost | ROC-AUC ≈ 0.85 |
| Random Forest | Compared |
| Logistic Regression | Compared |

> Final model selected based on best ROC-AUC and stability.

---

## Repository Structure
├── data/ # dataset (optional or sample)
├── notebooks/ # EDA + experiments
├── src/ # preprocessing + training scripts
├── models/ # saved models
├── app/ # API / inference (optional)
├── Dockerfile
├── requirements.txt
└── README.md


---

## How to Run Locally

### 1) Clone Repo
```bash
git clone https://github.com/<srishanthreddy456789>/<Customer-Churn-Prediction>.git
cd <Customer-Churn-Prediction>
2) Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3) Install Requirements
pip install -r requirements.txt
4) Train Model
python src/train.py
5) Run Inference (example)
python src/predict.py
6)Docker Run 
docker build -t churn-predictor .
docker run -p 8080:8080 churn-predictor
