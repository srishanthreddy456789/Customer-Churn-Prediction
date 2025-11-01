# Customer-Churn-Prediction

🎯 Project Description & Objective

Customer churn—the event of a customer leaving a service—is a critical problem for subscription-based businesses like telecommunications, streaming services, and banks. Acquiring a new customer is often far more expensive than retaining an existing one.

This project aims to solve this business problem by building an end-to-end machine learning pipeline. Using a dataset from a telecommunications company, we will analyze customer behavior and build a predictive model. The final model identifies customers at high risk of churning, allowing the business to take proactive measures—such as offering special discounts or support—to improve customer retention and reduce revenue loss.

This repository contains the complete Python script (customer_churn_project.py) and the original dataset.

Table of Contents

Workflow

Tech Stack

How to Run

Model Results

Workflow

The project follows a standard data science pipeline:

Data Loading & Cleaning: Loaded the raw .csv data and cleaned it (e.g., converted TotalCharges to numeric, handled missing values).

Exploratory Data Analysis (EDA): Analyzed features to understand their relationship with churn. Key findings include:

Customers on Month-to-Month contracts churn at a significantly higher rate.

Customers with low tenure (new customers) are more likely to leave.

Customers with higher monthly charges are more likely to leave.

Data Preprocessing:

Encoded all categorical features (like Contract, PaymentMethod) using One-Hot Encoding.

Scaled all numerical features (tenure, MonthlyCharges) using StandardScaler.

Handling Class Imbalance: Corrected the imbalanced dataset (73% 'No' vs. 27% 'Yes') using Random Under-sampling to create a balanced 50/50 training set.

Model Training: Trained three different classification models:

Logistic Regression

Random Forest

XGBoost

Model Evaluation: Evaluated all three models on the unseen test set using the ROC-AUC Score as the primary metric, which is ideal for imbalanced datasets.

🚀 Tech Stack

Python 3.x

Pandas: For data manipulation and analysis.

Scikit-learn (sklearn): For preprocessing, modeling (Logistic Regression, Random Forest), and evaluation.

XGBoost: For the XGBClassifier model.

Matplotlib: For data visualization and plotting the results.

🏁 How to Run

Make sure you have all the required libraries installed:

pip install pandas numpy matplotlib scikit-learn xgboost



Place the dataset WA_Fn-UseC_-Telco-Customer-Churn.csv in the same directory as the Python script.

Run the main project file from your terminal:

python customer_churn_project.py



The script will run the entire pipeline, print the classification reports for each model to the console, and save the final model comparison plot as roc_curve_comparison.png.

📊 Model Results

All models performed well, demonstrating a strong ability to distinguish between churning and non-churning customers. The Logistic Regression model had the best performance, with an AUC score of 0.85.

This means the model has an 85% chance of correctly identifying which of two random customers (one who will churn, one who won't) is the one who will churn.

ROC-AUC Curve Comparison

The plot below shows the performance of all three models. The closer the line is to the top-left corner, the better the model.

(This image will display automatically when roc_curve_comparison.png is in the same repository)
