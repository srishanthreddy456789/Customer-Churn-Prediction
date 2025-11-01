# Customer-Churn-Prediction
## Project Description & Objective

Customer churn—the event of a customer leaving a service—is a critical problem for subscription-based businesses like telecommunications, streaming services, and banks. Acquiring a new customer is often far more expensive than retaining an existing one.

This project aims to solve this business problem by building an end-to-end machine learning pipeline. Using a dataset from a telecommunications company, we analyze customer behavior and build a predictive model. The final model identifies customers at high risk of churning, allowing the business to take proactive measures (such as offering special discounts or support) to improve customer retention and reduce revenue loss.

This repository contains the complete Python script (`customer_churn_project.ipynb`) and the original dataset.

---

## Table of Contents

- [Workflow](#workflow)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Model Results](#model-results)


---

## Workflow

The project follows a standard data science pipeline:

1. **Data Loading & Cleaning**
   - Loaded the raw `WA_Fn-UseC_-Telco-Customer-Churn.csv` data and cleaned it (e.g., converted `TotalCharges` to numeric, handled missing values).
2. **ML WorkFlow**
      <img src="https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcRxVEjcOxp_Tfnn71PhTUhoJzPRfJpxx2JQEnxIcOoeKucb13FbZTekBYqcA_gO4QGSMQycBYhu84O8UDI6YotkJJjG7C5iPFqQcmS5W-BzAN_HVw4"/>
3. **Exploratory Data Analysis (EDA)**
   - Analyzed features to understand their relationship with churn.
     - Customers on Month-to-Month contracts churn at a significantly higher rate.
     - Customers with low tenure (new customers) are more likely to leave.
     - Customers with higher monthly charges are more likely to leave.

4. **Data Preprocessing**
   - Encoded all categorical features (like Contract, Payment Method) using One-Hot Encoding.
   - Scaled all numerical features (`tenure`, `MonthlyCharges`) using `StandardScaler`.

5. **Handling Class Imbalance**
   - Corrected the imbalanced dataset (73% 'No' vs. 27% 'Yes') using Random Under-sampling to create a balanced 50/50 training set.
6. **Model Training**
   - Trained three different classification models:
     - Logistic Regression
     - Random Forest
     - XGBoost

7. **Model Evaluation**
   - Evaluated all three models on the unseen test set using the ROC-AUC Score as the primary metric, ideal for imbalanced datasets.

---

## Tech Stack

- Python 3.x
- Pandas: For data manipulation and analysis.
- Scikit-learn (`sklearn`): For preprocessing, modeling (Logistic Regression, Random Forest), and evaluation.
- XGBoost: For the `XGBClassifier` model.
- Matplotlib: For data visualization and plotting the results.

---

## How to Run

1. Make sure you have all the required libraries installed:

    ```
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

2. Place the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same directory as the Python script.

3. Run the main project file from your terminal:

    ```
    python customer_churn_project.py
    ```

4. The script will run the entire pipeline, print the classification reports for each model to the console, and save the final model comparison plot as `roc_curve_comparison.png`.

---

## Model Results

All models performed well, demonstrating a strong ability to distinguish between churning and non-churning customers. The Logistic Regression model had the best performance, with an AUC score of 0.85.

This means the model has an 85% chance of correctly identifying which of two random customers (one who will churn, one who won't) is the one who will churn. ROC-AUC Curve Comparison: The closer the line is to the top-left corner, the better the model.
<img src="ROC-AUC_Curves_Model_Comparison.png"/>


## Contributors

- SRISHANTH REDDY NARRA (Reach out: [srishanthreddy456@gmail.com](mailto:srishanthreddy456@gmail.com))

## License

This project is licensed under the Apache License 2.0.

## Let's Collaborate!

Feel free to contribute and improve this project! Your insights and expertise are valuable.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srishanth-reddy-narra-61a1142a0) [![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instaram&logoColor=white)](https://www.instagram.com/srishanth._.reddy/)
