# Employee Attrition Prediction using XGBoost

## Overview

This project predicts whether an employee is likely to leave the company using machine learning models trained on HR analytics data.

The goal is to help organizations identify employees at high risk of attrition and take proactive measures to improve retention.

---

## Problem Statement

Employee attrition can significantly impact productivity, hiring costs, and organizational performance.

This project uses employee demographic, job-related, and satisfaction-related features to predict attrition.

Target Variable:

- Attrition = Yes (1)
- Attrition = No (0)

---

## Dataset

IBM HR Analytics Employee Attrition Dataset

Dataset contains information such as:

- Age
- Monthly Income
- Job Satisfaction
- Work-Life Balance
- Overtime
- Job Role
- Years at Company
- Performance Rating
- Environment Satisfaction

and other employee-related attributes.

---

## Exploratory Data Analysis

Performed analysis on:

- Attrition distribution
- Age vs Attrition
- Overtime vs Attrition
- Monthly Income vs Attrition
- Job Satisfaction vs Attrition
- Correlation analysis

Key insights were extracted before model development.

---

## Data Preprocessing

- Removed non-informative features
- Encoded categorical variables using Label Encoding
- Converted target variable into binary format
- Train-Test Split using stratification

---

## Machine Learning Models

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. AdaBoost Classifier
4. XGBoost Classifier

---

## Evaluation Metrics

Models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score

XGBoost achieved the best overall performance.

---

## Feature Importance

Feature importance analysis was performed using XGBoost.

Top contributing features included:

- Overtime
- Monthly Income
- Age
- Job Satisfaction
- Years At Company

---

## Explainable AI

SHAP (SHapley Additive Explanations) was used to understand model predictions and identify the most influential features.

---

## Deployment

The trained XGBoost model was deployed using Streamlit.

Users can input employee details and receive:

- Attrition Prediction
- Attrition Risk Probability

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn

---

## Project Structure

Employee-Attrition-Prediction/

├── app.py

├── employee_attrition_xgb.pkl

├── requirements.txt

├── README.md

├── notebooks/

│ └── attrition_analysis.ipynb

└── screenshots/

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- SMOTE for class imbalance handling
- Cloud deployment
- Real-time HR dashboard
- Employee retention recommendation system

---

## Author

Yash Mishra