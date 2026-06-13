# ============================
# EMPLOYEE ATTRITION PREDICTION
# ============================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================
# LOAD DATA
# ============================

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset Shape:", df.shape)

# ============================
# DATA CLEANING
# ============================

df.drop(
    columns=[
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber"
    ],
    inplace=True
)

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Encode categorical columns

categorical_cols = df.select_dtypes(include="object").columns

encoder_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoder_dict[col] = le

# ============================
# FEATURES / TARGET
# ============================

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ============================
# TRAIN TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# EVALUATION FUNCTION
# ============================

results = []

def evaluate_model(name, model):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n")
    print("=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    return model

# ============================
# LOGISTIC REGRESSION
# ============================

lr = LogisticRegression(max_iter=1000)

lr = evaluate_model(
    "Logistic Regression",
    lr
)

# ============================
# RANDOM FOREST
# ============================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf = evaluate_model(
    "Random Forest",
    rf
)

# ============================
# ADABOOST
# ============================

ada = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

ada = evaluate_model(
    "AdaBoost",
    ada
)

# ============================
# XGBOOST
# ============================

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb = evaluate_model(
    "XGBoost",
    xgb
)

# ============================
# MODEL COMPARISON
# ============================

results_df = pd.DataFrame(results)

print("\n")
print("MODEL COMPARISON")
print(results_df)

# ============================
# BEST MODEL
# ============================

best_model = xgb

# ============================
# CONFUSION MATRIX
# ============================

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================
# FEATURE IMPORTANCE
# ============================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features")
print(importance_df.head(15))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.show()

# ============================
# SAVE MODEL
# ============================

import pickle

with open("employee_attrition_xgb.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel Saved Successfully")