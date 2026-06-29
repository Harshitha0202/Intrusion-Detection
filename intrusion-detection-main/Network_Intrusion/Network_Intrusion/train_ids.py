import pandas as pd
import numpy as np
import os
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# ===============================
# CREATE MODELS DIRECTORY
# ===============================

os.makedirs("models", exist_ok=True)


# ===============================
# LOAD DATA
# ===============================

train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)


# ===============================
# REMOVE UNUSED COLUMNS
# ===============================

train.drop(columns=["id"], errors="ignore", inplace=True)
test.drop(columns=["id"], errors="ignore", inplace=True)


# ===============================
# ENCODE CATEGORICAL FEATURES
# ===============================

categorical_cols = ["proto", "service", "state"]

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    combined = pd.concat([train[col], test[col]])

    le.fit(combined)

    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

    encoders[col] = le


# ===============================
# FEATURES AND LABELS
# ===============================

X_train = train.drop(columns=["label", "attack_cat"])
y_train = train["label"]

X_test = test.drop(columns=["label", "attack_cat"])
y_test = test["label"]


# ===============================
# HANDLE CLASS IMBALANCE
# ===============================

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


# ===============================
# MODEL (XGBOOST)
# ===============================

model = XGBClassifier(

    n_estimators=400,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)


# ===============================
# EVALUATION
# ===============================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


# ===============================
# SAVE MODEL
# ===============================

joblib.dump(model, "models/ids_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("\nModel saved successfully!")


# ===============================
# SHAP EXPLAINABILITY
# ===============================

print("\nGenerating SHAP explanation...")

explainer = shap.TreeExplainer(model)

sample_data = X_test.iloc[:200]

shap_values = explainer.shap_values(sample_data)

shap.summary_plot(shap_values, sample_data)


# ===============================
# LIME EXPLANATION
# ===============================

print("\nGenerating LIME explanation...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(

    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=["Normal", "Attack"],
    mode="classification"
)

exp = lime_explainer.explain_instance(

    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

print("\nLIME Explanation:")

print(exp.as_list())