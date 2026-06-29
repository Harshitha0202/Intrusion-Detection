import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/ids_model.pkl")
encoders = joblib.load("models/encoders.pkl")

print("Model loaded")


# =========================
# LOAD DATA
# =========================

data = pd.read_csv("UNSW_NB15_testing-set.csv")

data.drop(columns=["id"], errors="ignore", inplace=True)

X = data.drop(columns=["label", "attack_cat"])


# =========================
# ENCODE CATEGORICAL DATA
# =========================

for col, encoder in encoders.items():
    X[col] = encoder.transform(X[col])


# =========================
# PREDICTION
# =========================

predictions = model.predict(X.iloc[:10])


# =========================
# DISPLAY RESULTS
# =========================

for i, p in enumerate(predictions):

    if p == 1:
        print(f"Sample {i} : ATTACK DETECTED")
    else:
        print(f"Sample {i} : Normal Traffic")