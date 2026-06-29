import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="Explainable IDS", layout="wide")

st.title("🔐 Network Intrusion Detection System")
st.write("Check whether network traffic is safe or under attack")

# ---------------- Load Model ----------------
model = joblib.load("models/ids_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.success("Model Loaded Successfully")

# ---------------- Upload CSV ----------------
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Traffic Data")
    st.dataframe(data.head())

    # Remove unnecessary columns
    data.drop(columns=["id"], errors="ignore", inplace=True)

    # Remove labels if present
    X = data.drop(columns=["label", "attack_cat"], errors="ignore")

    # ---------------- Encode Categorical Columns ----------------
    for col, encoder in encoders.items():
        if col in X.columns:
            try:
                X[col] = encoder.transform(X[col])
            except:
                st.warning(f"Encoding issue in column: {col}")

    # ---------------- Fix Feature Mismatch ----------------
    try:
        expected_features = model.get_booster().feature_names
        X = X.reindex(columns=expected_features, fill_value=0)
    except:
        st.error("Model feature mismatch. Please upload correct dataset.")
        st.stop()

    # ---------------- Prediction ----------------
    if st.button("Check Network Security"):

        predictions = model.predict(X)

        attack_count = int(sum(predictions))

        st.subheader("Prediction Result")

        if attack_count > 0:
            st.error("🚨 NETWORK UNDER ATTACK")
        else:
            st.success("✅ NETWORK IS SAFE")

        st.write("Total records analyzed:", len(predictions))
        st.write("Attack records detected:", attack_count)

        # Add predictions to dataframe
        result_df = data.copy()
        result_df["Prediction"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        # ---------------- SHAP Explainability (BAR GRAPH) ----------------
        st.subheader("Top Features Influencing Prediction (SHAP)")

        try:
            explainer = shap.TreeExplainer(model)

            sample_data = X.iloc[:100]

            shap_values = explainer.shap_values(sample_data)

            shap_df = pd.DataFrame(shap_values, columns=X.columns)

            top_features = (
                shap_df.abs()
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )

            fig1, ax1 = plt.subplots(figsize=(8,6))

            top_features.sort_values().plot.barh(ax=ax1)

            ax1.set_title("Top 10 Features Affecting Prediction")
            ax1.set_xlabel("Mean |SHAP Value|")

            st.pyplot(fig1)

        except:
            st.warning("SHAP visualization could not be generated.")

        # ---------------- Feature Importance ----------------
        st.subheader("Model Feature Importance")

        try:
            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            })

            importance = (
                importance
                .sort_values("Importance", ascending=False)
                .head(10)
            )

            fig2, ax2 = plt.subplots(figsize=(8,6))

            ax2.barh(
                importance["Feature"],
                importance["Importance"]
            )

            ax2.set_title("Top 10 Important Features")
            ax2.set_xlabel("Importance Score")

            plt.gca().invert_yaxis()

            st.pyplot(fig2)

        except:
            st.warning("Feature importance not available.")