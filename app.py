import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Explainable AI Dashboard", layout="wide")
st.title("🧠 Explainable AI Dashboard")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Target column selection
    target_col = st.selectbox("🎯 Select the target column", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if classification
        task_type = "classification" if y.dtype == "object" or len(y.unique()) <= 10 else "regression"
        st.markdown(f"🧐 Detected Task: `{task_type.capitalize()}`")

        if task_type == "classification":
            y = LabelEncoder().fit_transform(y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # SHAP explainer
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        # Global Explanation
        st.subheader("🌍 Global Feature Importance")
        try:
            if len(shap_values.shape) == 3:  # Multi-output case
                shap_fig = shap.plots.beeswarm(shap_values[:, :, 0], show=False)
            else:
                shap_fig = shap.plots.beeswarm(shap_values, show=False)

            st.pyplot(bbox_inches="tight", dpi=300)

            buf1 = io.BytesIO()
            shap_fig.figure.savefig(buf1, format="png")
            st.download_button("📥 Download Global SHAP Plot", data=buf1.getvalue(), file_name="global_shap.png")
        except Exception as e:
            st.error(f"Error generating global SHAP plot: {e}")

        # Local Explanation
        st.subheader("🔍 Local Explanation for a Single Sample")
        idx = st.number_input("Select row index for local explanation", min_value=0, max_value=len(X_test)-1, step=1)

        try:
            if len(shap_values.shape) == 3:
                shap_local = shap_values[idx, :, 0]
            else:
                shap_local = shap_values[idx]

            fig2 = shap.plots.waterfall(shap_local, max_display=10, show=False)
            st.pyplot(bbox_inches="tight", dpi=300)

            buf2 = io.BytesIO()
            fig2.figure.savefig(buf2, format="png")
            st.download_button("📥 Download Local SHAP Plot", data=buf2.getvalue(), file_name="local_shap.png")
        except Exception as e:
            st.error(f"Error generating local SHAP plot: {e}")
else:
    st.info("📄 Please upload a CSV file to begin.")
