import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Cancer Dashboard & Prediction", layout="wide")

# ------------------ Static cancer info ------------------
cancer_infos = {
    "Breast Cancer": "Breast cancer forms in breast tissue (ducts or lobules). Early detection improves outcomes.",
    "Lung Cancer": "Lung cancer starts in the lungs and may spread to other organs. Smoking is a key risk factor.",
    "Skin Cancer": "Skin cancer develops on sun-exposed skin. Monitoring moles is important.",
    "Blood Cancer": "Blood cancer affects blood cell production and bone marrow."
}

# ------------------ Load datasets ------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

try:
    df_analysis = load_csv("indiacancer.csv")
except:
    df_analysis = None

datasets = {
    "Breast Cancer": "breastcancer2.csv",
    "Lung Cancer": "lungcancer2.csv",
    "Skin Cancer": "skincancer2.csv",
    "Blood Cancer": "bloodcancer2.csv"
}

# ------------------ Sidebar ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Prediction"])

selected_cancer = st.sidebar.selectbox("Select Cancer Type:", list(cancer_infos.keys()))
st.sidebar.markdown("### Overview")
st.sidebar.write(cancer_infos[selected_cancer])

# ------------------ DASHBOARD ------------------
if page == "Dashboard":
    st.title("📊 Cancer Analysis Dashboard")

    if df_analysis is None:
        st.error("CSV file 'indiacancer.csv' not found or could not be read.")
    else:
        categories = df_analysis["Cancer_Type"].dropna().unique().tolist()
        selected_types = st.multiselect("Select Cancer Type(s):", options=categories)

        if selected_types:
            mdf = df_analysis[df_analysis["Cancer_Type"].isin(selected_types)]

            # Bar Chart
            if {"Cancer_Type", "Gender"}.issubset(mdf.columns):
                bd = mdf.groupby(["Cancer_Type", "Gender"]).size().reset_index(name="Count")
                bg = px.bar(bd, x="Cancer_Type", y="Count", color="Gender",
                            title="Gender Distribution by Cancer Type",
                            labels={"Count": "Patient Count", "Cancer_Type": "Cancer Type"})

            # Pie Chart
            if {"Stage", "Survival_Status"}.issubset(mdf.columns):
                pied = mdf.groupby(["Stage", "Survival_Status"]).size().reset_index(name="Count")
                pied["Label"] = pied["Stage"].astype(str) + " - " + pied["Survival_Status"].astype(str)
                pg = px.pie(pied, names="Label", values="Count", title="Survival Status by Stage")

            # Sunburst Chart
            needed_cols = {"Cancer_Type", "Stage", "Genetic_Mutation"}
            if needed_cols.issubset(mdf.columns):
                sunburst_df = mdf.dropna(subset=list(needed_cols))
                if len(sunburst_df) > 0:
                    sg = px.sunburst(
                        sunburst_df,
                        path=["Cancer_Type", "Stage", "Genetic_Mutation"],
                        values=[1] * len(sunburst_df),
                        title="Genetic Mutation Breakdown"
                    )

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(bg, use_container_width=True)
            with col2:
                st.plotly_chart(pg, use_container_width=True)
            st.plotly_chart(sg, use_container_width=True)
        else:
            st.warning("Please select one or more Cancer Types to view analysis.")

# ------------------ PREDICTION ------------------
elif page == "Prediction":
    st.title(" Cancer Prediction 🩺")

    st.subheader(f"About {selected_cancer}")
    st.write(cancer_infos[selected_cancer])

    st.markdown("---")
    st.subheader("Enter Patient Details")

    # Load dataset for selected cancer
    dataset_path = datasets[selected_cancer]
    try:
        df = load_csv(dataset_path)
    except:
        df = None

    if df is not None:
        # Identify target column
        target_col = None
        for col in ["Label", "target", "Cancer"]:
            if col in df.columns:
                target_col = col
                break

        if target_col:
            X = df.drop(target_col, axis=1)
            y = df[target_col]

            # Encode target
            if y.dtype == 'object':
                le_y = LabelEncoder()
                y = le_y.fit_transform(y)

            # Encode categorical features
            encoders = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le

            # Input fields (auto numeric or dropdown for categorical)
            inputs = {}
            for col in X.columns:
                if col in encoders:  # categorical → dropdown
                    options = encoders[col].classes_.tolist()
                    val = st.selectbox(f"{col}", options)
                    inputs[col] = encoders[col].transform([val])[0]
                else:  # numeric → number input
                    inputs[col] = st.number_input(f"{col}", value=float(X[col].mean()))

            if st.button("Predict"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                acc = accuracy_score(y_test, model.predict(X_test)) * 100
                st.info(f"Model trained. Accuracy: {acc:.2f}%")

                row = np.array([inputs[f] for f in X.columns]).reshape(1, -1)
                pred = model.predict(row)[0]
                prob = model.predict_proba(row).max() * 100

                st.subheader("Prediction Result")
                if pred == 1:
                    st.error(f"{selected_cancer}: Cancer Detected\nConfidence: {prob:.2f}%")
                else:
                    st.success(f"{selected_cancer}: No Cancer Detected\nConfidence: {prob:.2f}%")
        else:
            st.error("Dataset must have a 'Label', 'target', or 'Cancer' column for prediction.")
    else:
        st.error(f"Dataset for {selected_cancer} not found.")