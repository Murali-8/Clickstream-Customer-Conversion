import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# Load models
# ----------------------------
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        return None
    try:
        model = joblib.load(path)
        st.sidebar.success(f"✅ Model loaded: {os.path.basename(path)}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

classifier = load_model("/Users/muralidharanv/Documents/GUVI /PROJECTS/Cliskstream Customer Conversion/saved_models/classification_model_run_1.pkl")
regressor = load_model("/Users/muralidharanv/Documents/GUVI /PROJECTS/Cliskstream Customer Conversion/saved_models/regression_model_run_1.pkl")
clusterer = load_model("/Users/muralidharanv/Documents/GUVI /PROJECTS/Cliskstream Customer Conversion/saved_models/clustering_model_run_1.pkl")

# ----------------------------
# User credentials
# ----------------------------
USER_CREDENTIALS = {"Murali-44": "Murali94", "admin": "adminpass"}

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_data(df, task):
    df = df.copy()

    # Convert categorical to string
    cat_cols = ['country', 'main_category', 'clothing_model', 'colour',
                'location', 'photo_type', 'price_above_avg']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Label encode specific categorical
    label_encoding = {
        'location': {'top left': 1, 'top right': 2, 'middle left': 3,
                     'middle right': 4, 'bottom left': 5, 'bottom right': 6},
        'photo_type': {'product': 1, 'model': 2},
        'price_above_avg': {'0': 0, '1': 1},
        'clothing_model': {'C20': 1, 'C25': 2, 'C30': 3, 'C35': 4, 'C40': 5}
    }
    for col, mapping in label_encoding.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Convert boolean
    bool_cols = ['bounce', 'is_high_price']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=['country', 'main_category', 'colour'], drop_first=False)

    # Drop leakage depending on task
    if task == "regression":
        drop_cols = ['order', 'session_clicks', 'clicks_per_category', 'click_efficiency', 'purchase']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    elif task == "clustering":
        drop_cols = ['session_id', 'order', 'purchase', 'revenue', 'clicks_per_category',
                     'session_clicks', 'click_efficiency', 'clothing_model']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    elif task == "classification":
        drop_cols = ['session_id']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df

# ----------------------------
# Home page → Upload CSV
# ----------------------------
def home_page():
    st.title("🛍️ Clickstream ML App")
    st.subheader("Upload Your CSV File")

    uploaded_file = st.file_uploader("Upload a CSV file with clickstream data", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.user_df = df
        st.success("🎉 File uploaded and stored successfully!")
        st.write("📊 Uploaded Data Preview:", df.head())
    else:
        st.info("⬆️ Please upload a CSV file to continue.")

# ----------------------------
# Classification Tab
# ----------------------------
def classification_tab():
    st.header("📈 Classification")
    if "user_df" not in st.session_state:
        st.warning("Please upload a CSV file on the Home page first.")
        return

    if st.button("🔍 Run Classification"):
        with st.spinner("Predicting Purchases..."):
            df = preprocess_data(st.session_state.user_df, task="classification")
            preds = classifier.predict(df)
            results = st.session_state.user_df.copy()
            results["Predicted_Purchase"] = preds
            st.dataframe(results.head())

            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results (Classification)", csv, "classification_results.csv", "text/csv")

# ----------------------------
# Regression Tab
# ----------------------------
def regression_tab():
    st.header("💰 Regression")
    if "user_df" not in st.session_state:
        st.warning("Please upload a CSV file on the Home page first.")
        return

    if st.button("💸 Run Regression"):
        with st.spinner("Predicting Revenue..."):
            df = preprocess_data(st.session_state.user_df, task="regression")
            preds = regressor.predict(df)
            results = st.session_state.user_df.copy()
            results["Predicted_Revenue"] = preds
            st.dataframe(results.head())

            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results (Regression)", csv, "regression_results.csv", "text/csv")

# ----------------------------
# Clustering Tab
# ----------------------------
def clustering_tab():
    st.header("🔍 Clustering")
    if "user_df" not in st.session_state:
        st.warning("Please upload a CSV file on the Home page first.")
        return

    if st.button("🔎 Run Clustering"):
        with st.spinner("Assigning clusters..."):
            df = preprocess_data(st.session_state.user_df, task="clustering")
            preds = clusterer.predict(df)
            results = st.session_state.user_df.copy()
            results["Cluster_Label"] = preds
            st.dataframe(results.head())

            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results (Clustering)", csv, "clustering_results.csv", "text/csv")

# ----------------------------
# Login + Main Layout
# ----------------------------
def main():
    st.set_page_config(page_title="Clickstream ML App", page_icon="📊", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔐 Clickstream Login")
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
        return

    # Sidebar navigation
    st.sidebar.title("📂 Navigation")
    selection = st.sidebar.radio("Go to", ["🏠 Home", "📊 Classification", "💰 Regression", "🧠 Clustering"])

    if selection == "🏠 Home":
        home_page()
    elif selection == "📊 Classification":
        classification_tab()
    elif selection == "💰 Regression":
        regression_tab()
    elif selection == "🧠 Clustering":
        clustering_tab()

if __name__ == "__main__":
    main()
