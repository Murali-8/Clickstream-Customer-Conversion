import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

classifier = load_model("saved_models/classification_model_run_1.pkl")
regressor = load_model("saved_models/regression_model_run_1.pkl")
clusterer = load_model("saved_models/clustering_model_run_1.pkl")

# ----------------------------
# User credentials
# ----------------------------
USER_CREDENTIALS = {"Murali-44": "Murali94", "admin": "adminpass"}

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_data(df, task):
    df = df.copy()
    # Rename columns
    df.rename(columns={
    'page_1__main_category': 'main_category',
    'page_2__clothing_model': 'clothing_model',
    'model_photography': 'photo_type',
    'price_2': 'price_above_avg'}, inplace=True)

    # Convert categorical to string
    cat_cols = ['country', 'main_category', 'clothing_model', 'colour',
                'location', 'photo_type', 'price_above_avg']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)


    # feature enginering 
    df['purchase'] = df['order'].apply(lambda x: 1 if x > 12 else 2)

    # Feature creation
    df['session_clicks'] = df.groupby('session_id')['order'].transform('max')
    df['clicks_per_category'] = df.groupby(['session_id', 'main_category'])['order'].transform('sum')
    df['bounce'] = df['session_clicks'].apply(lambda x: 1 if x == 1 else 0)
    df['is_high_price'] = df['price'].apply(lambda x: 1 if x > 100 else 0)  # adjustable
    df['click_efficiency'] = df['order'] / df['page']
    # renaming countries to other category for (43,44,45,46,47)
    # df['country'] = df['country'].apply(lambda x: 'Other' if x in ['43', '44', '45', '46', '47'] else x)
    # Simplify country to top 5
    top_5 = df['country'].value_counts().nlargest(5).index.tolist()
    df['country'] = df['country'].apply(lambda x: x if x in top_5 else 'Other')
    #print("Current columns:", df.columns.tolist())

    # Encode features

    label_cols= ['location', 'photo_type', 'price_above_avg', 'clothing_model']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # one hot encoding
    ohe_cols = ['country', 'main_category', 'colour']
    df = pd.get_dummies(df, columns=ohe_cols)
    # convert boolean columns
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    
    # drop the columns mentioned in the list
    cols = ['year','month','day','session_id']
    df.drop(columns=cols, inplace=True)

    # Drop leakage depending on task
    #if task == "regression":
    #    drop_cols = ['order', 'session_clicks', 'clicks_per_category', 'click_efficiency', 'purchase']
    #    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    #if task == "clustering":
    #    drop_cols = ['session_id', 'order', 'purchase', 'revenue', 'clicks_per_category',
    #                 'session_clicks', 'click_efficiency', 'clothing_model']
    #    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    #elif task == "classification":
    #    drop_cols = ['order', 'session_clicks', 'clicks_per_category', 'click_efficiency']
    #    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

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
        # Load default CSV if no file is uploaded
        default_path = "data/default_clickstream.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            st.session_state.user_df = df
            st.info("📂 Using default dataset (no file uploaded).")
            st.write("📊 Default Data Preview:", df.head())
        else:
            st.error("❌ No file uploaded and default dataset not found.")

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
            drop_cols = ['order', 'session_clicks', 'clicks_per_category', 'click_efficiency']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            # ✅ Align features with trained model
            expected_features = classifier.feature_names_in_
            df = df.reindex(columns=expected_features, fill_value=0)

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
            df = df.drop(columns=['order', 'session_clicks', 'clicks_per_category', 'click_efficiency', 'purchase'], errors='ignore')

            # ✅ Align features with trained model
            expected_features = regressor.feature_names_in_
            df = df.reindex(columns=expected_features, fill_value=0)

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

            drop_cols = ['session_id', 'order', 'purchase', 'revenue', 'clicks_per_category',
                     'session_clicks', 'click_efficiency', 'clothing_model']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            cluster_features = ['price', 'page', 'price_above_avg', 'is_high_price',
                    'bounce', 'location', 'photo_type',
                    'colour_1', 'colour_2', 'colour_3', 'colour_4', 'colour_5',
                    'colour_6', 'colour_7', 'colour_8', 'colour_9', 'colour_10',
                    'colour_11', 'colour_12', 'colour_13', 'colour_14',
                    'country_16', 'country_24', 'country_29', 'country_9', 'country_Other',
                    'main_category_1', 'main_category_2', 'main_category_3', 'main_category_4']

            df_cluster = df[cluster_features]
        

            #expected_features = clusterer.feature_names_in_
            #df = df.reindex(columns=expected_features, fill_value=0)

            preds = clusterer.predict(df_cluster)
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
