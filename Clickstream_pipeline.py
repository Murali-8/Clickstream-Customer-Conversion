# clickstream_pipeline.py
# Combined pipeline for classification, regression, clustering with MLflow tracking

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn
import os
import joblib
import warnings
warnings.filterwarnings('ignore')


def save_model_with_version(model, model_name, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(model_name)]
    run_number = len(existing_files) + 1
    filename = f"{model_name}_run_{run_number}.pkl"
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

class ClickstreamPipeline:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.scaler = StandardScaler()
        mlflow.set_experiment("Clickstream_Pipeline")

    def preprocess(self):
        # Drop rows with missing values (or handle accordingly)
        self.df.dropna(inplace=True)

        # Convert specific columns (as described earlier)
        self.df['country'] = self.df['country'].astype(str)
        self.df['main_category'] = self.df['page1_main_category'].astype(str)
        self.df['clothing_model'] = self.df['page2_clothing_model'].astype(str)
        self.df['colour'] = self.df['colour'].astype(str)
        self.df['location'] = self.df['location'].astype(str)
        self.df['photo_type'] = self.df['model_photography'].astype(str)
        self.df['price_above_avg'] = self.df['price_2'].astype(str)

        self.df.drop(columns=['page1_main_category', 'page2_clothing_model', 'model_photography', 'price_2'], inplace=True)

        # Outlier treatment (IQR for order, price, page)
        for col in ['order', 'price', 'page']:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.df[col] = np.where(self.df[col] < lower, lower,
                              np.where(self.df[col] > upper, upper, self.df[col]))

    def feature_engineering(self):
        # Create classification target variable
        self.df['purchase'] = self.df['order'].apply(lambda x: 1 if x > 12 else 2)

        # Feature creation
        self.df['session_clicks'] = self.df.groupby('session_id')['order'].transform('max')
        self.df['clicks_per_category'] = self.df.groupby(['session_id', 'main_category'])['order'].transform('sum')
        self.df['bounce'] = self.df['session_clicks'].apply(lambda x: 1 if x == 1 else 0)
        self.df['is_high_price'] = self.df['price'].apply(lambda x: 1 if x > 100 else 0)  # adjustable
        self.df['click_efficiency'] = self.df['order'] / self.df['page']
        # renaming countries to other category for (43,44,45,46,47)
        self.df['country'] = self.df['country'].apply(lambda x: 'Other' if x in ['43', '44', '45', '46', '47'] else x)
        # Simplify country to top 5
        top_5 = self.df['country'].value_counts().nlargest(5).index.tolist()
        self.df['country'] = self.df['country'].apply(lambda x: x if x in top_5 else 'Other')
        # print 
        #print("Top 5 countries:", top_5)
        #print("tpo countries :", self.df['country'].unique().tolist())
        #print("Current columns:", self.df.columns.tolist())


    def encode_features(self):
        # Label encoding
        label_cols = ['location', 'photo_type', 'price_above_avg', 'clothing_model']
        for col in label_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])

        # One-hot encoding
        ohe_cols = ['country', 'main_category', 'colour']
        self.df = pd.get_dummies(self.df, columns=ohe_cols)

        # Convert boolean columns
        for col in self.df.select_dtypes(include='bool').columns:
            self.df[col] = self.df[col].astype(int)

        ## drop the columns mentioned in the list
        cols = ['year','month','day','session_id']
        self.df.drop(columns=cols, inplace=True)

        print("Encoded columns:", self.df.columns.tolist())

    def balance_classes(self):
        X = self.df.drop(columns=['purchase'], errors='ignore')
        y = self.df['purchase']

        # List of features to drop due to target leakage
        leakage_features = ['order', 'session_clicks', 'clicks_per_category', 'click_efficiency']
        X = X.drop(columns=leakage_features, errors='ignore')
        print("X columns:", X.columns.tolist())

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        y_res = y_res.replace({2: 0})
        return train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    def train_classifier(self):
        with mlflow.start_run(run_name="XGBoost_Classifier"):
            X_train, X_test, y_train, y_test = self.balance_classes()
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                  random_state=42, colsample_bytree=1.0, learning_rate=0.1,
                                  max_depth=5, n_estimators=250, subsample=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print("Classification Accuracy :",acc)
            mlflow.log_metric("accuracy", acc)
            #mlflow.sklearn.log_model(model, "xgb_classifier")
            save_model_with_version(model, "classification_model")

    def train_regressor(self):
        """
        Train XGBoost Regressor using engineered features and log metrics with MLflow.
        """
        print("\n[Regression] Starting training...")
        self.df['purchase'] = self.df['order'].apply(lambda x: 1 if x > 12 else 2)

        # Define leakage and non-regression columns
        regression_leakage_features = ['order', 'session_clicks', 'clicks_per_category','click_efficiency']
        self.df.drop(columns=regression_leakage_features, inplace=True)

        print("Regression columns:", self.df.columns.tolist())    

        # Create revenue target
        self.df['revenue'] = self.df['price'] * self.df['purchase']

        # Define features and target
        X = self.df.drop(columns=['revenue', 'purchase'])
        y = self.df['revenue']

        # Split
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        xgb_reg = XGBRegressor(
            random_state=42,
            eval_metric='rmse',
            colsample_bytree=1.0,
            learning_rate=0.1,
            max_depth=3,
            n_estimators=275,
            subsample=1.0
        )

        with mlflow.start_run(run_name="XGBoost_Regression"):
            xgb_reg.fit(X_train_reg, y_train_reg)
            y_pred_best_reg = xgb_reg.predict(X_test_reg)

            # Metrics
            r2 = r2_score(y_test_reg, y_pred_best_reg)
            mae = mean_absolute_error(y_test_reg, y_pred_best_reg)
            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_best_reg))

            print("Test R² Score:", r2)
            print("Test MAE     :", mae)
            print("Test RMSE    :", rmse)

            # Log
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            #mlflow.sklearn.log_model(xgb_reg, "xgb_regressor")
            save_model_with_version(xgb_reg, "regression_model")

        print("[Regression] XGBoost Regressor training completed.")


    def run_clustering(self):
        with mlflow.start_run(run_name="KMeans_Clustering"):
            # Select only useful columns for clustering
            clustering_df = self.df.drop(columns=['session_id', 'order', 'purchase', 'revenue',
                                                  'session_clicks', 'clicks_per_category',
                                                  'click_efficiency', 'clothing_model'], errors='ignore')
            scaled = self.scaler.fit_transform(clustering_df)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled)
            silhouette = silhouette_score(scaled, clusters)
            print("Shilhouette_score", silhouette)
            mlflow.log_metric("silhouette_score", silhouette)
            #mlflow.sklearn.log_model(kmeans, "kmeans_clustering")
            save_model_with_version(kmeans, "clustering_model")


    def run_pipeline(self):
        self.preprocess()
        self.feature_engineering()
        self.encode_features()
        self.train_classifier()
        self.train_regressor()
        self.run_clustering()

#  usage:
pipeline = ClickstreamPipeline("/Users/muralidharanv/Documents/GUVI /PROJECTS/Cliskstream Customer Conversion/train_data.csv")
#pipeline = ClickstreamPipeline("/Users/muralidharanv/Documents/GUVI /PROJECTS/Cliskstream Customer Conversion/test_data.csv")
pipeline.run_pipeline()
