import os
import glob
import pandas as pd
from scipy import stats
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange
import pprint
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve, learning_curve
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, roc_curve, precision_recall_curve, f1_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

import warnings 
warnings.filterwarnings("ignore")

import argparse

# to call this script: python scripts/model_train.py --snapshotdate "2024-06-01"

# Custom outlier handling transformer
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', factor=1.5, cap_method='percentile', lower_percentile=1, upper_percentile=99):
        """
        Custom outlier handler
        
        Parameters:
        - method: 'iqr' for IQR method, 'percentile' for percentile capping
        - factor: IQR factor (default 1.5)
        - cap_method: 'percentile' or 'iqr' for capping method
        - lower_percentile/upper_percentile: percentiles for capping (1-99)
        """
        self.method = method
        self.factor = factor
        self.cap_method = cap_method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col_idx in range(X_df.shape[1]):
            col_data = X_df.iloc[:, col_idx]
            
            if self.cap_method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
            else:  # percentile method
                lower_bound = col_data.quantile(self.lower_percentile / 100)
                upper_bound = col_data.quantile(self.upper_percentile / 100)
            
            self.bounds_[col_idx] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col_idx in range(X_df.shape[1]):
            if col_idx in self.bounds_:
                lower_bound, upper_bound = self.bounds_[col_idx]
                X_df.iloc[:, col_idx] = np.clip(X_df.iloc[:, col_idx], lower_bound, upper_bound)
        
        return X_df.values if not isinstance(X, pd.DataFrame) else X_df

# Simple log transformation skewness handler
class LogSkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        """
        Simple skewness handler using log transformation
        
        Parameters:
        - threshold: skewness threshold above which to apply log transformation
        """
        self.threshold = threshold
        self.apply_transform_ = {}
        self.shift_values_ = {}
        
    def fit(self, X, y=None):
        from scipy import stats
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col_idx in range(X_df.shape[1]):
            col_data = X_df.iloc[:, col_idx].dropna()
            skewness = abs(stats.skew(col_data))
            
            if skewness > self.threshold:
                min_val = col_data.min()
                
                if min_val > 0:
                    # Can use log directly
                    self.apply_transform_[col_idx] = True
                    self.shift_values_[col_idx] = 0
                elif min_val >= 0:
                    # Use log1p for non-negative values (handles zeros)
                    self.apply_transform_[col_idx] = True
                    self.shift_values_[col_idx] = 0
                else:
                    # Shift to make all values positive, then log
                    self.apply_transform_[col_idx] = True
                    self.shift_values_[col_idx] = abs(min_val) + 1
            else:
                self.apply_transform_[col_idx] = False
                self.shift_values_[col_idx] = 0
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col_idx in range(X_df.shape[1]):
            if self.apply_transform_.get(col_idx, False):
                try:
                    shift = self.shift_values_[col_idx]
                    
                    if shift > 0:
                        # Shift values to make positive, then log
                        X_df.iloc[:, col_idx] = np.log(X_df.iloc[:, col_idx] + shift)
                    else:
                        # Use log1p for non-negative values, log for positive
                        min_val = X_df.iloc[:, col_idx].min()
                        if min_val >= 0:
                            X_df.iloc[:, col_idx] = np.log1p(X_df.iloc[:, col_idx])
                        else:
                            X_df.iloc[:, col_idx] = np.log(X_df.iloc[:, col_idx])
                            
                except Exception as e:
                    print(f"Warning: Log transform failed for column {col_idx}: {str(e)}")
                    # Keep original values if transformation fails
                    pass
        
        return X_df.values if not isinstance(X, pd.DataFrame) else X_df
    
    def inverse_transform(self, X):
        """
        Inverse transform to get back original scale
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col_idx in range(X_df.shape[1]):
            if self.apply_transform_.get(col_idx, False):
                try:
                    shift = self.shift_values_[col_idx]
                    
                    if shift > 0:
                        # Reverse: exp then subtract shift
                        X_df.iloc[:, col_idx] = np.exp(X_df.iloc[:, col_idx]) - shift
                    else:
                        # Reverse log1p or log
                        if hasattr(self, '_used_log1p') and self._used_log1p.get(col_idx, False):
                            X_df.iloc[:, col_idx] = np.expm1(X_df.iloc[:, col_idx])
                        else:
                            X_df.iloc[:, col_idx] = np.exp(X_df.iloc[:, col_idx])
                            
                except Exception as e:
                    print(f"Warning: Inverse transform failed for column {col_idx}: {str(e)}")
                    pass
        
        return X_df.values if not isinstance(X, pd.DataFrame) else X_df

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Using pandas/pyarrow for data processing (more reliable than Spark in containers)
    print("\n=== Initializing Data Processing ===")
    print("Using pandas/pyarrow for data processing (container-optimized)")
    
    # --- Set up Configs --- 
    print("\n=== Setting up Configs ===")
    # Set up config
    model_train_date_str = snapshotdate
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] = oot_period_months
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
    config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
    config["oot_start_date"] = config['model_train_date'] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    config["train_test_ratio"] = train_test_ratio

    pprint.pprint(config)
    
    print("=== Training Timeline ===")
    print(f"Train/Test Period: {config['train_test_start_date']} to {config['train_test_end_date']}")
    print(f"OOT Period: {config['oot_start_date']} to {config['oot_end_date']}")
    print(f"Model Train Date: {config['model_train_date']}")
    
    # --- Load Data ---
    print("\n=== Loading Combined Store ===")
    # Use relative path from project root or current working directory
    base_path = os.getcwd()
    if not os.path.exists(os.path.join(base_path, "datamart")):
        # If running from scripts/ directory, go up one level
        base_path = os.path.dirname(base_path)
    
    combined_store_path = os.path.join(base_path, "datamart/gold/combined_store")
    print(f"Loading data from: {combined_store_path}")
    
    # Load parquet data using pandas (more reliable than Spark in containers)
    try:
        combined_store = pd.read_parquet(combined_store_path, engine='pyarrow')
        print(f"Successfully loaded combined store: {combined_store.shape[0]} rows, {combined_store.shape[1]} columns")
    except Exception as e:
        print(f"Error loading parquet data: {str(e)}")
        raise RuntimeError(f"Failed to load data from {combined_store_path}")

    # --- Split Data into Train - Test - OOT ---
    print("\n=== Splitting Data into Train - Test - OOT ===")
    # Filter data using pandas (replacing Spark operations)
    # IMPORTANT: Only use data up to snapshot date (no future data leakage)
    max_data_date = pd.to_datetime(config['model_train_date'] - timedelta(days=1))  # Day before snapshot
    
    # Ensure snapshot_date is in datetime format for filtering
    combined_store['snapshot_date'] = pd.to_datetime(combined_store['snapshot_date'])
    
    # Convert all config date values to pandas datetime for comparison
    config["train_test_start_date"] = pd.to_datetime(config["train_test_start_date"])
    config["train_test_end_date"] = pd.to_datetime(config["train_test_end_date"])
    config["oot_start_date"] = pd.to_datetime(config["oot_start_date"])
    config["oot_end_date"] = pd.to_datetime(config["oot_end_date"])
    
    # Apply filtering conditions
    combined_df = combined_store[
        (combined_store["snapshot_date"] >= config["train_test_start_date"]) & 
        (combined_store["snapshot_date"] <= config["oot_end_date"]) &
        (combined_store["snapshot_date"] <= max_data_date)  # No future data beyond snapshot
    ].copy()

    print(f"Combined dataset: {len(combined_df)} rows")

    # Train/Test dataset
    train_test_pdf = combined_df[
        (combined_df["snapshot_date"] >= config["train_test_start_date"]) &
        (combined_df["snapshot_date"] <= config["train_test_end_date"])
    ]
    print(f"Train/Test dataset: {len(train_test_pdf)} rows")

    # Out-of-time (OOT) dataset
    oot_pdf = combined_df[
        (combined_df["snapshot_date"] >= config["oot_start_date"]) &
        (combined_df["snapshot_date"] <= config["oot_end_date"])
    ]
    print(f"OOT dataset: {len(oot_pdf)} rows")

    # === TEMPORAL SPLIT (NEW) ===
    print("\n=== Performing Temporal Split ===")
    # Calculate the split date (80% of train/test period)
    split_months = int(config["train_test_period_months"] * config["train_test_ratio"])  # 80% of 12 = 9.6 ≈ 10 months
    train_end_date = config["train_test_start_date"] + relativedelta(months=split_months)

    print(f"Training period: {config['train_test_start_date'].date()} to {train_end_date.date()}")
    print(f"Testing period: {(train_end_date + timedelta(days=1)).date()} to {config['train_test_end_date'].date()}")

    # Temporal split instead of random split
    train_pdf = train_test_pdf[train_test_pdf["snapshot_date"] <= train_end_date]
    test_pdf = train_test_pdf[train_test_pdf["snapshot_date"] > train_end_date]

    print(f"Training dataset: {len(train_pdf)} rows")
    print(f"Testing dataset: {len(test_pdf)} rows")

    feature_cols = ['Age','occupation_developer', 'occupation_scientist', 'occupation_engineer',
        'occupation_teacher', 'occupation_manager', 'occupation_enterpreneur',
        'occupation_mechanic', 'occupation_musician', 'occupation_architect',
        'occupation_writer', 'occupation_accountant', 'occupation_journalist',
        'occupation_lawyer', 'occupation_doctor', 'occupation_media_manager',
        'occupation_unknown', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Delay_from_due_date', 'Changed_Credit_Limit', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance', 'Num_of_Delayed_Payment',
        'Credit_Mix_None', 'Credit_Mix_Good', 'Credit_Mix_Standard',
        'Payment_of_Min_Amount_Yes', 'Payment_of_Min_Amount_No',
        'Payment_of_Min_Amount_NM', 'Payment_of_Min_Amount_None',
        'Payment_Behaviour_High_spent_Large_value_payments',
        'Payment_Behaviour_High_spent_Medium_value_payments',
        'Payment_Behaviour_High_spent_Small_value_payments',
        'Payment_Behaviour_Low_spent_Large_value_payments',
        'Payment_Behaviour_Low_spent_Medium_value_payments',
        'Payment_Behaviour_Low_spent_Small_value_payments',
        'Payment_Behaviour_None', 'Loan_Not_Specified',
        'Loan_debt_consolidation_loan', 'Loan_personal_loan',
        'Loan_payday_loan', 'Loan_mortgage_loan', 'Loan_credit_builder_loan',
        'Loan_auto_loan', 'Loan_home_equity_loan', 'Loan_student_loan',
        'disposable_income', 'DTI', 'num_active_loans',
        'credit_history_bucket_Short', 'credit_history_bucket_Medium',
        'credit_history_bucket_Long', 'credit_history_bucket_None',
        'credit_inquiries_bucket_Low', 'credit_inquiries_bucket_Medium',
        'credit_inquiries_bucket_High', 'credit_inquiries_bucket_None',
        'avg_fe_1', 'avg_fe_2', 'avg_fe_3', 'avg_fe_4', 'avg_fe_5', 'avg_fe_6',
        'avg_fe_7', 'avg_fe_8', 'avg_fe_9', 'avg_fe_10', 'avg_fe_11',
        'avg_fe_12', 'avg_fe_13', 'avg_fe_14', 'avg_fe_15', 'avg_fe_16',
        'avg_fe_17', 'avg_fe_18', 'avg_fe_19', 'avg_fe_20'
    ]

    # Extract features and labels using temporal split
    X_train = train_pdf[feature_cols]
    y_train = train_pdf["label"]
    X_test = test_pdf[feature_cols] 
    y_test = test_pdf["label"]
    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]

    print(f'\nDataset Summary:')
    print(f'X_train: {X_train.shape[0]} rows')
    print(f'X_test: {X_test.shape[0]} rows')
    print(f'X_oot: {X_oot.shape[0]} rows')
    print(f'y_train: {y_train.shape[0]} rows, default rate: {round(y_train.mean(),2)}')
    print(f'y_test: {y_test.shape[0]} rows, default rate: {round(y_test.mean(),2)}')
    print(f'y_oot: {y_oot.shape[0]} rows, default rate: {round(y_oot.mean(),2)}')
    
    # --- Post Train-Test Split Data Preprocessing ---
    print("\n=== Post Train-Test Split Data Preprocessing ===")
        
    # Define column categories based on outlier analysis
    numerical_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',   
                    'Changed_Credit_Limit', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 
                    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 
                    'Num_of_Delayed_Payment', 'disposable_income', 'DTI', 'num_active_loans', 
                    'avg_fe_1', 'avg_fe_2', 'avg_fe_3', 'avg_fe_4', 'avg_fe_5', 'avg_fe_6', 
                    'avg_fe_7', 'avg_fe_8', 'avg_fe_9', 'avg_fe_10', 'avg_fe_11', 'avg_fe_12', 
                    'avg_fe_13', 'avg_fe_14', 'avg_fe_15', 'avg_fe_16', 'avg_fe_17', 'avg_fe_18', 
                    'avg_fe_19', 'avg_fe_20']

    # High VIF features to remove (based on multicollinearity analysis)
    high_vif_features = [
        'Monthly_Inhand_Salary', 'credit_history_bucket_Medium', 'num_active_loans', 'credit_inquiries_bucket_Medium'
    ]
    
    # Create reduced feature columns list for monitoring
    feature_cols_reduced = [col for col in feature_cols if col not in high_vif_features]
    print(f"Original features: {len(feature_cols)}, Reduced features: {len(feature_cols_reduced)}")
    print(f"Removed features: {high_vif_features}")

    # === SAVE DATASETS FOR EVIDENTLY MONITORING ===
    print("\n=== Saving Datasets for Monitoring ===")

    # Create monitoring directory
    monitoring_dir = os.path.join(base_path, "datamart/gold/model_monitoring")
    os.makedirs(monitoring_dir, exist_ok=True)
    print(f"Monitoring directory: {monitoring_dir}")

    # Save reference dataset (training data) for Evidently with reduced features
    reference_data = train_pdf[feature_cols_reduced + ['label', 'snapshot_date']].copy()
    reference_data.to_parquet(f"{monitoring_dir}/reference_data.parquet", index=False)
    print(f"Reference dataset saved: {len(reference_data)} rows with {len(feature_cols_reduced)} features")

    # === PRODUCTION DATASET: LOAD CURRENT MONTH DATA SEPARATELY ===
    # Extract year and month from model train date for production data
    model_train_year = config["model_train_date"].year
    model_train_month = config["model_train_date"].month

    # Create start and end dates for the current month
    production_start_date = pd.to_datetime(f"{model_train_year}-{model_train_month:02d}-01")
    last_day_of_month = monthrange(model_train_year, model_train_month)[1]
    production_end_date = pd.to_datetime(f"{model_train_year}-{model_train_month:02d}-{last_day_of_month}")

    print(f"Loading production data for period: {production_start_date.date()} to {production_end_date.date()}")

    # Load production data using pandas filtering
    try:
        production_data = combined_store[
            (combined_store["snapshot_date"] >= production_start_date) & 
            (combined_store["snapshot_date"] <= production_end_date)
        ].copy()
        
        print(f"Loaded production data: {len(production_data)} rows")
        
        if len(production_data) == 0:
            print("WARNING: No production data found for the specified period!")
            print("This might be expected if the data doesn't exist yet.")
            
            # Create empty production dataset with correct schema for monitoring
            production_data = pd.DataFrame(columns=feature_cols_reduced + ['label', 'snapshot_date'])
            print("Created empty production dataset with correct schema")
        else:
            # Filter to reduced features for consistency
            production_data = production_data[feature_cols_reduced + ['label', 'snapshot_date']].copy()
            
    except Exception as e:
        print(f"Error loading production data: {str(e)}")
        print("Creating empty production dataset with correct schema")
        production_data = pd.DataFrame(columns=feature_cols_reduced + ['label', 'snapshot_date'])

    # Save production dataset for Evidently
    production_data.to_parquet(f"{monitoring_dir}/production_data.parquet", index=False)
    print(f"Production dataset saved: {len(production_data)} rows with {len(feature_cols_reduced)} features")

    print(f"\nMonitoring datasets saved to: {monitoring_dir}/")
    print(f"- reference_data.parquet: {len(reference_data)} rows (Training: {config['train_test_start_date'].date()} to {train_end_date.date()})")
    print(f"- production_data.parquet: {len(production_data)} rows (Production: {production_start_date.date()} to {production_end_date.date()})")

    # Drop high VIF features from all datasets
    X_train_reduced = X_train.drop(columns=high_vif_features, errors='ignore')
    X_test_reduced = X_test.drop(columns=high_vif_features, errors='ignore')
    X_oot_reduced = X_oot.drop(columns=high_vif_features, errors='ignore')

    # Update numerical columns by removing the dropped features
    numerical_cols_reduced = [col for col in numerical_cols if col not in high_vif_features]

    # Get remaining engineered features (only the ones that weren't dropped)
    engineered_features_reduced = [f'avg_fe_{i}' for i in range(1, 21) if f'avg_fe_{i}' not in high_vif_features]

    # Get one-hot encoded columns (assuming you have all column names)
    # You need to define feature_cols or get all columns from your dataset
    all_cols = X_train.columns.tolist()  # Get all column names from your dataset
    onehot_cols = [col for col in all_cols if col not in numerical_cols]

    # Update one-hot encoded columns by removing dropped features
    onehot_cols_reduced = [col for col in onehot_cols if col not in high_vif_features]

    # Categorize features by outlier severity and business logic
    # High outlier columns (>5% outliers) - need aggressive treatment
    high_outlier_cols = ['Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 
                        'Monthly_Balance', 'DTI']

    # Medium outlier columns (2-5% outliers) - moderate treatment  
    # Note: Annual_Income was removed due to high VIF, so excluding it
    medium_outlier_cols = ['Monthly_Inhand_Salary', 'disposable_income']

    # Low/no outlier columns - minimal treatment
    low_outlier_cols = ['Age', 'Interest_Rate', 'Changed_Credit_Limit', 'Credit_Utilization_Ratio']

    # Count/discrete variables - business logic validation
    count_features = ['Num_of_Loan', 'Num_Credit_Card', 'Num_of_Delayed_Payment', 'Num_Bank_Accounts',
                    'Delay_from_due_date', 'num_active_loans']

    # Filter all feature categories to only include features that exist in reduced dataset
    high_outlier_cols = [col for col in high_outlier_cols if col in numerical_cols_reduced]
    medium_outlier_cols = [col for col in medium_outlier_cols if col in numerical_cols_reduced]
    low_outlier_cols = [col for col in low_outlier_cols if col in numerical_cols_reduced]
    count_features = [col for col in count_features if col in numerical_cols_reduced]

    # Create preprocessing pipeline
    # Note: You'll need to define OutlierHandler and LogSkewnessHandler classes
    # or replace them with available sklearn transformers

    preprocessor_robust = ColumnTransformer(
        transformers=[
            ('high_outlier_pipe', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('outlier_handler', OutlierHandler(cap_method='percentile', lower_percentile=1, upper_percentile=99)),
                ('skewness_handler', LogSkewnessHandler(threshold=0.5)),
                ('scaler', StandardScaler())
            ]), high_outlier_cols),

            ('medium_outlier_pipe', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('outlier_handler', OutlierHandler(cap_method='percentile', lower_percentile=2, upper_percentile=98)),
                ('skewness_handler', LogSkewnessHandler(threshold=0.5)),
                ('scaler', StandardScaler())
            ]), medium_outlier_cols),

            ('low_outlier_pipe', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('outlier_handler', OutlierHandler(cap_method='iqr', factor=2.0)),
                ('skewness_handler', LogSkewnessHandler(threshold=0.8)),
                ('scaler', StandardScaler())
            ]), low_outlier_cols),

            ('count_pipe', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('outlier_handler', OutlierHandler(cap_method='percentile', lower_percentile=0.5, upper_percentile=99.5)),
                ('scaler', StandardScaler())
            ]), count_features),

            ('engineered_pipe', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('outlier_handler', OutlierHandler(cap_method='iqr', factor=2.5)),
                ('scaler', StandardScaler())
            ]), engineered_features_reduced),

            ('passthrough', 'passthrough', onehot_cols_reduced)
        ])

    # Fit and transform
    print("Fitting preprocessing pipeline...")

    X_train_processed = preprocessor_robust.fit_transform(X_train_reduced)
    X_test_processed = preprocessor_robust.transform(X_test_reduced)
    X_oot_processed = preprocessor_robust.transform(X_oot_reduced)

    print('X_train_processed:', X_train_processed.shape)
    print('X_test_processed:', X_test_processed.shape)
    print('X_oot_processed:', X_oot_processed.shape)
    
    # --- Model Training ---
        
    def train_and_save_model(model_name, estimator, param_grid, search_type='random', model_dir="model_bank/"):
        print(f"\n=== Training {model_name} Model ===")
        
        scorer = make_scorer(roc_auc_score)
        
        if search_type == 'grid':
            search = GridSearchCV(estimator, param_grid, scoring=scorer, cv=3, verbose=1, n_jobs=-1)
        else:
            search = RandomizedSearchCV(estimator, param_grid, scoring=scorer, n_iter=2, cv=2, verbose=1, random_state=42, n_jobs=-1)
        
        search.fit(X_train_processed, y_train)
        
        best_model = search.best_estimator_
        print("Best parameters found:", search.best_params_)
        print("Best AUC score:", search.best_score_)
        
        def score_report(name, y_true, y_pred_proba):
            auc = roc_auc_score(y_true, y_pred_proba)
            print(f"{name} AUC score:", auc)
            print(f"{name} GINI score:", round(2 * auc - 1, 3))
            return auc

        auc_train = score_report("Train", y_train, best_model.predict_proba(X_train_processed)[:, 1])
        auc_test = score_report("Test", y_test, best_model.predict_proba(X_test_processed)[:, 1])
        auc_oot = score_report("OOT", y_oot, best_model.predict_proba(X_oot_processed)[:, 1])

        # Create artefact
        model_artefact = {
            'model': best_model,
            'model_version': f"{model_name}_model_" + config["model_train_date_str"].replace('-', '_'),
            'model_type': type(best_model).__name__,
            'creation_timestamp': datetime.now().isoformat(),
            'preprocessing_transformers': {
                'stdscaler': preprocessor_robust,
                'scaler_type': type(preprocessor_robust).__name__
            },
            'data_dates': config,
            'data_stats': {
                'X_train': X_train.shape[0],
                'X_test': X_test.shape[0],
                'X_oot': X_oot.shape[0],
                'y_train': round(y_train.mean(), 2),
                'y_test': round(y_test.mean(), 2),
                'y_oot': round(y_oot.mean(), 2)
            },
            'results': {
                'auc_train': auc_train,
                'auc_test': auc_test,
                'auc_oot': auc_oot,
                'gini_train': round(2 * auc_train - 1, 3),
                'gini_test': round(2 * auc_test - 1, 3),
                'gini_oot': round(2 * auc_oot - 1, 3)
            },
            'hp_params': search.best_params_
        }

        # Save model
        if not model_dir.startswith("/"):
            model_dir = os.path.join(base_path, model_dir)
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, model_artefact['model_version'] + '.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(model_artefact, file)

        print(f"{model_name} model saved to {file_path}")
        
        # Reload and test
        with open(file_path, 'rb') as file:
            loaded = pickle.load(file)
        y_pred_oot = loaded['model'].predict_proba(X_oot_processed)[:, 1]
        print(f"{model_name} model reloaded - OOT AUC:", roc_auc_score(y_oot, y_pred_oot))

    # XGBoost - Optimized for faster training
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss', 
        random_state=88,
        tree_method='hist',  # Faster histogram-based algorithm
        n_jobs=-1  # Use all available cores
    )
    xgb_param = {
        'n_estimators': [50, 100],  # Reduced for faster training
        'max_depth': [3, 4],  # Reduced from [3, 4, 5]
        'learning_rate': [0.1, 0.2],  # Higher values for faster convergence
        'subsample': [0.8],  # Fixed value
        'colsample_bytree': [0.8],  # Fixed value
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1]  # Fixed value
        'reg_lambda': [1]  # Fixed value
    }
    
    train_and_save_model("XGB", xgb_model, xgb_param, search_type='random')

    # Logistic Regression
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=88)
    logreg_param = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
    
    train_and_save_model("LOGREG", logreg, logreg_param, search_type='grid')

    # Random Forest - Optimized for faster training
    rf_model = RandomForestClassifier(random_state=88)
    rf_param = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    train_and_save_model("RF", rf_model, rf_param, search_type='random')
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)