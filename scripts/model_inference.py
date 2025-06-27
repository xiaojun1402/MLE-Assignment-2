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
import pprint
# Removed PySpark imports - using pandas only for consistency

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

# to call this script: python scripts/model_inference.py --snapshotdate "2024-06-01" --modelname "XGB_model_2024_06_01.pkl"

# --- define classes and functions ---     
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

def calculate_training_period_dates(model_train_date, train_test_period_months=12, oot_period_months=2, train_test_ratio=0.8):
    """
    Calculate training period dates based on model train date
    """
    # Calculate OOT period
    oot_end_date = model_train_date - timedelta(days=1)
    oot_start_date = model_train_date - relativedelta(months=oot_period_months)
    
    # Calculate train/test period
    train_test_end_date = oot_start_date - timedelta(days=1)
    train_test_start_date = oot_start_date - relativedelta(months=train_test_period_months)
    
    # Calculate training split (80% of train/test period)
    split_months = int(train_test_period_months * train_test_ratio)
    train_end_date = train_test_start_date + relativedelta(months=split_months)
    
    return {
        'train_test_start_date': train_test_start_date,
        'train_test_end_date': train_test_end_date,
        'train_end_date': train_end_date,
        'oot_start_date': oot_start_date,
        'oot_end_date': oot_end_date
    }

def process_inference_data(features_pdf, feature_cols, model_artefact, config, data_type="current"):
    """
    Process inference data for either training or current period
    """
    print(f"\n=== Processing {data_type} inference data ===")
    print(f"{data_type.capitalize()} data shape: {features_pdf.shape}")
    
    X_inference = features_pdf[feature_cols]
    
    # High VIF features to remove (based on multicollinearity analysis)
    high_vif_features = [
        'Monthly_Inhand_Salary', 'credit_history_bucket_Medium', 'num_active_loans', 'credit_inquiries_bucket_Medium'
    ]

    # Drop high VIF features from datasets
    X_inference_reduced = X_inference.drop(columns=high_vif_features, errors='ignore')

    # Apply preprocessing_transformers from model artefact
    preprocessing_transformer = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference_processed = preprocessing_transformer.transform(X_inference_reduced)

    print(f'X_inference_{data_type}', X_inference_processed.shape[0])

    # Model prediction inference
    model = model_artefact["model"]
    y_inference = model.predict_proba(X_inference_processed)[:, 1]
    
    # Prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    y_inference_pdf["data_type"] = data_type
    
    return y_inference_pdf

def main(snapshotdate, modelname):
    print('\n\n---starting enhanced inference job---\n\n')

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    
    # Use consistent path handling
    base_path = os.getcwd()
    if not os.path.exists(os.path.join(base_path, "datamart")):
        base_path = os.path.dirname(base_path)
    
    config["model_bank_directory"] = os.path.join(base_path, "model_bank/")
    config["model_artefact_filepath"] = os.path.join(config["model_bank_directory"], config["model_name"])
    
    # Calculate training period dates
    training_dates = calculate_training_period_dates(config["snapshot_date"].date())
    config.update(training_dates)
    
    print("=== Enhanced Inference Configuration ===")
    pprint.pprint(config)
    
    print("\n=== Timeline ===")
    print(f"Training Period: {config['train_test_start_date']} to {config['train_end_date']}")
    print(f"Current Snapshot Date: {config['snapshot_date'].date()}")

    # --- load model artefact from model bank ---
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print(f"\nModel loaded successfully! {config['model_artefact_filepath']}")

    # --- load feature store with fallback to combined_store ---
    feature_store_location = os.path.join(base_path, "datamart/gold/feature_store")
    combined_store_location = os.path.join(base_path, "datamart/gold/combined_store")
    
    # Try feature_store first, fallback to combined_store
    features_store = None
    
    if os.path.exists(feature_store_location):
        try:
            print(f"Loading from feature_store: {feature_store_location}")
            # Check if it's a directory with parquet files or a single file
            if os.path.isdir(feature_store_location):
                parquet_files = [f for f in os.listdir(feature_store_location) if f.endswith('.parquet')]
                if parquet_files:
                    features_store = pd.read_parquet(feature_store_location, engine='pyarrow')
                elif os.path.exists(os.path.join(feature_store_location, "feature_store.parquet")):
                    features_store = pd.read_parquet(os.path.join(feature_store_location, "feature_store.parquet"), engine='pyarrow')
            print(f"Feature store total rows: {len(features_store)}")
        except Exception as e:
            print(f"Error loading feature_store: {str(e)}, trying combined_store...")
    
    if features_store is None:
        try:
            print(f"Loading from combined_store: {combined_store_location}")
            # Check if it's a directory with parquet files or a single file
            if os.path.isdir(combined_store_location):
                parquet_files = [f for f in os.listdir(combined_store_location) if f.endswith('.parquet')]
                if parquet_files:
                    features_store = pd.read_parquet(combined_store_location, engine='pyarrow')
                elif os.path.exists(os.path.join(combined_store_location, "combined_store.parquet")):
                    features_store = pd.read_parquet(os.path.join(combined_store_location, "combined_store.parquet"), engine='pyarrow')
            print(f"Combined store total rows: {len(features_store)}")
        except Exception as e:
            print(f"Error loading combined_store: {str(e)}")
            raise RuntimeError(f"Failed to load data from both {feature_store_location} and {combined_store_location}")

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

    # Convert dates for pandas filtering
    features_store['snapshot_date'] = pd.to_datetime(features_store['snapshot_date'])
    train_test_start_date = pd.to_datetime(config["train_test_start_date"])
    train_end_date = pd.to_datetime(config["train_end_date"])
    snapshot_date = pd.to_datetime(config["snapshot_date"])

    # --- 1. Process Training Period Data ---
    print("\n" + "="*50)
    print("PROCESSING TRAINING PERIOD DATA")
    print("="*50)
    
    training_features = features_store[
        (features_store["snapshot_date"] >= train_test_start_date) & 
        (features_store["snapshot_date"] <= train_end_date)
    ].copy()
    print(f"Training period features: {len(training_features)} rows")
    
    training_predictions = process_inference_data(
        training_features, feature_cols, model_artefact, config, "training"
    )

    # --- 2. Process Current Snapshot Data ---
    print("\n" + "="*50)
    print("PROCESSING CURRENT SNAPSHOT DATA")
    print("="*50)
    
    current_features = features_store[
        features_store["snapshot_date"] == snapshot_date
    ].copy()
    print(f"Current snapshot features: {len(current_features)} rows")
    
    current_predictions = process_inference_data(
        current_features, feature_cols, model_artefact, config, "current"
    )

    # --- 3. Save Both Datasets ---
    print("\n" + "="*50)
    print("SAVING INFERENCE RESULTS")
    print("="*50)
    
    # Create output directory
    gold_directory = os.path.join(base_path, f"datamart/gold/model_predictions/{config['model_name'][:-4]}/")
    os.makedirs(gold_directory, exist_ok=True)
    
    # Save training predictions
    training_partition_name = f"{config['model_name'][:-4]}_training_predictions_{config['snapshot_date_str'].replace('-','_')}.parquet"
    training_filepath = os.path.join(gold_directory, training_partition_name)
    training_predictions.to_parquet(training_filepath, index=False, engine='pyarrow')
    print(f'Training predictions saved to: {training_filepath}')
    
    # Save current predictions  
    current_partition_name = f"{config['model_name'][:-4]}_current_predictions_{config['snapshot_date_str'].replace('-','_')}.parquet"
    current_filepath = os.path.join(gold_directory, current_partition_name)
    current_predictions.to_parquet(current_filepath, index=False, engine='pyarrow')
    print(f'Current predictions saved to: {current_filepath}')

    # --- Summary ---
    print("\n" + "="*50)
    print("INFERENCE JOB SUMMARY")
    print("="*50)
    print(f"Model: {config['model_name']}")
    print(f"Training Period: {config['train_test_start_date']} to {config['train_end_date']}")
    print(f"Current Snapshot: {config['snapshot_date'].date()}")
    print(f"Training Predictions: {len(training_predictions)} rows")
    print(f"Current Predictions: {len(current_predictions)} rows")
    
    print('\n\n---completed enhanced inference job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run enhanced inference job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)