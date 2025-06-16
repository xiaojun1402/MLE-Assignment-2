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
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

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

# to call this script: python model_inference.py --snapshotdate "2024-06-01" --modelname "XGB_model_2024_06_01.pkl"

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
    
def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])

    # --- load feature store ---
    feature_location = "datamart/gold/feature_store"

    # Load parquet into DataFrame - connect to feature store
    features_store_sdf = spark.read.parquet(feature_location)
    print("row_count:",features_store_sdf.count())

    # extract feature store
    features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date"])

    features_pdf = features_sdf.toPandas()

    # --- preprocess data for modeling ---
        
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

    X_inference = features_pdf[feature_cols]
        
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

    # Drop high VIF features from datasets
    X_inference_reduced = X_inference.drop(columns=high_vif_features, errors='ignore')

    # Update numerical columns by removing the dropped features
    numerical_cols_reduced = [col for col in numerical_cols if col not in high_vif_features]

    # Get remaining engineered features (only the ones that weren't dropped)
    engineered_features_reduced = [f'avg_fe_{i}' for i in range(1, 21) if f'avg_fe_{i}' not in high_vif_features]

    # One-hot encoded columns (categorical features)
    all_cols = X_inference.columns.tolist()  # Get all column names from your dataset
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

    # Fit and transform

    # apply preprocessing_transformers from model artefact
    preprocessing_transformer = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = preprocessing_transformer.transform(X_inference_reduced)

    print('X_inference', X_inference.shape[0])

    # --- model prediction inference ---
    # load model
    model = model_artefact["model"]
    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    
    # --- save model inference to datamart gold table ---
    # create bronze datalake
    gold_directory = f"datamart/gold/model_predictions/{config["model_name"][:-4]}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = config["model_name"][:-4] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    
    print('saved to:', filepath)

    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
