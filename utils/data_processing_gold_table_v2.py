from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit, max as spark_max, min as spark_min, sum, avg
from pyspark.sql.types import StringType, IntegerType, FloatType
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import shutil
from functools import reduce
from operator import add

def safe_count(df, description="DataFrame"):
    """Safely count rows in a DataFrame with error handling"""
    try:
        return df.count()
    except Exception as e:
        print(f"Warning: Could not count {description} due to: {str(e)}")
        return "Unknown"

def process_gold_feature_and_label_store(silver_base_dir, gold_base_dir, spark, dpd_cutoff=30, mob_cutoff=6, create_label_store=True):

    """
    Improved approach for multi-snapshot delayed labelling with separate feature and label stores
    
    Data characteristics:
    - Clickstream: 215K rows, 22 cols, 8,974 unique customers, snapshot table (Jan 2023 - Dec 2024)
    - Attributes: 12.5K rows, 6 cols, 12,500 unique customers, overwrite table (Jan 2023 - Jan 2025) 
    - Financials: 12.5K rows, 22 cols, 12,500 unique customers, overwrite table (Jan 2023 - Jan 2025)
    - LMS: 137.5K rows, 11 cols, 12,500 unique customers, fact table needs aggregation (Jan 2023 - Nov 2025)
    
    Creates:
    1. Feature Store: Features at different snapshot dates (without labels)
    2. Label Store: Simple binary labels, aggregated LMS data
    3. Combined Store: Features + Labels for immediate ML use
    """
    
    # Load all data sources
    print("=== Loading Data Sources ===")
    
    # Load attributes (monthly overwrite - only latest snapshot per month)
    attributes_folder_path = os.path.join(silver_base_dir, 'attributes')
    attributes_df = spark.read.parquet(f"{attributes_folder_path}/*.parquet")
    print(f'Loaded attributes_df: {attributes_df.count()} rows, {len(attributes_df.columns)} columns')
    print(f'Unique customers in attributes: {attributes_df.select("Customer_ID").distinct().count()}')
    
    # Load clickstream (snapshot table - multiple snapshots per customer)
    clickstream_folder_path = os.path.join(silver_base_dir, 'clickstream')
    clickstream_df = spark.read.parquet(f"{clickstream_folder_path}/*.parquet")
    print(f'Loaded clickstream_df: {clickstream_df.count()} rows, {len(clickstream_df.columns)} columns')
    print(f'Unique customers in clickstream: {clickstream_df.select("Customer_ID").distinct().count()}')
    
    # Load LMS (fact table - needs aggregation)
    lms_folder_path = os.path.join(silver_base_dir, 'lms')
    lms_df = spark.read.parquet(f"{lms_folder_path}/*.parquet")
    print(f'Loaded lms_df: {lms_df.count()} rows, {len(lms_df.columns)} columns')
    print(f'Unique customers in LMS: {lms_df.select("Customer_ID").distinct().count()}')
    
    # Load financials (monthly overwrite - only latest snapshot per month)
    financials_folder_path = os.path.join(silver_base_dir, 'financials')
    financials_df = spark.read.parquet(f"{financials_folder_path}/*.parquet")
    print(f'Loaded financials_df: {financials_df.count()} rows, {len(financials_df.columns)} columns')
    print(f'Unique customers in financials: {financials_df.select("Customer_ID").distinct().count()}')
    
    # Apply feature engineering to financials
    financials_df = apply_financial_feature_engineering(financials_df)
    
    # Get pre-loan clickstream features
    click_features = get_pre_loan_clickstream_features(clickstream_df, lms_df, spark)
    
    # Find valid feature dates (where we have data 6 months later for labels)
    print("\n=== Finding Valid Snapshot Dates ===")
    
    # Get available dates
    attr_dates = set([r.snapshot_date for r in attributes_df.select("snapshot_date").distinct().collect()])
    fin_dates = set([r.snapshot_date for r in financials_df.select("snapshot_date").distinct().collect()])
    lms_dates = set([r.snapshot_date for r in lms_df.select("snapshot_date").distinct().collect()])
    
    # Find feature dates where we have both feature data and label data 6 months later
    valid_feature_dates = []
    
    # Use the intersection of monthly data dates as potential feature dates
    # Since attributes and financials have the most customers (12.5K each), use their intersection
    potential_feature_dates = attr_dates.intersection(fin_dates)
    
    for feature_date in sorted(potential_feature_dates):
        label_date = feature_date + relativedelta(months=mob_cutoff)
        
        # Check if we have LMS data at the label date for creating labels
        if label_date in lms_dates:
            valid_feature_dates.append(feature_date)
        
    if not valid_feature_dates:
        raise RuntimeError("No valid feature-label date pairs found")
    
    # Process each valid snapshot date 
    print("\n=== Processing Feature and Label Stores ===")
    
    all_feature_snapshots = []
    all_combined_snapshots = []
    
    for feature_date in valid_feature_dates:        
        label_date = feature_date + relativedelta(months=mob_cutoff)
        
        # Get monthly features (exact date match)
        attr_snap = attributes_df.filter(col("snapshot_date") == feature_date)
        fin_snap = financials_df.filter(col("snapshot_date") == feature_date)
        
        
        # Join all features together - start with attributes (has all customers)
        combined_features = attr_snap.join(
            fin_snap.drop("snapshot_date"), 
            on="Customer_ID", 
            how="inner"  # Inner since both should have same customers
        )
                
        # Join clickstream features (left join since not all customers have clickstream)
        combined_features = combined_features.join(
            click_features, 
            on="Customer_ID", 
            how="left"
        )
                
        # Create feature snapshot (without labels)
        feature_snapshot = combined_features.withColumn("feature_snapshot_date", lit(feature_date))
        all_feature_snapshots.append(feature_snapshot)
        
        # Create label data for this feature date
        labels = lms_df.filter(
            (col("snapshot_date") == label_date) & 
            (col("mob") == mob_cutoff)
        ).withColumn(
            "label",
            when(col("dpd") >= dpd_cutoff, 1).otherwise(0).cast(IntegerType())
        ).select("Customer_ID", "label")
                
        # Join features with labels for combined store
        combined_snapshot = combined_features.join(
            labels, 
            on="Customer_ID", 
            how="inner"
        ).withColumn("feature_snapshot_date", lit(feature_date)) \
         .withColumn("label_snapshot_date", lit(label_date)) \
         .withColumn("mob_cutoff", lit(mob_cutoff)) \
         .withColumn("dpd_cutoff", lit(dpd_cutoff))
        
        all_combined_snapshots.append(combined_snapshot)
    
    # Create final datasets
    print(f"\n=== Creating Final Stores ===")
    
    # 1. Feature Store (union all feature snapshots)
    feature_store = all_feature_snapshots[0]
    for snapshot in all_feature_snapshots[1:]:
        feature_store = feature_store.unionByName(snapshot, allowMissingColumns=True)
    
    # Remove unwanted column from feature store
    feature_store = feature_store.drop("feature_snapshot_date")
    
    print(f"Feature Store: {safe_count(feature_store, 'feature store')} rows")
    
    # 2. Label Store (create using your simple approach)
    label_store = build_label_store(spark, lms_df, dpd_cutoff, mob_cutoff)
    
    # 3. Combined Store (union all combined snapshots)
    combined_store = all_combined_snapshots[0]
    for snapshot in all_combined_snapshots[1:]:
        combined_store = combined_store.unionByName(snapshot, allowMissingColumns=True)
    
   # Remove unwanted columns from combined store
    combined_store = combined_store.drop("feature_snapshot_date", "label_snapshot_date", "mob_cutoff", "dpd_cutoff")
    
    print(f"Combined Store: {safe_count(combined_store, 'combined store')} rows")
        
    # Save stores
    print("\n=== Saving Data Stores ===")
    
    # Feature Store
    feature_output_path = os.path.join(gold_base_dir, 'feature_store')
    os.makedirs(feature_output_path, exist_ok=True)
    try:
        feature_store.write.mode("overwrite").parquet(feature_output_path)
        print(f"Feature Store saved to: {feature_output_path}")
    except Exception as e:
        print(f"Error saving feature store: {str(e)}")
        # Try alternative approach - clear directory and use simple write without coalesce
        try:
            if os.path.exists(feature_output_path):
                shutil.rmtree(feature_output_path)
            os.makedirs(feature_output_path, exist_ok=True)
            # Avoid coalesce which can cause connection issues - just write as-is
            feature_store.write.mode("overwrite").option("maxRecordsPerFile", 10000).parquet(feature_output_path)
            print(f"Feature Store saved (chunked) to: {feature_output_path}")
        except Exception as e2:
            print(f"Failed to save feature store: {str(e2)}")
            print("Attempting to save as CSV fallback...")
            try:
                csv_path = feature_output_path.replace('.parquet', '.csv')
                feature_store.write.mode("overwrite").option("header", "true").csv(csv_path)
                print(f"Feature Store saved as CSV to: {csv_path}")
            except Exception as e3:
                print(f"All save attempts failed: {str(e3)}")
                # Don't re-raise - continue processing
    
    # Label Store  
    label_output_path = os.path.join(gold_base_dir, 'label_store')
    os.makedirs(label_output_path, exist_ok=True)
    try:
        label_store.write.mode("overwrite").parquet(label_output_path)
        print(f"Label Store saved to: {label_output_path}")
    except Exception as e:
        print(f"Error saving label store: {str(e)}")
        try:
            if os.path.exists(label_output_path):
                shutil.rmtree(label_output_path)
            os.makedirs(label_output_path, exist_ok=True)
            label_store.write.mode("overwrite").option("maxRecordsPerFile", 10000).parquet(label_output_path)
            print(f"Label Store saved (chunked) to: {label_output_path}")
        except Exception as e2:
            print(f"Failed to save label store: {str(e2)}")
            print("Attempting to save as CSV fallback...")
            try:
                csv_path = label_output_path.replace('.parquet', '.csv')
                label_store.write.mode("overwrite").option("header", "true").csv(csv_path)
                print(f"Label Store saved as CSV to: {csv_path}")
            except Exception as e3:
                print(f"All save attempts failed: {str(e3)}")
                # Don't re-raise - continue processing
    
    # Combined Store
    combined_output_path = os.path.join(gold_base_dir, 'combined_store')
    os.makedirs(combined_output_path, exist_ok=True)
    try:
        combined_store.write.mode("overwrite").parquet(combined_output_path)
        print(f"Combined Store saved to: {combined_output_path}")
    except Exception as e:
        print(f"Error saving combined store: {str(e)}")
        try:
            if os.path.exists(combined_output_path):
                shutil.rmtree(combined_output_path)
            os.makedirs(combined_output_path, exist_ok=True)
            combined_store.write.mode("overwrite").option("maxRecordsPerFile", 10000).parquet(combined_output_path)
            print(f"Combined Store saved (chunked) to: {combined_output_path}")
        except Exception as e2:
            print(f"Failed to save combined store: {str(e2)}")
            print("Attempting to save as CSV fallback...")
            try:
                csv_path = combined_output_path.replace('.parquet', '.csv')
                combined_store.write.mode("overwrite").option("header", "true").csv(csv_path)
                print(f"Combined Store saved as CSV to: {csv_path}")
            except Exception as e3:
                print(f"All save attempts failed: {str(e3)}")
                # Don't re-raise - continue processing
    
    return {
        'feature_store': feature_store,
        'label_store': label_store,
        'combined_store': combined_store
    }
    
def build_label_store(spark, lms_df, dpd_cutoff=30, mob_cutoff=6):
    """
    Build label store using your simple approach
    """
    print("📌 Building label store…")
    
    label_df = (
        lms_df
        .filter(col("mob") == mob_cutoff)
        .withColumn("label",
            when(col("dpd") >= dpd_cutoff, 1).otherwise(0).cast(IntegerType()))
        .withColumn("label_def", lit(f"{dpd_cutoff}dpd_{mob_cutoff}mob").cast(StringType()))
        .select("Customer_ID", "loan_id", "label", "snapshot_date", "label_def")
    )
    
    print("✅ Label store created")
    return label_df

def apply_financial_feature_engineering(financials_df):
    """Apply the feature engineering from the original code"""
    
    # 1. Disposable Income
    financials_df = financials_df.withColumn(
        "disposable_income", 
        when(
            col("Monthly_Inhand_Salary").isNull() | col("Total_EMI_per_month").isNull(), 
            None
        ).otherwise(
            col("Monthly_Inhand_Salary") - col("Total_EMI_per_month")
        ).cast(FloatType())
    )

    # 2. Debt-to-Income Ratio (DTI)
    financials_df = financials_df.withColumn(
        "DTI", 
        when(
            col("Monthly_Inhand_Salary").isNull() | (col("Monthly_Inhand_Salary") == 0), 
            None
        ).otherwise(
            col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")
        ).cast(FloatType())
    )
    
    # 3. Number of Active Loans - check for loan columns in financials
    loan_cols = ["Loan_personal_loan", "Loan_mortgage_loan", "Loan_payday_loan", 
                 "Loan_debt_consolidation_loan", "Loan_credit_builder_loan", "Loan_auto_loan", 
                 "Loan_home_equity_loan", "Loan_student_loan"]
    
    # Filter to only include columns that actually exist in the DataFrame
    existing_loan_cols = [c for c in loan_cols if c in financials_df.columns]

    if existing_loan_cols:
        loan_sum = reduce(add, [col(c) for c in existing_loan_cols])
        financials_df = financials_df.withColumn("num_active_loans", loan_sum.cast(IntegerType()))
    else:
        # Create placeholder if no loan columns found
        financials_df = financials_df.withColumn("num_active_loans", lit(0).cast(IntegerType()))
        
    # 4. Credit History Length Bucket
    def credit_history_bucket(chm):
        if chm is None:
            return None
        elif chm < 12:
            return "Short"
        elif 12 <= chm < 36:
            return "Medium"
        else:
            return "Long"
    
    bucket_udf = F.udf(credit_history_bucket, StringType())
    
    # Check if Credit_History_Months column exists
    if "Credit_History_Months" in financials_df.columns:
        financials_df = financials_df.withColumn(
            "credit_history_bucket", 
            bucket_udf(col("Credit_History_Months"))
        )
    else:
        financials_df = financials_df.withColumn("credit_history_bucket", lit(None).cast(StringType()))
    
    # Define the categories
    buckets = ["Short", "Medium", "Long"]

    # One-hot encode each non-null category
    for b in buckets:
        col_name = f"credit_history_bucket_{b}"
        financials_df = financials_df.withColumn(
            col_name,
            when(col("credit_history_bucket") == b, 1).otherwise(0)
        )

    # Handle null values separately
    financials_df = financials_df.withColumn(
        "credit_history_bucket_None",
        when(col("credit_history_bucket").isNull(), 1).otherwise(0)
    )
    
    # Drop the original bucket column
    financials_df = financials_df.drop("credit_history_bucket")
    financials_df = financials_df.drop("Credit_History_Months")

    # 5. Credit Inquiries Bucket 
    def credit_inquiries_bucket(nci):
        if nci is None:
            return None
        elif nci < 6:
            return "Low"
        elif 6 <= nci < 11:
            return "Medium"
        else:
            return "High"
    
    bucket_udf2 = F.udf(credit_inquiries_bucket, StringType())
    
    # Check if Num_Credit_Inquiries column exists
    if "Num_Credit_Inquiries" in financials_df.columns:
        financials_df = financials_df.withColumn(
            "credit_inquiries_bucket", 
            bucket_udf2(col("Num_Credit_Inquiries"))
        )
    else:
        financials_df = financials_df.withColumn("credit_inquiries_bucket", lit(None).cast(StringType()))
    
    # Define the bucket categories
    inquiry_buckets = ["Low", "Medium", "High"]

    # One-hot encode each non-null category
    for b in inquiry_buckets:
        col_name = f"credit_inquiries_bucket_{b}"
        financials_df = financials_df.withColumn(
            col_name,
            when(col("credit_inquiries_bucket") == b, 1).otherwise(0)
        )

    # Handle null values separately
    financials_df = financials_df.withColumn(
        "credit_inquiries_bucket_None",
        when(col("credit_inquiries_bucket").isNull(), 1).otherwise(0)
    )
    
    # Drop the original bucket column
    financials_df = financials_df.drop("credit_inquiries_bucket")
    financials_df = financials_df.drop("Num_Credit_Inquiries")

    return financials_df

def get_pre_loan_clickstream_features(clickstream_df, lms_df, spark):
    """
    Aggregate clickstream features for each customer's entire pre-loan period
    Only calculates average (mean) of fe_1 to fe_20 features
    
    Args:
        clickstream_df: Clickstream data with Customer_ID, snapshot_date, fe_1 to fe_20 columns
        lms_df: LMS data to identify first loan date (MOB=0)
        spark: Spark session
    
    Returns:
        DataFrame with averaged pre-loan clickstream features per customer
    """
    
    # Get first loan date (MOB=0) for each customer
    first_loan_dates = lms_df.filter(col("mob") == 0) \
                            .select("Customer_ID", col("snapshot_date").alias("first_loan_date")) \
                            .distinct()
        
    # Join clickstream with first loan dates
    clickstream_with_loan_date = clickstream_df.join(
        first_loan_dates, 
        on="Customer_ID", 
        how="inner"  # Only customers with loans
    )
    
    # Filter to pre-loan period only
    pre_loan_clickstream = clickstream_with_loan_date.filter(
        col("snapshot_date") < col("first_loan_date")  # Strictly before loan
    )
        
    # Create aggregation expressions for fe_1 to fe_20 (average only)
    agg_exprs = [avg(col(f'fe_{i}')).alias(f"avg_fe_{i}") for i in range(1, 21)]
    
    # Aggregate by customer
    customer_features = pre_loan_clickstream.groupBy("Customer_ID").agg(*agg_exprs)
        
    return customer_features