import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import argparse
import re

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace, col, when, regexp_extract, lower, lit, split, explode, array_contains, trim, udf, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, ArrayType
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer

def process_silver_loan_table(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table 
    partition_name = "bronze_lms_" + snapshot_date_str.replace("-", "_") + '.csv'
    filepath = bronze_base_dir + "lms/" + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded bronze table from: ', filepath, 'row count: ', df.count())
    
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("loan_start_date", to_date(col("loan_start_date"), "yyyy-MM-dd")) \
        .withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))
    
    df = df.withColumn("mob", col("installment_num"))
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
    # Save silver table as parquet
    partition_name = "silver_lms_" + snapshot_date_str.replace("-", "_") + '.parquet'
    filepath = silver_base_dir + "lms/" + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved silver table to: ', filepath, 'row count: ', df.count())
    
    return df

def process_silver_financials_table(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_base_dir + "financials/" + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    column_type_map = {
        'Customer_ID': StringType(),
        'Monthly_Inhand_Salary': FloatType(),
        'Num_Credit_Inquiries': IntegerType(),
        'Credit_Utilization_Ratio': FloatType(),
        'Payment_of_Min_Amount': StringType(),
        'Total_EMI_per_month': FloatType(),
        'Monthly_Balance': FloatType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

    df = df.withColumn("Annual_Income", regexp_replace(col("Annual_Income"), "_", "").cast(FloatType()))
    df = df.withColumn("Num_Bank_Accounts", when(col("Num_Bank_Accounts") < 0, -col("Num_Bank_Accounts")).otherwise(col("Num_Bank_Accounts")).cast(IntegerType()))
    df = df.withColumn("Num_Bank_Accounts", when(col("Num_Bank_Accounts") > 15, None).otherwise(col("Num_Bank_Accounts")).cast(IntegerType()))
    df = df.withColumn("Num_Credit_Card", when(col("Num_Credit_Card") > 15, None).otherwise(col("Num_Credit_Card")).cast(IntegerType()))
    df = df.withColumn("Interest_Rate", when(col("Interest_Rate") > 40, None).otherwise(col("Interest_Rate")).cast(IntegerType()))

    df = df.withColumn("Num_of_Loan", regexp_replace(col("Num_of_Loan"), "_", "").cast(IntegerType()))
    df = df.withColumn("Num_of_Loan", when(col("Num_of_Loan") < 0, -col("Num_of_Loan")).otherwise(col("Num_of_Loan")).cast(IntegerType()))
    df = df.withColumn("Num_of_Loan", when(col("Num_of_Loan") > 15, None).otherwise(col("Num_of_Loan")).cast(IntegerType()))

    df = df.withColumn("Delay_from_due_date", when(col("Delay_from_due_date") < 0, -col("Delay_from_due_date")).otherwise(col("Delay_from_due_date")).cast(IntegerType()))

    df = df.withColumn("Num_cleaned", regexp_replace(col("Num_of_Delayed_Payment"), "_", ""))
    df = df.withColumn("Num_cleaned", when(col("Num_cleaned").rlike(r"^\d+$"), col("Num_cleaned")).otherwise(None))
    df = df.withColumn("Num_int", col("Num_cleaned").cast(IntegerType()))
    df = df.withColumn("Num_abs", when(col("Num_int") < 0, -col("Num_int")).otherwise(col("Num_int")))
    df = df.withColumn("Num_final", when(col("Num_abs") > 50, None).otherwise(col("Num_abs")))
    df = df.drop("Num_of_Delayed_Payment", "Num_cleaned", "Num_int", "Num_abs") \
            .withColumnRenamed("Num_final", "Num_of_Delayed_Payment")

    df = df.withColumn("Changed_Credit_Limit", when(col("Changed_Credit_Limit") == "_", None).otherwise(col("Changed_Credit_Limit")))
    df = df.withColumn("Changed_Credit_Limit", col("Changed_Credit_Limit").cast(FloatType()))
    df = df.withColumn("Changed_Credit_Limit", when(col("Changed_Credit_Limit") < 0, -col("Changed_Credit_Limit")).otherwise(col("Changed_Credit_Limit")))

    df = df.withColumn("Num_Credit_Inquiries", when(col("Num_Credit_Inquiries") > 30, None).otherwise(col("Num_Credit_Inquiries")).cast(IntegerType()))

    df = df.withColumn("Credit_Mix", when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))
    
    credit_mix_categories = ["Good", "Standard", "Bad"]

    # Create one-hot encoded columns
    df = df.withColumn("Credit_Mix_None", when(col("Credit_Mix").isNull(), 1).otherwise(0))

    for cat in credit_mix_categories[:-1]:
        df = df.withColumn(f"Credit_Mix_{cat}",
            when(col("Credit_Mix") == cat, 1).otherwise(0)
        )
    df = df.drop("Credit_Mix")
        
    categories = ["Yes", "No", "NM"]

    # One-hot encode known categories
    for cat in categories:
        col_name = f"Payment_of_Min_Amount_{cat}"
        df = df.withColumn(col_name, when(col("Payment_of_Min_Amount") == cat, 1).otherwise(0))

    # One-hot encode null (None)
    df = df.withColumn("Payment_of_Min_Amount_None", when(col("Payment_of_Min_Amount").isNull(), 1).otherwise(0))

    df = df.drop("Payment_of_Min_Amount")

    df = df.withColumn("Outstanding_Debt", regexp_replace(col("Outstanding_Debt"), "_", "").cast(FloatType()))

    df = df.withColumn("Years", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast(IntegerType()))
    df = df.withColumn("Months", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast(IntegerType()))
    df = df.fillna({"Years": 0, "Months": 0})
    df = df.withColumn("Credit_History_Months", (col("Years") * 12 + col("Months")).cast(IntegerType()))
    df = df.drop("Years", "Months", "Credit_History_Age")

    df = df.withColumn("Total_EMI_per_month", when(col("Total_EMI_per_month") > 1000, 1000).otherwise(col("Total_EMI_per_month")))
    df = df.withColumn("Amount_invested_monthly", when(col("Amount_invested_monthly") == "__10000__", None).otherwise(col("Amount_invested_monthly")).cast(FloatType()))

    pattern = r"^(Low|High)_spent_(Small|Medium|Large)_value_payments$"
    df = df.withColumn("Payment_Behaviour", when(col("Payment_Behaviour").rlike(pattern), col("Payment_Behaviour")).otherwise(None).cast(StringType()))
    
    categories = [
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "None"
    ]

    # Add one-hot encoded columns
    for cat in categories:
        if cat == "Other":
            df = df.withColumn("Payment_Behaviour_" + cat,
                when(~col("Payment_Behaviour").isin(categories[:-1]), 1).otherwise(0)
            )
        else:
            df = df.withColumn("Payment_Behaviour_" + cat,
                when(col("Payment_Behaviour") == cat, 1).otherwise(0)
            )
    df = df.drop("Payment_Behaviour")

    df = df.withColumn("Type_of_Loan", lower(col("Type_of_Loan")))
    df = df.withColumn("Type_of_Loan", regexp_replace(col("Type_of_Loan"), r"\s*and\s*", ","))
    df = df.withColumn("Type_of_Loan", regexp_replace(col("Type_of_Loan"), r"\s*,\s*", ","))
    df = df.withColumn("Type_of_Loan", regexp_replace(col("Type_of_Loan"), r"^,|,$", ""))
    df = df.withColumn("Type_of_Loan", split(col("Type_of_Loan"), ","))
    
    def trim_list(lst):
        if lst is None:
            return []
        return [x.strip() for x in lst if x.strip() != ""]
    
    trim_udf = udf(trim_list, ArrayType(StringType()))
    df = df.withColumn("Type_of_Loan_List", trim_udf(col("Type_of_Loan")))
    df = df.withColumn("Loan_Not_Specified", when(array_contains(col("Type_of_Loan_List"), "not specified"), lit(1)).otherwise(lit(0)))
    loan_types = (df.select(explode(col("Type_of_Loan_List")).alias("loan")).distinct().rdd.flatMap(lambda x: x).collect())
    
    for loan in loan_types:
        if loan == "not specified":
            continue
        col_name = "Loan_" + loan.replace(" ", "_").replace("-", "_")
        df = df.withColumn(col_name, when(array_contains(col("Type_of_Loan_List"), loan), lit(1)).otherwise(lit(0)))
    
    df = df.drop("Type_of_Loan_List")
    df = df.drop("Type_of_Loan")

    # Save silver table as parquet
    partition_name = "silver_financials_" + snapshot_date_str.replace("-", "_") + '.parquet'
    filepath = silver_base_dir + "financials/" + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved silver table to:', filepath, 'row count:', df.count())
    
    return df

def process_silver_attributes_table(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_base_dir + "attributes/" + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    column_type_map = {
        'Customer_ID': StringType(),
        'Name': StringType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))
    
    df = df.withColumn("Name", when(col("Name").isNull(),None).otherwise(trim(regexp_replace(regexp_replace(col("Name"), r'["""''`]', ""), r'[^a-zA-Z\s\-]', ""))).cast(StringType()))

    df = df.withColumn("Age", regexp_replace(col("Age"), "_", "").cast(IntegerType()))
    df = df.withColumn("Age", when(col("Age") < 0, -col("Age")).otherwise(col("Age")).cast(IntegerType()))
    df = df.withColumn("Age_clean", when((col("Age") < 0) | (col("Age") > 120), None).otherwise(col("Age")).cast(IntegerType()))
    df = df.drop("Age").withColumnRenamed("Age_clean", "Age")
    df = df.select("*")

    valid_ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn("SSN", when(col("SSN").rlike(valid_ssn_pattern), col("SSN")).otherwise(None).cast(StringType()))

    df = df.withColumn("Occupation", when(col("Occupation").rlike(r"^_+$"), None).otherwise(trim(lower(regexp_replace(col("Occupation"), "_", " ")))).cast(StringType()))
    
    occupation_categories = [
        "developer",
        "scientist", 
        "engineer",
        "teacher",
        "manager",
        "enterpreneur",
        "mechanic",
        "musician",
        "architect",
        "writer",
        "accountant",
        "journalist",
        "lawyer",
        "doctor",
        "media manager" 
    ]

    for occupation in occupation_categories:
        col_name = f"occupation_{occupation.replace(' ', '_')}"
        df = df.withColumn(
            col_name,
            when(col("Occupation") == occupation, 1).otherwise(0)
        )

    df = df.withColumn(
        "occupation_unknown",
        when(col("Occupation").isNull(), 1).otherwise(0)
    )
    
    df = df.drop("Occupation")
    
    # Save silver table as parquet
    partition_name = "silver_attributes_" + snapshot_date_str.replace("-", "_") + '.parquet'
    filepath = silver_base_dir + "attributes/" + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved silver table to:', filepath, 'row count:', df.count())
    
    return df

def process_silver_clickstream_table(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_base_dir + "clickstream/" + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    column_type_map = {
            'Customer_ID': StringType()
    }
    
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))
    
    # save silver table as parquet
    partition_name = "silver_clickstream_" + snapshot_date_str.replace("-", "_") + '.parquet'
    filepath = silver_base_dir + "clickstream/" + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved silver table to:', filepath, 'row count:', df.count())
    
    return df

   