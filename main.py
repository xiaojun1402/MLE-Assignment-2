### Core python libs 
import os
import glob
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

### Data Science Tools 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### PySpark 
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace, col, when, regexp_extract, lower, lit, split, explode, array_contains, trim, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, ArrayType
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf

### Custom scripts from utils folder 
from utils.data_processing_bronze_table import process_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table_v2


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_base_dir = "datamart/bronze/"

# Create the base directory if it doesn't exist
if not os.path.exists(bronze_base_dir):
    os.makedirs(bronze_base_dir)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_base_dir, spark)

# create silver datalake
silver_base_dir = "datamart/silver/"

# Create the base directory if it doesn't exist
if not os.path.exists(silver_base_dir):
    os.makedirs(silver_base_dir)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_base_dir, silver_base_dir, spark)

# create gold datalake
gold_base_dir = "datamart/gold/"

if not os.path.exists(gold_base_dir):
    os.makedirs(gold_base_dir)

# run gold backfill
utils.data_processing_gold_table_v2.process_gold_feature_and_label_store(silver_base_dir, gold_base_dir, spark, dpd_cutoff = 30, mob_cutoff = 6)

# Load them separately
label_df = spark.read.parquet("datamart/gold/label_store/")
feature_df = spark.read.parquet("datamart/gold/feature_store/")
combined_df = spark.read.parquet("datamart/gold/combined_store/")

# Print row counts and preview
print("Label Store Row Count:", label_df.count())
label_df.show()

print("Feature Store Row Count:", feature_df.count())
feature_df.show()

print("Combined Store Row Count:", combined_df.count())




    