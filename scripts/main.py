### Core python libs 
import argparse
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

# to call this script: python main.py --snapshotdate "2024-06-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Set minimal Java configuration for Spark
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-arm64'
    os.environ['JAVA_OPTS'] = '-Xmx512m -Xms256m'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python'
    
    # Try minimal Spark configuration
    try:
        spark = pyspark.sql.SparkSession.builder \
            .appName("ml_pipeline_data_processing") \
            .master("local[1]") \
            .config("spark.driver.memory", "512m") \
            .config("spark.driver.maxResultSize", "256m") \
            .config("spark.sql.shuffle.partitions", "1") \
            .getOrCreate()
    except Exception as e:
        print(f"Failed to initialize Spark with minimal config: {e}")
        print("Attempting with basic configuration...")
        spark = pyspark.sql.SparkSession.builder \
            .appName("ml_pipeline_data_processing") \
            .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # Single date processing for proper Airflow backfill
    date_str = snapshotdate
    print(f"Processing snapshot date: {date_str}")
    
    # Create datalake directories
    bronze_base_dir = "/opt/airflow/datamart/bronze/"
    silver_base_dir = "/opt/airflow/datamart/silver/"
    gold_base_dir = "/opt/airflow/datamart/gold/"
    
    for base_dir in [bronze_base_dir, silver_base_dir, gold_base_dir]:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    # Process single date for bronze and silver
    print(f"\n=== Processing Bronze & Silver for {date_str} ===")
    
    # run data processing - bronze 
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_base_dir, spark)

    # run data processing - silver 
    utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_base_dir, silver_base_dir, spark)

    # run data processing - gold (process all dates together to find valid feature-label pairs)
    print("\n=== Processing Gold Table (All Dates) ===")
    utils.data_processing_gold_table_v2.process_gold_feature_and_label_store(silver_base_dir, gold_base_dir, spark, dpd_cutoff = 30, mob_cutoff = 6, snapshot_date = date_str)

    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments 
    parser = argparse.ArgumentParser(description = "run job")
    parser.add_argument("--snapshotdate", type = str, required = True, help = "YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)

    