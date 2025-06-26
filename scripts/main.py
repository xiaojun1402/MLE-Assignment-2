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
    
    # Initialize SparkSession with optimized settings for containerized environment
    spark = pyspark.sql.SparkSession.builder \
        .appName("ml_pipeline_data_processing") \
        .master("local[2]") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate 

    # create bronze datalake
    bronze_base_dir = "/opt/airflow/datamart/bronze/"
    
    if not os.path.exists(bronze_base_dir):
        os.makedirs(bronze_base_dir)

    # run data processing - bronze 
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_base_dir, spark)

    # create silver datalake
    silver_base_dir = "/opt/airflow/datamart/silver/"

    if not os.path.exists(silver_base_dir):
        os.makedirs(silver_base_dir)

    # run data processing - silver 
    utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_base_dir, silver_base_dir, spark)
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_base_dir, silver_base_dir, spark)

    # create gold datalake
    gold_base_dir = "/opt/airflow/datamart/gold/"

    if not os.path.exists(gold_base_dir):
        os.makedirs(gold_base_dir)

    # run data processing - gold
    utils.data_processing_gold_table_v2.process_gold_feature_and_label_store(silver_base_dir, gold_base_dir, spark, dpd_cutoff = 30, mob_cutoff = 6)

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

    