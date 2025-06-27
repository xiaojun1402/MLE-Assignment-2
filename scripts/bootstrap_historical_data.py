#!/usr/bin/env python3
"""
Bootstrap script to process all historical data through bronze and silver layers.
This creates the foundation for gold layer processing.

Usage: python bootstrap_historical_data.py
"""

### Core python libs 
import argparse
import os
import glob
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY

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

def bootstrap_historical_data(start_date="2023-01-01", end_date="2024-12-01"):
    """
    Bootstrap all historical data through bronze and silver layers.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    print('\n\n=== BOOTSTRAP HISTORICAL DATA PROCESSING ===\n')
    
    # Set minimal Java configuration for Spark
    import subprocess
    import sys
    
    # Set all necessary environment variables before any Spark imports
    java_home = '/usr/lib/jvm/java-17-openjdk-arm64'
    os.environ['JAVA_HOME'] = java_home
    os.environ['SPARK_HOME'] = '/home/airflow/.local/lib/python3.7/site-packages/pyspark'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # Verify Java is accessible
    try:
        result = subprocess.run([f'{java_home}/bin/java', '-version'], 
                               capture_output=True, text=True, check=True)
        print(f"Java verification successful")
    except Exception as e:
        print(f"Java verification failed: {e}")
        # Fallback to system java
        os.environ['JAVA_HOME'] = ''
    
    # Try to initialize Spark with robust error handling
    spark = None
    try:
        spark = pyspark.sql.SparkSession.builder \
            .appName("bootstrap_historical_data") \
            .master("local[1]") \
            .config("spark.driver.memory", "512m") \
            .config("spark.driver.maxResultSize", "256m") \
            .config("spark.sql.shuffle.partitions", "1") \
            .getOrCreate()
        print("Spark initialized successfully")
    except Exception as e:
        print(f"Spark initialization failed: {e}")
        return False

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")
    
    # Generate all first-of-month dates between start and end
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_list = [dt.strftime("%Y-%m-01") for dt in rrule(MONTHLY, dtstart=start_dt, until=end_dt)]
    print(f"Processing {len(date_list)} months from {start_date} to {end_date}")
    print(f"Dates to process: {date_list[:3]}...{date_list[-3:]}")
    
    # Create datalake directories
    bronze_base_dir = "/opt/airflow/datamart/bronze/"
    silver_base_dir = "/opt/airflow/datamart/silver/"
    
    for base_dir in [bronze_base_dir, silver_base_dir]:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    # Track processing statistics
    stats = {
        'total_dates': len(date_list),
        'processed_dates': 0,
        'failed_dates': [],
        'bronze_files_created': 0,
        'silver_files_created': 0
    }
    
    # Process each date for bronze and silver
    for i, date_str in enumerate(date_list, 1):
        try:
            print(f"\n=== [{i}/{len(date_list)}] Processing snapshot date: {date_str} ===")
            
            # Check if data already exists (skip if already processed)
            bronze_lms_file = f"{bronze_base_dir}/lms/bronze_lms_{date_str.replace('-', '_')}.csv"
            silver_lms_file = f"{silver_base_dir}/lms/silver_lms_{date_str.replace('-', '_')}.parquet"
            
            if os.path.exists(bronze_lms_file) and os.path.exists(silver_lms_file):
                print(f"⏭️  Data for {date_str} already exists, skipping...")
                stats['processed_dates'] += 1
                continue
            
            # Run data processing - bronze 
            print(f"📥 Processing bronze layer for {date_str}")
            utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_base_dir, spark)
            stats['bronze_files_created'] += 4  # lms, attributes, financials, clickstream

            # Run data processing - silver 
            print(f"🔄 Processing silver layer for {date_str}")
            utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_base_dir, silver_base_dir, spark)
            utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_base_dir, silver_base_dir, spark)
            utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_base_dir, silver_base_dir, spark)
            utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_base_dir, silver_base_dir, spark)
            stats['silver_files_created'] += 4  # lms, attributes, financials, clickstream
            
            stats['processed_dates'] += 1
            print(f"✅ Successfully processed {date_str}")
            
            # Progress update every 5 dates
            if i % 5 == 0:
                progress = (i / len(date_list)) * 100
                print(f"\n📊 Progress: {progress:.1f}% ({i}/{len(date_list)} dates processed)")
            
        except Exception as e:
            print(f"❌ Error processing {date_str}: {str(e)}")
            stats['failed_dates'].append(date_str)
            continue
    
    # End spark session
    if spark:
        spark.stop()
    
    # Print final statistics
    print('\n\n=== BOOTSTRAP PROCESSING COMPLETED ===')
    print(f"📊 Final Statistics:")
    print(f"   • Total dates to process: {stats['total_dates']}")
    print(f"   • Successfully processed: {stats['processed_dates']}")
    print(f"   • Failed dates: {len(stats['failed_dates'])}")
    print(f"   • Bronze files created: {stats['bronze_files_created']}")
    print(f"   • Silver files created: {stats['silver_files_created']}")
    
    if stats['failed_dates']:
        print(f"\n❌ Failed dates: {stats['failed_dates']}")
    
    success_rate = (stats['processed_dates'] / stats['total_dates']) * 100
    print(f"\n✅ Success rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 Bootstrap completed successfully!")
        print("💡 You can now run gold layer processing to create feature-label pairs.")
        return True
    else:
        print("\n⚠️  Bootstrap completed with some failures.")
        print("🔧 Please check the failed dates and retry if needed.")
        return False

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments 
    parser = argparse.ArgumentParser(description="Bootstrap historical data processing")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-12-01", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Run bootstrap processing
    success = bootstrap_historical_data(args.start_date, args.end_date)
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Run: python scripts/main.py --snapshotdate '2024-06-01' (to test gold layer)")
        print("2. Or use Airflow backfill for ongoing incremental processing")
    else:
        print("\n🔧 Please fix the errors and retry bootstrap processing")