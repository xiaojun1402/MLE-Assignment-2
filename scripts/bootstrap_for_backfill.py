#!/usr/bin/env python3
"""
Bootstrap script to prepare data for temporal Airflow backfill.
Processes all available data (Jan 2023 - Dec 2024) through bronze and silver layers.

This enables the DAG backfill to work correctly with snapshot date constraints.
"""

### Core python libs 
import argparse
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY

### PySpark 
import pyspark
import sys

### Custom scripts from utils folder 
from utils.data_processing_bronze_table import process_bronze_table
import utils.data_processing_silver_table

def bootstrap_for_backfill():
    """
    Bootstrap all historical data to support DAG backfill from 2024-03-01 to 2024-12-01.
    
    This processes ALL available data (Jan 2023 - Dec 2024) through bronze/silver layers
    so that each DAG run can apply snapshot date constraints dynamically.
    """
    print('\n\n=== BOOTSTRAP FOR TEMPORAL BACKFILL ===\n')
    
    # Set minimal Java configuration for Spark
    java_home = '/usr/lib/jvm/java-17-openjdk-arm64'
    os.environ['JAVA_HOME'] = java_home
    os.environ['SPARK_HOME'] = '/home/airflow/.local/lib/python3.7/site-packages/pyspark'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # Initialize Spark
    try:
        spark = pyspark.sql.SparkSession.builder \
            .appName("bootstrap_for_backfill") \
            .master("local[1]") \
            .config("spark.driver.memory", "512m") \
            .config("spark.driver.maxResultSize", "256m") \
            .config("spark.sql.shuffle.partitions", "1") \
            .getOrCreate()
        print("Spark initialized successfully")
    except Exception as e:
        print(f"Spark initialization failed: {e}")
        return False

    spark.sparkContext.setLogLevel("ERROR")
    
    # Process ALL available data (full dataset range)
    start_date = "2023-01-01"
    end_date = "2024-12-01"
    
    # Generate all first-of-month dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [dt.strftime("%Y-%m-01") for dt in rrule(MONTHLY, dtstart=start_dt, until=end_dt)]
    
    print(f"Processing ALL {len(date_list)} months from {start_date} to {end_date}")
    print("This creates the foundation for temporal DAG backfill...")
    
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
        'skipped_dates': 0,
        'failed_dates': []
    }
    
    # Process each date for bronze and silver
    for i, date_str in enumerate(date_list, 1):
        try:
            print(f"\\n=== [{i}/{len(date_list)}] Processing: {date_str} ===")
            
            # Check if data already exists
            bronze_lms_file = f"{bronze_base_dir}/lms/bronze_lms_{date_str.replace('-', '_')}.csv"
            silver_lms_file = f"{silver_base_dir}/lms/silver_lms_{date_str.replace('-', '_')}.parquet"
            
            if os.path.exists(bronze_lms_file) and os.path.exists(silver_lms_file):
                print(f"⏭️  Data for {date_str} already exists, skipping...")
                stats['skipped_dates'] += 1
                continue
            
            # Process bronze layer
            print(f"📥 Processing bronze layer for {date_str}")
            utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_base_dir, spark)

            # Process silver layer  
            print(f"🔄 Processing silver layer for {date_str}")
            utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_base_dir, silver_base_dir, spark)
            utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_base_dir, silver_base_dir, spark)
            utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_base_dir, silver_base_dir, spark)
            utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_base_dir, silver_base_dir, spark)
            
            stats['processed_dates'] += 1
            print(f"✅ Successfully processed {date_str}")
            
            # Progress update every 6 dates
            if i % 6 == 0:
                progress = (i / len(date_list)) * 100
                print(f"\\n📊 Progress: {progress:.1f}% ({i}/{len(date_list)} dates processed)")
            
        except Exception as e:
            print(f"❌ Error processing {date_str}: {str(e)}")
            stats['failed_dates'].append(date_str)
            continue
    
    # End spark session
    if spark:
        spark.stop()
    
    # Print final statistics
    print('\\n\\n=== BOOTSTRAP COMPLETED ===')
    print(f"📊 Final Statistics:")
    print(f"   • Total dates: {stats['total_dates']}")
    print(f"   • Newly processed: {stats['processed_dates']}")
    print(f"   • Already existed: {stats['skipped_dates']}")
    print(f"   • Failed: {len(stats['failed_dates'])}")
    
    if stats['failed_dates']:
        print(f"\\n❌ Failed dates: {stats['failed_dates']}")
    
    total_ready = stats['processed_dates'] + stats['skipped_dates']
    success_rate = (total_ready / stats['total_dates']) * 100
    print(f"\\n✅ Data readiness: {success_rate:.1f}% ({total_ready}/{stats['total_dates']} months)")
    
    if success_rate >= 95:
        print("\\n🎉 Bootstrap completed successfully!")
        print("\\n🚀 Ready for Airflow backfill:")
        print("   airflow dags backfill ml_pipeline_loan_default -s 2024-03-01 -e 2024-12-01")
        print("\\n📈 Expected DAG runs: 10 months (March 2024 to December 2024)")
        print("📋 Each run will:")
        print("   • Use snapshot date constraints (no future data)")
        print("   • Create valid feature-label pairs within dataset range")
        print("   • Train models respecting 10/2/2 month methodology")
        return True
    else:
        print("\\n⚠️  Bootstrap completed with issues.")
        print("🔧 Please check failed dates and retry if needed.")
        return False

if __name__ == "__main__":
    success = bootstrap_for_backfill()
    
    if success:
        print("\\n📝 Next Steps:")
        print("1. Run: docker-compose restart  (restart Airflow with new DAG config)")
        print("2. Access Airflow UI: http://localhost:8080")
        print("3. Trigger backfill or individual DAG runs")
        print("4. Monitor temporal model performance across snapshots")
    else:
        print("\\n🔧 Please fix errors and retry bootstrap")