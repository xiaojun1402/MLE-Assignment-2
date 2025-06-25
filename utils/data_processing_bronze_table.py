'''
This script builds the Bronze layer of your data lake. 
In the Medallion Architecture, the Bronze layer stores raw ingested data (as-is or lightly cleaned) from the source system - in this case, a CSV file. 
For each given date (e.g. every first of the month), this function: 
1. Reads the entire raw datasets (different csv files)  
2. Filters the rows for that snapshot date 
3. Saves the filtered data to a new CSV file named with the snapshot date and in different folders 
4. Logs the process so you can track progress 
'''

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_bronze_table(snapshot_date_str, bronze_base_dir, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    datasets = {
        "lms": "Data/lms_loan_daily.csv",
        "financials": "Data/features_financials.csv",
        "attributes": "Data/features_attributes.csv",
        "clickstream": "Data/feature_clickstream.csv"
    }

    for name, path in datasets.items():
        # Read the raw CSV
        df = spark.read.csv(path, header=True, inferSchema=True)

        # Filter by snapshot date
        filtered_df = df.filter(col("snapshot_date") == snapshot_date)

        # Print row count for verification
        print(f"{name} ({snapshot_date_str}) row count:", filtered_df.count())

        # Build save path
        bronze_subdir = os.path.join(bronze_base_dir, name)
        os.makedirs(bronze_subdir, exist_ok=True)

        filename = f"bronze_{name}_{snapshot_date_str.replace('-', '_')}.csv"
        filepath = os.path.join(bronze_subdir, filename)

        # Save as Pandas CSV
        filtered_df.toPandas().to_csv(filepath, index=False)
        print(f"Saved {name} to: {filepath}")

    return

