#!/usr/bin/env python3
"""
Verification script to check data coverage after bootstrap processing.
Shows which dates have been processed through bronze/silver layers.
"""

import os
import glob
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
from dateutil.relativedelta import relativedelta

def verify_data_coverage(start_date="2023-01-01", end_date="2024-12-01"):
    """
    Verify which dates have been processed through bronze and silver layers.
    """
    print('\n=== DATA COVERAGE VERIFICATION ===\n')
    
    # Generate expected date list
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    expected_dates = [dt.strftime("%Y-%m-01") for dt in rrule(MONTHLY, dtstart=start_dt, until=end_dt)]
    
    print(f"Expected date range: {start_date} to {end_date}")
    print(f"Expected months: {len(expected_dates)}")
    
    # Check bronze layer coverage
    print(f"\n📥 BRONZE LAYER COVERAGE:")
    bronze_base_dir = "/opt/airflow/datamart/bronze/"
    bronze_coverage = {}
    
    for table in ['lms', 'attributes', 'financials', 'clickstream']:
        bronze_coverage[table] = []
        table_dir = os.path.join(bronze_base_dir, table)
        
        if os.path.exists(table_dir):
            files = glob.glob(f"{table_dir}/bronze_{table}_*.csv")
            for file in files:
                # Extract date from filename: bronze_lms_2023_01_01.csv -> 2023-01-01
                basename = os.path.basename(file)
                date_part = basename.replace(f"bronze_{table}_", "").replace(".csv", "")
                date_str = date_part.replace("_", "-")
                bronze_coverage[table].append(date_str)
        
        bronze_coverage[table] = sorted(bronze_coverage[table])
        coverage_pct = (len(bronze_coverage[table]) / len(expected_dates)) * 100
        print(f"   • {table:12}: {len(bronze_coverage[table]):2d}/{len(expected_dates)} files ({coverage_pct:5.1f}%)")
    
    # Check silver layer coverage
    print(f"\n🔄 SILVER LAYER COVERAGE:")
    silver_base_dir = "/opt/airflow/datamart/silver/"
    silver_coverage = {}
    
    for table in ['lms', 'attributes', 'financials', 'clickstream']:
        silver_coverage[table] = []
        table_dir = os.path.join(silver_base_dir, table)
        
        if os.path.exists(table_dir):
            files = glob.glob(f"{table_dir}/silver_{table}_*.parquet")
            for file in files:
                # Extract date from filename: silver_lms_2023_01_01.parquet -> 2023-01-01
                basename = os.path.basename(file)
                date_part = basename.replace(f"silver_{table}_", "").replace(".parquet", "")
                date_str = date_part.replace("_", "-")
                silver_coverage[table].append(date_str)
        
        silver_coverage[table] = sorted(silver_coverage[table])
        coverage_pct = (len(silver_coverage[table]) / len(expected_dates)) * 100
        print(f"   • {table:12}: {len(silver_coverage[table]):2d}/{len(expected_dates)} files ({coverage_pct:5.1f}%)")
    
    # Find missing dates
    print(f"\n❓ MISSING DATA ANALYSIS:")
    
    all_bronze_dates = set()
    for table_dates in bronze_coverage.values():
        all_bronze_dates.update(table_dates)
    
    all_silver_dates = set()
    for table_dates in silver_coverage.values():
        all_silver_dates.update(table_dates)
    
    missing_bronze = set(expected_dates) - all_bronze_dates
    missing_silver = set(expected_dates) - all_silver_dates
    
    if missing_bronze:
        print(f"   • Missing bronze dates: {sorted(missing_bronze)}")
    else:
        print(f"   • ✅ All bronze dates present")
    
    if missing_silver:
        print(f"   • Missing silver dates: {sorted(missing_silver)}")
    else:
        print(f"   • ✅ All silver dates present")
    
    # Check for feature-label pair potential
    print(f"\n🎯 FEATURE-LABEL PAIR ANALYSIS:")
    
    # For 6-month MOB cutoff, check which feature dates have corresponding label dates
    lms_dates = set(silver_coverage.get('lms', []))
    attr_dates = set(silver_coverage.get('attributes', []))
    fin_dates = set(silver_coverage.get('financials', []))
    
    # Potential feature dates (intersection of attributes and financials)
    potential_feature_dates = attr_dates.intersection(fin_dates)
    
    valid_pairs = []
    for feature_date in sorted(potential_feature_dates):
        feature_dt = datetime.strptime(feature_date, "%Y-%m-%d")
        label_dt = feature_dt + relativedelta(months=6)
        label_date = label_dt.strftime("%Y-%m-%d")
        
        if label_date in lms_dates:
            valid_pairs.append((feature_date, label_date))
    
    print(f"   • Potential feature dates: {len(potential_feature_dates)}")
    print(f"   • Valid feature-label pairs: {len(valid_pairs)}")
    
    if valid_pairs:
        print(f"   • Example pairs:")
        for i, (feat_date, label_date) in enumerate(valid_pairs[:5]):
            print(f"     - Features: {feat_date} → Labels: {label_date}")
        if len(valid_pairs) > 5:
            print(f"     - ... and {len(valid_pairs) - 5} more pairs")
    
    # Overall assessment
    print(f"\n📊 OVERALL ASSESSMENT:")
    
    total_expected_files = len(expected_dates) * 4  # 4 tables per date
    total_bronze_files = sum(len(dates) for dates in bronze_coverage.values())
    total_silver_files = sum(len(dates) for dates in silver_coverage.values())
    
    bronze_completeness = (total_bronze_files / total_expected_files) * 100
    silver_completeness = (total_silver_files / total_expected_files) * 100
    
    print(f"   • Bronze layer completeness: {bronze_completeness:.1f}%")
    print(f"   • Silver layer completeness: {silver_completeness:.1f}%")
    print(f"   • Valid ML pairs available: {len(valid_pairs)}")
    
    if bronze_completeness >= 90 and silver_completeness >= 90 and len(valid_pairs) > 0:
        print(f"\n✅ SUCCESS: Data is ready for gold layer processing!")
        print(f"🎯 You can create {len(valid_pairs)} feature-label pairs for ML model training.")
        return True
    else:
        print(f"\n⚠️  WARNING: Data coverage is incomplete.")
        print(f"🔧 Consider running bootstrap script to fill missing data.")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify data coverage")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-12-01", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    ready = verify_data_coverage(args.start_date, args.end_date)
    
    if ready:
        print(f"\n🚀 Ready for next steps!")
    else:
        print(f"\n🔧 Please run bootstrap processing first.")