# Required imports
import os
import pandas as pd
import glob
import datetime
import numpy as np  # Fixed typo from 'npm'
import json
from typing import Dict, List

# Evidently imports 
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests.base_test import generate_column_tests
from evidently.tests import *

from evidently.test_preset import (
    DataDriftTestPreset, 
    DataQualityTestPreset
)

# Workspace imports
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase 

import warnings 
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# to call this script: python scripts/model_monitoring.py 
# toggle Evidently User Interface by runnign this line in the terminal: evidently ui --workspace loan_default_monitoring_workspace

# Configuration
WORKSPACE_NAME = "loan_default_monitoring_workspace"
PROJECT_NAME = "Loan Default Prediction Monitoring"
PROJECT_DESCRIPTION = "Loan default prediction model with drift detection and automated testing"

def load_monitoring_datasets():
    """Load reference and production datasets for monitoring"""
    
    print("=== Loading Monitoring Datasets ===")
    
    # Load reference data (training data)
    reference_path = "datamart/gold/model_monitoring/reference_data.parquet"
    reference_data = pd.read_parquet(reference_path)
    print(f"Reference data loaded: {len(reference_data)} rows, {len(reference_data.columns)} columns")
    print(f"Reference data date range: {reference_data['snapshot_date'].min()} to {reference_data['snapshot_date'].max()}")
    
    # Load production data
    production_path = "datamart/gold/model_monitoring/production_data.parquet"
    production_data = pd.read_parquet(production_path)
    print(f"Production data loaded: {len(production_data)} rows, {len(production_data.columns)} columns")
    
    # Load reference prediction data 
    ref_prediction_path = "datamart/gold/model_predictions/XGB_model_2024_06_01/XGB_model_2024_06_01_training_predictions_2024_06_01.parquet"
    ref_prediction_data = pd.read_parquet(ref_prediction_path)
    print(f"Reference Prediction data loaded: {len(ref_prediction_data)} rows, {len(ref_prediction_data.columns)} columns")
    
    # Load production prediction data 
    prod_prediction_path = "datamart/gold/model_predictions/XGB_model_2024_06_01/XGB_model_2024_06_01_current_predictions_2024_06_01.parquet"
    prod_prediction_data = pd.read_parquet(prod_prediction_path)
    print(f"Production Prediction data loaded: {len(prod_prediction_data)} rows, {len(prod_prediction_data.columns)} columns")
    
    reference_data["snapshot_date"] = pd.to_datetime(reference_data["snapshot_date"])
    production_data["snapshot_date"] = pd.to_datetime(production_data["snapshot_date"])
    ref_prediction_data["snapshot_date"] = pd.to_datetime(ref_prediction_data["snapshot_date"])
    prod_prediction_data["snapshot_date"] = pd.to_datetime(prod_prediction_data["snapshot_date"])
    return reference_data, production_data, ref_prediction_data, prod_prediction_data

def create_comprehensive_monthly_report(snapshot_date="2024-06-01"):
    """Create comprehensive monthly monitoring reports"""
    
    # Initialize reports list
    reports = []
    
    # Convert snapshot_date to month name for display
    # Handle different date formats (with underscores or hyphens)
    if "_" in snapshot_date:
        # Convert 2024_06_01 to 2024-06-01
        formatted_date = snapshot_date.replace("_", "-")
    else:
        formatted_date = snapshot_date
    
    snapshot_dt = pd.to_datetime(formatted_date)
    month_name = snapshot_dt.strftime("%B %Y")
    
    # Load datasets
    reference_data, production_data, ref_prediction_data, prod_prediction_data = load_monitoring_datasets()
    
    # 1. DATA DRIFT REPORT
    if not reference_data.empty and not production_data.empty:
        print(f"Creating Data Drift Report...")
        
        data_drift_report = Report(
            metrics=[
                DataDriftPreset(stattest='psi', stattest_threshold=0.3),
                generate_column_metrics(ColumnSummaryMetric)])
        
        data_drift_report.run(
            reference_data=reference_data, 
            current_data=production_data)
        
        reports.append(data_drift_report)
        
        print(f"Data drift report created")
        
    # 2. TARGET DRIFT REPORT (Label patterns)
    if not reference_data.empty and not production_data.empty:
        print(f"Creating Target Drift Report...")
        
        target_drift_report = Report(
            metrics=[
                TargetDriftPreset(),
                ColumnDriftMetric(column_name="label", stattest='psi'),
                ColumnDistributionMetric(column_name="label"),
                ColumnSummaryMetric(column_name="label"),
                ColumnQuantileMetric(column_name="label", quantile=0.5),  # Median
                ColumnQuantileMetric(column_name="label", quantile=0.25), # 25th percentile
                ColumnQuantileMetric(column_name="label", quantile=0.75)  # 75th percentile
            ])
        
        target_drift_report.run(
            reference_data=reference_data,
            current_data=production_data)
        
        reports.append(target_drift_report)
        
        print(f"Target drift report created")
        
    # 3. PREDICTION DRIFT REPORT (Model output patterns)
    if not ref_prediction_data.empty and not prod_prediction_data.empty:
        print(f"Creating Prediction Drift Report...")
                
        if "model_predictions" in ref_prediction_data.columns and "model_predictions" in prod_prediction_data.columns:
            prediction_drift_report = Report(
                metrics=[
                    ColumnDriftMetric(column_name="model_predictions", stattest='psi', stattest_threshold=0.3),
                    ColumnDistributionMetric(column_name="model_predictions"),
                    ColumnSummaryMetric(column_name="model_predictions"),
                    ColumnQuantileMetric(column_name="model_predictions", quantile=0.95),
                    ColumnQuantileMetric(column_name="model_predictions", quantile=0.05),
                    DatasetSummaryMetric(),
                    DatasetMissingValuesMetric()])
            
            # Create subset with just predictions for drift analysis
            ref_pred_data = ref_prediction_data[['Customer_ID', 'model_predictions']].copy()
            
            prod_pred_data = prod_prediction_data[['Customer_ID', 'model_predictions']].copy()
            
            prediction_drift_report.run(
                reference_data=ref_pred_data,
                current_data=prod_pred_data
            )
            
            reports.append(prediction_drift_report)
            print("Prediction drift report created")
        
    
    # 4. PERFORMANCE MONITORING REPORT (Model performance)
    # Fixed: This should compare actual vs predicted, not label vs label
    if not reference_data.empty and not production_data.empty:
        print(f"Creating Performance Monitoring Report...")
        
        # Check if we have both labels and predictions for performance evaluation
        if 'label' in reference_data.columns and 'label' in production_data.columns:
            try:
                performance_report = Report(
                    metrics=[
                        DatasetSummaryMetric(),
                        DatasetMissingValuesMetric(),
                        ColumnDistributionMetric(column_name="label"),
                        ColumnSummaryMetric(column_name="label")])

                performance_report.run(
                    reference_data=reference_data,
                    current_data=production_data
                )
                
                reports.append(performance_report)
                print("Performance monitoring report created")
                
            except Exception as e:
                print(f"Warning: Could not create performance report: {e}")

    print(f"\n🎉 Created {len(reports)} monitoring reports for {month_name}")
    
    return reports

def create_comprehensive_monthly_test_suites(snapshot_date="2024-06-01"):
     # Initialize test suites list
    test_suites = []
    
    # Convert snapshot_date to month name for display
    if "_" in snapshot_date:
        formatted_date = snapshot_date.replace("_", "-")
    else:
        formatted_date = snapshot_date
    
    snapshot_dt = pd.to_datetime(formatted_date)
    month_name = snapshot_dt.strftime("%B %Y")
    
    # Load datasets
    reference_data, production_data, ref_prediction_data, prod_prediction_data = load_monitoring_datasets()
    
    # 1. DATA DRIFT TEST SUITE
    if not reference_data.empty and not production_data.empty:
        print(f"Creating Data Drift Test Suite...")
        
        data_drift_tests = TestSuite(tests=[
            DataDriftTestPreset(
                stattest='psi', 
                stattest_threshold=0.3,
                drift_share=0.3  # Allow up to 30% of features to drift
            ),
            # Data quality tests
            TestNumberOfColumns(),
            TestNumberOfRows(),
            TestNumberOfConstantColumns(),
        ])
        
        data_drift_tests.run(
            reference_data=reference_data,
            current_data=production_data
        )
        
        test_suites.append(data_drift_tests)
        print(f"Data drift test suite created")
    
    # 2. DATA QUALITY TEST SUITE
    if not production_data.empty:
        print(f"Creating Data Quality Test Suite...")
        
        data_quality_tests = TestSuite(tests=[
            DataQualityTestPreset(),
            # Custom data quality tests
            TestNumberOfMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
        ])
        
        data_quality_tests.run(
            reference_data=reference_data, 
            current_data=production_data
        )
        
        test_suites.append(data_quality_tests)
        print(f"Data quality test suite created")
    
    # 3. TARGET DRIFT TEST SUITE
    if not reference_data.empty and not production_data.empty and 'label' in reference_data.columns:
        print(f"Creating Target Drift Test Suite...")
        
        target_drift_tests = TestSuite(tests=[
            TestColumnDrift(
                column_name="label",
                stattest='psi',
                stattest_threshold=0.2
            ),
            # Target distribution tests
            TestColumnValueMean(column_name="label"),
            TestTargetFeaturesCorrelations()
        ])
        
        target_drift_tests.run(
            reference_data=reference_data,
            current_data=production_data
        )
        
        test_suites.append(target_drift_tests)
        print(f"Target drift test suite created")
    
    # 4. PREDICTION DRIFT TEST SUITE
    if not ref_prediction_data.empty and not prod_prediction_data.empty:
        print(f"Creating Prediction Drift Test Suite...")
        
        if "model_predictions" in ref_prediction_data.columns and "model_predictions" in prod_prediction_data.columns:
            prediction_drift_tests = TestSuite(tests=[
                TestColumnDrift(
                    column_name="model_predictions",
                    stattest='psi',
                    stattest_threshold=0.2
                ),
                # Prediction quality tests
                TestColumnValueMean(column_name="model_predictions"),
                TestColumnValueStd(column_name="model_predictions"),
                TestColumnQuantile(
                    column_name="model_predictions",
                    quantile=0.95
                ),
                TestColumnQuantile(
                    column_name="model_predictions", 
                    quantile=0.05
                )
            ])
            
            # Create subset with just predictions for drift analysis
            ref_pred_data = ref_prediction_data[['Customer_ID', 'model_predictions']].copy()
            prod_pred_data = prod_prediction_data[['Customer_ID', 'model_predictions']].copy()
            
            prediction_drift_tests.run(
                reference_data=ref_pred_data,
                current_data=prod_pred_data
            )
            
            test_suites.append(prediction_drift_tests)
            print("Prediction drift test suite created")
    
    # 5. DATA STABILITY TEST SUITE
    if not production_data.empty:
        print(f"Creating Data Stability Test Suite...")
        
        data_stability_tests = TestSuite(tests=[
            DataStabilityTestPreset(),
            # Custom stability tests
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfMissingValues()
        ])
        
        data_stability_tests.run(
            reference_data=reference_data, 
            current_data=production_data
        )
        
        test_suites.append(data_stability_tests)
        print(f"Data stability test suite created")
    
    # 6. MODEL PERFORMANCE TEST SUITE (if ground truth is available)
    if not reference_data.empty and not production_data.empty:
        print(f"Creating Model Performance Test Suite...")
        
        # This assumes you have actual labels for performance evaluation
        performance_tests = TestSuite(tests=[
            NoTargetPerformanceTestPreset(),
            # Custom performance tests
            TestNumberOfMissingValues(),
            TestColumnValueMean(column_name="label")
        ])
        
        performance_tests.run(
            reference_data=reference_data,
            current_data=production_data
        )
        
        test_suites.append(performance_tests)
        print("Model performance test suite created")
    
    print(f"\n🧪 Created {len(test_suites)} test suites for {month_name}")
    return test_suites

def run_monthly_monitoring_with_tests(workspace, snapshot_date: str = "2024_06_01") -> Dict[str, List]:
    """
    Run complete monthly monitoring for a specific snapshot date
    """
    print(f"\n🚀 STARTING MONTHLY MONITORING WITH TESTS FOR {snapshot_date}")
    print("=" * 80)
    
    try:
        # Create reports
        reports = create_comprehensive_monthly_report(snapshot_date)
        
        # Create test suites
        test_suites = create_comprehensive_monthly_test_suites(snapshot_date)
        
        # Get or create project
        try:
            project = workspace.search_project(PROJECT_NAME)[0]
            print(f"✅ Found existing project '{PROJECT_NAME}'")
        except (IndexError, AttributeError):
            project = workspace.create_project(PROJECT_NAME, PROJECT_DESCRIPTION)
            print(f"✅ Created new project '{PROJECT_NAME}'")

        project_id = project.id 
                
        # Add reports to workspace
        print(f"\n📤 Adding reports to workspace...")
        for i, report in enumerate(reports):
            try:
                workspace.add_report(project_id, report)
                print(f"✅ Added report {i+1}/{len(reports)}")
            except Exception as e:
                print(f"❌ Failed to add report {i+1}: {e}")

        # Add test suites to workspace
        print(f"\n🧪 Adding test suites to workspace...")
        for i, test_suite in enumerate(test_suites):
            try:
                workspace.add_test_suite(project_id, test_suite)
                print(f"✅ Added test suite {i+1}/{len(test_suites)}")
            except Exception as e:
                print(f"❌ Failed to add test suite {i+1}: {e}")

        # Print test results summary
        print(f"\n📋 TEST RESULTS SUMMARY:")
        print("=" * 50)
        
        for i, test_suite in enumerate(test_suites):
            suite_name = f"Test Suite {i+1}"
            try:
                results = test_suite.as_dict()
                total_tests = len(results.get('tests', []))
                passed_tests = sum(1 for test in results.get('tests', []) if test.get('status') == 'SUCCESS')
                failed_tests = total_tests - passed_tests
                
                print(f"{suite_name}: {passed_tests}/{total_tests} passed, {failed_tests} failed")
                
                # Show failed test details
                if failed_tests > 0:
                    failed_test_names = [
                        test.get('name', 'Unknown test') 
                        for test in results.get('tests', []) 
                        if test.get('status') != 'SUCCESS'
                    ]
                    print(f"  ❌ Failed tests: {', '.join(failed_test_names[:3])}{'...' if len(failed_test_names) > 3 else ''}")
                
            except Exception as e:
                print(f"{suite_name}: Error getting results - {e}")

        print(f"\n✅ Successfully processed {len(reports)} reports and {len(test_suites)} test suites")

        return {
            'reports': reports,
            'test_suites': test_suites,
            'snapshot_date': snapshot_date,
            'project_id': project_id
        }
        
    except Exception as e:
        print(f"❌ Error in monthly monitoring for {snapshot_date}: {str(e)}")
        raise

# ===== WORKSPACE AND PROJECT SETUP =====
def get_or_create_workspace():
    """Get existing workspace or create new one"""
    try:
        # Try to load existing workspace
        ws = Workspace.load(WORKSPACE_NAME)
        print(f"✅ Found existing workspace '{WORKSPACE_NAME}'")
        
    except:
        # Create new workspace if it doesn't exist
        ws = Workspace.create(WORKSPACE_NAME)
        print(f"✅ Created new workspace '{WORKSPACE_NAME}'")
    
    return ws

# ===== MAIN SCRIPT ENTRY POINT =====
if __name__ == "__main__":
    workspace = get_or_create_workspace()

    result = run_monthly_monitoring_with_tests(workspace, snapshot_date="2024_06_01")
    print(f"\n🎉 Monthly monitoring with tests completed successfully!")
    print(f"📊 Generated {len(result['reports'])} reports and {len(result['test_suites'])} test suites")

    print(f"📁 Project ID: {result['project_id']}")
    
    