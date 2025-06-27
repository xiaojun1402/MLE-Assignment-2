from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
import pandas as pd

# Evidently imports for model monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.ui.workspace import Workspace 

# Default snapshot date
DEFAULT_SNAPSHOT_DATE = "2024-06-01"

def get_snapshot_date(**context):
    """
    Get snapshot date from Airflow execution date for backfill support
    For manual runs, users can override by providing 'snapshot_date' in DAG config
    """
    dag_run = context.get('dag_run')
    execution_date = context.get('execution_date')
    
    # Priority 1: User provided custom date in DAG config (for manual overrides)
    if dag_run and dag_run.conf and dag_run.conf.get('snapshot_date'):
        custom_date = dag_run.conf.get('snapshot_date')
        print(f"Using custom snapshot date from config: {custom_date}")
        return custom_date
    
    # Priority 2: Use execution date for backfill (first day of month)
    elif execution_date:
        # Convert execution date to first day of month (YYYY-MM-01 format)
        snapshot_date = execution_date.strftime("%Y-%m-01")
        print(f"Using execution date as snapshot date: {snapshot_date}")
        return snapshot_date
    
    # Priority 3: Fallback to default
    else:
        print(f"Using default snapshot date: {DEFAULT_SNAPSHOT_DATE}")
        return DEFAULT_SNAPSHOT_DATE

def run_evidently_monitoring(**context):
    """
    Run Evidently AI monitoring for model drift detection
    """
    # Get snapshot date from previous task
    snapshot_date = context['ti'].xcom_pull(task_ids='get_snapshot_date')
    snapshot_formatted = snapshot_date.replace("-", "_")
    
    print(f"=== Running Evidently Monitoring for {snapshot_date} ===")
    
    # Load reference and current data
    reference_path = "/opt/airflow/datamart/gold/model_monitoring/reference_data.parquet"
    production_path = "/opt/airflow/datamart/gold/model_monitoring/production_data.parquet"
    
    # Check if files exist
    if not os.path.exists(reference_path):
        print(f"Warning: Reference data not found at {reference_path}")
        return {"status": "skipped", "reason": "Reference data not found"}
    
    if not os.path.exists(production_path):
        print(f"Warning: Production data not found at {production_path}")
        return {"status": "skipped", "reason": "Production data not found"}
    
    # Load data
    reference_data = pd.read_parquet(reference_path)
    current_data = pd.read_parquet(production_path)
    
    print(f"Reference data: {len(reference_data)} rows, {len(reference_data.columns)} columns")
    print(f"Current data: {len(current_data)} rows, {len(current_data.columns)} columns")
    
    # Create Evidently reports
    reports_created = []
    
    # 1. Data Drift Report
    try:
        data_drift_report = Report(metrics=[
            DataDriftPreset(stattest='psi', stattest_threshold=0.3),
            DatasetSummaryMetric(),
            DatasetMissingValuesMetric()
        ])
        
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Save report
        report_path = f"/opt/airflow/monitoring_reports/data_drift_report_{snapshot_formatted}.html"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        data_drift_report.save_html(report_path)
        
        reports_created.append(f"Data Drift Report saved to {report_path}")
        print(f"✅ Data drift report created: {report_path}")
        
    except Exception as e:
        print(f"❌ Error creating data drift report: {e}")
    
    # 2. Target Drift Report (if label column exists)
    if 'label' in reference_data.columns and 'label' in current_data.columns:
        try:
            target_drift_report = Report(metrics=[
                TargetDriftPreset(),
                ColumnDriftMetric(column_name="label", stattest='psi'),
                ColumnDistributionMetric(column_name="label")
            ])
            
            target_drift_report.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            # Save report
            report_path = f"/opt/airflow/monitoring_reports/target_drift_report_{snapshot_formatted}.html"
            target_drift_report.save_html(report_path)
            
            reports_created.append(f"Target Drift Report saved to {report_path}")
            print(f"✅ Target drift report created: {report_path}")
            
        except Exception as e:
            print(f"❌ Error creating target drift report: {e}")
    
    # Setup Evidently workspace
    workspace_path = "/opt/airflow/monitoring_workspace/loan_default_monitoring_workspace"
    try:
        os.makedirs(os.path.dirname(workspace_path), exist_ok=True)
        
        # Create or load workspace
        try:
            workspace = Workspace.load(workspace_path)
            print(f"✅ Loaded existing workspace")
        except:
            workspace = Workspace.create(workspace_path)
            print(f"✅ Created new workspace")
        
        # Create or get project
        project_name = "Loan Default Prediction Monitoring"
        try:
            projects = workspace.search_project(project_name)
            if projects:
                project = projects[0]
            else:
                project = workspace.create_project(project_name, "Model monitoring for loan default prediction")
            
            # Add reports to workspace
            if 'data_drift_report' in locals():
                workspace.add_report(project.id, data_drift_report)
                print(f"✅ Added data drift report to workspace")
            
            if 'target_drift_report' in locals():
                workspace.add_report(project.id, target_drift_report)
                print(f"✅ Added target drift report to workspace")
                
        except Exception as e:
            print(f"❌ Error adding reports to workspace: {e}")
            
    except Exception as e:
        print(f"❌ Error setting up workspace: {e}")
    
    result = {
        "status": "completed",
        "snapshot_date": snapshot_date,
        "reports_created": reports_created,
        "workspace_path": workspace_path
    }
    
    print(f"✅ Evidently monitoring completed successfully!")
    print(f"📊 Created {len(reports_created)} reports")
    print(f"💡 To view dashboard: evidently ui --workspace {workspace_path}")
    
    return result

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_pipeline_loan_default',
    default_args=default_args,
    description=f'End-to-end ML pipeline with temporal constraints',
    schedule_interval='@monthly',  # Monthly schedule for backfill
    start_date=datetime(2024, 3, 1),  # First valid snapshot (needs 14 months prior data)
    end_date=datetime(2024, 12, 1),   # Last snapshot with full dataset
    catchup=True,  # Enable backfill
    max_active_runs=2,  # Allow some parallelism but not too much
    # Add DAG documentation
    doc_md=f"""
    ## ML Pipeline for Loan Default Prediction
    
    **Schedule:** Monthly backfill enabled (2024-03-01 to 2024-12-01)
    
    ### Training Methodology:
    - **Training Period**: 10 months (80% of 12-month window)
    - **Test Period**: 2 months (20% of 12-month window)  
    - **OOT Period**: 2 months (out-of-time validation)
    - **Total Window**: 14 months before snapshot date
    
    ### No Data Leakage Constraints:
    - Features: Only use data up to snapshot date
    - Labels: Available from full dataset (Jan 2023 - Dec 2024)
    - Training: Respects temporal boundaries
    
    ### Backfill Usage:
    1. **Historical Processing:** Use Airflow backfill:
       ```bash
       airflow dags backfill ml_pipeline_loan_default -s 2024-03-01 -e 2024-12-01
       ```
    
    2. **Single Month:** Trigger manually for specific snapshot date
    
    3. **Custom Date Override:** Click "Trigger DAG w/ Config" and enter:
       ```json
       {{"snapshot_date": "2024-07-01"}}
       ```
    
    ### Example Timeline (Snapshot: 2024-07-01):
    - Train: 2023-05-01 to 2024-02-01 (10 months)
    - Test: 2024-03-01 to 2024-04-01 (2 months)
    - OOT: 2024-05-01 to 2024-06-01 (2 months)
    - Feature-Label: Use all valid pairs with labels ≤ Dec 2024
    
    **Valid Snapshots:** 2024-03-01 to 2024-12-01 (10 runs)
    """,
) as dag:

    # =============================================================================
    # GET SNAPSHOT DATE
    # =============================================================================
    
    get_date_task = PythonOperator(
        task_id='get_snapshot_date',
        python_callable=get_snapshot_date,
        do_xcom_push=True,
    )

    # =============================================================================
    # DATA PIPELINE 
    # =============================================================================
        
    data_processing_start = DummyOperator(task_id='data_processing_start')

    data_processing = BashOperator(
        task_id='data_processing',
        bash_command=(
            'cd /opt/airflow && '  # Changed from /opt/airflow/scripts
            'python3 scripts/main.py '
            '--snapshotdate "{{ ti.xcom_pull(task_ids="get_snapshot_date") }}"'
        ),
    )

    data_processing_completed = DummyOperator(task_id="data_processing_completed")
        
    # =============================================================================
    # MACHINE LEARNING PIPELINE 
    # =============================================================================

    machine_learning_pipeline_start = DummyOperator(task_id='machine_learning_pipeline_start')
    model_training_start = DummyOperator(task_id='model_training_start')

    model_training = BashOperator(
        task_id='model_training',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_train.py '
            '--snapshotdate "{{ ti.xcom_pull(task_ids="get_snapshot_date") }}"'
        ),
    )

    model_inference_start = DummyOperator(task_id='model_inference_start')

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_inference.py '
            '--snapshotdate "{{ ti.xcom_pull(task_ids="get_snapshot_date") }}" '
            '--modelname "XGB_model_{{ ti.xcom_pull(task_ids="get_snapshot_date") | replace("-", "_") }}.pkl"'
        ),
    )

    model_monitoring_start = DummyOperator(task_id='model_monitoring_start')

    model_monitoring = PythonOperator(
        task_id='model_monitoring',
        python_callable=run_evidently_monitoring,
        provide_context=True,
    )

    machine_learning_pipeline_completed = DummyOperator(task_id="machine_learning_pipeline_completed")
        
    # =============================================================================
    # TASK DEPENDENCIES
    # =============================================================================

    get_date_task >> data_processing_start >> data_processing >> data_processing_completed >> machine_learning_pipeline_start >> model_training_start >> model_training >> model_inference_start >> model_inference >> model_monitoring_start >> model_monitoring >> machine_learning_pipeline_completed