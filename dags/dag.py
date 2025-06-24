from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os 

# Default snapshot date
DEFAULT_SNAPSHOT_DATE = "2024-06-01"

def get_snapshot_date(**context):
    """
    Get snapshot date from DAG config or use default
    Users can override by providing 'snapshot_date' in DAG config
    """
    dag_run = context.get('dag_run')
    if dag_run and dag_run.conf:
        # User provided custom date in DAG config
        custom_date = dag_run.conf.get('snapshot_date', DEFAULT_SNAPSHOT_DATE)
        print(f"Using custom snapshot date: {custom_date}")
        return custom_date
    else:
        # Use default date
        print(f"Using default snapshot date: {DEFAULT_SNAPSHOT_DATE}")
        return DEFAULT_SNAPSHOT_DATE

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_pipeline_loan_default',
    default_args=default_args,
    description=f'End-to-end ML pipeline - Default: {DEFAULT_SNAPSHOT_DATE} (customizable)',
    schedule_interval=None,  # Manual only
    start_date=datetime(2024, 6, 1),
    catchup=False,
    max_active_runs=1,
    # Add DAG documentation
    doc_md=f"""
    ## ML Pipeline for Loan Default Prediction
    
    **Default Date:** {DEFAULT_SNAPSHOT_DATE}
    
    ### How to use custom dates:
    1. Click "Trigger DAG w/ Config"
    2. Enter: {{"snapshot_date": "YYYY-MM-DD"}}
    3. Example: {{"snapshot_date": "2024-07-01"}}
    
    ### Simple trigger uses default date: {DEFAULT_SNAPSHOT_DATE}
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
            'cd /opt/airflow/scripts && '
            'python3 model_train.py '
            '--snapshotdate "{{ ti.xcom_pull(task_ids="get_snapshot_date") }}"'
        ),
    )

    model_inference_start = DummyOperator(task_id='model_inference_start')

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py '
            '--snapshotdate "{{ ti.xcom_pull(task_ids="get_snapshot_date") }}" '
            '--modelname "XGB_model_{{ ti.xcom_pull(task_ids="get_snapshot_date") | replace("-", "") }}.pkl"'
        ),
    )

    model_monitoring_start = DummyOperator(task_id='model_monitoring_start')

    model_monitoring = BashOperator(
        task_id='model_monitoring',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_monitoring.py '
            '--snapshotdate "{{ ti.xcom_pull(task_ids="get_snapshot_date") }}"'
        ),
    )

    machine_learning_pipeline_completed = DummyOperator(task_id="machine_learning_pipeline_completed")
        
    # =============================================================================
    # TASK DEPENDENCIES
    # =============================================================================

    get_date_task >> data_processing_start >> data_processing >> data_processing_completed >> machine_learning_pipeline_start >> model_training_start >> model_training >> model_inference_start >> model_inference >> model_monitoring_start >> model_monitoring >> machine_learning_pipeline_completed