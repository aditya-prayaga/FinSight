from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os
import logging

# Path handling
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from FinSight.finsight_pipeline_functions import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# DAG definition
dag = DAG(
    'FinSight_pipeline',
    default_args={
        'owner': 'Group 2',
        'start_date': datetime(2024, 6, 6),
        'email': ['prayaga.a@northeastern.edu'],
        'email_on_failure': True,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='DAG for financial insights pipeline',
    schedule_interval=None,
    catchup=False,
)

# Tasks
download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_and_uploadToDVCBucket,
    op_kwargs={'ticker_symbol': 'NFLX', 'start_date': '2002-01-01', 'end_date': '2022-12-31'},
    dag=dag,
)

visualize_task = PythonOperator(
    task_id='visualize_data',
    python_callable=visualize_raw_data,
    op_args=[download_task.output],
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    op_args=[visualize_task.output],
    dag=dag,
)

# Dynamic task generation for handling different data splits
for split in ['train', 'eval', 'test']:
    cleanup_task = PythonOperator(
        task_id=f'cleanup_{split}',
        python_callable=cleanup_data,
        op_kwargs={'data_split': split},
        dag=dag,
    )

    transform_task >> cleanup_task

# More tasks can be added here following the pattern above

# Set the task dependencies
download_task >> visualize_task >> transform_task
