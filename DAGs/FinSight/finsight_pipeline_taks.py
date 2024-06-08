from airflow.operators.python_operator import PythonOperator
from airflow import DAG
from datetime import datetime
from airflow import configuration as conf
import sys, os
import functools
 
# Add the directory containing finsight_pipeline_functions.py to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
 
from FinSight.finsight_pipeline_functions import *
 
# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')
 
# Define the DAG
dag = DAG(
    dag_id='FinSight_pipeline',
    default_args={
        'owner': 'Group 2',
        'start_date': datetime(2024, 6, 6),
    },
    schedule_interval=None,
)
 
# Define the tasks
download_and_uploadToDVCBucket_task = PythonOperator(
    task_id='download_upload_data',
    python_callable=functools.partial(download_and_uploadToDVCBucket, ticker_symbol='NFLX', start_date='2002-01-01', end_date='2022-12-31', gcs_location="gs://data_finsight/"),
    dag=dag,
)
 
visualize_raw_data_task = PythonOperator(
    task_id='visualize_data',
    python_callable=functools.partial(
        visualize_raw_data,
        file_path='gs://data_finsight/NFLX_stock_data_2002-01-01_2022-12-31.csv'
    ),
    dag=dag,
)