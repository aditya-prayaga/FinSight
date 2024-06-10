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
    python_callable=download_and_uploadToDVCBucket,
    op_kwargs={
        'ticker_symbol': 'NFLX',
        'start_date': '2002-01-01',
        'end_date': '2022-12-31'
    },
    dag=dag,
    provide_context=True,
)

visualize_raw_data_task = PythonOperator(
    task_id='visualize_data',
    python_callable=functools.partial(
        visualize_raw_data, 
    ),
    provide_context=True,
)

divide_train_eval_test_splits_task = PythonOperator(
    task_id='divide_train_eval_test_splits',
    python_callable=divide_train_eval_test_splits,
    op_args=[visualize_raw_data_task.output],
    # op_kwargs={'file_path': 'gs://data_finsight/NFLX_stock_data_2002-01-01_2022-12-31.csv'},
    provide_context=True,
    dag=dag,
)

def handle_missing_values_task(ti, split):
    df = ti.xcom_pull(task_ids='divide_train_eval_test_splits', key=split)
    handled_df = handle_missing_values(df)
    return handled_df

handle_missing_values_in_training_data_task = PythonOperator(
    task_id='handle_missing_values_in_training_data',
    python_callable=functools.partial(handle_missing_values_task, split='train'),
    provide_context=True,
    dag=dag,
)

handle_missing_values_in_evaluation_data_task = PythonOperator(
    task_id='handle_missing_values_in_evaluation_data',
    python_callable=functools.partial(handle_missing_values_task, split='eval'),
    provide_context=True,
    dag=dag,
)

handle_missing_values_in_test_data_task = PythonOperator(
    task_id='handle_missing_values_in_test_data',
    python_callable=functools.partial(handle_missing_values_task, split='test'),
    provide_context=True,
    dag=dag,
)

handle_outliers_in_training_data_task = PythonOperator(
    task_id='handle_outliers_in_training_data',
    python_callable=handle_outliers,
    provide_context=True,
    op_args=[handle_missing_values_in_training_data_task.output],
    dag=dag,
)
handle_missing_values_in_training_data_task.set_downstream(handle_outliers_in_training_data_task)

handle_outliers_in_evaluation_data_task = PythonOperator(
    task_id='handle_outliers_in_evaluation_data',
    python_callable=handle_outliers,
    provide_context=True,
    op_args=[handle_missing_values_in_evaluation_data_task.output],
    dag=dag,
)
handle_missing_values_in_evaluation_data_task.set_downstream(handle_outliers_in_evaluation_data_task)

handle_outliers_in_test_data_task = PythonOperator(
    task_id='handle_outliers_in_test_data',
    python_callable=handle_outliers,
    provide_context=True,
    op_args=[handle_missing_values_in_test_data_task.output],
    dag=dag,
)
handle_missing_values_in_test_data_task.set_downstream(handle_outliers_in_test_data_task)

##
generate_scheme_and_stats_training_task = PythonOperator(
    task_id='generate_scheme_and_stats_training',
    python_callable=generate_scheme_and_stats,
    provide_context=True,
    op_args=[handle_outliers_in_training_data_task.output],
    dag=dag,
)
handle_outliers_in_training_data_task.set_downstream(generate_scheme_and_stats_training_task)

generate_scheme_and_stats_eval_task = PythonOperator(
    task_id='generate_scheme_and_stats_eval',
    python_callable=generate_scheme_and_stats,
    provide_context=True,
    op_args=[handle_outliers_in_evaluation_data_task.output],
    dag=dag,
)
handle_outliers_in_evaluation_data_task.set_downstream(generate_scheme_and_stats_eval_task)

generate_scheme_and_stats_test_task = PythonOperator(
    task_id='generate_scheme_and_stats_test',
    python_callable=generate_scheme_and_stats,
    provide_context=True,
    op_args=[handle_outliers_in_test_data_task.output],
    dag=dag,
)
handle_outliers_in_test_data_task.set_downstream(generate_scheme_and_stats_test_task)

calculate_and_display_anomalies_training_task = PythonOperator(
    task_id='calculate_and_display_anomalies_training',
    python_callable=calculate_and_display_anomalies,
    provide_context=True,
    op_args=[generate_scheme_and_stats_training_task.output],
    dag=dag,
)
generate_scheme_and_stats_training_task.set_downstream(calculate_and_display_anomalies_training_task)

calculate_and_display_anomalies_eval_task = PythonOperator(
    task_id='calculate_and_display_anomalies_eval',
    python_callable=calculate_and_display_anomalies,
    provide_context=True,
    op_args=[generate_scheme_and_stats_eval_task.output],
    dag=dag,
)
generate_scheme_and_stats_eval_task.set_downstream(calculate_and_display_anomalies_eval_task)

calculate_and_display_anomalies_test_task = PythonOperator(
    task_id='calculate_and_display_anomalies_test',
    python_callable=calculate_and_display_anomalies,
    op_args=[generate_scheme_and_stats_test_task.output],
    provide_context=True,
    dag=dag,
)
generate_scheme_and_stats_test_task.set_downstream(calculate_and_display_anomalies_test_task)

##

apply_transformation_training_task = PythonOperator(
    task_id='apply_transformation_training',
    python_callable=apply_transformation,
    provide_context=True,
    op_args=[calculate_and_display_anomalies_training_task.output],
    dag=dag,
)
calculate_and_display_anomalies_training_task.set_downstream(apply_transformation_training_task)

apply_transformation_eval_task = PythonOperator(
    task_id='apply_transformation_eval',
    python_callable=apply_transformation,
    provide_context=True,
    op_args=[calculate_and_display_anomalies_eval_task.output],
    dag=dag,
)
calculate_and_display_anomalies_eval_task.set_downstream(apply_transformation_eval_task)

apply_transformation_test_task = PythonOperator(
    task_id='apply_transformation_test',
    python_callable=apply_transformation,
    provide_context=True,
    op_args=[calculate_and_display_anomalies_test_task.output],
    dag=dag,
)
calculate_and_display_anomalies_test_task.set_downstream(apply_transformation_test_task)



# 

visualize_training_refined_data_task = PythonOperator(
    task_id='visualize_training_refined_data',
    python_callable=visualize_df,
    provide_context=True,
    op_args=[apply_transformation_training_task.output],
    dag=dag,
)
apply_transformation_training_task.set_downstream(visualize_training_refined_data_task)

visualize_evaluation_refined_data_task = PythonOperator(
    task_id='visualize_evaluation_refined_data',
    python_callable=visualize_df,
    provide_context=True,
    op_args=[apply_transformation_eval_task.output],
    dag=dag,
)
apply_transformation_eval_task.set_downstream(visualize_evaluation_refined_data_task)

visualize_test_refined_data_task = PythonOperator(
    task_id='visualize_test_refined_data',
    python_callable=visualize_df,
    provide_context=True,
    op_args=[apply_transformation_test_task.output],
    dag=dag,
)
apply_transformation_test_task.set_downstream(visualize_test_refined_data_task)


# Set the task dependencies
download_and_uploadToDVCBucket_task >> visualize_raw_data_task >> divide_train_eval_test_splits_task >> [
    handle_missing_values_in_training_data_task, 
    handle_missing_values_in_evaluation_data_task, 
    handle_missing_values_in_test_data_task
]
