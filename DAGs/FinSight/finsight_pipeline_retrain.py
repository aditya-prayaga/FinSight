import mlflow
import numpy as np
import time
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import os
import glob
from airflow.models.xcom_arg import XComArg
from FinSight.finsight_pipeline_functions import *
from keras.models import load_model

# Suppress logging from the git module to avoid clutter
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Define the DAG with the necessary metadata and scheduling options
retrain_dag = DAG(
    dag_id='Retrain_FinSight_pipeline',
    default_args={
        'owner': 'Group 2',
        'start_date': datetime(2024, 6, 6),
        'email_on_failure': True,
    },
    schedule_interval=None,  # No scheduling, trigger manually or externally
)

def get_retrain_dataset(file_pattern):
    try:
        logging.info("Starting Retraining")
        # Wait until files are available in the specified pattern
        while not glob.glob(file_pattern):
            logging.info(f"File {file_pattern} not found, checking again in 10 seconds...")
            time.sleep(10)
        
        logging.info(f"Files matching pattern {file_pattern} found, proceeding with retraining...")
        # Combine data from all found files into a single DataFrame
        data_frames = [pd.read_csv(file) for file in glob.glob(file_pattern)]
        combined_df = pd.concat(data_frames, ignore_index=True)
        logging.info("All files read and combined into a single DataFrame.")
        return combined_df
    except Exception as e:
        logging.error(f"Failed to Retrain: {e}")
        raise

def divide_features_and_labels(retrain_dataset, ti):
    try:
        x, y = [], []
        window_size = 50  # Window size for rolling lookback for features
        for i in range(window_size, retrain_dataset.shape[0]):
            x.append(retrain_dataset.iloc[i-window_size:i, 0]) 
            y.append(retrain_dataset.iloc[i, 0]) 
        x, y = pd.DataFrame(x), pd.DataFrame(y)
        ti.xcom_push(key='x', value=x)
        ti.xcom_push(key='y', value=y)
        return x, y
    except Exception as e:
        logging.error(f"Error in Dividing Features and Labels: {e}")
        raise

def retraining(model_file_path, x_train, y_train, best_params):
    try:
        model = load_model(model_file_path)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=best_params["batch_size"], verbose=1)
        output_path = os.path.join(os.path.dirname(model_file_path), 'retrained_stock_prediction.h5')
        model.save(output_path)
        return output_path
    except Exception as e:
        logging.error(f"Error during retraining: {e}")
        raise

def evaluate_model():
    # This function should be implemented to evaluate the retrained model
    pass

# Set up the DAG tasks using the PythonOperator
search_for_retraining_dataset_task = PythonOperator(
    task_id='search_for_retraining_dataset',
    python_callable=get_retrain_dataset,
    provide_context=True,
    op_args=["./mlruns/retraining-data/*.csv"],
    dag=retrain_dag,
)

handle_missing_values_in_retraining_data_task = PythonOperator(
    task_id='handle_missing_values_in_retraining_data',
    python_callable=handle_missing_values,
    op_args=[search_for_retraining_dataset_task.output],
    provide_context=True,
    dag=retrain_dag,
)

handle_outliers_in_retraining_data_task = PythonOperator(
    task_id='handle_outliers_in_retraining_data',
    python_callable=handle_outliers,
    provide_context=True,
    op_args=[handle_missing_values_in_retraining_data_task.output],
    dag=retrain_dag,
)

apply_transformation_retraining_task = PythonOperator(
    task_id='apply_transformation_retraining',
    python_callable=apply_transformation,
    provide_context=True,
    op_args=[search_for_retraining_dataset_task.output],
    dag=retrain_dag,
)

visualize_retraining_refined_data_task = PythonOperator(
    task_id='visualize_retraining_refined_data',
    python_callable=visualize_df,
    provide_context=True,
    op_args=[apply_transformation_retraining_task.output, "./visualizations/retrain-processed-data.png"],
    dag=retrain_dag,
)

divide_features_and_labels_task = PythonOperator(
    task_id='divide_features_and_labels',
    python_callable=divide_features_and_labels,
    provide_context=True,
    op_args=[visualize_retraining_refined_data_task.output],
    dag=retrain_dag,
)

retraining_task = PythonOperator(
    task_id='retraining',
    python_callable=retraining,
    provide_context=True,
    op_kwargs={
        "model_file_path": "./model/trained_stock_prediction.h5",
        'best_params': {
            'batch_size': 75,
            # Additional params can be added here
        },
        'x_train': divide_features_and_labels_task.output['x'], 
        'y_train': divide_features_and_labels_task.output['y']
    },
    dag=retrain_dag,
)

evaluating_models_task = PythonOperator(
    task_id='evaluating_models',
    python_callable=evaluate_model,
    provide_context=True,
    dag=retrain_dag,
)

# Set up the dependencies between tasks
search_for_retraining_dataset_task >> \
handle_missing_values_in_retraining_data_task >> \
handle_outliers_in_retraining_data_task >> \
apply_transformation_retraining_task >> \
visualize_retraining_refined_data_task >> \
divide_features_and_labels_task >> \
retraining_task >> \
evaluating_models_task
