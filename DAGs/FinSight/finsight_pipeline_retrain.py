import mlflow
import numpy as np
import time
import logging
from datetime import datetime
import os
import glob
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from FinSight.finsight_pipeline_functions import *
from keras.models import load_model

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Define the DAG
retrain_dag = DAG(
    dag_id='Retrain_FinSight_pipeline',
    default_args={
        'owner': 'Group 2',
        'start_date': datetime(2024, 6, 6),
        'email_on_failure': True,
    },
    schedule_interval=None,
    catchup=False
)

def get_retrain_dataset(file_pattern):
    logging.info("Starting Retraining")
    # Loop to continuously check for the file
    while not glob.glob(file_pattern):
        logging.info(f"File {file_pattern} not found, checking again in 10 seconds...")
        time.sleep(10)
    
    logging.info(f"Files matching pattern {file_pattern} found, proceeding with retraining...")
    data_frames = [pd.read_csv(file) for file in glob.glob(file_pattern)]
    combined_df = pd.concat(data_frames, ignore_index=True)
    
    logging.info("All files read and combined into a single DataFrame.")
    return combined_df

def retraining(model_file_path, x_train, y_train, best_params):
    model = load_model(model_file_path)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=best_params["batch_size"], verbose=1)
    output_path = os.path.join(Variable.get("model_path"), 'retrained_stock_prediction.h5')
    model.save(output_path)

# Define tasks
search_for_retraining_dataset_task = PythonOperator(
    task_id='search_for_retraining_dataset',
    python_callable=get_retrain_dataset,
    op_args=["./mlruns/retraining-data/*.csv"],
    dag=retrain_dag,
)

# Other tasks would be similarly updated for consistency

retraining_task = PythonOperator(
    task_id='retraining',
    python_callable=retraining,
    op_kwargs={
        "model_file_path": "./model/trained_stock_prediction.h5",
        'best_params': Variable.get("best_retrain_params", deserialize_json=True),
        'x_train': XComArg(divide_features_and_labels_task, key='x'),
        'y_train': XComArg(divide_features_and_labels_task, key='y')
    },
    dag=retrain_dag,
)

# Sequence of tasks
search_for_retraining_dataset_task >> \
handle_missing_values_in_retraining_data_task >> \
handle_outliers_in_retraining_data_task >> \
apply_transformation_retraining_task >> \
visualize_retraining_refined_data_task >> \
divide_features_and_labels_task >> \
retraining_task >> \
evaluating_models_task
