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

# Set environment variable to suppress GitPython warnings
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
)

def get_retrain_dataset(file_pattern):
    """
    Continuously checks for CSV files matching the pattern and reads them into a DataFrame.
    """
    try:
        logging.info("Starting Retraining")
        
        # Loop to continuously check for the file
        while not glob.glob(file_pattern):
            logging.info(f"File {file_pattern} not found, checking again in 10 seconds...")
            time.sleep(10)
        
        logging.info(f"Files matching pattern {file_pattern} found, proceeding with retraining...")
        
        # Get a list of all files matching the pattern
        file_list = glob.glob(file_pattern)
        
        # Read each file into a DataFrame and concatenate them
        data_frames = [pd.read_csv(file) for file in file_list]
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        df = combined_df['Open'].values.reshape(-1, 1)
        df = pd.DataFrame(df)
        
        logging.info("All files read and combined into a single DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Failed to Retrain: {e}")
        raise

def divide_features_and_labels(retrain_dataset, ti):
    """
    Divides the dataset into features (X) and labels (Y) for training.
    """
    try:
        x = []
        y = []
        
        for i in range(50, retrain_dataset.shape[0]):
            x.append(retrain_dataset.iloc[i-50:i, 0]) 
            y.append(retrain_dataset.iloc[i, 0]) 
            
        x = pd.DataFrame(x) 
        y = pd.DataFrame(y)

        # Push x and y to XCom for downstream tasks
        ti.xcom_push(key='x', value=x)
        ti.xcom_push(key='y', value=y)

        return x, y
    except Exception as e:
        logging.error(f"Error in Dividing Features and Labels: {e}")
        raise

def retraining(model_file_path, x_train, y_train, best_params):
    """
    Retrains the model with new data and saves the updated model.
    """
    try:
        # Load the existing model
        model = load_model(model_file_path)

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Retrain the model
        model.fit(x_train, y_train, epochs=1, batch_size=best_params["batch_size"], verbose=1)

        # Define output path for saving the retrained model
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", 'retrained_stock_prediction.h5')
        
        # Save the retrained model
        model.save(output_path)
        logging.info(f"Model retrained and saved at {output_path}")
    except Exception as e:
        logging.error(f"Error in Retraining Model: {e}")
        raise

def evaluate_model():
    """
    Placeholder function for model evaluation.
    """
    pass

# Define tasks for the DAG
search_for_retraining_dataset_task = PythonOperator(
    task_id='search_for_retraining_dataset',
    python_callable=get_retrain_dataset,
    op_args=["./mlruns/retraining-data/*.csv"],
    dag=retrain_dag,
)

handle_missing_values_in_retraining_data_task = PythonOperator(
    task_id='handle_missing_values_in_retraining_data',
    python_callable=handle_missing_values,
    op_args=[XComArg(search_for_retraining_dataset_task)],
    dag=retrain_dag,
)

handle_outliers_in_retraining_data_task = PythonOperator(
    task_id='handle_outliers_in_retraining_data',
    python_callable=handle_outliers,
    op_args=[XComArg(handle_missing_values_in_retraining_data_task)],
    dag=retrain_dag,
)

apply_transformation_retraining_task = PythonOperator(
    task_id='apply_transformation_retraining',
    python_callable=apply_transformation,
    op_args=[XComArg(search_for_retraining_dataset_task)],
    dag=retrain_dag,
)

visualize_retraining_refined_data_task = PythonOperator(
    task_id='visualize_retraining_refined_data',
    python_callable=visualize_df,
    op_args=[XComArg(apply_transformation_retraining_task), "./visualizations/retrain-processed-data.png"],
    dag=retrain_dag,
)

divide_features_and_labels_task = PythonOperator(
    task_id='divide_features_and_labels',
    python_callable=divide_features_and_labels,
    op_args=[XComArg(visualize_retraining_refined_data_task)],
    dag=retrain_dag,
)

retraining_task = PythonOperator(
    task_id='retraining',
    python_callable=retraining,
    op_kwargs={
        "model_file_path": "./model/trained_stock_prediction.h5",
        'best_params': {'units': 106, 'num_layers': 1, 'dropout_rate': 0.13736332505446322, 'learning_rate': 0.0008486320428172737, 'batch_size': 75},
        'x_train': XComArg(divide_features_and_labels_task, key='x'), 
        'y_train': XComArg(divide_features_and_labels_task, key='y')
    },
    dag=retrain_dag,
)

evaluating_models_task = PythonOperator(
    task_id='evaluating_models',
    python_callable=evaluate_model,
    dag=retrain_dag,
)

# Define task dependencies
search_for_retraining_dataset_task >> \
handle_missing_values_in_retraining_data_task >> \
handle_outliers_in_retraining_data_task >> \
apply_transformation_retraining_task >> \
visualize_retraining_refined_data_task >> \
divide_features_and_labels_task >> \
retraining_task >> \
evaluating_models_task
