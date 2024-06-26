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

def fetch_retrain_dataset(file_pattern):
    """
    Continuously checks for CSV files matching the pattern and reads them into a DataFrame.
    """
    try:
        logging.info("Starting Retraining Process")
        
        # Loop to continuously check for the file
        while not glob.glob(file_pattern):
            logging.info(f"No file found for pattern {file_pattern}, retrying in 10 seconds...")
            time.sleep(10)
        
        logging.info(f"Files matching pattern {file_pattern} found. Preparing to load data...")

        # Get a list of all files matching the pattern
        csv_files = glob.glob(file_pattern)
        
        # Read each file into a DataFrame and concatenate them
        data_frames = [pd.read_csv(file) for file in csv_files]
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        open_prices = combined_df['Open'].values.reshape(-1, 1)
        open_prices_df = pd.DataFrame(open_prices)
        
        logging.info("Successfully loaded and combined data into a single DataFrame.")
        return open_prices_df
    except Exception as e:
        logging.error(f"Failed to Fetch Retrain Dataset: {e}")
        raise

def split_features_labels(dataset, ti):
    """
    Splits the dataset into features (X) and labels (Y) for training.
    """
    try:
        features = []
        labels = []
        
        for i in range(50, dataset.shape[0]):
            features.append(dataset.iloc[i-50:i, 0]) 
            labels.append(dataset.iloc[i, 0]) 
            
        features_df = pd.DataFrame(features) 
        labels_df = pd.DataFrame(labels)

        # Push features and labels to XCom for downstream tasks
        ti.xcom_push(key='features', value=features_df)
        ti.xcom_push(key='labels', value=labels_df)

        return features_df, labels_df
    except Exception as e:
        logging.error(f"Error in Splitting Features and Labels: {e}")
        raise

def retrain_model(model_path, x_train, y_train, best_params):
    """
    Retrains the model with new data and saves the updated model.
    """
    try:
        # Load the existing model
        model = load_model(model_path)

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Retrain the model
        model.fit(x_train, y_train, epochs=1, batch_size=best_params["batch_size"], verbose=1)

        # Define output path for saving the retrained model
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", 'retrained_stock_prediction.h5')
        
        # Save the retrained model
        model.save(output_path)
        logging.info(f"Model successfully retrained and saved at {output_path}")
    except Exception as e:
        logging.error(f"Error in Retraining Model: {e}")
        raise

def evaluate_model():
    """
    Placeholder function for model evaluation.
    """
    pass

# Define tasks for the DAG
fetch_retrain_dataset_task = PythonOperator(
    task_id='fetch_retrain_dataset',
    python_callable=fetch_retrain_dataset,
    op_args=["./mlruns/retraining-data/*.csv"],
    dag=retrain_dag,
)

handle_missing_values_task = PythonOperator(
    task_id='handle_missing_values',
    python_callable=handle_missing_values,
    op_args=[XComArg(fetch_retrain_dataset_task)],
    dag=retrain_dag,
)

handle_outliers_task = PythonOperator(
    task_id='handle_outliers',
    python_callable=handle_outliers,
    op_args=[XComArg(handle_missing_values_task)],
    dag=retrain_dag,
)

apply_transformation_task = PythonOperator(
    task_id='apply_transformation',
    python_callable=apply_transformation,
    op_args=[XComArg(handle_outliers_task)],
    dag=retrain_dag,
)

visualize_data_task = PythonOperator(
    task_id='visualize_data',
    python_callable=visualize_df,
    op_args=[XComArg(apply_transformation_task), "./visualizations/retrain-processed-data.png"],
    dag=retrain_dag,
)

split_features_labels_task = PythonOperator(
    task_id='split_features_labels',
    python_callable=split_features_labels,
    op_args=[XComArg(visualize_data_task)],
    dag=retrain_dag,
)

retrain_model_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    op_kwargs={
        "model_path": "./model/trained_stock_prediction.h5",
        'best_params': {'units': 106, 'num_layers': 1, 'dropout_rate': 0.13736332505446322, 'learning_rate': 0.0008486320428172737, 'batch_size': 75},
        'x_train': XComArg(split_features_labels_task, key='features'), 
        'y_train': XComArg(split_features_labels_task, key='labels')
    },
    dag=retrain_dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=retrain_dag,
)

# Define task dependencies
fetch_retrain_dataset_task >> \
handle_missing_values_task >> \
handle_outliers_task >> \
apply_transformation_task >> \
visualize_data_task >> \
split_features_labels_task >> \
retrain_model_task >> \
evaluate_model_task
