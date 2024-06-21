import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import numpy as np
import optuna
import time
from functools import partial
import os
from flask import Flask, jsonify, request, redirect, render_template

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuring Mlflow settings
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
experiment_name = "LSTM Stock Prediction"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Check if the experiment exists, and create it if it doesn't
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=os.path.abspath(os.path.join(os.getcwd(), "mlruns","artifacts")),
        tags={"version": "v2", "priority": "P1"},
    )
else:
    experiment_id = experiment.experiment_id

# Set the experiment
mlflow.set_experiment(experiment_name)
mlflow.autolog()
mlflow.enable_system_metrics_logging()

def download_and_uploadToDVCBucket(ticker_symbol, start_date, end_date, ti):
    """
    Download stock data from Yahoo Finance and upload it to a Google Cloud Storage bucket.
    """
    mlflow.start_run(run_name="Download Data")
    time.sleep(15)
    try:
        logging.info(f"Downloading data for {ticker_symbol} from {start_date} to {end_date}.")
        mlflow.log_param("ticker_symbol", ticker_symbol)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        filename = os.path.abspath(os.path.join(os.getcwd(), "mlruns","artifacts")) + "/" + f"{ticker_symbol}_stock_data_{start_date}_{end_date}.csv"
        
        # Save stock data to CSV
        stock_data.to_csv(filename)
        logging.info(f"Data downloaded and saved as {filename}")
        
        # # Log the CSV file as an artifact
        # print("hello1:", filename)
        # import getpass
        # print("user: ",         getpass.getuser()) 
        # print("world:", os.path.abspath(os.path.join(os.getcwd(), "..", "mlruns","artifacts")))
        # # os.environ['MLFLOW_TRACKING_URI'] = "/mlflow/artifacts"
        # print(mlflow.get)
        # mlflow.log_artifact(filename)
        
        # Push the stock_data to XCom
        ti.xcom_push(key='stock_data', value=stock_data)

        # Log dataset information
        # mlflow.log_input(name=os.path.basename(filename), context="dataset", path=filename)

    except Exception as e:
        logging.error(f"Failed to download or upload data: {e}")
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()

def visualize_raw_data(ti):
    """
    Read stock data from a CSV file and visualize it, saving the plot to a specified GCS location.
    """
    mlflow.start_run(run_name="Visualize Data")
    time.sleep(15)
    try:
        # logging.info(f"Reading data from {file_path}")
        # Pull the DataFrame from XCom
        stock_data_dict = ti.xcom_pull(task_ids='download_upload_data', key='stock_data')
        df = pd.DataFrame(stock_data_dict)

        logging.info("Converting 'Date' column to datetime format and setting it as index.")
        df['Date'] = df.index
        df.set_index('Date', inplace=True)

        logging.info("Plotting data.")
        plt.figure(figsize=(14, 7))
        plt.suptitle('Stock Data Visualizations', fontsize=16)

        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['Open'], label='Open Price', color='blue')
        plt.title('Open Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['Close'], label='Close Price', color='green')
        plt.title('Close Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['Volume'], label='Volume', color='red')
        plt.title('Trading Volume Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.savefig("/opt/airflow/visualizations/data1-viz.png")
        # return df
        # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        #     tmp_file_name = tmp_file.name
        #     plt.savefig(tmp_file_name)
        #     plt.close()
        
        # logging.info(f"Uploading plot to GCS.")
        # storage_client = storage.Client()
        # bucket_name = "data_finsight"
        # destination_blob_name = 'data-viz/data-viz.png'
        
        # bucket = storage_client.bucket(bucket_name)
        # blob = bucket.blob(destination_blob_name)
        # blob.upload_from_filename(tmp_file_name)
        
        # os.remove(tmp_file_name)

        # logging.info(f"Plot saved to 'gs://{bucket_name}/{destination_blob_name}'")
    except Exception as e:
        logging.error(f"Failed to visualize Raw data: {e}")
        raise
    finally:
        mlflow.end_run()

def get_retrain_dataset(ticker_symbol, start_date, end_date, ti):

    mlflow.start_run(run_name="Retraining")
    time.sleep(15)
    try:
        logging.info("Starting Retraining")

        # download_and_uploadToDVCBucket(ticker_symbol, start_date, end_date, ti)

        # stock_data_dict = ti.xcom_pull(task_ids='download_upload_data', key='stock_data')
        # df = pd.DataFrame(stock_data_dict)

        # handle_missing_values(df)

        # handle_outliers(df)

        # apply_transformation(df,ti)

        # generate_scheme_and_stats(df,ti)

        # detect_anomalies(df, generate_schema(df), generate_statistics(df))

        # calculate_and_display_anomalies(df, scheme, stats)

        # divide_train_eval_test_splits(file_path, test_size=0.2, eval_size=0.1, random_state=42, ti=None)

        # divide_features_and_labels(train_df, eval_df, test_df, ti)

        # objective(trial, x, y)

        # hyper_parameter_tuning(x,y)

        # training(best_params, x, y)

        # load_and_predict(best_params, x, file_path,ti)

        # evaluate_and_visualize(predictions, y, ti, save_path='actual_vs_predicted.png')

    except Exception as e:
        logging.error(f"Failed to Retrain: {e}")
        raise
    finally:
        mlflow.end_run()

def divide_train_eval_test_splits(file_path, test_size=0.2, eval_size=0.1, random_state=42, ti=None):
    """
    - file_path (str): Path to the CSV file containing stock data.
    - test_size (float): Proportion of the dataset to include in the test split.
    - eval_size (float): Proportion of the train dataset to include in the eval split.
    - random_state (int): Random seed for reproducibility.
    
    Returns:
    - train_df (pd.DataFrame): DataFrame containing the training data.
    - eval_df (pd.DataFrame): DataFrame containing the evaluation data.
    - test_df (pd.DataFrame): DataFrame containing the testing data.
    """
    mlflow.start_run(run_name="Divide data set")
    time.sleep(15)
    # mlflow.log_params(test_size,eval_size,random_state)
    try:
        logging.info(f"Reading data from {file_path}")
        df = ti.xcom_pull(task_ids='download_upload_data', key='stock_data') #pd.read_csv(file_path)
        # dataset: PandasDataset = mlflow.data.from_pandas(df, source=file_path)

        df = df['Open'].values

        # Reshape the data
        df = df.reshape(-1, 1) 
        
        train_df = np.array(df[:int(df.shape[0]*0.7)])
        eval_df = np.array(df[int(df.shape[0]*0.7):int(df.shape[0]*0.8)])
        test_df = np.array(df[int(df.shape[0]*0.8):])

        logging.info("Splitting data into train+eval and test sets.")
        # train_eval_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        logging.info("Further splitting train+eval set into train and eval sets.")
        # train_df, eval_df = train_test_split(train_eval_df, test_size=eval_size, random_state=random_state)

        train_dataset = mlflow.data.from_numpy(train_df, source=file_path)
        eval_dataset = mlflow.data.from_numpy(eval_df, source=file_path)
        test_dataset = mlflow.data.from_numpy(test_df, source=file_path)

        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(eval_dataset, context="Eval") 
        mlflow.log_input(test_dataset, context="Test") 

        train_df = pd.DataFrame(train_df)
        eval_df = pd.DataFrame(eval_df)
        test_df = pd.DataFrame(test_df)


        

        logging.info("Pushing data splits to XCom.")
        ti.xcom_push(key='train', value=train_df)
        ti.xcom_push(key='eval', value=eval_df)
        ti.xcom_push(key='test', value=test_df)
        return pd.DataFrame(train_df), pd.DataFrame(eval_df), pd.DataFrame(test_df)
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise
    finally:
        mlflow.end_run()

def handle_missing_values(df):
    """
    Handles null values in the DataFrame:
    - Forward fills null values in all columns.

    Parameters:
    df: Input stock data.

    Returns:
    pd.DataFrame: DataFrame with null values handled.
    """
    mlflow.start_run(run_name="Handle Missing Values - PreProcessing Step 1")
    try:
        logging.info("Handling missing values.")
        logging.info("Dataset before handling missing values:\n{}".format(df))

        # df = handle_null_open(df)
        df.fillna(method='ffill', inplace=True)

        # logging.info("Dataset after handling missing values:\n{}".format(df.head()))

        return df
    except Exception as e:
        logging.error(f"Failed to handle missing values: {e}")
        raise
    finally:
        mlflow.end_run()

def handle_outliers(df):
    """
    Removes outliers from the specified columns in the DataFrame using the IQR method.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to check for outliers.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    mlflow.start_run(run_name="Handle Outlier Values - PreProcessing Step 2")
    try:
        logging.info("Handling outliers.")
        columns = df.columns.tolist()
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
            df = df[~outliers]
            logging.info(f"Removed outliers from column: {column}")

        return df
    except Exception as e:
        logging.error(f"Failed to handle outliers: {e}")
        raise
    finally:
        mlflow.end_run()


def visualize_df(df):
    """
    Placeholder function for visualizing DataFrame.
    """
    mlflow.start_run(run_name="Visualize Preprocessed Data")
    try:
        logging.info("Visualizing DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Failed to visualize DataFrame: {e}")
        raise
    finally:
        mlflow.end_run()


def apply_transformation(df,ti):
    """
    Normalizes the columns using MinMaxScaler, checks the data, and saves the preprocessed data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with stock data.

    Returns:
    pd.DataFrame: DataFrame with scaled columns.
    """
    mlflow.start_run(run_name="Apply Transformations on Train Data Sets")    
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = MinMaxScaler(feature_range=(0,1))
        df= scaler.fit_transform(df)
        df = pd.DataFrame(df)
        ti.xcom_push(key='scalar', value=scaler)
        return df
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise
    finally:
        mlflow.end_run()

def apply_transformation_eval_test(df,ti):
    """
    Normalizes the columns using MinMaxScaler, checks the data, and saves the preprocessed data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with stock data.

    Returns:
    pd.DataFrame: DataFrame with scaled columns.
    """
    mlflow.start_run(run_name="Apply Transformations on Eval and Test Data Sets")    
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = ti.xcom_pull(task_ids='apply_transformation_training', key="scalar")
        df = scaler.transform(df)        
        df = pd.DataFrame(df)
        ti.xcom_push(key='scalar', value=scaler)
        return df
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise
    finally:
        mlflow.end_run()

def generate_schema(df):
    """
    Generate schema from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input data frame.

    Returns:
    dict: Schema definition with column types.
    """
    mlflow.start_run(run_name="Generate Schema")   
    try: 
        schema = {}
        for column in df.columns:
            schema[column] = df[column].dtype
        mlflow.log_param("Scheme", schema)
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise    
    finally:
        mlflow.end_run()
        return schema

def generate_statistics(df):
    """
    Generate descriptive statistics from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input data frame.

    Returns:
    dict: Dictionary with descriptive statistics.
    """
    mlflow.start_run(run_name="Generate Schema")   
    try: 
        # Generate descriptive statistics
        statistics = df.describe(include='all').transpose()

        # Convert the DataFrame to a dictionary
        statistics_dict = statistics.to_dict()
        mlflow.log_param("Stats", statistics_dict)

    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise  
    finally:
        mlflow.end_run()
        return statistics_dict

def generate_scheme_and_stats(df,ti):
    """
    Placeholder function for generating and validating scheme.
    """
    mlflow.start_run(run_name="Generate Schema & Statistics")   
    try:
        logging.info("Generating scheme and stats.")
        
        # Scheme
        schema = generate_schema(df)
        logging.info(f"Schema: {schema}")

        # Stats
        data_stats = generate_statistics(df)
        logging.info(f"Statistics: \n{data_stats}")
        
        logging.info("Pushing data splits to XCom.")
        ti.xcom_push(key='schema', value=schema)
        ti.xcom_push(key='stats', value=data_stats)
        
    except Exception as e:
        logging.error(f"Failed to generate and validate scheme: {e}")
        raise
    finally:
        mlflow.end_run()
        return df

def detect_anomalies(eval_df, training_schema, training_stats):
    """
    Detect anomalies in the evaluation DataFrame by comparing it against the training schema and statistics.

    Parameters:
    eval_df (pd.DataFrame): Evaluation data frame.
    training_schema (dict): Schema of the training data.
    training_stats (dict): Statistics of the training data.

    Returns:
    dict: Detected anomalies including missing values and outliers.
    """
    mlflow.start_run(run_name="Detecting Anomalies")   
    try:
        anomalies = {'missing_values': {}, 'outliers': {}, 'schema_mismatches': {}, 'statistical_anomalies': {}}

        # Detect missing values in the evaluation data
        missing_values = eval_df.isnull().sum()
        anomalies['missing_values'] = {col: count for col, count in missing_values.items() if count > 0}

        # Detect outliers in the evaluation data
        numeric_cols = eval_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            Q1 = eval_df[col].quantile(0.25)
            Q3 = eval_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = eval_df[(eval_df[col] < lower_bound) | (eval_df[col] > upper_bound)][col]
            if not outliers.empty:
                anomalies['outliers'][col] = outliers.tolist()

        # Compare schema and detect schema mismatches
        for col in training_schema:
            if col not in eval_df.columns:
                anomalies['schema_mismatches'][col] = 'Column missing in evaluation data'
            elif eval_df[col].dtype != training_schema[col]:
                anomalies['schema_mismatches'][col] = f'Type mismatch: expected {training_schema[col]}, got {eval_df[col].dtype}'

        # Compare statistical properties
        for col in training_stats:
            if col in eval_df.columns:
                eval_mean = eval_df[col].mean()
                eval_std = eval_df[col].std()
                train_mean = training_stats[col]['mean']
                train_std = training_stats[col]['std']
                if abs(eval_mean - train_mean) > 3 * train_std:
                    anomalies['statistical_anomalies'][col] = {'eval_mean': eval_mean, 'train_mean': train_mean}
                if abs(eval_std - train_std) > 3 * train_std:
                    anomalies['statistical_anomalies'][col].update({'eval_std': eval_std, 'train_std': train_std}) 
    except Exception as e:
        logging.error(f"Failed to generate and validate scheme: {e}")
        raise
    finally:
        mlflow.end_run()
        return anomalies

def calculate_and_display_anomalies(eval_df, training_schema, training_stats):
    """
    Calculate and display anomalies in the evaluation DataFrame by comparing it against the training schema and statistics.

    Parameters:
    eval_df (pd.DataFrame): Evaluation data frame.
    ti (TaskInstance): Airflow TaskInstance for XCom operations.
    training_schema (dict): Schema of the training data.
    training_stats (dict): Statistics of the training data.

    Returns:
    pd.DataFrame: The original evaluation DataFrame after anomaly detection.
    """
    mlflow.start_run(run_name="Calculating anomalies from Eval Df and Training (Schema, Stats)")   
    try:
        logging.info("Calculating and Displaying Anomalies")

        # Log the values of training schema and stats for debugging purposes
        logging.info(f"Training Schema: {training_schema}")
        logging.info(f"Training Statistics: {training_stats}")

        # Detect anomalies
        anomalies = detect_anomalies(eval_df, training_schema, training_stats)
        logging.info(f"Anomalies: {anomalies}")

        
    except Exception as e:
        logging.error(f"Failed to calculate and display anomalies: {e}")
        raise
    finally:
        mlflow.end_run()
        return eval_df
    

# Training Phase

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create the function that will help us to create the datasets
def divide_features_and_labels(train_df, eval_df, test_df, ti):
    """
    Divide the data into features and labels.

    Parameters:
    - train_df (pd.DataFrame): DataFrame containing the training data.
    - eval_df (pd.DataFrame): DataFrame containing the evaluation data.
    - test_df (pd.DataFrame): DataFrame containing the testing data.
    - ti (TaskInstance): Airflow TaskInstance for XCom operations.
    """
    mlflow.start_run(run_name="Divide Data set into features and labels")   
    try:
        dfs = [train_df, eval_df, test_df]
        x_train = []
        x_eval = []
        x_test = []
        y_train = []
        y_eval = []
        y_test = []
        x = [x_train, x_eval, x_test]
        y = [y_train, y_eval, y_test]
        for ind, df in enumerate(dfs):
            for i in range(50, df.shape[0]):
                x[ind].append(df.iloc[i-50:i, 0].values) 
                y[ind].append(df.iloc[i, 0]) 
        
        ti.xcom_push(key='x', value=x)
        ti.xcom_push(key='y', value=y)

        mlflow.log_params({"x": x,"y": y})

    except Exception as e:
        logging.error(f"Error in Dividing Features and Labels: {e}")
        raise
    finally:
        mlflow.end_run()

def objective(trial, x , y):
    """
    Objective Function for Hyperparameter Tuning

    Parameters:
    - trail:
    - x: Features to train on 
    - y: Labels to evaluate against

    Returns:
    Loss Val: The loss value of the trail.
    """
    mlflow.start_run(run_name="Objective Function to run experiments on, used by optuna", nested=True)   
    try:
        # Define hyperparameters to be optimized
        units = trial.suggest_int('units', 32, 128)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
        batch_size = trial.suggest_int('batch_size', 32, 128)

        x_train, y_train = np.array(x[0]), np.array(y[0])
        x_eval, y_eval = np.array(x[1]), np.array(y[1])

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
        x_eval = np.reshape(x_eval, (x_eval.shape[0], x_eval.shape[1], 1))
        # y_eval = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Create the model
        model = Sequential()
        for _ in range(num_layers):
            model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(rate=dropout_rate))
        model.add(LSTM(units=units))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(x_train, y_train, validation_data=(x_eval, y_eval), epochs=10, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

        # Evaluate the model
        val_loss = model.evaluate(x_eval, y_eval, verbose=0)

    except Exception as e:
        logging.error(f"Error in Objective Function: {e}")
        raise
    finally:
        mlflow.end_run()
        return val_loss
    

def hyper_parameter_tuning(x,y):
    """
    Objective Function for Hyperparameter Tuning

    Parameters:
    - x: Features to train on 
    - y: Labels to evaluate against

    Returns:
    Loss Val: The loss value of the trail.
    """
    mlflow.start_run(run_name="Hyper-parameter Tuning", nested=True)   
    try:
        # Partial function to pass x and y to the objective
        objective_fn = partial(objective, x=x, y=y)

        # Define the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_fn, n_trials=1)

        # Get the best trial
        best_trial = study.best_trial
        best_params = best_trial.params

    except Exception as e:
        logging.error(f"Error in Hyper Parameter Tuning: {e}")
        raise
    finally:
        mlflow.end_run()
        return best_params   
    # return {'units': 106, 'num_layers': 1, 'dropout_rate': 0.13736332505446322, 'learning_rate': 0.0008486320428172737, 'batch_size': 75}
    # return {'units': 96, 'num_layers': 1, 'dropout_rate': 0.2, 'batch_size': 64}


def training(best_params, x, y):
    """
   Train the model with the best hyperparameters

    Parameters:
    - best_params: Best parameters from Hyperparameter Tuning
    - x: Features to train on
    - y: Labels to evaluate against

    Returns:
    output_path: Saved model output path.
    """
    mlflow.start_run(run_name="training")  
    try:
        x_train, y_train = np.array(x[0]), np.array(y[0])
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
        # Create the model
        model = Sequential()
        # for _ in range(best_params['num_layers']):
        model.add(LSTM(units=best_params['units'], return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(rate=best_params['dropout_rate']))
        model.add(LSTM(units=best_params['units'], return_sequences=True))
        model.add(Dropout(rate=best_params['dropout_rate']))
        model.add(LSTM(units=best_params['units'], return_sequences=True))
        model.add(Dropout(rate=best_params['dropout_rate']))
        model.add(LSTM(units=best_params['units']))
        model.add(Dropout(rate=best_params['dropout_rate']))
        model.add(Dense(units=1))

        model.summary()

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Log parameters with MLflow
        mlflow.log_params(best_params)

        # Train the model
        # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=1, batch_size=best_params["batch_size"], verbose=1)

        # Save the model with MLflow
        # mlflow.keras.log_model(model, "model")

        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", 'best_stock_prediction.h5')
        model.save(output_path)

    except Exception as e:
        logging.error(f"Error in Training: {e}")
        raise
    finally:
        mlflow.end_run()
        return output_path

def load_and_predict(x, file_path,ti):
    """
   Load and Predict from trained model

    Parameters:
    - x: Features to train on
    - file_path: Trained Model File

    Returns:
    predictions: Predictions on the test feature set.
    """
    mlflow.start_run(run_name="Load and Prediction")  
    try:

        # Load and predict
        model = load_model(file_path)

        x_test = np.array(x[2])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
        scaler = ti.xcom_pull(task_ids='apply_transformation_training', key='scalar')

        predictions =  model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

    except Exception as e:
        logging.error(f"Error in Load and Prediction: {e}")
        raise
    finally:
        mlflow.end_run()
        return predictions

def evaluate_and_visualize(predictions, y, ti, save_path='/opt/airflow/visualizations/act.png'):
    """
   Evaluate and visualize the predictions 

    Parameters:
    - predictions: Predictions on the test feature set.
    - y: Label set

    Returns:
    predictions: Predictions on the test feature set.
    """
    mlflow.start_run(run_name="Evaluate and Visualize")  
    try:
        # Visualize the difference between actual vs predicted y-values
        plt.figure(figsize=(10, 6))
        y_test_actual = np.array(y[2])
        
        scaler = ti.xcom_pull(task_ids='apply_transformation_training', key='scalar')
        y_test_scaled = scaler.inverse_transform(y_test_actual.reshape(-1, 1))
        
        plt.plot(y_test_scaled, label='Actual Values')
        plt.plot(predictions, label='Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(save_path)
        plt.show()

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_actual, predictions)
        mse = mean_squared_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions)


        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R-squared (R2 ): {r2}')
    except Exception as e:
        logging.error(f"Error in Evaluate and Visualize: {e}")
        raise
    finally:
        mlflow.end_run()
 
# Lets create a Flask API to show sucess or failure of the main dag
app = Flask(__name__)

# Function to start Flask app
def start_flask_app():
    app.run(host='0.0.0.0', port=5001)

# Flask routes
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data as JSON
    Open = float(data['Open'])
    Close = float(data['Close'])
    # petal_length = float(data['petal_length'])
    # petal_width = float(data['petal_width'])

    print(Open, Close)