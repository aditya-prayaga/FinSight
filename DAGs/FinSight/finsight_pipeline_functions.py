import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score
import logging
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
from FinSight.LSTM_model import *
import optuna
from torch.utils.data import DataLoader, TensorDataset
from mlflow.tracking import MlflowClient
from functools import partial
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
mlflow.autolog()

def download_and_uploadToDVCBucket(ticker_symbol, start_date, end_date, ti):
    """
    Download stock data from Yahoo Finance and upload it to a Google Cloud Storage bucket.
    """
    try:
        logging.info(f"Downloading data for {ticker_symbol} from {start_date} to {end_date}.")
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        filename = f"{ticker_symbol}_stock_data_{start_date}_{end_date}.csv"
        # file_loc = gcs_location + filename 
        stock_data.to_csv("./"+filename)
        ti.xcom_push(key='stock_data', value=stock_data)
        logging.info(f"Data downloaded and saved as {filename}")
        # logging.info(f"Data uploaded and saved } as {filename}")
        # return stock_data
    except Exception as e:
        logging.error(f"Failed to download or upload data: {e}")
        raise

def visualize_raw_data(ti):
    """
    Read stock data from a CSV file and visualize it, saving the plot to a specified GCS location.
    """
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
        
        plt.savefig("./data-viz.png")
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
        logging.error(f"Failed to visualize or upload data: {e}")
        raise

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
    try:
        logging.info(f"Reading data from {file_path}")
        df = ti.xcom_pull(task_ids='download_upload_data', key='stock_data') #pd.read_csv(file_path)
        
        logging.info("Splitting data into train+eval and test sets.")
        train_eval_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        logging.info("Further splitting train+eval set into train and eval sets.")
        train_df, eval_df = train_test_split(train_eval_df, test_size=eval_size, random_state=random_state)

        logging.info("Pushing data splits to XCom.")
        ti.xcom_push(key='train', value=train_df)
        ti.xcom_push(key='eval', value=eval_df)
        ti.xcom_push(key='test', value=test_df)
        return train_df, eval_df, test_df
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise

def handle_missing_values(df):
    """
    Handles null values in the DataFrame:
    - Fills null values in 'Open' column with 'Close' value of the previous row.
    - Forward fills null values in all columns.

    Parameters:
    df: Input stock data.

    Returns:
    pd.DataFrame: DataFrame with null values handled.
    """
    try:
        logging.info("Handling missing values.")
        logging.info("Dataset before handling missing values:\n{}".format(df.head()))

        def handle_null_open(df):
            for i in range(1, len(df)):
                if pd.isnull(df.iloc[i]['Open']):
                    if not pd.isnull(df.iloc[i-1]['Close']):
                        df.at[i, 'Open'] = df.iloc[i-1]['Close']
            df['Open'].fillna(method='ffill', inplace=True)
            return df

        df = handle_null_open(df)
        df.fillna(method='ffill', inplace=True)

        logging.info("Dataset after handling missing values:\n{}".format(df.head()))

        return df
    except Exception as e:
        logging.error(f"Failed to handle missing values: {e}")
        raise

def handle_outliers(df):
    """
    Removes outliers from the specified columns in the DataFrame using the IQR method.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to check for outliers.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    try:
        logging.info("Handling outliers.")
        columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
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

def visualize_df(df):
    """
    Placeholder function for visualizing DataFrame.
    """
    try:
        logging.info("Visualizing DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Failed to visualize DataFrame: {e}")
        raise

def apply_transformation(df,ti):
    """
    Normalizes the columns using MinMaxScaler, checks the data, and saves the preprocessed data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with stock data.

    Returns:
    pd.DataFrame: DataFrame with scaled columns.
    """
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = MinMaxScaler(feature_range=(0,1))
        df[['Volume', 'Open', 'Close', 'High', 'Low', 'Adj Close']] = scaler.fit_transform(df[['Volume', 'Open', 'Close', 'High', 'Low', 'Adj Close']])
        ti.xcom_push(key='scalar', value=scaler)
        return df
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise

def apply_transformation_eval_test(df,ti):
    """
    Normalizes the columns using MinMaxScaler, checks the data, and saves the preprocessed data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with stock data.

    Returns:
    pd.DataFrame: DataFrame with scaled columns.
    """
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = ti.xcom_pull(task_ids='apply_transformation_training', key="scalar")
        df[['Volume', 'Open', 'Close', 'High', 'Low', 'Adj Close']] = scaler.transform(df[['Volume', 'Open', 'Close', 'High', 'Low', 'Adj Close']])
        ti.xcom_push(key='scalar', value=scaler)
        return df
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise

def generate_schema(df):
    """
    Generate schema from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input data frame.

    Returns:
    dict: Schema definition with column types.
    """
    schema = {}
    for column in df.columns:
        schema[column] = df[column].dtype
    return schema

def generate_statistics(df):
    """
    Generate descriptive statistics from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input data frame.

    Returns:
    dict: Dictionary with descriptive statistics.
    """
    # Generate descriptive statistics
    statistics = df.describe(include='all').transpose()

    # Convert the DataFrame to a dictionary
    statistics_dict = statistics.to_dict()

    return statistics_dict

def generate_scheme_and_stats(df,ti):
    """
    Placeholder function for generating and validating scheme.
    """
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
        return df
    except Exception as e:
        logging.error(f"Failed to generate and validate scheme: {e}")
        raise

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
    try:
        logging.info("Calculating and Displaying Anomalies")

        # Log the values of training schema and stats for debugging purposes
        logging.info(f"Training Schema: {training_schema}")
        logging.info(f"Training Statistics: {training_stats}")

        # Detect anomalies
        anomalies = detect_anomalies(eval_df, training_schema, training_stats)
        logging.info(f"Anomalies: {anomalies}")

        return eval_df
    except Exception as e:
        logging.error(f"Failed to calculate and display anomalies: {e}")
        raise


# Training Phase

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create the function that will help us to create the datasets
def divide_features_and_labels(train_df, eval_df, test_df, ti):
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
        # x = np.array(x) 
        # y = np.array(y)
    
    ti.xcom_push(key='x', value=x)
    ti.xcom_push(key='y', value=y)
    # ti.xcom_push(key='x_train', value=x[0])
    # ti.xcom_push(key='x_eval', value=x[1])
    # ti.xcom_push(key='x_test', value=x[2])
    # ti.xcom_push(key='y_train', value=y[0])
    # ti.xcom_push(key='y_eval', value=y[1])
    # ti.xcom_push(key='y_test', value=y[2])


    # return x,y


def objective(trial, x , y):
    # Define hyperparameters to be optimized
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_int('batch_size', 32, 128)

    # Create the model
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x_train, y_train = np.array(x[0]), np.array(y[0])
    x_eval, y_eval = np.array(x[1]), np.array(y[1])
    # x_test, y_test = np.array(x[2]), np.array(y[2])
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    x_eval = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # y_eval = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Ensure x_train and y_train are numpy arrays
    x_train = x_train.values if isinstance(x_train, (pd.DataFrame, pd.Series)) else x_train
    y_train = y_train.values if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train

    # Create DataLoader
    train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Train the model
    model.train()
    for epoch in range(10):  # Set a small number of epochs for tuning
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    val_data = torch.from_numpy(x_eval).float().to(device)
    val_targets = torch.from_numpy(y_eval).float().to(device)
    with torch.no_grad():
        predictions = model(val_data)
        val_loss = criterion(predictions, val_targets).item()

    return val_loss

def hyper_parameter_tuning(x,y):

    objective1 = partial(objective, x = x, y = y)

    # Define the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective1, n_trials=1) #40

    # Get the best trial
    best_trial = study.best_trial
    best_params = best_trial.params

    return best_params


def training(best_params, x, y):
    # Train the final model with best parameters
    model = LSTMModel(input_size=1, hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], dropout=best_params['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    x_train, y_train = np.array(x[0]), np.array(y[0])
    # Reshape the feature for the LSTM layer 
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)

    os.environ["GIT_PYTHON_REFRESH"] = "quiet"

    experiment_name = "LSTM Stock Prediction"
    # current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    # experiment_id=current_experiment['experiment_id']

    mlflow.set_experiment(experiment_name)
    logging.info("MLflow trackng URI: %s", mlflow.get_tracking_uri())

    logging.info("Experiment name is %s","LSTM Stock Prediction")
    # mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    # logging.info("Experiment was set as: %s", EXPERIMENT_NAME)

    MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # with mlflow.start_run():
        # Track the experiment with MLflow
        # MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
        
    # mlflow.log_params(best_params)

    model.train()
    for epoch in range(1):  # Original number of epochs 40
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Save the model with MLflow
    # mlflow.pytorch.log_model(model, "model")
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", 'best_stock_prediction.pth')
    
    torch.save(model.state_dict(), output_path)
    return output_path

def load_and_predict(best_params, x, file_path,ti):
    # Load and predict
    model = LSTMModel(input_size=1, hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], dropout=best_params['dropout']).to(device)
    model.load_state_dict(torch.load(file_path))

    x_test = np.array(x[2])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
    # xcom_pull(task_ids='download_upload_data', key='stock_data')
    scaler = ti.xcom_pull(task_ids='apply_transformation_training', key='scalar')
    model.eval()
    test_data = torch.from_numpy(x_test).float().to(device)
    with torch.no_grad():
        predictions = model(test_data).cpu().numpy()
    
    # # Ensure the scaler expects the correct number of features
    # if len(predictions.shape) == 1:
    #     predictions = predictions.reshape(-1, 1)

    # # Check if predictions need to be reshaped further
    # if predictions.shape[1] != scaler.n_features_in_:
    #     try:
    predictions = np.tile(predictions, (1, scaler.n_features_in_))
    #     except ValueError as e:
    #         print(f"Error reshaping predictions: {e}")
    #         raise

    print(f"Shape of predictions after reshaping: {predictions.shape}")
    # Inverse transform the predictions if necessary    
    predictions = scaler.inverse_transform(predictions)
    # y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    return predictions
    # Log final results
    # mlflow.log_metric("final_val_loss", val_loss)
    # mlflow.end_run()

def evaluate_and_visualize(predictions, y, ti, save_path='actual_vs_predicted.png'):
    # Visualize the difference between actual vs predicted y-values
    plt.figure(figsize=(10, 6))
    
    # Reshape predictions if needed
    # predictions = predictions.reshape(-1)
    
    # Get actual y values and inverse transform if necessary
    y_test_actual = np.array(y[2])
    
    # # Ensure y_test_actual has the correct shape for scaler.inverse_transform
    # if y_test_actual.ndim == 1:
    #     y_test_actual = y_test_actual.reshape(-1, 1)
    # elif y_test_actual.ndim == 2 and y_test_actual.shape[1] != 1:
    #     raise ValueError("y_test_actual should be 1-dimensional or 2-dimensional with a single feature.")
    
    scaler = ti.xcom_pull(task_ids='apply_transformation_test', key='scalar')
    y_test_scaled = scaler.inverse_transform(y_test_actual.reshape(-1,1))
    
    
    plt.plot(y_test_scaled, label='Actual Values')
    plt.plot(predictions, label='Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_actual, predictions)
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, predictions)
    accuracy = accuracy_score(y_test_actual, predictions.round())
    precision = precision_score(y_test_actual, predictions.round(), average='weighted')


    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2 ): {r2}')
    print(f'Accuracy: {accuracy}')
    print(f'Precesion: {precision}')