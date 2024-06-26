import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import logging
import mlflow
import numpy as np
import optuna
import time
from functools import partial
import os
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from FinSight.model import Model

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
        artifact_location=os.path.abspath(os.path.join(os.getcwd(), "mlruns", "artifacts")),
        tags={"version": "v2", "priority": "P1"},
    )
else:
    experiment_id = experiment.experiment_id

# Set the experiment
mlflow.set_experiment(experiment_name)
mlflow.autolog()
mlflow.enable_system_metrics_logging()

def download_and_uploadToDVCBucket(ticker_symbol, start_date, end_date, ti=None):
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
        filename = os.path.abspath(os.path.join(os.getcwd())) + "/" + f"{ticker_symbol}_stock_data_{start_date}_{end_date}.csv"
        
        # Save stock data to CSV
        stock_data.to_csv(filename)
        logging.info(f"Data downloaded and saved as {filename}")
        
        # Log the CSV file as an artifact
        mlflow.log_artifact(filename)
        
        return stock_data
    except Exception as e:
        logging.error(f"Failed to download or upload data: {e}")
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()

def visualize_raw_data(stock_data, file_path):
    """
    Read stock data from a CSV file and visualize it, saving the plot to a specified GCS location.
    """
    mlflow.start_run(run_name="Visualize Data")
    time.sleep(15)
    try:
        df = pd.DataFrame(stock_data)

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
        plt.savefig(file_path)
    except Exception as e:
        logging.error(f"Failed to visualize Raw data: {e}")
        raise
    finally:
        mlflow.end_run()
        return df

def divide_train_eval_test_splits(df, ti=None):
    """
    Divide the data into training, evaluation, and testing sets.
    """
    mlflow.start_run(run_name="Divide data set")
    time.sleep(15)
    try {
        df = df['Open'].values
        df = df.reshape(-1, 1)
        
        train_df = np.array(df[:int(df.shape[0]*0.7)])
        eval_df = np.array(df[int(df.shape[0]*0.7):int(df.shape[0]*0.8)])
        test_df = np.array(df[int(df.shape[0]*0.8):])
        
        train_dataset = mlflow.data.from_numpy(train_df)
        eval_dataset = mlflow.data.from_numpy(eval_df)
        test_dataset = mlflow.data.from_numpy(test_df)
        
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(eval_dataset, context="eval")
        mlflow.log_input(test_dataset, context="test")
        
        train_df = pd.DataFrame(train_df)
        eval_df = pd.DataFrame(eval_df)
        test_df = pd.DataFrame(test_df)
        
        if ti:
            ti.xcom_push(key='train', value=train_df)
            ti.xcom_push(key='eval', value=eval_df)
            ti.xcom_push(key='test', value=test_df)
        
        return train_df, eval_df, test_df
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise
    finally:
        mlflow.end_run()
}

def handle_missing_values(df):
    """
    Handles null values in the DataFrame by forward filling them.
    """
    mlflow.start_run(run_name="Handle Missing Values - PreProcessing Step 1")
    try:
        logging.info("Handling missing values.")
        df.fillna(method='ffill', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to handle missing values: {e}")
        raise
    finally:
        mlflow.end_run()

def handle_outliers(df):
    """
    Removes outliers from the specified columns in the DataFrame using the IQR method.
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

def visualize_df(df, file_path):
    """
    Visualize the preprocessed DataFrame, saving the plot to a specified location.
    """
    mlflow.start_run(run_name="Visualize Preprocessed Data")
    try:
        logging.info("Visualizing DataFrame.")
        
        df = pd.DataFrame(df)

        plt.figure(figsize=(14, 10))
        plt.suptitle('Preprocessed Data Visualizations', fontsize=16)

        plt.subplot(3, 1, 1)
        plt.plot(df.index, df.iloc[:, 0], label='Preprocessed Data', color='blue')
        plt.title('Preprocessed Data Over Time')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.hist(df.iloc[:, 0], bins=30, color='green', edgecolor='black')
        plt.title('Distribution of Preprocessed Data')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.subplot(3, 1, 3)
        plt.boxplot(df.iloc[:, 0], vert=False, patch_artist=True, boxprops=dict(facecolor='red'))
        plt.title('Box Plot of Preprocessed Data')
        plt.xlabel('Value')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.savefig(file_path)
        plt.show()
    except Exception as e:
        logging.error(f"Failed to visualize DataFrame: {e}")
        raise
    finally:
        mlflow.end_run()
        return df

def apply_transformation(df, ti=None):
    """
    Normalizes the columns using MinMaxScaler and saves the scaler.
    """
    mlflow.start_run(run_name="Apply Transformations on Train Data Sets")
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(df)
        df = pd.DataFrame(df)
        if ti:
            ti.xcom_push(key='scaler', value=scaler)
        return df
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise
    finally:
        mlflow.end_run()

def apply_transformation_eval_test(df, ti=None):
    """
    Normalizes the columns using a pre-fitted MinMaxScaler.
    """
    mlflow.start_run(run_name="Apply Transformations on Eval/Test Data Sets")
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = ti.xcom_pull(key='scaler', task_ids='apply_transformation')
        df = scaler.transform(df)
        df = pd.DataFrame(df)
        return df
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise
    finally:
        mlflow.end_run()

def generate_schema(df, scaler, schema_file_path):
    """
    Generates and logs the schema of the dataset.
    """
    try:
        logging.info("Generating schema.")
        schema = {
            "columns": [
                {"name": column, "type": str(df[column].dtype), "min": float(df[column].min()), "max": float(df[column].max())}
                for column in df.columns
            ],
            "scaler_min": float(scaler.data_min_[0]),
            "scaler_max": float(scaler.data_max_[0])
        }
        
        with open(schema_file_path, "w") as schema_file:
            json.dump(schema, schema_file)
        mlflow.log_artifact(schema_file_path)
    except Exception as e:
        logging.error(f"Failed to generate schema: {e}")
        raise

def generate_statistics(df, statistics_file_path):
    """
    Generates and logs descriptive statistics of the dataset.
    """
    try:
        logging.info("Generating statistics.")
        statistics = df.describe().to_dict()
        with open(statistics_file_path, "w") as statistics_file:
            json.dump(statistics, statistics_file)
        mlflow.log_artifact(statistics_file_path)
    except Exception as e:
        logging.error(f"Failed to generate statistics: {e}")
        raise

def generate_scheme_and_stats(df, scaler):
    """
    Generate and log schema and statistics of the dataset.
    """
    mlflow.start_run(run_name="Generate Scheme and Stats")
    try:
        schema_file_path = "schema.json"
        statistics_file_path = "statistics.json"
        generate_schema(df, scaler, schema_file_path)
        generate_statistics(df, statistics_file_path)
    except Exception as e:
        logging.error(f"Failed to generate schema and statistics: {e}")
        raise
    finally:
        mlflow.end_run()

def detect_anomalies(eval_df, schema, statistics):
    """
    Detect anomalies in the evaluation dataset based on the schema and statistics.
    """
    try:
        anomalies = pd.DataFrame()
        for column in eval_df.columns:
            column_min = schema["columns"][column]["min"]
            column_max = schema["columns"][column]["max"]
            column_mean = statistics[column]["mean"]
            column_std = statistics[column]["std"]

            anomalies[column] = (eval_df[column] < column_min) | (eval_df[column] > column_max) | \
                                (abs(eval_df[column] - column_mean) > 3 * column_std)
        return anomalies
    except Exception as e:
        logging.error(f"Failed to detect anomalies: {e}")
        raise

def calculate_and_display_anomalies(eval_df, schema, statistics):
    """
    Calculate and display anomalies, and log them with MLflow.
    """
    mlflow.start_run(run_name="Calculate and Display Anomalies")
    try:
        anomalies = detect_anomalies(eval_df, schema, statistics)
        anomaly_count = anomalies.sum().sum()
        mlflow.log_metric("anomaly_count", anomaly_count)
    except Exception as e:
        logging.error(f"Failed to calculate and display anomalies: {e}")
        raise
    finally:
        mlflow.end_run()

def divide_features_and_labels(train_df, eval_df, test_df):
    """
    Divide the data into features and labels for training, evaluation, and testing.
    """
    mlflow.start_run(run_name="Divide Features and Labels")
    try:
        train_X = train_df[:-1]
        train_y = train_df.shift(-1).dropna()
        eval_X = eval_df[:-1]
        eval_y = eval_df.shift(-1).dropna()
        test_X = test_df[:-1]
        test_y = test_df.shift(-1).dropna()
        
        return train_X, train_y, eval_X, eval_y, test_X, test_y
    except Exception as e:
        logging.error(f"Failed to divide features and labels: {e}")
        raise
    finally:
        mlflow.end_run()

def objective(train_X, train_y, trial):
    """
    Define the objective function for hyperparameter tuning.
    """
    mlflow.start_run(run_name="Hyperparameter Tuning")
    try:
        units = trial.suggest_int("units", 50, 200)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        batch_size = trial.suggest_int("batch_size", 16, 64)
        epochs = trial.suggest_int("epochs", 10, 50)

        model = Model(units, dropout)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=[metrics.MeanAbsoluteError()])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            train_X, train_y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )

        val_loss = min(history.history['val_loss'])
        mlflow.log_metric("val_loss", val_loss)
        return val_loss
    except Exception as e:
        logging.error(f"Failed to complete objective function: {e}")
        raise
    finally:
        mlflow.end_run()

def hyper_parameter_tuning(train_X, train_y, n_trials=50):
    """
    Perform hyperparameter tuning using Optuna.
    """
    mlflow.start_run(run_name="Hyperparameter Tuning")
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(partial(objective, train_X, train_y), n_trials=n_trials)
        best_trial = study.best_trial

        logging.info(f"Best trial: {best_trial.params}")
        mlflow.log_params(best_trial.params)
        
        return best_trial.params
    except Exception as e:
        logging.error(f"Failed to complete hyperparameter tuning: {e}")
        raise
    finally:
        mlflow.end_run()

def training(train_X, train_y, params):
    """
    Train the LSTM model with the best hyperparameters.
    """
    mlflow.start_run(run_name="Training Model")
    try:
        model = Model(params["units"], params["dropout"])
        model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss=MeanSquaredError(), metrics=[metrics.MeanAbsoluteError()])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            train_X, train_y,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stopping]
        )

        model.save("trained_model.h5")
        mlflow.log_artifact("trained_model.h5")

        return model
    except Exception as e:
        logging.error(f"Failed to complete model training: {e}")
        raise
    finally:
        mlflow.end_run()

def load_and_predict(test_X):
    """
    Load the trained model and make predictions on the test dataset.
    """
    mlflow.start_run(run_name="Load Model and Predict")
    try:
        model = load_model("trained_model.h5")
        predictions = model.predict(test_X)
        return predictions
    except Exception as e:
        logging.error(f"Failed to complete prediction: {e}")
        raise
    finally:
        mlflow.end_run()
def evaluate_model(model, test_X, test_y, scaler):
    """
    Evaluate the trained model on the test dataset and log the evaluation metrics.
    """
    mlflow.start_run(run_name="Evaluate Model")
    try:
        predictions = model.predict(test_X)
        
        # Inverse transform the predictions and true values
        predictions = scaler.inverse_transform(predictions)
        test_y = scaler.inverse_transform(test_y)
        
        mse = MeanSquaredError()(test_y, predictions).numpy()
        mae = metrics.mean_absolute_error(test_y, predictions).numpy()
        
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("mean_absolute_error", mae)
        
        return mse, mae
    except Exception as e:
        logging.error(f"Failed to evaluate the model: {e}")
        raise
    finally:
        mlflow.end_run()

def visualize_predictions(test_y, predictions, file_path):
    """
    Visualize the true vs predicted values and save the plot to a specified location.
    """
    mlflow.start_run(run_name="Visualize Predictions")
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(test_y, label='True Values', color='blue')
        plt.plot(predictions, label='Predicted Values', color='red')
        plt.title('True vs Predicted Stock Prices')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        
        plt.savefig(file_path)
        plt.show()
        
        mlflow.log_artifact(file_path)
    except Exception as e:
        logging.error(f"Failed to visualize predictions: {e}")
        raise
    finally:
        mlflow.end_run()

# Main execution steps
def main():
    ticker_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    raw_data_path = f"{ticker_symbol}_raw_data.png"
    preprocessed_data_path = f"{ticker_symbol}_preprocessed_data.png"
    predictions_path = f"{ticker_symbol}_predictions.png"

    stock_data = download_and_uploadToDVCBucket(ticker_symbol, start_date, end_date)
    df = visualize_raw_data(stock_data, raw_data_path)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    visualize_df(df, preprocessed_data_path)

    train_df, eval_df, test_df = divide_train_eval_test_splits(df)
    train_df = apply_transformation(train_df)
    eval_df = apply_transformation_eval_test(eval_df)
    test_df = apply_transformation_eval_test(test_df)

    scaler = MinMaxScaler().fit(df)
    generate_scheme_and_stats(df, scaler)

    schema_file_path = "schema.json"
    statistics_file_path = "statistics.json"
    with open(schema_file_path, "r") as schema_file:
        schema = json.load(schema_file)
    with open(statistics_file_path, "r") as statistics_file:
        statistics = json.load(statistics_file)

    calculate_and_display_anomalies(eval_df, schema, statistics)

    train_X, train_y, eval_X, eval_y, test_X, test_y = divide_features_and_labels(train_df, eval_df, test_df)

    best_params = hyper_parameter_tuning(train_X, train_y)
    model = training(train_X, train_y, best_params)

    mse, mae = evaluate_model(model, test_X, test_y, scaler)
    logging.info(f"Test MSE: {mse}, Test MAE: {mae}")

    predictions = load_and_predict(test_X)
    visualize_predictions(test_y, predictions, predictions_path)

if __name__ == "__main__":
    main()
