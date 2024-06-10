import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def apply_transformation(df):
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
        df['Volume'] = scaler.fit_transform(df[['Volume']])
        df['Open'] = scaler.fit_transform(df[['Open']])
        df['Close'] = scaler.fit_transform(df[['Close']])
        df['High'] = scaler.fit_transform(df[['High']])
        df['Low'] = scaler.fit_transform(df[['Low']])
        df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
        return df
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise

def generate_scheme_and_stats(df,ti):
    """
    Placeholder function for generating and validating scheme.
    """
    try:
        logging.info("Generating scheme and stats.")
        
        # Scheme
        # schema = tfdv.infer_schema(df)
        # tfdv.display_schema(schema)

        # # Stats
        # data_stats = tfdv.generate_statistics_from_dataframe(df)
        # tfdv.visualize_statistics(data_stats)
        
        # logging.info("Pushing data splits to XCom.")
        # ti.xcom_push(key='schema', value=schema)
        # ti.xcom_push(key='stats', value=data_stats)
        return df
    except Exception as e:
        logging.error(f"Failed to generate and validate scheme: {e}")
        raise

def calculate_and_display_anomalies(df, ti):
    '''
    Calculate and display anomalies.

            Parameters:
                    statistics : Data statistics in statistics_pb2.DatasetFeatureStatisticsList format
                    schema : Data schema in schema_pb2.Schema format

            Returns:
                    display of calculated anomalies
    '''
    try:
        logging.info("Calculating and Displaying Anomalies")
        # schema = ti.xcom_pull(task_ids='generate_scheme_and_stats', key='schema')
        # statistics = ti.xcom_pull(task_ids='generate_scheme_and_stats', key='statistics')
        # anomalies = tfdv.validate_statistics(schema=schema, statistics=statistics)
        # tfdv.display_anomalies(anomalies=anomalies)
        return df
    except Exception as e:
        logging.error(f"Failed to generate and validate example generator: {e}")
        raise
