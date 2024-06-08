import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import storage
import os
import tempfile
import gcsfs

 
def download_and_uploadToDVCBucket(ticker_symbol, start_date, end_date, gcs_location):
    """
    Download stock data from Yahoo Finance and upload it to a Google Cloud Storage bucket.
    """
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    filename = f"{ticker_symbol}_stock_data_{start_date}_{end_date}.csv"
    file_loc = gcs_location + filename
    stock_data.to_csv(file_loc)
    print(f"Data downloaded and saved as {filename}")
    print(f"Data uploaded and saved to {gcs_location} as {filename}")
    return stock_data

def visualize_raw_data(file_path):
    """
    Read stock data from a CSV file and visualize it, saving the plot to a specified GCS location.
    """
    # Read data from CSV file on GCS
    fs = gcsfs.GCSFileSystem()
    with fs.open(file_path) as f:
        df = pd.read_csv(f)
    
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set 'Date' as the index for better plotting
    df.set_index('Date', inplace=True)
    
    # Plot settings
    plt.figure(figsize=(14, 7))
    plt.suptitle('Stock Data Visualizations', fontsize=16)
 
    # Plot Open Prices
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Open'], label='Open Price', color='blue')
    plt.title('Open Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
 
    # Plot Close Prices
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['Close'], label='Close Price', color='green')
    plt.title('Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
 
    # Plot Volume
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['Volume'], label='Volume', color='red')
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
 
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
        plt.savefig(tmp_file_name)
        plt.close()
    
    # Upload the plot to GCS
    storage_client = storage.Client()
    bucket_name = "data_finsight"
    destination_blob_name = 'data-viz/data-viz.png'
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(tmp_file_name)
    
    # Clean up the local temporary file
    os.remove(tmp_file_name)
 
    print(f"Plot saved to 'gs://{bucket_name}/{destination_blob_name}'")