import pytest
from pytest_mock import mocker
from unittest import mock
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
import logging

# Assuming these are defined in FinSight/finsight_pipeline_functions.py
from .finsight_pipeline_functions import (
    download_and_uploadToDVCBucket,
    visualize_raw_data,
    divide_train_eval_test_splits,
    handle_missing_values,
    handle_outliers,
    visualize_df,
    apply_transformation,
    generate_scheme_and_stats,
    calculate_and_display_anomalies
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock Airflow task instance
class MockTaskInstance:
    def __init__(self):
        self.xcom_pull_results = {}
        self.xcom_push_results = {}

    def xcom_pull(self, task_ids, key):
        return self.xcom_pull_results.get((task_ids, key), None)

    def xcom_push(self, key, value):
        self.xcom_push_results[key] = value

# Test for download_and_uploadToDVCBucket
def test_download_and_uploadToDVCBucket_positive(mocker):
    ticker = "NFLX"
    start = "2002-01-01"
    end = "2022-12-31"
    mock_task_instance = MockTaskInstance()

    # Mock yfinance download
    mock_stock_data = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })
    mocker.patch('yfinance.download', return_value=mock_stock_data)

    download_and_uploadToDVCBucket(ticker_symbol=ticker, start_date=start, end_date=end, ti=mock_task_instance)
   
    assert 'stock_data' in mock_task_instance.xcom_push_results
    pd.testing.assert_frame_equal(mock_task_instance.xcom_push_results['stock_data'], mock_stock_data)

def test_download_and_uploadToDVCBucket_negative(mocker):
    ticker = "INVALID_TICKER"
    start = "2002-01-01"
    end = "2022-12-31"
    mock_task_instance = MockTaskInstance()

    # Mock yfinance download to raise an exception
    mocker.patch('yfinance.download', side_effect=Exception("Invalid ticker"))

    with pytest.raises(Exception, match="Invalid ticker"):
        download_and_uploadToDVCBucket(ticker_symbol=ticker, start_date=start, end_date=end, ti=mock_task_instance)

# # Test for visualize_raw_data
# def test_visualize_raw_data_positive(mocker):
#     mock_task_instance = MockTaskInstance()
#     mock_stock_data = pd.DataFrame({
#         'Open': [100, 200],
#         'Close': [110, 210],
#         'Volume': [1000, 1500],
#     })
#     mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = mock_stock_data

#     mocker.patch('matplotlib.pyplot.savefig')

#     visualize_raw_data(ti=mock_task_instance)

#     # Check if the plot was saved
#     assert mocker.patch('matplotlib.pyplot.savefig').call_count == 0

# def test_visualize_raw_data_negative(mocker):
#     mock_task_instance = MockTaskInstance()
#     mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = None

#     with pytest.raises(Exception, match="Failed to visualize or upload data"):
#         visualize_raw_data(ti=mock_task_instance)

# Test for divide_train_eval_test_splits
def test_divide_train_eval_test_splits_positive(mocker):
    mock_task_instance = MockTaskInstance()
    mock_stock_data = pd.DataFrame({
        'Open': [100, 200, 300],
        'Close': [110, 210, 310],
        'Volume': [1000, 1500, 2000],
    })
    mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = mock_stock_data

    divide_train_eval_test_splits(file_path="dummy_path", ti=mock_task_instance)

    assert 'train' in mock_task_instance.xcom_push_results
    assert 'eval' in mock_task_instance.xcom_push_results
    assert 'test' in mock_task_instance.xcom_push_results

def test_divide_train_eval_test_splits_negative(mocker):
    mock_task_instance = MockTaskInstance()
    mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = None

    with pytest.raises(Exception, match="Failed to split data"):
        divide_train_eval_test_splits(file_path="dummy_path", ti=mock_task_instance)

# Test for handle_missing_values
def test_handle_missing_values_positive():
    mock_df = pd.DataFrame({
        'Open': [100, None, 300],
        'Close': [110, 210, 310],
        'Volume': [1000, 1500, 2000],
    })

    result_df = handle_missing_values(mock_df.copy())

    expected_df = pd.DataFrame({
        'Open': [100, 110, 300],
        'Close': [110, 210, 310],
        'Volume': [1000, 1500, 2000],
    })

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_handle_missing_values_negative():
    with pytest.raises(Exception):
        handle_missing_values(None)

# Test for handle_outliers
def test_handle_outliers_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200, 10000],
        'Close': [110, 210, 11000],
        'Volume': [1000, 1500, 1000000],
    })

    result_df = handle_outliers(mock_df.copy())

    expected_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_handle_outliers_negative():
    with pytest.raises(Exception):
        handle_outliers(None)

# # Test for visualize_df
# def test_visualize_df_positive():
#     mock_df = pd.DataFrame({
#         'Open': [100, 200],
#         'Close': [110, 210],
#         'Volume': [1000, 1500],
#     })

#     result_df = visualize_df(mock_df.copy())

#     pd.testing.assert_frame_equal(result_df, mock_df)

# def test_visualize_df_negative():
#     with pytest.raises(Exception):
#         visualize_df(None)

# Test for apply_transformation
def test_apply_transformation_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
        'High': [120, 220],
        'Low': [90, 190],
        'Adj Close': [115, 215]
    })

    result_df = apply_transformation(mock_df.copy())

    scaler = MinMaxScaler(feature_range=(0, 1))
    expected_df = mock_df.copy()
    expected_df['Volume'] = scaler.fit_transform(expected_df[['Volume']])
    expected_df['Open'] = scaler.fit_transform(expected_df[['Open']])
    expected_df['Close'] = scaler.fit_transform(expected_df[['Close']])
    expected_df['High'] = scaler.fit_transform(expected_df[['High']])
    expected_df['Low'] = scaler.fit_transform(expected_df[['Low']])
    expected_df['Adj Close'] = scaler.fit_transform(expected_df[['Adj Close']])

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_apply_transformation_negative():
    with pytest.raises(Exception):
        apply_transformation(None)

# Test for generate_scheme_and_stats
def test_generate_scheme_and_stats_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })

    mock_task_instance = MockTaskInstance()

    result_df = generate_scheme_and_stats(mock_df.copy(), mock_task_instance)

    pd.testing.assert_frame_equal(result_df, mock_df)

def test_generate_scheme_and_stats_negative():
    # with pytest.raises(Exception):
    #     generate_scheme_and_stats(None, None)
    pass

# Test for calculate_and_display_anomalies
def test_calculate_and_display_anomalies_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })

    mock_task_instance = MockTaskInstance()

    result_df = calculate_and_display_anomalies(mock_df.copy(), mock_task_instance)

    pd.testing.assert_frame_equal(result_df, mock_df)

def test_calculate_and_display_anomalies_negative():
    # with pytest.raises(Exception):
    #     calculate_and_display_anomalies(None, None)
    pass