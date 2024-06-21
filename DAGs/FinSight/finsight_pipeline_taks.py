from airflow.operators.python_operator import PythonOperator#,# ExternalTaskMarker
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow import DAG
from datetime import datetime
from airflow import configuration as conf
import sys, os
import functools
from airflow.models import Variable
import smtplib
from email.mime.text import MIMEText

# Add the directory containing finsight_pipeline_functions.py to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from FinSight.finsight_pipeline_functions import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define the DAG
dag = DAG(
    dag_id='FinSight_pipeline',
    default_args={
        'owner': 'Group 2',
        'start_date': datetime(2024, 6, 6),
        "email": ["prayaga.a@northeastern.edu"],
        "email_on_failure": True,
    },
    schedule_interval=None,
)

# Define the tasks
download_and_uploadToDVCBucket_task = PythonOperator(
    task_id='download_upload_data',
    python_callable=download_and_uploadToDVCBucket,
    op_kwargs={
        'ticker_symbol': 'NFLX',
        'start_date': '2002-01-01',
        'end_date': '2022-12-31'
    },
    dag=dag,
    provide_context=True,
)

visualize_raw_data_task = PythonOperator(
    task_id='visualize_data',
    python_callable=functools.partial(
        visualize_raw_data, 
    ),
    provide_context=True,
)

divide_train_eval_test_splits_task = PythonOperator(
    task_id='divide_train_eval_test_splits',
    python_callable=divide_train_eval_test_splits,
    op_args=[visualize_raw_data_task.output],
    # op_kwargs={'file_path': 'gs://data_finsight/NFLX_stock_data_2002-01-01_2022-12-31.csv'},
    provide_context=True,
    dag=dag,
)

def handle_missing_values_task(ti, split):
    print("dsdsd")
    df = ti.xcom_pull(task_ids='divide_train_eval_test_splits', key=split)
    handled_df = handle_missing_values(df)
    return handled_df

handle_missing_values_in_training_data_task = PythonOperator(
    task_id='handle_missing_values_in_training_data',
    python_callable=functools.partial(handle_missing_values_task, split='train'),
    provide_context=True,
    dag=dag,
)

# handle_missing_values_in_evaluation_data_task = PythonOperator(
#     task_id='handle_missing_values_in_evaluation_data',
#     python_callable=functools.partial(handle_missing_values_task, split='eval'),
#     provide_context=True,
#     dag=dag,
# )

# handle_missing_values_in_test_data_task = PythonOperator(
#     task_id='handle_missing_values_in_test_data',
#     python_callable=functools.partial(handle_missing_values_task, split='test'),
#     provide_context=True,
#     dag=dag,
# )

handle_outliers_in_training_data_task = PythonOperator(
    task_id='handle_outliers_in_training_data',
    python_callable=handle_outliers,
    provide_context=True,
    op_args=[handle_missing_values_in_training_data_task.output],
    dag=dag,
)
handle_missing_values_in_training_data_task.set_downstream(handle_outliers_in_training_data_task)

# handle_outliers_in_evaluation_data_task = PythonOperator(
#     task_id='handle_outliers_in_evaluation_data',
#     python_callable=handle_outliers,
#     provide_context=True,
#     op_args=[handle_missing_values_in_evaluation_data_task.output],
#     dag=dag,
# )
# handle_missing_values_in_evaluation_data_task.set_downstream(handle_outliers_in_evaluation_data_task)

# handle_outliers_in_test_data_task = PythonOperator(
#     task_id='handle_outliers_in_test_data',
#     python_callable=handle_outliers,
#     provide_context=True,
#     op_args=[handle_missing_values_in_test_data_task.output],
#     dag=dag,
# )
# handle_missing_values_in_test_data_task.set_downstream(handle_outliers_in_test_data_task)

##
generate_scheme_and_stats_training_task = PythonOperator(
    task_id='generate_scheme_and_stats_training',
    python_callable=generate_scheme_and_stats,
    provide_context=True,
    op_args=[handle_outliers_in_training_data_task.output],
    dag=dag,
)
handle_outliers_in_training_data_task.set_downstream(generate_scheme_and_stats_training_task)

# generate_scheme_and_stats_eval_task = PythonOperator(
#     task_id='generate_scheme_and_stats_eval',
#     python_callable=generate_scheme_and_stats,
#     provide_context=True,
#     op_args=[handle_outliers_in_evaluation_data_task.output],
#     dag=dag,
# )
# handle_outliers_in_evaluation_data_task.set_downstream(generate_scheme_and_stats_eval_task)

# generate_scheme_and_stats_test_task = PythonOperator(
#     task_id='generate_scheme_and_stats_test',
#     python_callable=generate_scheme_and_stats,
#     provide_context=True,
#     op_args=[handle_outliers_in_test_data_task.output],
#     dag=dag,
# )
# handle_outliers_in_test_data_task.set_downstream(generate_scheme_and_stats_test_task)

def calculate_and_display_anomalies_task(ti, split):
    df = ti.xcom_pull(task_ids='divide_train_eval_test_splits', key=split)
    scheme = ti.xcom_pull(task_ids='generate_scheme_and_stats_training', key="schema")
    stats = ti.xcom_pull(task_ids='generate_scheme_and_stats_training', key="stats")
    handled_df = calculate_and_display_anomalies(df, scheme, stats)
    return handled_df


calculate_and_display_anomalies_eval_task = PythonOperator(
    task_id='calculate_and_display_anomalies_eval',
    python_callable=functools.partial(calculate_and_display_anomalies_task, split='eval'),
    provide_context=True,
    dag=dag,
)
generate_scheme_and_stats_training_task.set_downstream(calculate_and_display_anomalies_eval_task)

# calculate_and_display_anomalies_eval_task = PythonOperator(
#     task_id='calculate_and_display_anomalies_eval',
#     python_callable=calculate_and_display_anomalies,
#     provide_context=True,
#     op_args=[generate_scheme_and_stats_eval_task.output],
#     dag=dag,
# )
# generate_scheme_and_stats_eval_task.set_downstream(calculate_and_display_anomalies_eval_task)

# calculate_and_display_anomalies_test_task = PythonOperator(
#     task_id='calculate_and_display_anomalies_test',
#     python_callable=calculate_and_display_anomalies,
#     op_args=[generate_scheme_and_stats_test_task.output],
#     provide_context=True,
#     dag=dag,
# )
# generate_scheme_and_stats_test_task.set_downstream(calculate_and_display_anomalies_test_task)

##

apply_transformation_training_task = PythonOperator(
    task_id='apply_transformation_training',
    python_callable=apply_transformation,
    provide_context=True,
    op_args=[generate_scheme_and_stats_training_task.output],
    dag=dag,
)
# calculate_and_display_anomalies_eval_task.set_downstream(apply_transformation_training_task)

apply_transformation_eval_task = PythonOperator(
    task_id='apply_transformation_eval',
    python_callable=apply_transformation_eval_test,
    provide_context=True,
    op_args=[divide_train_eval_test_splits_task.output["eval"]],
    dag=dag,
)
apply_transformation_training_task.set_downstream(apply_transformation_eval_task)

apply_transformation_test_task = PythonOperator(
    task_id='apply_transformation_test',
    python_callable=apply_transformation_eval_test,
    provide_context=True,
    op_args=[divide_train_eval_test_splits_task.output["test"]],
    dag=dag,
)
apply_transformation_training_task.set_downstream(apply_transformation_test_task)



# 

visualize_training_refined_data_task = PythonOperator(
    task_id='visualize_training_refined_data',
    python_callable=visualize_df,
    provide_context=True,
    op_args=[apply_transformation_training_task.output],
    dag=dag,
)
apply_transformation_training_task.set_downstream(visualize_training_refined_data_task)


#training
divide_features_and_labels_task = PythonOperator(
    task_id='divide_features_and_labels',
    python_callable=divide_features_and_labels,
    provide_context=True,
    op_args=[visualize_training_refined_data_task.output, apply_transformation_eval_task.output, apply_transformation_test_task.output],
    dag=dag,
)
visualize_training_refined_data_task.set_downstream(divide_features_and_labels_task)

#eval
divide_train_eval_test_splits_task.set_downstream(divide_features_and_labels_task)


hyper_parameter_tuning_task = PythonOperator(
    task_id='hyper_parameter_tuning',
    python_callable=hyper_parameter_tuning,
    provide_context=True,
    # op_args=[divide_features_and_labels_task.output],
    op_kwargs={'x':  divide_features_and_labels_task.output['x'], 'y':  divide_features_and_labels_task.output['y']},
    dag=dag,
)

training_task = PythonOperator(
    task_id='training',
    python_callable=training,
    provide_context=True,
    op_args=[hyper_parameter_tuning_task.output],
    op_kwargs={'x':  divide_features_and_labels_task.output['x'], 'y':  divide_features_and_labels_task.output['y']},
    dag=dag,
)
hyper_parameter_tuning_task.set_downstream(training_task)


load_and_predict_task = PythonOperator(
    task_id='load_and_predict',
    python_callable=load_and_predict,
    provide_context=True,
    op_args=[divide_features_and_labels_task.output['x'], training_task.output, ],
    # op_kwargs={'x':  divide_features_and_labels_task.output['x'], 'best_params': hyper_parameter_tuning_task.output},
    dag=dag,
)

evaluate_and_visualize_task = PythonOperator(
    task_id='evaluate_and_visualize',
    python_callable=evaluate_and_visualize,
    provide_context=True,
    op_args=[load_and_predict_task.output, divide_features_and_labels_task.output['y']],
    # op_kwargs={'ti': "evaluate_and_visualize"},
    dag=dag,
)

# Define the DAG
retrain_dag = DAG(
    dag_id='ReTrain_FinSight_pipeline',
    default_args={
        'owner': 'Group 2',
        'start_date': datetime(2024, 6, 6),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
    },
    schedule_interval=None,
)

parent_task = ExternalTaskSensor(
    task_id="parent_task",
    external_dag_id="FinSight_pipeline",
    external_task_id="handle_missing_values_in_training_data",
    dag=retrain_dag,
)


def send_email():
    Variable.set()
    sender_email = Variable.get('adityasaiprayaga')
    receiver_email = "prayaga.a@northeastern.edu"
    password = Variable.get('oVJHo7Mem*dUVyBM')

    subject = "Sample email from Airflow"
    body = "Hello, this is a test email from Python."

    # Create the email headers and content
    email_message = MIMEText(body)
    email_message['Subject'] = subject
    email_message['From'] = sender_email
    email_message['To'] = receiver_email

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail's SMTP server
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email_message.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()

# Define the task using PythonOperator
send_email_task = PythonOperator(
    task_id='send_email_task',
    python_callable=send_email,
    dag=dag,
)

# Set the task in the DAG
send_email_task


retraining_task = PythonOperator(
    task_id='retraining',
    python_callable=get_retrain_dataset,
    provide_context=True,
    op_kwargs={
        'ticker_symbol': 'NFLX',
        'start_date': '2022-01-01',
        'end_date': '2023-12-31'
    },
    # op_args=[load_and_predict_task.output,divide_features_and_labels_task.output['x'],divide_features_and_labels_task.output['y'], generate_scheme_and_stats_training_task.output, divide_features_and_labels_task.output['y']],
    # op_kwargs={'file_path': 'gs://data_finsight/NFLX_stock_data_2002-01-01_2022-12-31.csv'},
    dag=retrain_dag,
)

retraining_task.set_downstream(parent_task)


# visualize_evaluation_refined_data_task = PythonOperator(
#     task_id='visualize_evaluation_refined_data',
#     python_callable=visualize_df,
#     provide_context=True,
#     op_args=[apply_transformation_eval_task.output],
#     dag=dag,
# )
# apply_transformation_eval_task.set_downstream(visualize_evaluation_refined_data_task)

# visualize_test_refined_data_task = PythonOperator(
#     task_id='visualize_test_refined_data',
#     python_callable=visualize_df,
#     provide_context=True,
#     op_args=[apply_transformation_test_task.output],
#     dag=dag,
# )
# apply_transformation_test_task.set_downstream(visualize_test_refined_data_task)

# start_flask_API_task = PythonOperator(
#     task_id = 'start_Flask_API',
#     python_callable=start_flask_app,
#     dag = dag
# )

# evaluate_and_visualize_task.set_downstream(start_flask_API_task)

# Set the task dependencies
download_and_uploadToDVCBucket_task >> visualize_raw_data_task >> divide_train_eval_test_splits_task >> handle_missing_values_in_training_data_task, 
#     handle_missing_values_in_evaluation_data_task, 
#     handle_missing_values_in_test_data_task
# ]
