from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from hdfs import InsecureClient
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

with DAG(
    'drift_check_task',
    description='DAG that executes the drift check pipeline',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    HDFS_HOST = 'namenode'
    HDFS_PORT = '9000'

    def hdfs_upload():
        client = InsecureClient('http://namenode:9870')
        client.upload('/data/raw/original-data.csv', '/opt/airflow/data/raw/original-data.csv',
                      overwrite=True)
    
    distribution_path = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/distribution"
    is_retrain_flag_file = '/opt/airflow/data/tmp/is_retrain_flag'
    
    upload_to_hdfs = PythonOperator(
        task_id='upload_to_hdfs',
        python_callable=hdfs_upload
    )

    drift_simulation = SparkSubmitOperator(
        task_id='drift_simulation',
        application='/opt/airflow/src/preprocess_drift/drift_simulation.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/raw", '--output_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/drifted", '--drift_intensity', '0']
    )

    cleaning_data = SparkSubmitOperator(
        task_id='cleaning_data',
        application='/opt/airflow/src/preprocess_drift/cleaning.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/drifted", '--output_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/cleaned", '--file_type', 'parquet']
    )

    check_psi = SparkSubmitOperator(
        task_id='check_psi',
        application='/opt/airflow/src/preprocess_drift/psi.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/cleaned", '--distribution_path', distribution_path, '--is_retrain_flag_file', is_retrain_flag_file, '--file_type', 'parquet']
    )

    def check_retrain():
        with open(is_retrain_flag_file, "r") as f:
            return f.read() == 'True'

    check_retrain_task = ShortCircuitOperator(
        task_id='check_retrain',
        python_callable=check_retrain
    )

    preprocess_data = SparkSubmitOperator(
        task_id='preprocess_data',
        application='/opt/airflow/src/preprocess_drift/preprocess.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/cleaned", '--output_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/preprocessed", '--file_type', 'parquet']
    )

    train_model = SparkSubmitOperator(
        task_id='train_model',
        application='/opt/airflow/src/ml/train.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/preprocessed"]
    )

    upload_to_hdfs >> drift_simulation >> cleaning_data >> check_psi >> check_retrain_task >> preprocess_data >> train_model

    
        