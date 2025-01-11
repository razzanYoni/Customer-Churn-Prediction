from airflow import DAG
from airflow.operators.python import PythonOperator
from hdfs import InsecureClient
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import pickle

with DAG(
    'drift_check_task',
    description='DAG that executes the drift check pipeline',
    schedule_interval=timedelta(minutes=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    HDFS_HOST = 'namenode'
    HDFS_PORT = '9000'

    preprocess_data = SparkSubmitOperator(
        task_id='preprocess_data',
        application='/opt/airflow/src/preprocess_drift/preprocess.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/raw", '--output_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/output", '--checkpoint_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/checkpoint", '--file_type', 'parquet']
    )

    bins_file = '/opt/airflow/data/bins'

    def load_bins(ti):
        bins = ti.xcom_pull(task_ids='store_bins', key='bins')

        with open(bins_file, "wb") as f:
            pickle.dump(bins, f)

    load_bins_task = PythonOperator(
        task_id='load_bins',
        python_callable=load_bins
    )

    check_and_retrain_model = SparkSubmitOperator(
        task_id='check_and_retrain_model',
        application='/opt/airflow/src/ml/check_and_retrain.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/output", '--bins_file', bins_file]
    )

    preprocess_data >> load_bins_task >> check_and_retrain_model

    
        