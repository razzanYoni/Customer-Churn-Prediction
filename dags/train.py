from airflow import DAG
from airflow.operators.python import PythonOperator
from hdfs import InsecureClient
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime
import pickle

with DAG(
    'train_task',
    description='DAG that executes the train pipeline',
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
    
    upload_to_hdfs = PythonOperator(
        task_id='upload_to_hdfs',
        python_callable=hdfs_upload
    )

    preprocess_data = SparkSubmitOperator(
        task_id='preprocess_data',
        application='/opt/airflow/src/preprocess_drift/preprocess.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/raw", '--output_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/output", '--checkpoint_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/checkpoint", '--file_type', 'csv']
    )

    bins_file = '/opt/airflow/data/bins'

    train_model = SparkSubmitOperator(
        task_id='train_model',
        application='/opt/airflow/src/ml/train.py',
        packages='org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0',
        conn_id='spark_default',
        application_args=['--input_path', f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/output", '--bins_file', bins_file]
    )

    def store_bins(ti):
        with open(bins_file, "rb") as f:
            bins = pickle.load(f)
        ti.xcom_push(key='bins', value=bins)

    store_bins_task = PythonOperator(
        task_id='store_bins',
        python_callable=store_bins
    )

    upload_to_hdfs >> preprocess_data >> train_model >> store_bins_task

    
        