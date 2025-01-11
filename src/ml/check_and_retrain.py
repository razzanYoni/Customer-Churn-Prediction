import argparse
from ml import monitor_drift_and_retrain
from pyspark.sql import SparkSession
import mlflow
import pickle

def main():
    parser = argparse.ArgumentParser(description="Run Model Training")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--bins_file", type=str, required=True, help="File path to the bins")
    args = parser.parse_args()

    with open(args.bins_file, "rb") as f:
        bins = pickle.load(f)

    mlflow.set_tracking_uri("http://mlflow_server:5000")

    spark: SparkSession = SparkSession.builder    \
    .appName("Check and Retrain Model") \
    .getOrCreate()

    mlflow.pyspark.ml.autolog()

    monitor_drift_and_retrain(bins, args.input_path, args.latest_model_path)

    spark.stop()

if __name__ == "__main__":
    main()
