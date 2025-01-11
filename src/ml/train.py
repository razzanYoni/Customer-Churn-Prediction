import argparse
from ml import train_model_neural_network, calculate_initial_model_psi
from pyspark.sql import SparkSession
import mlflow
import pickle

def main():
    parser = argparse.ArgumentParser(description="Run Model Training")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--bins_file", type=str, required=True, help="File path to the bins")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://mlflow_server:5000")

    spark: SparkSession = SparkSession.builder    \
    .appName("Train Initial Model") \
    .getOrCreate()
    
    mlflow.pyspark.ml.autolog()

    df = spark.read.parquet(args.input_path, header=True, inferSchema=True)

    data_train, data_test = df.randomSplit([0.8, 0.2], seed=488)

    train_model_neural_network(data_train, data_test, latest_model=None, isRetrain=False)

    bins = calculate_initial_model_psi("MonthlyCharges", df)

    with open(args.bins_file, "wb") as f:
        pickle.dump(bins, f)

    spark.stop()

if __name__ == "__main__":
    main()
