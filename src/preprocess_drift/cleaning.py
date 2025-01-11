"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, trim, when, log
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description="Data Preprocessing and Drift Simulation")
parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the drifted dataset")
parser.add_argument("--file_type", type=str, required=True, help="File type, csv or parquet")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
file_type = args.file_type

spark: SparkSession = SparkSession.builder.appName("preprocess").getOrCreate()

df: DataFrame = spark.read \
    .format(file_type) \
    .option("header", True) \
    .option("inferSchema", True) \
    .load(input_path)

df = df.drop('customerID')

df = df.withColumn(
    "SeniorCitizen",
    df.SeniorCitizen.cast("String")
)

df = df.dropna()

df = df.filter(trim(col("TotalCharges")) != "")

df = df.withColumn(
    "TotalCharges",
    df.TotalCharges.cast("Double")
)

df.printSchema()

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .option("header", True) \
  .save(output_path)

spark.stop()