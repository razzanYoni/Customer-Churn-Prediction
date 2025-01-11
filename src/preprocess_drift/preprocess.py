"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col, when
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

stringIndexer = StringIndexer(inputCol="Churn", outputCol="label", handleInvalid="skip")
df = stringIndexer.fit(df).transform(df)
df = df.drop('Churn')

categories = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Bank transfer (automatic)", "Credit card (automatic)", 
        "Electronic check", "Mailed check"
    ],
}

categorical_distribution = {}

# Apply one-hot encoding and filter valid rows
for column, valid_values in categories.items():

    # Create one-hot encoded columns for each valid category
    for value in valid_values:
        new_column = f"{column}_{value.replace(' ', '_').lower()}"
        df = df.withColumn(new_column, when(col(column) == value, 1).otherwise(0))

    df = df.drop(column)


feature_col = df.columns
feature_col.remove("label")
vector_assembler = VectorAssembler(inputCols=feature_col, outputCol="features")
df = vector_assembler.transform(df)

df = df.select("features", "label")
df.printSchema()

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .option("header", True) \
  .save(output_path)

spark.stop()