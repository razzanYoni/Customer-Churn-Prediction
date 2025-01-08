"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer

# Argument Parser
parser = argparse.ArgumentParser(description="Data Preprocessing and Drift Simulation")
parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the drifted dataset")
parser.add_argument("--drift_intensity", type=float, default=0.2, help="Drift intensity (default: 0.2)")
args = parser.parse_args()

path = "../../data/raw/original-data.csv"
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()


df = spark.read.csv(path, header=True, inferSchema=True)

# Drop customerID Column because it is not predictive
df.drop('customerID')

# Drop Rows with Missing Churn Values
df.na.drop(subset=["Churn"])

# Replace empty strings in 'TotalCharges' with 0
df = df.withColumn(
    "TotalCharges",
    df.TotalCharges.cast("double")
)

# Handle Missing Values using Imputer
imputer = Imputer(
    inputCols=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
    outputCols=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
).setStrategy("mean")

df_imputed = imputer.fit(df).transform(df)


# Drop rows where more than 4 columns are NA
df = df.dropna()

df.write.csv()

spark.stop()