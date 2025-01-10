"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, OneHotEncoder
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType
from pyspark.sql.functions import col, trim, when

# Argument Parser
parser = argparse.ArgumentParser(description="Data Preprocessing and Drift Simulation")
parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the drifted dataset")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to save the checkpoint")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
checkpoint_path = args.checkpoint_path

spark = SparkSession.builder.appName("preprocess").getOrCreate()

dataSchema = StructType([
    StructField("customerID", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("SeniorCitizen", IntegerType(), True),
    StructField("Partner", StringType(), True),
    StructField("Dependents", StringType(), True),
    StructField("tenure", IntegerType(), True),
    StructField("PhoneService", StringType(), True),
    StructField("MultipleLines", StringType(), True),
    StructField("InternetService", StringType(), True),
    StructField("OnlineSecurity", StringType(), True),
    StructField("OnlineBackup", StringType(), True),
    StructField("DeviceProtection", StringType(), True),
    StructField("TechSupport", StringType(), True),
    StructField("StreamingTV", StringType(), True),
    StructField("StreamingMovies", StringType(), True),
    StructField("Contract", StringType(), True),
    StructField("PaperlessBilling", StringType(), True),
    StructField("PaymentMethod", StringType(), True),
    StructField("MonthlyCharges", DoubleType(), True),
    StructField("TotalCharges", StringType(), True),
    StructField("Churn", StringType(), True),
])

df = spark \
    .readStream \
    .schema(dataSchema) \
    .option("header", True) \
    .parquet(input_path) \

# Drop customerID Column because it is not predictive
df = df.drop('customerID')

df = df.dropna()

df = df.filter(trim(col("TotalCharges")) != "")

df = df.withColumn(
    "TotalCharges",
    df.TotalCharges.cast("Double")
)

categories = {
    "gender": ["Male", "Female"],
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

# Apply one-hot encoding and filter valid rows
for column, valid_values in categories.items():
    # Create one-hot encoded columns for each valid category
    for value in valid_values:
        new_column = f"{column}_{value.replace(' ', '_').lower()}"
        df = df.withColumn(new_column, when(col(column) == value, 1).otherwise(0))
    
    # TOO MEMORY CONSUMING, MY JAVA HEAP SPACE IS NOT ENOUGH (FAKHRI)
    # Filter rows to include only valid values for the column
    # valid_condition = " OR ".join([f"{column} = '{value}'" if isinstance(value, str) else f"{column} = {value}" for value in valid_values])
    # df = df.filter(valid_condition)

    df = df.drop(column)

df.printSchema()

query = df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("checkpointLocation", checkpoint_path) \
    .option("path", output_path) \
    .option("header", True) \
    .start()

query.awaitTermination()

# df.write.mode('overwrite').option("header", True).parquet(args.output_path)

# spark.stop()