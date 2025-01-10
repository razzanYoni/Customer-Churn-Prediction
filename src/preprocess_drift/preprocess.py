"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType

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
    StructField("TotalCharges", DoubleType(), True),
    StructField("Churn", StringType(), True),
])


df = spark \
    .readStream \
    .schema(dataSchema) \
    .option("header", True) \
    .parquet(input_path) \

# Drop customerID Column because it is not predictive
df = df.drop('customerID')

# Drop Rows with Missing Churn Values
df = df.na.drop(subset=["Churn"])

# Replace empty strings in 'TotalCharges' with 0
df = df.withColumn(
    "TotalCharges",
    df.TotalCharges.cast("float")
)

df = df.withColumn(
    "MonthlyCharges",
    df.TotalCharges.cast("float")
)

df.printSchema()

# # Handle Missing Values using Imputer
# imputer = Imputer(
#     inputCols=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
#     outputCols=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
# ).setStrategy("mean")

# df_imputed = imputer.fit(df).transform(df)


# Drop rows where there is NA
df = df.dropna()

query = df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("checkpointLocation", checkpoint_path) \
    .option("path", output_path) \
    .start()

query.awaitTermination()

# df.write.mode('overwrite').option("header", True).parquet(args.output_path)

# spark.stop()