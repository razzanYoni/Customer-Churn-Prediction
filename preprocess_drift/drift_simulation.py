"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, lit, rand

# Argument Parser
parser = argparse.ArgumentParser(description="Data Preprocessing and Drift Simulation")
parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the drifted dataset")
parser.add_argument("--drift_intensity", type=float, default=0.2, help="Drift intensity (default: 0.2)")
args = parser.parse_args()

path = args.input_path
output_path = args.output_path
drift_intensity = args.drift_intensity

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

df = spark.read.csv(path, header=True, inferSchema=True)

# Replace empty strings in 'TotalCharges' with 0
df = df.withColumn(
    "TotalCharges",
    df.TotalCharges.cast("double")
)

# Drop rows where more than 4 columns are NA
df = df.dropna()

# Function to Simulate Data Drift
def simulate_data_drift(df, drift_intensity=0.2):
    
    # Drift for Numerical Columns
    df = df.withColumn(
        "MonthlyCharges_drifted",
        col("MonthlyCharges") * (1 + lit(drift_intensity) * rand())
    ).withColumn(
        "TotalCharges_drifted",
        col("TotalCharges") * (1 - lit(drift_intensity) * rand())
    )
    
    # Drift for Categorical Columns
    df = df.withColumn(
        "gender_drifted",
        when(rand() > 0.7, "Male").otherwise("Female")
    ).withColumn(
        "Contract_drifted",
        when(rand() > 0.5, "One year").otherwise(col("Contract"))
    )
    
    return df

# Simulate Drift
drifted_df = simulate_data_drift(df, drift_intensity=0.2)

# Save Drifted Dataset
drifted_df.write.mode('overwrite').parquet(output_path)
# drifted_df.write.csv(output_path)  

spark.stop()