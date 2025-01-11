"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, lit, rand
from pyspark.sql import DataFrame

# Argument Parser
parser = argparse.ArgumentParser(description="Data Preprocessing and Drift Simulation")
parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the drifted dataset")
parser.add_argument("--drift_intensity", type=float, default=0.2, help="Drift intensity (default: 0.2)")
args = parser.parse_args()

path = args.input_path
output_path = args.output_path
drift_intensity = args.drift_intensity

spark : SparkSession = SparkSession.builder.appName("SimpleApp").getOrCreate()

df : DataFrame = spark.read.csv(path, header=True, inferSchema=True)

# Function to Simulate Data Drift
def simulate_data_drift(df: DataFrame, drift_intensity=0.2):
    # Drift for Numerical Columns
    df = df.withColumn(
        "MonthlyCharges",
        col("MonthlyCharges") * (1 + lit(drift_intensity) * rand())
    ).withColumn(
        "TotalCharges",
        col("TotalCharges") * (1 - lit(drift_intensity) * rand())
    ).withColumn(
        "tenure",
        (col("tenure") + (lit(drift_intensity * 10) * rand()).cast("int"))
    )

    # Drift for Categorical Columns
    df = df.withColumn(
        "gender",
        when(rand() > 0.9, "Male").otherwise("Female")
    ).withColumn(
        "Partner",
        when(rand() > 0.7, "Yes").otherwise("No")
    ).withColumn(
        "InternetService",
        when(rand() > 0.6, "Fiber optic")
        .when(rand() > 0.3, "DSL")
        .otherwise("No")
    ).withColumn(
        "PaymentMethod",
        when(rand() > 0.75, "Electronic check")
        .when(rand() > 0.5, "Credit card (automatic)")
        .when(rand() > 0.25, "Mailed check")
        .otherwise("Bank transfer (automatic)")
    ).withColumn(
        "Contract",
        when(rand() > 0.7, "Two year")
        .when(rand() > 0.4, "One year")
        .otherwise("Month-to-month")
    )

    return df

# Simulate Drift
drifted_df = simulate_data_drift(df, drift_intensity)

# Save Drifted Dataset
drifted_df.write.mode('overwrite').option('header', True).parquet(output_path)

spark.stop()