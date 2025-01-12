import argparse
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import Bucketizer
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType
from pyspark.sql.functions import col
import numpy as np
import pathlib
import os

def is_retrain(psi_values) -> bool:
    return any(psi > 0.1 for psi in psi_values.values())

def get_psi(new_distribution: DataFrame, old_distribution: DataFrame):
    new_row = new_distribution.collect()[0]
    old_row = old_distribution.collect()[0]
    
    psi_values = {}
    
    # Iterate over columns in the new_distribution
    for col_name in new_distribution.columns:
        # Extract the array from each DF row
        new_arr = new_row[col_name]  # e.g., [0.2, 0.3, 0.5]
        old_arr = old_row[col_name]  # e.g., [0.25, 0.25, 0.5]

        # Basic validation - skip if arrays are None or empty
        if not new_arr or not old_arr:
            psi_values[col_name] = None
            print(f"PSI for {col_name} could not be calculated (empty array).")
            continue
        
        # Ensure arrays are the same length
        if len(new_arr) != len(old_arr):
            raise ValueError(
                f"Arrays for column '{col_name}' have different lengths: "
                f"{len(new_arr)} vs {len(old_arr)}"
            )
        
        psi = 0.0
        
        # Calculate PSI for each bin/category
        for i in range(len(new_arr)):
            # Avoid division by zero: replace 0 with a small number
            old_val = old_arr[i] if old_arr[i] != 0 else 1e-6
            new_val = new_arr[i] if new_arr[i] != 0 else 1e-6
            
            # PSI formula: (p - q) * ln(p/q)
            psi += (new_val - old_val) * np.log(new_val / old_val)
        
        psi_values[col_name] = psi
        print(f"PSI for {col_name}: {psi}")
    
    return psi_values

def get_numerical_distribution(df: DataFrame):
    numerical_distribution = {}
    num_bucket = {
        "tenure" : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('Inf')],
        "MonthlyCharges" : [0, 20, 40, 60, 80, 100, float('Inf')],
        "TotalCharges" : [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, float('Inf')]
    }

    for col_name, splits in num_bucket.items():
        bucketizer = Bucketizer(splits=splits,
                                inputCol=col_name,
                                outputCol=col_name + "_bucket",)
        df_bucket = bucketizer.transform(df)

        df_bucket = df_bucket.select(col_name + "_bucket")

        all_buckets = list(range(len(splits) - 1))
        dummy_data = [(float(b),) for b in all_buckets]
        dummy_df = df.sparkSession.createDataFrame(
            dummy_data, [col_name + "_bucket"]
        )

        df_union = df_bucket.union(dummy_df)

        # distribution = (
        #     df_union.groupBy(col_name + "_bucket")
        #             .agg((count("*") - 1).alias("count"))  # subtract dummy row
        #             .orderBy(col_name + "_bucket")
        # )

        distribution = df_union.groupBy(col_name+ "_bucket").count()
        distribution = distribution.withColumn("count", col("count") - 1)
        distribution.show()
        
        total = distribution.select("count").agg({"count": "sum"}).collect()[0][0]

        numerical_distribution[col_name] = [row["count"] / total for row in distribution.collect()]


    return numerical_distribution
# Argument Parser
parser = argparse.ArgumentParser(description="Data Preprocessing and Drift Simulation")
parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--is_retrain_flag_file", type=str, required=True, help="File to save retrain flag")
parser.add_argument("--distribution_path", type=str, required=True, help="Path to save distribution")
parser.add_argument("--file_type", type=str, required=True, help="File type, csv or parquet")
args = parser.parse_args()

input_path = args.input_path
is_retrain_flag_file = args.is_retrain_flag_file
distribution_path = args.distribution_path
file_type = args.file_type

def write_is_retrain(is_retrain: bool):
    pathlib.Path(os.path.dirname(is_retrain_flag_file)).mkdir(parents=True, exist_ok=True)
    with open(is_retrain_flag_file, "w") as f:
        f.write(str(is_retrain))

spark: SparkSession = SparkSession.builder.appName("preprocess").getOrCreate()

df: DataFrame = spark.read \
    .format(file_type) \
    .option("header", True) \
    .option("inferSchema", True) \
    .load(input_path)

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

    categorical_distribution[column] = []

    count_dict = {}

    total = 0
    for value in valid_values:
        count_dict[value] = df.filter(col(column) == value).count()
        total += count_dict[value]

    # Calculate the distribution
    for value in valid_values:
        categorical_distribution[column].append(count_dict[value] / total)


numerical_distribution = get_numerical_distribution(df)
print(categorical_distribution)
print(numerical_distribution)

complete_distribution = {**categorical_distribution, **numerical_distribution}

rdd = spark.sparkContext.parallelize([complete_distribution])

dataSchema = StructType([
    StructField("gender", ArrayType(DoubleType()), True),
    StructField("SeniorCitizen", ArrayType(DoubleType()), True),
    StructField("Partner", ArrayType(DoubleType()), True),
    StructField("Dependents", ArrayType(DoubleType()), True),
    StructField("PhoneService", ArrayType(DoubleType()), True),
    StructField("MultipleLines", ArrayType(DoubleType()), True),
    StructField("InternetService", ArrayType(DoubleType()), True),
    StructField("OnlineSecurity", ArrayType(DoubleType()), True),
    StructField("OnlineBackup", ArrayType(DoubleType()), True),
    StructField("DeviceProtection", ArrayType(DoubleType()), True),
    StructField("TechSupport", ArrayType(DoubleType()), True),
    StructField("StreamingTV", ArrayType(DoubleType()), True),
    StructField("StreamingMovies", ArrayType(DoubleType()), True),
    StructField("Contract", ArrayType(DoubleType()), True),
    StructField("PaperlessBilling", ArrayType(DoubleType()), True),
    StructField("tenure", ArrayType(DoubleType()), True),
    StructField("MonthlyCharges", ArrayType(DoubleType()), True),
    StructField("TotalCharges", ArrayType(DoubleType()), True),
])



new_distribution = spark.createDataFrame(rdd, dataSchema)

new_distribution.printSchema()

new_distribution.show(5)

try:
    old_distribution = spark.read \
        .format("json") \
        .option("header", True) \
        .load(distribution_path)
    
    print("Getting old distribution successful")
    old_distribution.show(5)
except:
    new_distribution.write \
        .mode("overwrite") \
        .format("json") \
        .option("header", True) \
        .save(distribution_path)
    
    write_is_retrain(True)
    
    spark.stop()
    exit(0)

print("Old Distribution")
psi = get_psi(new_distribution, old_distribution)

print(psi)

if is_retrain(psi):
    write_is_retrain(True)
    
    new_distribution.write \
        .mode("overwrite") \
        .format("json") \
        .option("header", True) \
        .save(distribution_path)

else:
    write_is_retrain(False)

spark.stop()