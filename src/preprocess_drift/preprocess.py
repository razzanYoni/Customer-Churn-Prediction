"""SimpleApp.py"""
import argparse
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType, ArrayType
from pyspark.sql.functions import col, trim, when, log
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description="Data Preprocessing and Drift Simulation")
parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the drifted dataset")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to save the checkpoint")
parser.add_argument("--file_type", type=str, required=True, help="File type, csv or parquet")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
checkpoint_path = args.checkpoint_path
file_type = args.file_type

# def get_psi(new_distribution: DataFrame, old_distribution: DataFrame):
#     psi_values = {}
#     for column in new_distribution:
#         psi = 0
#         for i in range(len(new_distribution[column])):
#             if value == 0:
#                 value = 0.000001
#             psi += (value - old_distribution[column][i]) * log(value / old_distribution[column][i])
#         psi_values[column] = psi
#         print(f"PSI for {column}: {psi}")

def get_psi(new_distribution: DataFrame, old_distribution: DataFrame):
    """
    Calculate PSI (Population Stability Index) for each column in the given DataFrames.
    Assumes each DataFrame has one row, and each column in that row is an Array[Double].
    
    :param new_distribution: Spark DataFrame with columns of ArrayType(DoubleType())
    :param old_distribution: Spark DataFrame with columns of ArrayType(DoubleType())
    :return: A dictionary {column_name: PSI_value}
    """
    
    # Collect the single row from each distribution as Python objects
    # (we expect exactly one row per DataFrame)
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

spark: SparkSession = SparkSession.builder.appName("preprocess").getOrCreate()

df: DataFrame = spark.read \
    .format(file_type) \
    .option("header", True) \
    .option("inferSchema", True) \
    .load(input_path)

df = df.withColumn(
    "SeniorCitizen",
    df.SeniorCitizen.cast("String")
)

stringIndexer = StringIndexer(inputCol="Churn", outputCol="label", handleInvalid="skip")

df = stringIndexer.fit(df).transform(df)

df = df.drop('Churn')

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
    # Create one-hot encoded columns for each valid category
    for value in valid_values:
        count_dict[value] = df.filter(col(column) == value).count()
        total += count_dict[value]
        new_column = f"{column}_{value.replace(' ', '_').lower()}"
        df = df.withColumn(new_column, when(col(column) == value, 1).otherwise(0))

    
    # TOO MEMORY CONSUMING, MY JAVA HEAP SPACE IS NOT ENOUGH (FAKHRI)
    # Filter rows to include only valid values for the column
    # valid_condition = " OR ".join([f"{column} = '{value}'" if isinstance(value, str) else f"{column} = {value}" for value in valid_values])
    # df = df.filter(valid_condition)

    # Calculate the distribution
    for value in valid_values:
        categorical_distribution[column].append(count_dict[value] / total)
    
    df = df.drop(column)



print(categorical_distribution)

rdd = spark.sparkContext.parallelize([categorical_distribution])

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
])



cd_dataframe = spark.createDataFrame(rdd, dataSchema)

cd_dataframe.printSchema()

cd_dataframe.show(5)

cd_dataframe.write \
  .mode("overwrite") \
  .format("json") \
  .option("header", True) \
  .save("../../data/out/cd2")


old_cd = spark.read \
    .format("json") \
    .option("header", True) \
    .load("../../data/out/cd")

psi = get_psi(cd_dataframe, old_cd)

print(psi)

feature_col = df.columns
feature_col.remove("label")

vector_assembler = VectorAssembler(inputCols=feature_col, outputCol="features")

df = vector_assembler.transform(df)

df = df.select("features", "label", "MonthlyCharges")

df.printSchema()

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .option("header", True) \
  .save(output_path)

spark.stop()