import mlflow
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import numpy as np


# https://arize.com/blog-course/population-stability-index-psi/
# https://github.com/mwburke/population-stability-index
def calculate_initial_model_psi(column_name, baseline, bins=10):
    """Calculate Population Stability Index (PSI) for a single feature."""
    baseline_bins = baseline.select(F.histogram_numeric(col=column_name, nBins=F.lit(bins))).collect()[0][0]

    # divide as proportion
    normalized = np.array([bin_info.y for bin_info in baseline_bins]) / baseline.count()

    return normalized

def calculate_psi(column_name, baseline, current, bins=10):
    """Calculate Population Stability Index (PSI) for a single feature."""
    baseline_bins = baseline.select(F.histogram_numeric(col=column_name, nBins=F.lit(bins))).collect()[0][0]
    current_bins = current.select(F.histogram_numeric(col=column_name, nBins=F.lit(bins))).collect()[0][0]

    baseline_bins = np.array([bin_info.y for bin_info in baseline_bins]) / baseline.count()
    current_bins = np.array([bin_info.y for bin_info in current_bins]) / current.count()

    psi = np.sum((baseline_bins - current_bins) * np.log(baseline_bins / current_bins))
    return psi, baseline_bins, current_bins

def calculate_psi_multiple_features(baseline, current, columns, bins=10):
    """Calculate PSI for multiple features."""
    psi_values = {}
    for col in columns:
        psi = calculate_psi(baseline, current, bins)
        psi_values[col] = psi
    return psi_values

def train_model_neural_network(data_train, data_test, latest_model = None, isRetrain=False):
    """Train the initial neural network model and log it to MLflow."""
    with mlflow.start_run(run_name="Initial_Model_Training" if not isRetrain else "Retrain_Model"):
        # get weights from the latest model
        mlp = MultilayerPerceptronClassifier(
            layers = [45, 10, 45, 20, 2],
            seed = 123,
            maxIter = 500,
            labelCol = "label",
            featuresCol="features",
            rawPredictionCol="rawPrediction",
            predictionCol="prediction"
        )
        if isRetrain:
            mlp.setInitialWeights(latest_model.weights)

        mlp_model = mlp.fit(data_test)

        # Log model parameters
        mlflow.log_param("model_type", "NeuralNetwork")

        # Log model metrics
        pred_df = mlp_model.transform(data_train)
        pred_df.select('features','label', 'prediction', 'rawPrediction', 'probability').show(5)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        mlpacc = evaluator.evaluate(pred_df)
        mlflow.log_metric("accuracy", mlpacc)

        mlflow.spark.log_model(mlp_model, "model", registered_model_name="CustomerChurnModel")

        return mlp_model

# https://stackoverflow.com/questions/70111193/how-can-i-load-the-latest-model-version-from-mlflow-model-registry
# https://medium.com/swlh/pysparks-multi-layer-perceptron-classifier-on-iris-dataset-dcf70d553cd8 
def monitor_drift_and_retrain(baseline, input_data_path, latest_model_path, psi_threshold = 0.1):
    spark = SparkSession.builder.appName("Train Model").getOrCreate()
    df = spark.read.parquet(input_data_path, header=True, inferSchema=True)

    # get all columns without the Churn column
    feature_columns = [col for col in df.columns if col != "Churn"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data = assembler.transform(df)

    # Split into features and target
    train_df, test_df = data.randomSplit(weights=[0.8, 0.2], seed=123)

    # calculate PSI for only the numerical columns
    # ? choose columns to calculate psi
    columns = ['MonthlyCharges', 'TotalCharges']
    current = df[columns]
    psi_value, baseline_bins, current_bins = calculate_psi_multiple_features(baseline, current, columns)
    print(f"Calculated PSI: {psi_value}")

    if psi_value > psi_threshold:
        # TODO: store current bins
        print("Drift detected. Retraining the model...")
        latest_model = MultilayerPerceptronClassifier.load(latest_model_path)
        # Retrain the model
        train_model_neural_network(
            data_train=train_df,
            data_test=test_df,
            latest_model=latest_model,
            isRetrain=True
        )
    else:
        print("No significant drift detected. No retraining required.")