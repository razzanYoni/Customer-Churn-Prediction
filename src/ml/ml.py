import mlflow
import mlflow.sklearn
import numpy as np
    
mlflow.set_tracking_uri("http://mlflow_server:5000")

# https://arize.com/blog-course/population-stability-index-psi/
# https://github.com/mwburke/population-stability-index
def calculate_psi(baseline, current, bins=10):
    """Calculate Population Stability Index (PSI) for a single feature."""
    baseline_bins = np.histogram(baseline, bins=bins)[0] / len(baseline)
    current_bins = np.histogram(current, bins=bins)[0] / len(current)

    # Avoid division by zero by adding a small value to bins
    baseline_bins = np.clip(baseline_bins, 1e-6, None)
    current_bins = np.clip(current_bins, 1e-6, None)

    psi = np.sum((baseline_bins - current_bins) * np.log(baseline_bins / current_bins))
    return psi

def calculate_psi_multiple_features(baseline, current, columns, bins=10):
    """Calculate PSI for multiple features."""
    psi_values = {}
    for col in columns:
        psi = calculate_psi(baseline[col], current[col], bins)
        psi_values[col] = psi
    return psi_values

def train_model_neural_network(input_data_path, latest_model=None, isRetrain=False):
    """Train the initial neural network model using PySpark's MultilayerPerceptronClassifier and log it to MLflow."""
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.sql import SparkSession
    
    # Create SparkSession
    spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

    mlflow.pyspark.ml.autolog()
    
    df = spark.read.parquet(input_data_path, header=True, inferSchema=True)

    train, test = df.randomSplit([0.8, 0.2], seed=488)
    
    with mlflow.start_run(run_name="Initial_Model_Training"):
        # Define the neural network architecture
        layers = [45, 64, 64, 2]  # Input layer, two hidden layers, output layer
        
        # Create and train the model
        mlp = MultilayerPerceptronClassifier(
            maxIter=400,
            layers=layers,
            blockSize=32,
            seed=488,
            featuresCol="features",
            labelCol="label"
        )
        
        if isRetrain and latest_model is not None:
            # In PySpark, we can't directly set weights like in Keras
            # We'll use the same architecture but retrain from scratch
            pass
            
        model = mlp.fit(train)
        
        # Make predictions on test data
        predictions = model.transform(test)
        
        # Evaluate the model
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        test_accuracy = evaluator.evaluate(predictions)
        
        # Log model parameters
        mlflow.log_param("model_type", "MultilayerPerceptronClassifier")
        mlflow.log_param("max_iterations", 10)
        mlflow.log_param("layers", layers)
        
        # Log model metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log the model
        mlflow.spark.log_model(model, "model", registered_model_name="CustomerChurnModel")
        
        if isRetrain:
            print(f"Retrained model logged with test accuracy: {test_accuracy}")
        else:
            print(f"Model trained with test accuracy: {test_accuracy}")
        
        return model

# https://stackoverflow.com/questions/70111193/how-can-i-load-the-latest-model-version-from-mlflow-model-registry
# def monitor_drift_and_retrain(baseline, current, input_data_path, latest_model_path, psi_threshold = 0.1):
#     spark = SparkSession.builder.appName("Train Model").getOrCreate()
#     df = spark.read.parquet(input_data_path, header=True, inferSchema=True)

#     # Split into features and target
#     X = df.drop(columns=['Churn'])
#     y = df['Churn']

#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=488)

#     # calculate PSI for only the numerical columns
#     columns = ['MonthlyCharges', 'TotalCharges']
#     psi_values = calculate_psi_multiple_features(baseline, current, columns)
#     # max of psi_values
#     psi_value = np.max(list(psi_values.values()))
#     print(f"Calculated PSI: {psi_value}")

#     if psi_value > psi_threshold:
#         print("Drift detected. Retraining the model...")
#         latest_model = keras.models.load_model(latest_model_path)
#         # Retrain the model
#         with mlflow.start_run(run_name="Model_Retraining"):
#             # Neural network model
#             train_model_neural_network(
#                 input_data_path=input_data_path,
#                 latest_model=latest_model,
#                 isRetrain=True
#             )
#     else:
#         print("No significant drift detected. No retraining required.")
