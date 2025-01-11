import argparse
import mlflow
import mlflow.keras as keras
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from pyspark.sql import SparkSession

parser = argparse.ArgumentParser(description="Data Drift Monitoring and Model Retraining")
parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--latest_model_path", type=str, required=True, help="Path to the model")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.autolog()

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

def train_model_neural_network(X_train, y_train, X_test, y_test, latest_model = None, isRetrain=False):
    """Train the initial neural network model and log it to MLflow."""
    with mlflow.start_run(run_name="Initial_Model_Training"):
        # get weights from the latest model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        if isRetrain:
            model.set_weights(latest_model.get_weights())
        model.compile(optimizer=keras.optimizers.Adam, loss=keras.losses.BinaryCrossentropy, metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Log model parameters
        mlflow.log_param("model_type", "NeuralNetwork")
        mlflow.log_param("epochs", 10)

        # Log model metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log the model
        mlflow.keras.log_model(model, "model", registered_model_name="CustomerChurnModel")

        if isRetrain:
            print(f"Retrained model logged with test loss: {test_loss}, test accuracy: {test_accuracy}")
        else:
            print(f"Model trained with test loss: {test_loss}, test accuracy: {test_accuracy}")

# https://stackoverflow.com/questions/70111193/how-can-i-load-the-latest-model-version-from-mlflow-model-registry
def monitor_drift_and_retrain(baseline, current, input_data_path, latest_model_path, psi_threshold = 0.1):
    spark = SparkSession.builder.appName("Train Model").getOrCreate()
    df = spark.read.parquet(input_data_path, header=True, inferSchema=True)

    # Split into features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=488)

    # calculate PSI for only the numerical columns
    columns = ['MonthlyCharges', 'TotalCharges']
    psi_values = calculate_psi_multiple_features(baseline, current, columns)
    # max of psi_values
    psi_value = np.max(list(psi_values.values()))
    print(f"Calculated PSI: {psi_value}")

    if psi_value > psi_threshold:
        print("Drift detected. Retraining the model...")
        latest_model = keras.models.load_model(latest_model_path)
        # Retrain the model
        with mlflow.start_run(run_name="Model_Retraining"):
            # Neural network model
            train_model_neural_network(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                latest_model=latest_model,
                isRetrain=True
            )
    else:
        print("No significant drift detected. No retraining required.")
