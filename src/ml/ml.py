import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime

mlflow.set_tracking_uri("http://localhost:5000")

def preprocess_data(data):
    """Preprocess the dataset by encoding categorical variables and scaling numerical features."""
    # Encode categorical variables
    data.drop(columns=['customerID'], inplace=True)

    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]

    for col in categorical_columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Scale numerical features
    numerical_columns = ['MonthlyCharges', 'TotalCharges']
    data[numerical_columns] = StandardScaler().fit_transform(data[numerical_columns])

    target_column = [
    'Churn'
    ]
    data['Churn'] = LabelEncoder().fit_transform(data['Churn'])

    return data

def train_initial_model(X_train, y_train, X_test, y_test):
    """Train the initial logistic regression model and log it to MLflow."""
    with mlflow.start_run(run_name="Initial_Model_Training"):
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Log model parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Log model metrics
        accuracy = accuracy_score(y_test, model.predict(X_test))
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained with accuracy: {accuracy}, ROC AUC: {roc_auc}")

def calculate_psi(baseline, current, bins=10):
    """Calculate Population Stability Index (PSI) for a given feature."""
    baseline_bins = np.histogram(baseline, bins=bins)[0] / len(baseline)
    current_bins = np.histogram(current, bins=bins)[0] / len(current)

    psi = np.sum((baseline_bins - current_bins) * np.log(baseline_bins / current_bins))
    return psi

def monitor_drift_and_retrain(baseline, current, psi_threshold, X_train, y_train, X_test, y_test):
    """Monitor drift using PSI and retrain the model if necessary."""
    psi_value = calculate_psi(baseline, current)
    print(f"Calculated PSI: {psi_value}")

    # Log PSI to MLflow
    with mlflow.start_run(run_name="Drift_Monitoring"):
        mlflow.log_metric("PSI", psi_value)

        if psi_value > psi_threshold:
            print("Drift detected. Retraining the model...")

            # Retrain the model
            with mlflow.start_run(run_name="Model_Retraining"):
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)

                # Log retrained model parameters
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("max_iter", 1000)

                # Log retrained model metrics
                accuracy = accuracy_score(y_test, model.predict(X_test))
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("roc_auc", roc_auc)

                # Log the retrained model
                mlflow.sklearn.log_model(model, "model", registered_model_name="CustomerChurnModel")

                print(f"Retrained model logged with accuracy: {accuracy}, ROC AUC: {roc_auc}")
        else:
            print("No significant drift detected. No retraining required.")

def analyze_correlation(data):
    """Analyze and print correlation between features."""
    correlation_matrix = data.corr()
    print("Feature Correlation Matrix:\n", correlation_matrix)
    return correlation_matrix

# Example usage
if __name__ == "__main__":
    # Load data from CSV
    data = pd.read_csv("./data/raw/manual-delete-data.csv")

    # Preprocess data
    data = preprocess_data(data)
    # Analyze correlation
    analyze_correlation(data)

    # Split into features and target
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

    # Train initial model
    train_initial_model(X_train, y_train, X_test, y_test)

    # Monitor drift and retrain if necessary
    monitor_drift_and_retrain(X_train['MonthlyCharges'], X_test['MonthlyCharges'], psi_threshold=0.1, 
                              X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
