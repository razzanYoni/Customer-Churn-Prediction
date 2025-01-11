import argparse

import mlflow.artifacts
from ml import train_initial_model, retrain_model
import mlflow

mlflow.set_tracking_uri("http://mlflow_server:5000")

def get_latest_model():
    client = mlflow.tracking.MlflowClient()
    model_name = "CustomerChurnModel"
    try:
        latest_version_info = client.get_latest_versions(model_name, stages=["None"])
        if latest_version_info:
            latest_model_version = latest_version_info[0].version
            print(f"Latest model version: {latest_model_version}")
            model_uri = client.get_model_version_download_uri(model_name, latest_model_version)
            return model_uri
        else:
            print("No versions found for the model.")
            return None
    except:
        print("No versions found for the model.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run Model Training")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
    args = parser.parse_args()

    latest_model_uri = get_latest_model()

    if latest_model_uri is not None:
        retrain_model(args.input_path, latest_model_uri)
    else:
        train_initial_model(args.input_path)

if __name__ == "__main__":
    main()
