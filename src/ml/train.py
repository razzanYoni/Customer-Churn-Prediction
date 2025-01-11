import argparse
from ml import train_model_neural_network

def main():
    parser = argparse.ArgumentParser(description="Run Model Training")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset")
    args = parser.parse_args()

    train_model_neural_network(args.input_path, latest_model=None, isRetrain=False)

if __name__ == "__main__":
    main()
