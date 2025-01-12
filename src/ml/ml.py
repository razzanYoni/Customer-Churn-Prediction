import mlflow
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


# https://arize.com/blog-course/population-stability-index-psi/
# https://github.com/mwburke/population-stability-index

def train_model_neural_network(data_train, data_test, isRetrain=False):
    with mlflow.start_run(run_name="Initial_Model_Training" if not not isRetrain else "Retrain_Model"):
        # get weights from the latest model
        mlp = MultilayerPerceptronClassifier(
            layers = [data_train.schema["features"].metadata["ml_attr"]["num_attrs"], 10, 45, 20, 2],
            seed = 123,
            maxIter = 400,
            labelCol = "label",
            featuresCol="features",
        )

        mlp_model = mlp.fit(data_train)

        # Log model parameters
        mlflow.log_param("model_type", "NeuralNetwork")

        # Log model metrics
        pred_df = mlp_model.transform(data_test)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        mlpacc = evaluator.evaluate(pred_df)
        mlflow.log_metric("accuracy", mlpacc)

        mlflow.spark.log_model(mlp_model, "model", registered_model_name="CustomerChurnModel")

        return mlp_model

def train_initial_model(input_data_path):
    spark = SparkSession.builder.appName("Train Model").getOrCreate()
    mlflow.pyspark.ml.autolog()
    df = spark.read.parquet(input_data_path, header=True, inferSchema=True)

    # Split into features and target
    train_df, test_df = df.randomSplit(weights=[0.8, 0.2], seed=123)

    # Train the initial model
    train_model_neural_network(data_train=train_df, data_test=test_df)
    print("Model trained successfully.")

# https://stackoverflow.com/questions/70111193/how-can-i-load-the-latest-model-version-from-mlflow-model-registry
# https://medium.com/swlh/pysparks-multi-layer-perceptron-classifier-on-iris-dataset-dcf70d553cd8 
def retrain_model(input_data_path):
    spark = SparkSession.builder.appName("Retrain Model").getOrCreate()
    mlflow.pyspark.ml.autolog()
    df = spark.read.parquet(input_data_path, header=True, inferSchema=True)

    # Split into features and target
    train_df, test_df = df.randomSplit(weights=[0.8, 0.2], seed=123)

    # Retrain the model
    train_model_neural_network(
        data_train=train_df,    
        data_test=test_df,
        isRetrain=True
    )
    print("Model retrained successfully.")