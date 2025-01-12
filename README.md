# Customer Churn Prediction Ops Pipeline

## System Requirements

- Docker
- Docker Compose

## Setup

1. Clone the repository
2. move to infra directory `cd ./infra`
3. Run `docker compose up -d` to start the services
4. Airflow will be available at [http://localhost:8090](http://localhost:8090). Use username `admin` and password `admin` to login.
5. The DAGs are located in the `dags` directory.
6. The source code is located in the `src` directory.
7. MLFlow web UI will be available at [http://localhost:5001](http://localhost:5001).
8. You can run the DAG by triggering it from the Airflow web UI at [http://localhost:8090](http://localhost:8090).

## DAG

`drift_check.py` is the DAG that checks for simulating data drift. The pipelines are as follows:

1. `upload_to_hdfs`: Upload the raw data to HDFS.
2. `drift_simulation`: Simulate data drift by adding noise to the data.
3. `cleaning_data`: Clean the data.
4. `check_psi`: Check the data drift by calculating the PSI. It will check whether the workflow should be continued or not.
5. `train_model`: Train the model based on the new data. The trained model will be saved in the MLFlow server and can be monitored through the MLFlow web UI.

## Members

- 13521045 - Fakhri Muhammad Mahendra
- 13521087 - Razzan Daksana Yoni
- 13521101 - Arsa Izdihar Islam
