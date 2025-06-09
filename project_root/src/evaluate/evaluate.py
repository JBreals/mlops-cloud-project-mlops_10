import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from icecream import ic
import numpy as np
import os

# MLflow 설정 - CI 환경에서는 비활성화
ENABLE_MLFLOW = os.getenv('ENABLE_MLFLOW', 'true').lower() == 'true'

if ENABLE_MLFLOW:
    import mlflow


def evaluate_prophet(model_path, test_csv, run_id):
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv, parse_dates=['time'])
    test_data = df[['time', 'temp']].copy()
    test_data.columns = ['ds', 'y']
    forecast = model.predict(test_data[['ds']])
    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
    ic(mae, rmse)

    # MLflow 로깅 (CI 환경에서는 건너뛰기)
    if ENABLE_MLFLOW:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
    else:
        ic("MLflow disabled - skipping metric logging")

    return {'mae': mae, 'rmse': rmse}


def evaluate_sarimax(model_path, test_csv, run_id):
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv, parse_dates=['time'])
    y_true = df['temp'].values
    start = 0
    end = len(y_true) - 1

    ic("Running SARIMAX forecast...")
    y_pred = model.predict(start=start, end=end)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ic(mae, rmse)

    # MLflow 로깅 (CI 환경에서는 건너뛰기)
    if ENABLE_MLFLOW:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
    else:
        ic("MLflow disabled - skipping metric logging")

    return {'mae': mae, 'rmse': rmse}