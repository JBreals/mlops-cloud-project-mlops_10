import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from tqdm import tqdm
from icecream import ic
from src.utils.utils import model_dir, ensure_dir
import os
import uuid

# MLflow 설정 - CI 환경에서는 비활성화
ENABLE_MLFLOW = os.getenv('ENABLE_MLFLOW', 'true').lower() == 'true'

if ENABLE_MLFLOW:
    import mlflow


def train_prophet(
        train_csv,
        model_name='prophet_model.pkl',
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=15.0,
        **kwargs
):
    ensure_dir(model_dir())
    df = pd.read_csv(train_csv, parse_dates=['time'])
    data = df[['time', 'temp']].copy()
    data.columns = ['ds', 'y']

    # MLflow 또는 Mock run 처리
    if ENABLE_MLFLOW:
        run_context = mlflow.start_run()
    else:
        # Mock run_id 생성
        class MockRun:
            def __init__(self):
                self.info = type('obj', (object,), {'run_id': str(uuid.uuid4())})()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        run_context = MockRun()

    with run_context as run:
        # 파라미터 로깅 (MLflow가 활성화된 경우에만)
        if ENABLE_MLFLOW:
            mlflow.log_param("seasonality_mode", seasonality_mode)
            mlflow.log_param("daily_seasonality", daily_seasonality)
            mlflow.log_param("weekly_seasonality", weekly_seasonality)
            mlflow.log_param("yearly_seasonality", yearly_seasonality)
            mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
            mlflow.log_param("seasonality_prior_scale", seasonality_prior_scale)

        model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            **kwargs
        )

        ic("Fitting Prophet model...")
        model.fit(data)
        model_path = model_dir(model_name)
        joblib.dump(model, model_path)
        ic(f"Model saved to {model_path}")

        return model_path, run.info.run_id


def train_sarimax(
        train_csv,
        model_name="sarimax_model.pkl",
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 24),  # 시간 단위 seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
):
    ensure_dir(model_dir())
    df = pd.read_csv(train_csv, parse_dates=["time"])
    df = df.sort_values("time")

    y = df["temp"].values

    ic(f"Training SARIMAX model with order={order} and seasonal_order={seasonal_order}")

    # MLflow 또는 Mock run 처리
    if ENABLE_MLFLOW:
        run_context = mlflow.start_run()
    else:
        # Mock run_id 생성
        class MockRun:
            def __init__(self):
                self.info = type('obj', (object,), {'run_id': str(uuid.uuid4())})()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        run_context = MockRun()

    with run_context as run:
        # 파라미터 로깅 (MLflow가 활성화된 경우에만)
        if ENABLE_MLFLOW:
            mlflow.log_param("order", order)
            mlflow.log_param("seasonal_order", seasonal_order)
            mlflow.log_param("enforce_stationarity", enforce_stationarity)
            mlflow.log_param("enforce_invertibility", enforce_invertibility)

        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        results = model.fit(disp=False)
        model_path = model_dir(model_name)
        joblib.dump(results, model_path)
        ic(f"SARIMAX model saved to {model_path}")

        return model_path, run.info.run_id