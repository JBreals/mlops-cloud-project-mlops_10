import pandas as pd
import joblib
from datetime import timedelta
from tqdm import tqdm
from icecream import ic
from src.utils.utils import dataset_dir
from datetime import datetime, timedelta
import os


def predict_future(model, last_date, days=7, save_name='future_temperature.csv'):
    # 🔽 1. 모델 경로일 경우 로드
    if isinstance(model, str):
        model = joblib.load(model)
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, "%Y-%m-%d")

    future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=days * 24, freq='h')
    future_df = pd.DataFrame({'ds': future_dates})
    ic(f"Predicting {len(future_dates)} future time points")

    # Prophet 모델의 경우 predict 메서드 사용
    if hasattr(model, 'predict'):
        forecast = model.predict(future_df)
        result = pd.DataFrame({
            'datetime': future_dates,
            'pred_temp': forecast['yhat'],
            'temp_min': forecast['yhat_lower'],
            'temp_max': forecast['yhat_upper'],
        })
    else:
        # SARIMAX 모델의 경우 다른 방식으로 예측
        # 이 경우 모델이 이미 fitted SARIMAX 결과여야 함
        forecast_values = model.forecast(steps=len(future_dates))
        result = pd.DataFrame({
            'datetime': future_dates,
            'pred_temp': forecast_values,
            'temp_min': forecast_values * 0.95,  # 간단한 신뢰구간 근사
            'temp_max': forecast_values * 1.05,
        })

    save_path = f"{dataset_dir()}/{save_name}"
    result.to_csv(save_path, index=False)
    ic(f"Saved future predictions to {save_path}")
    return save_path