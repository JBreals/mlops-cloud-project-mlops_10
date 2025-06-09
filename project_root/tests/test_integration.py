# tests/test_integration.py
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock


def test_full_pipeline_without_mlflow():
    """전체 파이프라인 통합 테스트 (MLflow 없이)"""
    os.environ['ENABLE_MLFLOW'] = 'false'

    # 테스트 데이터 생성
    dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
    temps = 15 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 1, len(dates))
    test_data = pd.DataFrame({'time': dates, 'temp': temps})

    # 데이터 분할
    split_idx = int(len(test_data) * 0.8)
    train_data = test_data[:split_idx]
    test_data_split = test_data[split_idx:]

    with tempfile.TemporaryDirectory() as temp_dir:
        train_csv = os.path.join(temp_dir, 'train.csv')
        test_csv = os.path.join(temp_dir, 'test.csv')

        train_data.to_csv(train_csv, index=False)
        test_data_split.to_csv(test_csv, index=False)

        # 모델 훈련
        from src.train.train import train_prophet
        from src.evaluate.evaluate import evaluate_prophet

        with patch('src.utils.utils.model_dir') as mock_model_dir, \
                patch('src.utils.utils.ensure_dir'), \
                patch('joblib.dump') as mock_dump, \
                patch('joblib.load') as mock_load:
            model_path = os.path.join(temp_dir, 'model.pkl')
            mock_model_dir.return_value = model_path

            # Mock Prophet 모델
            mock_model = MagicMock()
            mock_model.predict.return_value = pd.DataFrame({
                'yhat': np.random.normal(15, 2, len(test_data_split)),
                'yhat_lower': np.random.normal(13, 2, len(test_data_split)),
                'yhat_upper': np.random.normal(17, 2, len(test_data_split))
            })
            mock_load.return_value = mock_model

            # 훈련 및 평가
            model_path, run_id = train_prophet(train_csv)
            metrics = evaluate_prophet(model_path, test_csv, run_id)

            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert metrics['mae'] >= 0
            assert metrics['rmse'] >= 0