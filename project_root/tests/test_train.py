# tests/test_train.py - 수정된 테스트 파일
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.train.train import train_prophet, train_sarimax


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터 생성"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    temps = 15 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 1, len(dates))
    df = pd.DataFrame({'time': dates, 'temp': temps})
    return df


@pytest.fixture
def temp_csv_file(sample_data):
    """임시 CSV 파일 생성"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


def test_train_prophet_without_mlflow(temp_csv_file):
    """MLflow 없이 Prophet 모델 훈련 테스트"""
    os.environ['ENABLE_MLFLOW'] = 'false'

    # train.py에서 import한 model_dir를 직접 패치
    with patch('src.train.train.model_dir') as mock_model_dir, \
            patch('src.train.train.ensure_dir') as mock_ensure_dir, \
            patch('joblib.dump') as mock_dump, \
            patch('mlflow.start_run'), \
            patch('mlflow.log_params'), \
            patch('mlflow.log_artifacts'), \
            patch('mlflow.end_run'):
        expected_path = '/tmp/test_model.pkl'
        mock_model_dir.return_value = expected_path

        model_path, run_id = train_prophet(temp_csv_file, model_name='test_model.pkl')

        assert model_path == expected_path
        assert run_id is not None
        mock_dump.assert_called_once()


def test_train_sarimax_without_mlflow(temp_csv_file):
    """MLflow 없이 SARIMAX 모델 훈련 테스트"""
    os.environ['ENABLE_MLFLOW'] = 'false'

    # train.py에서 import한 model_dir를 직접 패치
    with patch('src.train.train.model_dir') as mock_model_dir, \
            patch('src.train.train.ensure_dir') as mock_ensure_dir, \
            patch('joblib.dump') as mock_dump, \
            patch('mlflow.start_run'), \
            patch('mlflow.log_params'), \
            patch('mlflow.log_artifacts'), \
            patch('mlflow.end_run'):
        expected_path = '/tmp/test_sarimax_model.pkl'
        mock_model_dir.return_value = expected_path

        model_path, run_id = train_sarimax(
            temp_csv_file,
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            model_name='test_sarimax_model.pkl'
        )

        assert model_path == expected_path
        assert run_id is not None
        mock_dump.assert_called_once()