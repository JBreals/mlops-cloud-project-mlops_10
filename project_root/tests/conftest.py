# tests/conftest.py - 강화된 설정
import pytest
import os
from unittest.mock import patch


@pytest.fixture(autouse=True)
def setup_test_environment():
    """테스트 환경 설정"""
    # MLflow 비활성화
    os.environ['ENABLE_MLFLOW'] = 'false'
    os.environ['CI'] = 'true'

    # 테스트 디렉토리 생성
    test_dirs = ['data/dataset', 'data/models', 'data/results']
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # MLflow 전역 mock
    with patch('mlflow.tracking.MlflowClient'), \
         patch('mlflow.start_run'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_artifacts'), \
         patch('mlflow.end_run'):
        yield

    # 테스트 후 정리 (필요한 경우)
    pass