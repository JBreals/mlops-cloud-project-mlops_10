import os
import random
import numpy as np
import boto3
import re


def init_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def project_path():
    # 프로젝트 루트: utils.py에서 두 단계 위
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def dataset_dir():
    return os.path.join(project_path(), "dataset")


def model_dir(model_name=None):
    base = os.path.join(project_path(), "models")
    if model_name:
        return os.path.join(base, model_name)
    return base


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def upload_to_s3(bucket, bucket_path, key, file_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=key.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=key.get("AWS_SECRET_ACCESS_KEY"),
        region_name="ap-northeast-2"
    )
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket, bucket_path)

    print(f"Uploaded {file_path} to {bucket}/{bucket_path}")


def load_from_s3(bucket, bucket_path, key, file_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=key.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=key.get("AWS_SECRET_ACCESS_KEY"),
        region_name="ap-northeast-2"
    )
    s3.download_file(bucket, bucket_path, file_path)


def get_next_deployment_experiment_name(base_name="deployment"):
    # MLflow가 비활성화된 경우 기본값 반환
    if os.getenv('ENABLE_MLFLOW', 'true').lower() == 'false':
        return f"{base_name}-1"

    from mlflow.tracking import MlflowClient
    import re

    client = MlflowClient()
    experiments = client.list_experiments()

    # 정규식으로 'deployment-N' 패턴 추출
    pattern = re.compile(f"^{base_name}-(\\d+)$")
    max_id = 0
    found = False

    for exp in experiments:
        match = pattern.match(exp.name)
        if match:
            found = True
            num = int(match.group(1))
            max_id = max(max_id, num)

    if not found:
        return f"{base_name}-1"
    else:
        return f"{base_name}-{max_id + 1}"