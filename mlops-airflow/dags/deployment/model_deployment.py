import sys
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import mlflow
from airflow.operators.python_operator import PythonOperator
import requests

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from plugins.utils import get_next_deployment_experiment_name


def generate_experiment_name(mlflow_uri,**context):
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = get_next_deployment_experiment_name()
    context['ti'].xcom_push(key='experiment_name', value=experiment_name)


# def request_model_loading(**context):
#     api_url = "http://api-server:8000/model_upload"  # 도커 네트워크 상에서 접근 가능
#     res = requests.get(api_url)
    
#     if res.status_code != 200:
#         raise Exception(f"Model load failed: {res.text}")
    
#     print("Model load response:", res.json())




with DAG(
    dag_id='ml_pipeline',
    start_date=datetime.now() - timedelta(days=1),
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 's3'],
) as dag:

    mlflow_uri = "http://mlflow-server:5050"
    api_url = "http://api-server:8000/model_upload"  # 도커 네트워크 상에서 접근 가능

    experiment_name = PythonOperator(
        task_id='generate_experiment_name',
        python_callable=generate_experiment_name,
        provide_context=True,
        op_kwargs={'mlflow_uri': mlflow_uri}
    )

    COMMON_ENV = {
      "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
      "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
      "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
      "MLFLOW_TRACKING_URI": mlflow_uri,
      "MLFLOW_ARTIFACT_LOCATION": f"s3://mlops-weather/data/deploy_volume/model",
      "API_URL": api_url
    }
    
    preprocess = DockerOperator(
    task_id='preprocess_data',
    image='jbreal/mlops-deployment:latest',
    platform='linux/amd64',  # Add this line
    api_version='auto',
    auto_remove=True,
    environment={
      **COMMON_ENV,
      "MLFLOW_EXPERIMENT_NAME": "{{ ti.xcom_pull(task_ids='generate_experiment_name', key='experiment_name') }}"
    },
    command='python src/preprocess/preprocess_deploy.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-net'
)

    train_prophet = DockerOperator(
    task_id='train_prophet',
    image='jbreal/mlops-deployment:latest',
    platform='linux/amd64',  # Add this line
    api_version='auto',
    auto_remove=True,
    environment={
      **COMMON_ENV,
      "MLFLOW_EXPERIMENT_NAME": "{{ ti.xcom_pull(task_ids='generate_experiment_name', key='experiment_name') }}"
    },
    command='python src/train/train_prophet_deploy.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-net'
)

    train_sarimax = DockerOperator(
    task_id='train_sarimax',
    image='jbreal/mlops-deployment:latest',
    platform='linux/amd64',  # Add this line
    api_version='auto',
    auto_remove=True,
    environment={
      **COMMON_ENV,
      "MLFLOW_EXPERIMENT_NAME": "{{ ti.xcom_pull(task_ids='generate_experiment_name', key='experiment_name') }}"
    },
    command='python src/train/train_sarimax_deploy.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-net',
    mem_limit='3g'
)

    evaluate_prophet = DockerOperator(
    task_id='evaluate_prophet',
    image='jbreal/mlops-deployment:latest',
    platform='linux/amd64',  # Add this line
    api_version='auto',
    auto_remove=True,
    environment={
      **COMMON_ENV,
      "MLFLOW_EXPERIMENT_NAME": "{{ ti.xcom_pull(task_ids='generate_experiment_name', key='experiment_name') }}"
    },
    command='python src/evaluate/evaluate_prophet_deploy.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-net'
)

    evaluate_sarimax = DockerOperator(
    task_id='evaluate_sarimax',
    image='jbreal/mlops-deployment:latest',
    platform='linux/amd64',  # Add this line
    api_version='auto',
    auto_remove=True,
    environment={
      **COMMON_ENV,
      "MLFLOW_EXPERIMENT_NAME": "{{ ti.xcom_pull(task_ids='generate_experiment_name', key='experiment_name') }}"
    },
    command='python src/evaluate/evaluate_sarimax_deploy.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-net'
)

    model_select = DockerOperator(
    task_id='model_select',
    image='jbreal/mlops-deployment:latest',
    platform='linux/amd64',  # Add this line
    api_version='auto',
    auto_remove=True,
    environment={
      **COMMON_ENV,
      "MLFLOW_EXPERIMENT_NAME": "{{ ti.xcom_pull(task_ids='generate_experiment_name', key='experiment_name') }}"
    },
    command='python src/model_select/modelselect_deploy.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-net'
)

    # request_model_loading = PythonOperator(
    #     task_id='request_model_loading',
    #     python_callable=request_model_loading,
    #     provide_context=True
    # )



    experiment_name >> preprocess >> train_prophet >> evaluate_prophet >> model_select
    experiment_name >> preprocess >> train_sarimax >> evaluate_sarimax >> model_select
