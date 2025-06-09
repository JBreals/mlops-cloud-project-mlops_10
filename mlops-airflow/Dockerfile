FROM apache/airflow:2.8.1-python3.10

ENV AIRFLOW_HOME=/opt/airflow

USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /opt/airflow/datas

USER root