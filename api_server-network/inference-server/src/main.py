from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
import boto3
import joblib
import pandas as pd
import psycopg2


class ModelUploadRequest(BaseModel):
    exp_name: str
    run_id: str
    pkl_file: str


app = FastAPI()

# 🔐 환경 변수 기반 (AWS는 모델 다운로드용)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "ap-northeast-2"

# 🔐 PostgreSQL 환경 변수
DB_HOST = os.getenv("DB_HOST", "serving-db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "serving")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

os.makedirs("model", exist_ok=True)

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run_inference")
def run_inference(request: ModelUploadRequest):
    exp_name = request.exp_name
    run_id = request.run_id
    pkl_file = request.pkl_file

    BUCKET_NAME = "mlops-weather"
    MODEL_S3_KEY = (
        f"data/deploy_volume/model/{exp_name}/{run_id}/artifacts/model/artifacts/{pkl_file}"
    )
    LOCAL_MODEL_PATH = os.path.join("model", pkl_file)

    try:
        # 1. 모델 다운로드
        s3.download_file(BUCKET_NAME, MODEL_S3_KEY, LOCAL_MODEL_PATH)

        # 2. 추론
        model = joblib.load(LOCAL_MODEL_PATH)
        future = pd.date_range(start=pd.Timestamp.now(), periods=168, freq="H")
        df_future = pd.DataFrame({"ds": future})
        forecast = model.predict(df_future)
        result = forecast[["ds", "yhat"]].copy()
        result.columns = ["datetime", "pred_temp"]
        result["datetime"] = pd.to_datetime(result["datetime"])
        result["datetime"] = result["datetime"].dt.floor("H")

        # 3. DB 연결
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()

        # 4. 테이블 없으면 생성
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                datetime TIMESTAMP UNIQUE,
                pred_temp FLOAT
            );
        """)
        conn.commit()

        # 5. 결과 삽입
        for _, row in result.iterrows():
            cur.execute("""
                INSERT INTO predictions (datetime, pred_temp)
                VALUES (%s, %s)
                ON CONFLICT (datetime)
                DO UPDATE SET pred_temp = EXCLUDED.pred_temp;
            """, (row["datetime"], row["pred_temp"]))
        conn.commit()

        cur.close()
        conn.close()

        return {
            "status": "success",
            "rows": len(result)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
