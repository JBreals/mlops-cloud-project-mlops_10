from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
import psycopg2
import requests

# ⛅ 옷 추천 기준 함수
def recommend(temp):
    if temp >= 28:
        return '반팔, 반바지, 샌들 (매우 더움)'
    elif temp >= 23:
        return '반팔, 긴바지, 운동화 (더움)'
    elif temp >= 18:
        return '긴팔, 긴바지 (적당함)'
    elif temp >= 12:
        return '긴팔, 니트, 자켓 (쌀쌀함)'
    elif temp >= 5:
        return '코트, 니트, 긴바지 (추움)'
    else:
        return '두꺼운 코트, 목도리 (매우 추움)'

# 🧾 모델 요청 바디 정의
class ModelUploadRequest(BaseModel):
    exp_name: str
    run_id: str
    pkl_file: str

# 📡 환경 변수 설정
INFERENCE_TRIGGER_URL = f"{os.getenv('INFERENCE_URL')}/run_inference"
DB_HOST = os.getenv("DB_HOST", "serving-db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "serving")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# 🔄 DB에서 예측 데이터 불러오기
def get_base_df():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        query = "SELECT datetime, pred_temp FROM predictions ORDER BY datetime LIMIT 168"
        df = pd.read_sql(query, conn)
        conn.close()

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        return df
    except Exception as e:
        print("DB Error:", e)
        return None

# 📅 일별 평균/최소/최대 및 옷 추천 가공
def get_daily_df(df):
    if df is None:
        return None
    daily = df.groupby('date').agg({
        'pred_temp': ['min', 'max', 'mean']
    })
    daily.columns = ['min_temp', 'max_temp', 'avg_temp']
    daily = daily.reset_index()
    daily['clothing'] = daily['avg_temp'].apply(recommend)
    return daily

# 🧩 FastAPI 앱 생성
app = FastAPI()
app.state.df = None
app.state.daily = None

@app.get("/")
def hello():
    return {"message": "hello world 8000"}

@app.get("/result_load")
def result_load():
    app.state.df = get_base_df()
    app.state.daily = get_daily_df(app.state.df)
    if app.state.df is None:
        return {"status": "error", "message": "No base data available"}
    if app.state.daily is None:
        return {"status": "error", "message": "No daily data available"}
    return {
        "status": "success",
        "df": app.state.df.to_dict(orient="records"),
        "daily": app.state.daily.to_dict(orient="records")
    }

@app.get("/current_data")
def current_data():
    if app.state.df is None:
        return {"status": "error", "message": "No base data available"}
    return {"status": "success", "df": app.state.df.to_dict(orient="records")}

@app.get("/forecast")
def get_forecast():
    if app.state.df is None:
        app.state.df = get_base_df()
    return app.state.df.to_dict(orient="records")

@app.get("/clothing")
def get_clothing():
    if app.state.daily is None:
        if app.state.df is None:
            app.state.df = get_base_df()
        app.state.daily = get_daily_df(app.state.df)
    return app.state.daily.to_dict(orient="records")

@app.post("/model_upload")
def model_upload(request: ModelUploadRequest):
    exp_name = request.exp_name
    run_id = request.run_id
    pkl_file = request.pkl_file
    try:
        response = requests.post(
            INFERENCE_TRIGGER_URL,
            json={"exp_name": exp_name, "run_id": run_id, "pkl_file": pkl_file}
        )
        # 추론 후 바로 데이터 불러오기
        app.state.df = get_base_df()
        app.state.daily = get_daily_df(app.state.df)
        return {
            "status": "success",
            "df": app.state.df.to_dict(orient="records") if app.state.df is not None else None,
            "daily": app.state.daily.to_dict(orient="records") if app.state.daily is not None else None,
        }
    except requests.exceptions.RequestException as e:
        return {"status": 500, "error": str(e)}

        # try:
#     response = s3.get_object(Bucket="mlops-weather", Key="data/deploy_volume/result/prediction.csv")
#     data = response['Body'].read().decode('utf-8')
#     df = pd.read_csv(StringIO(data))

#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df['date'] = df['datetime'].dt.date
#     daily = df.groupby('date').agg({
#         'pred_temp': ['min', 'max', 'mean']
#     })
#     daily.columns = ['min_temp', 'max_temp', 'avg_temp']
#     daily = daily.reset_index()

    
#     daily['clothing'] = daily['avg_temp'].apply(recommend)

# except s3.exceptions.NoSuchKey:
#     df = None