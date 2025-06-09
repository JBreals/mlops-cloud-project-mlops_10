from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
import psycopg2
import requests
import time

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

class ModelUploadRequest(BaseModel):
    exp_name: str
    run_id: str
    pkl_file: str

# 환경변수
INFERENCE_TRIGGER_URL = f"{os.getenv('INFERENCE_URL')}/run_inference"
DB_HOST = os.getenv("DB_HOST", "serving-db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "serving")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# 데이터 불러오기
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

# FastAPI 앱 생성
app = FastAPI()
app.state.df = None
app.state.daily = None

# ✅ 수정된 Rate Limiting 미들웨어
request_log = []
REQUEST_LIMIT = 200
TIME_WINDOW = 60

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    global request_log

    now = time.time()
    request_log = [t for t in request_log if now - t < TIME_WINDOW]

    if len(request_log) >= REQUEST_LIMIT:
        print("🚨 Rate limit exceeded!")
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    request_log.append(now)
    return await call_next(request)

# 엔드포인트 정의
@app.get("/")
def hello():
    return {"message": "hello world 8000"}

#from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
#from fastapi.responses import Response

#@app.get("/metrics")
#def metrics():
    #return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
        app.state.df = get_base_df()
        app.state.daily = get_daily_df(app.state.df)
        return {
            "status": "success",
            "df": app.state.df.to_dict(orient="records") if app.state.df is not None else None,
            "daily": app.state.daily.to_dict(orient="records") if app.state.daily is not None else None,
        }
    except requests.exceptions.RequestException as e:
        return {"status": 500, "error": str(e)}
