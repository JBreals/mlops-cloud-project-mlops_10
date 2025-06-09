from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path


def load_forecast(path):
    app = FastAPI()
    CSV_PATH = Path(path)
    @app.get("/forecast")
    def get_forecast():
        if not CSV_PATH.exists():
            return JSONResponse(status_code=404, content={"error": "Prediction file not found"})
        df = pd.read_csv(CSV_PATH)
        return df.to_dict(orient="records")

def load_clothing(path):
    app = FastAPI()
    CSV_PATH = Path(path)
    @app.get("/clothing")
    def get_clothing():
        if not CSV_PATH.exists():
            return JSONResponse(status_code=404, content={"error": "Clothing file not found"})
        df = pd.read_csv(CSV_PATH)
        return df.to_dict(orient="records")