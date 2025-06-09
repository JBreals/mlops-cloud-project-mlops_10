import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# MLflow 설정
mlflow.set_tracking_uri("http://localhost:5050")  # log (mlruns) 저장 위치
mlflow.set_experiment("demo_experiment")  # 실험 이름

# 실험 시작
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = mse ** 0.5

    # 로그 기록
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)

    # 모델 저장 (아티팩트 루트는 MLflow UI 실행 시 이미 /app/model로 지정됨)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"  # -> 실제로는 /app/model/model/ 이하에 저장됨
    )

    print(f"RMSE: {rmse:.4f}")
