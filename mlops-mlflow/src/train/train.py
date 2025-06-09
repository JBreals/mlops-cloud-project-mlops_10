import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_model_with_autolog(model, model_name, X_train, X_test, y_train, y_test):

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    experiment_name = "japan_weather_prediction"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
    
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id) as run:
        print(f"\nStarting MLflow run for {model_name} (Run ID: {run.info.run_id})")

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)

        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)

        print(f"\n{model_name} Results:")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"MLflow UI link: {mlflow.get_tracking_uri()}#/experiments/{experiment_id}/runs/{run.info.run_id}")
