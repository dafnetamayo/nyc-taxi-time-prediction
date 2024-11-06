import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
import xgboost as xgb
from hyperopt.pyll import scope
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

@task(name="Read Data", retries=4, retry_delay_seconds=10)
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name = "Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name="Hyper-Parameter Tunning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv):
    mlflow.start_run()      
    
    def objective(params):
        with mlflow.start_run(nested=True):

            # Tag model
            mlflow.set_tag("model_family", "RandomForest")

            # Log parameters
            mlflow.log_params(params)

            # Train model
            rf_model = rfr(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                random_state=42
            )
            rf_model.fit(X_train, y_train)

            # Predict
            y_pred = rf_model.predict(X_val)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # Log RMSE
            mlflow.log_metric("rmse", rmse)

            # Calculate R²
            r2 = r2_score(y_val, y_pred)

            # Log R²
            mlflow.log_metric("r_squared", r2)
        
        return {'loss': rmse, 'status': STATUS_OK}

    # Define search space for Random Forest
    search_space = {
        'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    }

    # Hyperparameter optimization
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )

    # Convert best params to integers where necessary
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_samples_split"] = int(best_params["min_samples_split"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

    # Log best model parameters
    mlflow.log_params(best_params)

    mlflow.end_run()
    
    return best_params

@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best model ever (RF)"):
        # Log the best parameters
        mlflow.log_params(best_params)

        # Initialize and train the Random Forest model
        rf_model = rfr(
            n_estimators=int(best_params['n_estimators']),
            max_depth=int(best_params['max_depth']),
            min_samples_split=int(best_params['min_samples_split']),
            min_samples_leaf=int(best_params['min_samples_leaf']),
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = rf_model.predict(X_val)


        # Calculate R²
        r2 = r2_score(y_val, y_pred)
        mlflow.log_metric("r_squared", r2)


        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        # Save preprocessor (if applicable)
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        # Log the preprocessor as an artifact
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    return None

@task(name="Update Best Model")
def update_best_model(model_name) -> None:
    """update the best model"""

    df = mlflow.search_runs(order_by=["metrics.rmse"])
    run_id = df.loc[df['metrics.rmse'].idxmin()]['run_id']
    run_uri = f"runs:/{run_id}/model"

    from mlflow import MlflowClient
    client = MlflowClient()
    
    # Get the latest model version
    model_versions = client.search_model_versions("name='nyc-taxi-model-perfect'")
    latest_version = max(version.version for version in model_versions)
    
    client.set_registered_model_alias("nyc-taxi-model-perfect", "Challenger", latest_version)

@task(name="Champion vs Challenger")
def epic_fight(X_val, y_val):
    """Compare the champion model with the challenger and promote the challenger if it's better."""
    
    from mlflow import MlflowClient
    client = MlflowClient()

    # Fetch the latest version of the champion model
    champion_model = mlflow.pyfunc.load_model("models:/nyc-taxi-model-perfect/latest")
    
    # Predict using the current champion model
    y_pred_champion = champion_model.predict(X_val)
    champion_rmse = np.sqrt(mean_squared_error(y_val, y_pred_champion))
    
    # Fetch the challenger model (the latest run)
    df = mlflow.search_runs(order_by=["metrics.rmse"], max_results=1)
    challenger_run_id = df.loc[0, 'run_id']
    challenger_model = mlflow.pyfunc.load_model(f"runs:/{challenger_run_id}/model")
    
    # Predict using the challenger model
    y_pred_challenger = challenger_model.predict(X_val)
    challenger_rmse = np.sqrt(mean_squared_error(y_val, y_pred_challenger))
    
    # Compare RMSE of both models
    if challenger_rmse < champion_rmse:
        print(f"Challenger model wins with RMSE: {challenger_rmse} (Champion RMSE: {champion_rmse})")
        
        # Register the challenger as the new champion
        client.transition_model_version_stage(
            name="nyc-taxi-model-perfect",
            version=client.get_latest_versions("nyc-taxi-model-perfect", stages=["None"])[0].version,
            stage="Production"
        )
        print("Challenger promoted to Champion!")
    else:
        print(f"Champion retains the title with RMSE: {champion_rmse} (Challenger RMSE: {challenger_rmse})")
    
    return None

@flow(name="Main Flow")
def main_flow(year: str, month_train: str, month_val: str) -> None:
    """The main training pipeline"""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    # MLflow settings
    dagshub.init(url="https://dagshub.com/Pepe-Chuy/nyc-taxi-time-prediction", mlflow=True)
    
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # Hyper-parameter Tunning
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    
    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    # Compare Champion vs Challenger
    epic_fight(X_val, y_val)

main_flow("2024","01","02")






