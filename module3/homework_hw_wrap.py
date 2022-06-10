
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import timedelta
from datetime import date

import mlflow

from prefect import flow, task

from prefect import get_run_logger

from prefect.task_runners import SequentialTaskRunner

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):

    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        #prefect logger
        logger.info(print(f"The mean duration of training is {mean_duration}"))
    else:
        #prefect logger
        logger.info(print(f"The mean duration of validation is {mean_duration}"))
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical, date):

    logger = get_run_logger()

    with mlflow.start_run():
        mlflow.set_tag("model", "LinearRegression")
        # mlflow.log_params(params)

        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts) 
        y_train = df.duration.values

        logger.info(print(f"The shape of X_train is {X_train.shape}"))
        # get_run_logger
        logger.info(print(f"The DictVectorizer has {len(dv.feature_names_)} features"))
        # get_run_logger

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)
        
        logger.info(print(f"The MSE of training is: {mse}")) # get_run_logger

        mlflow.log_metric("Training mse", mse)
        #mlflow.log_params()

        with open(f"models/dv-{date}.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(f"models/dv-{date}.pkl", artifact_path="models_mlflow")

        with open(f"models/model-{date}.bin", "wb") as f_out2:
            pickle.dump(lr, f_out2)

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")

    return lr, dv

@task
def run_model(df, categorical, dv, lr):

    logger = get_run_logger()

    with mlflow.start_run():

        val_dicts = df[categorical].to_dict(orient='records')
        X_val = dv.transform(val_dicts) 
        y_pred = lr.predict(X_val)
        y_val = df.duration.values

        mse = mean_squared_error(y_val, y_pred, squared=False)

        logger.info(print(f"The MSE of validation is: {mse}"))
        mlflow.log_metric("Validation mse", mse)

    # get_run_logger
    return

@task
def get_paths(dt = None):  #date = date.today()
    logger = get_run_logger()

    if dt == None:
        dt = date.today()
        
        dt = dt.strftime("%Y-%m-%d")
        logger.info(print("date is =", dt)) #logger 
        
        

    months = ["01" ,"02" ,"03" ,"04" ,"05" ,"06" ,"07" ,"08" ,"09" ,"10" , "11", "12"]
    month = dt.split("-")[1]
    year = dt.split("-")[0]
    
    for i in range(12):
        if month == months[i]:
            if i > 1:
                month_tra = months[i-2]
                month_val = months[i-1]
                year_tra = year
                year_val = year
            else:
                month_tra = months[10+i]
                year_tra = str(int(year) - 1)
                if i > 0: 
                    month_val = months[0]
                    year_val = year
                else:
                    month_val = months[11]
                    year_val = str(int(year) - 1)

    date_tra = year_tra + "-" + month_tra
    date_val = year_val + "-" + month_val

    train_path = f'./data/fhv_tripdata_{date_tra}.parquet'
    val_path = f'./data/fhv_tripdata_{date_val}.parquet'

    return train_path, val_path      

@flow(task_runner=SequentialTaskRunner())
def main(date = "2021-08-15"):

    mlflow.set_tracking_uri("sqlite:///prefect_hw.db")
    mlflow.set_experiment("homework-nyc-taxi-experiment-prefect")

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical, date).result() #result() ojo date
    run_model(df_val_processed, categorical, dv, lr) 



main()
