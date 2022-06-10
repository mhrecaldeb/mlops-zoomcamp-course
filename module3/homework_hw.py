from datetime import datetime
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import timedelta
from datetime import date

def read_data(path):
    df = pd.read_parquet(path)
    return df

def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


def get_paths(dt = None):  #date = date.today()

    if dt == None:
        dt = date.today()
        
        dt = dt.strftime("%Y-%m-%d")
        print("date is =", dt)
        

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


def main(date = "2021-08-15"):

    train_path, val_path = get_paths(date)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)



main()
