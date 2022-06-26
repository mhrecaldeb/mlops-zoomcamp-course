#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
# import sys


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr

def read_data(filename):
    
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def prepare_dict(df):
    categorical = ['PUlocationID', 'DOlocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts


# taxi_type = 'fhv'
# year = 2021
# month = 4

# df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet')


def fit_model(dv, lr, dicts):
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(y_pred.mean())
    return X_val, y_pred

def save_output(df, y_pred, year, month, taxi_type):
    df['ride_id'] = f"{year:04d}/{month:02d}_" + f"{df.index.astype('str')}"
    df['results'] = y_pred
    df_result = df

    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    

def run():

    taxi_type = 'fhv'
    year = 2021  # int(sys.argv[1])  #
    month = 4  # int(sys.argv[2])  #

    dv, lr = load_model()

    filename = f'https://nyc-tlc.s3.amazonaws.com/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'

    df = read_data(filename)

    dicts = prepare_dict(df)

    _, y_pred = fit_model(dv, lr, dicts)

    save_output(df, y_pred, year, month, taxi_type)

if __name__ == '__main__':
    run()

