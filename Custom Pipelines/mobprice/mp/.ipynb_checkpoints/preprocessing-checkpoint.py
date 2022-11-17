
import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd
import pathlib
import logging
import boto3


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/Bank_Personal_Loan_Modelling.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)
    
    x_new = df.drop(['price_range'],axis=1)
    y_new = df['price_range']
    print(x_new.shape)
    print(y_new.shape)
    
    x_new['screen_res'] = x_new['px_width'] * x_new['px_height']
    x_new = x_new.drop(labels=['px_height','px_width','battery_power'],axis=1)
    
    x_train_new,x_test_new,y_train_new,y_test_new = train_test_split(x_new,y_new,test_size=0.3,random_state=42)
    
    train = pd.concat((y_train_new,x_train_new),axis=1)
    test  = pd.concat((y_test_new,x_test_new),axis=1)
    
    train.to_csv(f"{base_dir}/train/train.csv",index=False)
    test.to_csv(f"{base_dir}/test/test.csv", index=False)
