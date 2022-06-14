import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y)
    return z

if __name__ = "__main__":
    base_dir = '/opt/ml/preprocessing'
    df = pd.read_csv(
        f"{base_dir}/input/train.csv"
    )
    
    x_new = df.drop('price_range',axis=1)
    y_new = df['price_range']
    print(x_new.shape)
    print(y_new.shape)
    
    x_new['screen_res'] = x_new['px_width'] * x_new['px_height']
    x_new = x_new.drop(labels=['px_height','px_width'],axis=1)
    
    x_train_new,x_test_new,y_train_new,y_test_new = train_test_split(x_new,y_new,test_size=0.3,random_state=42)
    
    train = np.concat((y_train_new,x_train_new),axis=1)
    test  = np.concat((y_test_new,x_test_new),axis=1)
    
    train.to_csv(f"{base_dir}/train/train.csv",index=False)
    test.to_csv(f"{base_dir}/test/test.csv", index=False)
