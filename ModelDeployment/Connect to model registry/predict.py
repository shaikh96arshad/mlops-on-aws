#%%writefile '/home/ec2-user/notebooks/scripts/predict.py'
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import mlflow
import boto3


from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://65.1.107.253:5000")
#RUN_ID = 'ebea9df24cd44f5a8617c8db2a4b09dd'
#logged_model = f'runs:/{RUN_ID}/model'
ssm = boto3.client('ssm',region_name='ap-south-1')
RUN_ID = ssm.get_parameter(Name='RUN_ID')['Parameter']['Value']
logged_model = f's3://ars-mlops-projects/mlflow/{RUN_ID}/artifacts/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
model = mlflow.pyfunc.load_model(logged_model)


def predictc(features):
    print("Inside predictc")
    predictions = model.predict(features)
    print(predictions)
    return predictions[0]
app = Flask('app')

@app.route('/predict',methods=['POST'])
def predict_endpoint():
    features = request.get_json()
    print("Inside predict_endpoint")
    print(features)
    feature_df = pd.DataFrame.from_dict([features])
    
    predictions = predictc(feature_df)
    print(predictions)
    
    result = {
        'Class' : str(predictions),
        'ModelVersion' : RUN_ID
    }
    print(result)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9697)
