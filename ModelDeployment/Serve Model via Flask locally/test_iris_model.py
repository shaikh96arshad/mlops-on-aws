import predict
import pandas as pd
import numpy as np
import requests

"""df  = pd.DataFrame.from_dict([{
    'sepal length (cm)': 5.1,
    'sepal width (cm)' : 3.5,
    'petal length (cm)': 1.4,
    'petal width (cm)' : 0.2
}])"""
features = {
    'sepal length (cm)': 5.1,
    'sepal width (cm)' : 3.5,
    'petal length (cm)': 1.4,
    'petal width (cm)' : 0.2
}

url = 'http://localhost:9697/predict'
print(features)
response = requests.post(url,json=features)
print(response.json)

#predictions = predict.predict(df)
#print(predictions[0])
