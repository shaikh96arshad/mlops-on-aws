import json
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost
import logging

from sklearn import metrics
from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    logger.debug("Reading model")
    model = pickle.load(open("xgboost-model", "rb"))
    logger.debug("Reading test path")
    logger.info("Reading test file")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path)
    
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    #df.drop(index=0,axis=0,inplace=True)
    X_test = xgboost.DMatrix(data=df.values,label=y_test)
    
    predictions = model.predict(X_test)
    logger.debug("predictions done.")

    #acc = metrics.accuracy_score(y_test, predictions)
    #std = np.std(y_test - predictions)
    mse = mean_squared_error(y_out,y_test)
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
