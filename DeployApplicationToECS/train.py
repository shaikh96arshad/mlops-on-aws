
def main():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
    
    iris_df['target']  = iris.target
    
    
    x = iris_df.drop('target',axis=1)
    y = iris_df['target']
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
    
    xgb_classifer = XGBClassifier(objective='binary:logistic')
    xgb_classifer.fit(X_train,y_train)
    y_pred = xgb_classifer.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    print("model accuracy is ")
    print(accuracy)
    xgb_classifer.save_model('model')
#     pickle.dump(xgb_classifer,open('models/model','wb'))
    print("Model export to " +  os.path.join(os.getcwd(),'model'))
    
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV,train_test_split
    import os
    from sklearn import datasets
    main()
    