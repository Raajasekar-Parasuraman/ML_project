import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def model_evaluate(xtrain,ytrain,xtest,ytest,models):
    try:
        report=dict()
        for i in models.keys():
            model=models[i]
            model.fit(xtrain,ytrain)
            pred_train=model.predict(xtrain)
            pred_test=model.predict(xtest)
            train_score=r2_score(ytrain,pred_train)
            test_score=r2_score(ytest,pred_test)
            report[i]=[train_score,test_score]
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
    