import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model_name=list(models.keys())[i]
            para=params[model_name]

            rs=RandomizedSearchCV(model, para, cv=3, n_jobs=-1, verbose=1, n_iter=15, random_state=42, scoring='accuracy')
            rs.fit(x_train, y_train)

            best_model=rs.best_estimator_

            models[model_name]=best_model

            y_train_pred=best_model.predict(x_train)
            y_test_pred=best_model.predict(x_test)

            train_model_score=accuracy_score(y_train, y_train_pred)
            test_model_score=accuracy_score(y_test, y_test_pred)

            report[model_name]=test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)