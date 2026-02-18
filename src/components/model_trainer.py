import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self, x_train, y_train, x_test, y_test):
        try:
            logging.info("Received training and test input data")

            models={
                "Logistic Regression":LogisticRegression(max_iter=1000),
                "Decision Tree":DecisionTreeClassifier(),
                "Random Forest":RandomForestClassifier(),
                "XGBClassifier":XGBClassifier(eval_metric='logloss'),
                "CatBoost Classifier":CatBoostClassifier(verbose=False),
                "AdaBoost Classifier":AdaBoostClassifier()
            }
            params={
                "Logistic Regression": {'C': [1.0, 10.0]},
                "Decision Tree": {'max_depth': [10, 20]},
                "Random Forest": {'n_estimators': [100], 'max_depth': [20]},
                "XGBClassifier": {'learning_rate': [0.1], 'n_estimators': [100], 'max_depth': [7]},
                "CatBoost Classifier": {'depth': [8], 'learning_rate': [0.1], 'iterations': [100]},
                "AdaBoost Classifier": {'learning_rate': [0.1], 'n_estimators': [100]}
            }
            
            model_report: dict=evaluate_models(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params
            )

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.7:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            predicted=best_model.predict(x_test)
            accuracy=accuracy_score(y_test, predicted)

            return accuracy
        except Exception as e:
            raise CustomException(e, sys)