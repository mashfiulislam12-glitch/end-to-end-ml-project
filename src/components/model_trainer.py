import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import (mean_absolute_error,r2_score)
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomExeption
from src.logger import logging
from src.utils import save_obj,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')

@dataclass
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_data,test_data):
        try:
            X_train,X_test,y_train,y_test=(
                train_data[:,:-1],
                test_data[:,:-1],
                train_data[:,-1],
                test_data[:,-1],
            )


            models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'Extra Trees': ExtraTreesRegressor(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor(verbose=-1),
            'CatBoost': CatBoostRegressor(verbose=False)
            }

            model_report_test_data:dict

            model_report_test_data=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report_test_data.values()))

            best_model_name=list(model_report_test_data.keys())[
                list(model_report_test_data.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            save_obj(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2=r2_score(y_test,predicted)
            print(r2)
            print("Train shape:", X_train.shape)
            print("Test shape:", X_test.shape)

            print("Train R2:", r2_score(y_train, best_model.predict(X_train)))
            print("Test R2:", r2_score(y_test, best_model.predict(X_test)))
        except Exception as e:
            raise CustomExeption(e,sys)