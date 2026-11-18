import os
import sys 
import numpy as np
import pandas as pd
from dataclasses import dataclass 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from src.logger import logging
from src.exception import CustomExeption
from src.utils import save_obj


@dataclass
class datatransformationconfig:
    preoprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

class data_transformation:
    def __init__(self):
        self.data_trans_config=datatransformationconfig()

    def get_data_trans_obj(self):
        try:
            cat_enc_cols=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
            scale_cols=['area']

            num_pipe=Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipe=Pipeline(
                steps=[
                    ('ohe',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
                ]
            )

            clf=ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipe,scale_cols),
                    ('cat_pipiline',cat_pipe,cat_enc_cols)
                ]
            )

            return clf
        except Exception as e:
            raise CustomExeption(e,sys)


    def initiate_transformation(self):
        try:
            train_data=pd.read_csv('artifacts/train.csv')
            test_data=pd.read_csv('artifacts/test.csv')

            preprocessor=self.get_data_trans_obj()

            target_col='price'

            train_features=train_data.drop(target_col,axis=1)
            train_target=train_data[target_col]

            test_features=test_data.drop(target_col,axis=1)
            test_target=test_data[target_col]

            train_features_arr=preprocessor.fit_transform(train_features)
            test_features_arr=preprocessor.transform(test_features)

            train_data_arr=np.c_[train_features_arr,np.array(train_target)]
            test_data_arr=np.c_[test_features_arr,np.array(test_target)]

            save_obj(
                self.data_trans_config.preoprocessor_obj_file,
                preprocessor
            )

        
            return(
                train_data_arr,
                test_data_arr,
                self.data_trans_config.preoprocessor_obj_file
            )
        except Exception as e:
            raise CustomExeption(e,sys)