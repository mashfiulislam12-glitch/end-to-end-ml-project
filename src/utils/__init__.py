import os 
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.exception import CustomExeption
from src.logger import logging
import dill
from sklearn.metrics import r2_score

load_dotenv()

def read_sql_data(query:str)-> pd.DataFrame:
    logging.info('reading data from postgresql')
    try:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        connection_url=f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}'
        engine=create_engine(connection_url)

        with engine.connect() as conn:
            df=pd.read_sql_query(query,conn)
        logging.info('successfully fetched data from postgresql')
        return df
    except Exception as e:
        raise CustomExeption(e,sys)




def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as f:
            dill.dump(obj,f)
        pass
    except Exception as e:
        raise CustomExeption(e,sys)





def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        tarin_data_report={}
        test_data_report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            test_data_report[list(models.keys())[i]]=test_model_score
        return (
        test_data_report
        )
    except Exception as e:
        raise CustomExeption(e,sys)
