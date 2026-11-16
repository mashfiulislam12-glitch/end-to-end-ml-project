import os 
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.exception import CustomExeption
from src.logger import logging

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