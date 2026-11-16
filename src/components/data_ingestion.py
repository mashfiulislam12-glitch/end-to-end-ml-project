import os
import sys 
from src.utils import read_sql_data
from src.exception import CustomExeption
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')


class Data_Ingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('starting data ingestion')

        try:
            query= 'SELECT * FROM housing'

            df = read_sql_data(query)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train-test split completed successfully")

            return (
                "artifacts/train.csv",
                "artifacts/test.csv"
            )

        except Exception as e:
            logging.info('logging got an error')
            raise CustomExeption(e,sys)



if __name__=='__main__':
    obj=Data_Ingestion()
    obj.initiate_data_ingestion()