import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transform import DataTransformation
from src.decorators import handle_exception
from src.components.model_trainer import ModelTrainer
PATH_TO_RAW_DATASET = 'notebooks/data/StudentsPerformance.csv'

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    @handle_exception
    def initiate_data_ingestion(self):
        """
            This function is for collection dataset from data source.
        """
        logging.info("Entering Data Ingestion method or component")

        df = pd.read_csv(PATH_TO_RAW_DATASET)
        logging.info("Read student data as dataframe")
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok = True)

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

        logging.info("Train Test Split initiated")
        
        train_set, test_set = train_test_split(df,test_size=0.2,random_state = 42, shuffle=True)

        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        logging.info("Ingestion of data is done")

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
        
        

if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformer = DataTransformation()

    train_arr, test_arr, path = data_transformer.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    path = trainer.initiate_model_traning(train_arr,test_arr)
    # print(path)

