import sys 
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


logging.info("take the decorator class for not use the init methos to create a varible")
@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts/raw.csv")
    train_data_path:str=os.path.join("artifacts/train.csv")
    test_data_path:str=os.path.join("artifacts/test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def intiate_data_ingestion(self):
        try:
            logging.info("read the data")
            df=pd.read_csv(r"D:\RESUME ML PROJECTS\Customer Churn Prediction\notebooks\cleaned.csv")
            logging.info("create a directory for the arifacts folder")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info("pass the raw data to the folder")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)


            logging.info("split the data train and test data") 
            train_set,test_set=train_test_split(df,test_size=0.2)

            logging.info("pass the train data to the path")
            train_set.to_csv(self.ingestion_config.train_data_path)

            logging.info("pass the test data to the path")
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("return the train and test data for the next process")
            return (
                train_set,
                test_set
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.intiate_data_ingestion()

    transformation=DataTransformation()
    transformation.initiate_data_transformation(train_data,test_data)


