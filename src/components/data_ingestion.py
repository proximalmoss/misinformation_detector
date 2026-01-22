import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifact","train.csv")
    test_data_path: str=os.path.join("artifact", "test.csv")
    raw_data_path: str= os.path.join("artifact", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method and component")

        try:
            fake_df= pd.read_csv("notebook/data/Fake.csv")
            real_df= pd.read_csv("notebook/data/True.csv")
            logging.info("Read fake and real new datasets as dataframes")

            fake_df['label']=0
            real_df['label']=1
            logging.info("Labels added to datasets")

            df=pd.concat([fake_df, real_df], axis=0, ignore_index=True)
            logging.info("Datasets combined sucessfully")

            df=df.drop_duplicates()
            logging.info("Duplicates removed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Copy of combined dataset saved")

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    x_train, y_train, x_test, y_test, preprocessor_path=data_transformation.initiate_data_transformation(train_data, test_data)