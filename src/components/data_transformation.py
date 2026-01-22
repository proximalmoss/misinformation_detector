import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import string

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
        preprocessor_obj_file_path= os.path.join("artifact", "preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            text_feature='text'
            numerical_columns=['text_length','title_length', 'word_count', 'punct_density', 'capital_density', 'avg_word_length']

            tfidf_pipeline=TfidfVectorizer(max_features=3000, ngram_range=(1,2))
            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("TF-IF vectorication and numerical scaling pipelines created")

            return tfidf_pipeline, num_pipeline, numerical_columns
        except Exception as e:
            raise CustomException(e,sys)
        
    def engineer_features(self, df):
        try:
            df['text_length']=df['text'].apply(lambda x: len(str(x)))
            df['title_length']=df['title'].apply(lambda x: len(str(x)))
            df['word_count']=df['text'].apply(lambda x: len(str(x).split()))
            df['punct_count']=df['text'].apply(lambda x: sum([1 for c in str(x) if c in string.punctuation]))
            df['capital_count']=df['text'].apply(lambda x: sum([1 for c in str(x) if c.isupper()]))
            df['numeric_count']=df['text'].apply(lambda x: sum([1 for c in str(x) if c.isdigit()]))

            df['text_length'] = df['text_length'].replace(0, np.nan)
            df['word_count'] = df['word_count'].replace(0, np.nan)

            df['punct_density']=df['punct_count']/df['text_length']
            df['capital_density']=df['capital_count']/df['text_length']

            df['avg_word_length']=df['text_length']/df['word_count']

            df['avg_word_length']=df['avg_word_length'].replace([np.inf, -np.inf], np.nan)
            df['avg_word_length']=df['avg_word_length'].fillna(df['avg_word_length'].median())

            logging.info("Feature engineering completed")

            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")

            train_df=self.engineer_features(train_df)
            test_df=self.engineer_features(test_df)

            self.le=LabelEncoder()
            train_df['subject_encoded']=self.le.fit_transform(train_df['subject'])
            test_df['subject_encoded']=self.le.transform(test_df['subject'])

            logging.info("Obtaining preprocessing object")

            tfidf_pipeline, num_pipeline, numerical_columns=self.get_data_transformer_object()

            target_column_name='label'

            input_text_train=train_df['text']
            input_numerical_train=train_df[numerical_columns+['subject_encoded']]
            target_feature_train=train_df[target_column_name]

            input_text_test=test_df['text']
            input_numerical_test=test_df[numerical_columns+['subject_encoded']]
            target_feature_test=test_df[target_column_name]

            logging.info('Applying preprocessing object on train and test dataset')

            input_text_train_arr=tfidf_pipeline.fit_transform(input_text_train)
            input_text_test_arr=tfidf_pipeline.transform(input_text_test)

            input_numerical_train_arr=num_pipeline.fit_transform(input_numerical_train)
            input_numerical_test_arr=num_pipeline.transform(input_numerical_test)

            input_feature_train_arr=hstack([input_text_train_arr, input_numerical_train_arr])
            input_feature_test_arr=hstack([input_text_test_arr, input_numerical_test_arr])

            x_train = input_feature_train_arr
            y_train = target_feature_train.values
            x_test = input_feature_test_arr
            y_test = target_feature_test.values

            logging.info("Saved preprocessing object")

            preprocessor={
                'tfidf_pipeline':tfidf_pipeline,
                'num_pipeline':num_pipeline,
                'label_encoder':self.le,
                'numerical_columns':numerical_columns
            }

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return(x_train, y_train, x_test, y_test, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)