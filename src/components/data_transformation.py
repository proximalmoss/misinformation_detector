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
    def __init__(self):
        preprocessor_obj_file_path= os.path.join("artifact", "preprocessor.pkl")

    def get_data_transformer_object(self):
        try:
            text_feature='text'
            numerical_columns=['text_length','title_length', 'word_count', 'punct_density', 'capital_density', 'avg_word_length']

            tfidf_piepline=TfidfVectorizer(max_features=3000, ngram_range=(1,2))
            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("TF-IF vectorication and numerical scaling pipelines created")

            return tfidf_piepline, num_pipeline, numerical_columns
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

            df['punct_density']=df['punct_count']/df['text_length']
            df['capital_density']=df['capital_count']/df['text_length']

            df['avg_word_length']=df['text_length']/df['word_count']

            df['avg_word_length']=df['avg_word_length'].replace([np.inf, -np.inf], np.nan)
            df['avg_word_length']=df['avg_word_length'].fillna(df['avg_word_length'].median())

            le=LabelEncoder()
            df['subject_encoded']=le.fit_transform(df['subject'])

            logging.info("Feature engineering completed")

            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def inititate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")

            train_df=self.engineer_features()
        except:
            pass