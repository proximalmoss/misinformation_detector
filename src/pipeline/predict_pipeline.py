import sys
import pandas as pd
import numpy as np
import string
from scipy.sparse import hstack
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path='artifact/model.pkl'
            preprocessor_path='artifact/preprocessor.pkl'

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            tfidf_pipeline=preprocessor['tfidf_pipeline']
            num_pipeline=preprocessor['num_pipeline']
            numerical_columns=preprocessor['numerical_columns']

            features=self.engineer_features(features)

            text_features=features['text']
            numerical_features=features[numerical_columns]

            text_transformed=tfidf_pipeline.transform(text_features)
            numerical_transformed=num_pipeline.transform(numerical_features)

            data_scaled=hstack([text_transformed, numerical_transformed])

            preds=model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
    def engineer_features(self, df):
        try:
            df['text_length']=df['text'].apply(lambda x: len(str(x)))
            df['title_length']=df['title'].apply(lambda x: len(str(x)))
            df['word_count']=df['text'].apply(lambda x: len(str(x).split()))
            df['punct_count']=df['text'].apply(lambda x: sum([1 for c in str(x) if c in string.punctuation]))
            df['capital_count']=df['text'].apply(lambda x: sum([1 for c in str(x) if c.isupper()]))
            df['numeric_count']=df['text'].apply(lambda x: sum([1 for c in str(x) if c.isdigit()]))

            df['text_length']=df['text_length'].replace(0, np.nan)
            df['word_count']=df['word_count'].replace(0, np.nan)

            df['punct_density']=df['punct_density']/df['text_length']
            df['capital_density']=df['capital_count']/df['text_length']
            df['avg_word_length']=df['text_length']/df['word_count']

            df['avg_word_length']=df['avg_word_length'].replace([np.inf, -np.inf], np.nan)
            df['avg_word_length']=df['avg_word_length'].fillna(df['avg_word_length'].median())

            return df
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, title: str, text: str):
        self.title=title
        self.text=text
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "title": [self.title],
                "text": [self.text]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)