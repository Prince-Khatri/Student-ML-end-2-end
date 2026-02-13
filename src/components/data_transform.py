import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.decorators import handle_exception
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    @handle_exception
    def get_data_preprocessor(self):
        """
            This Function is responsible for getting preprocessor object.
        """
        
        num_features = ['reading score', 'writing score']
        cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

        num_pipeline = Pipeline([
            # (name, transformation)
            ('imputer',SimpleImputer(strategy="median")),
            ('scaler',StandardScaler())
        ])
        logging.info('Numerical Features scaling done.')

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('one_hot_encoder',OneHotEncoder())
        ])
        logging.info('Categorical Features encoding done.')

        preprocessor = ColumnTransformer([
            # (name, object/Transformer, columns)
            ('numerical_tranformer',num_pipeline, num_features),
            ('categorical_tranformer',cat_pipeline, cat_features)
        ])
        logging.info("PreProcessor object created.")

        return preprocessor

    @handle_exception   
    def initiate_data_transformation(self, train_path, test_path):
        """
            This function uses preprocessor to transform train and test set.
        """
        logging.info("Reading traing and test dataset.")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info("Creating preprocessor.")

        preprocessor = self.get_data_preprocessor()

        target_column_name = 'math score'
        num_features = ['reading score', 'writing score']

        input_feature_train_df = train_df.drop(columns=[target_column_name])
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name])
        target_feature_test_df = test_df[target_column_name]

        logging.info("Applying the preprocessor object on traning and test set.")

        input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df) # fit and transform train, can also use .fit, .transform
        input_feature_test_arr = preprocessor.transform(input_feature_test_df)

        logging.info("Concatenating the transformed train and test input columns with target columns into np array.")
        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]

        test_arr = np.c_[
            input_feature_test_arr, np.array(target_feature_test_df)
        ]

        logging.info("Saving preprocessor object.")
        save_object(
            file_path = self.data_transformation_config.preprocessor_path,
            obj = preprocessor
        )


        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_path
        )
