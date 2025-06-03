import sys, os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numeric_features = ['writing score', 'reading score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Categorical columns scaling done")
            
            logging.info("Numerical columns encoding done")

            # Combining numeric and categorical pipelines into a single preprocessor object
            # to apply transformations to respective columns in the dataset.
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("categorical_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return(
                preprocessor
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiaite_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")
            
            logging.info("Obtaining preprocessing object")

            preprocessor = self.get_data_transformer_object()

            target_col = 'math score'
            numeric_features = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns = [target_col], axis = 1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns = [target_col], axis = 1)
            target_feature_test_df = test_df[target_col]

            logging.info("appplying the transformation on training and test data")

            # Learns parameters from the data (fit) and applies the transformation (transform) in one step
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            # Only applies the transformation using parameters already learned from a prior fit()
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saving preprocessing object')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)