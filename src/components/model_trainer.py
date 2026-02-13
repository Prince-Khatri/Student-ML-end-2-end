import numpy as np
import pandas as pd
import sys,os
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_models, save_as_csv
from src.exception import CustomException
from src.logger import logging
from src.decorators import handle_exception

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts","model.pkl")
    model_history_path = os.path.join("artifacts","model_history","model_history.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    @handle_exception
    def initiate_model_traning(self, train_arr, test_arr):
        """
            This is function to Train and find best model from choosen models.
        """
        logging.info("Splitting traning and test dataset.")
        X_train, y_train, X_test, y_test = (
            train_arr[:,:-1], # just before last column, 2nd dimenshion
            train_arr[:,-1], # only last column, 2nd dimenshion
            test_arr[:,:-1],
            test_arr[:,-1]
        )

        logging.info("Chooing best choice of models available.")
        models = {
            "Linear Regression": LinearRegression(), 
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "AdaBoostRegressor": AdaBoostRegressor(), 
            "RandomForestRegressor": RandomForestRegressor(), 
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=False),
            "XGBRegressor": XGBRegressor(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
        }

        model_train_metrics, model_test_metrics, best_model = evaluate_models(
            X_train = X_train,
            y_train = y_train,
            X_test = X_test,
            y_test = y_test,
            models = models
        )
        logging.info("Saving train and test history.")
        metrics_name = ['MSE','RMSE','MAE','R2']
        save_as_csv(file_name = f"train_history", file_path = self.model_trainer_config.model_history_path,columns=['Algo Name']+metrics_name,rows=model_train_metrics)
        save_as_csv(file_name = f"test_history", file_path = self.model_trainer_config.model_history_path,columns=['Algo Name']+metrics_name,rows=model_test_metrics)

        logging.info("Saving best model and ")
        save_object(self.model_trainer_config.train_model_file_path,best_model)

        return self.model_trainer_config.train_model_file_path


