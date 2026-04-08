import os
import sys
import numpy as np
import pandas as pd
import dill
from .exception import CustomException
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from .logger import logging
from .decorators import handle_exception

@handle_exception
def save_object(file_path, obj):
    """ 
        This is a function to save an python object
    """
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)
    with open(file_path, 'wb') as file_obj:
        dill.dump(obj, file_obj)

@handle_exception
def load_object(file_path):
    with open(file_path, 'rb') as file_obj:
        object = dill.load(file_obj)
    return object
    

@handle_exception
def save_as_csv(file_name, file_path, columns ,rows):
    """
        This function create and saves dataframes from rows and columns provided.
    """
    df = pd.DataFrame(columns = columns, data = rows)
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)
    csv_file_path = os.path.join(dir_name,f"{file_name}_{len(os.listdir(dir_name))//2}.csv")
    df.to_csv(csv_file_path)
    

   



def evaluate_performance(y_true, y_pred):
    """
        This function returns performance of a model based on:-
        1. Mean Squared Error
        2. Root Mean Squared Error
        3. Mean Absolute Error
        4. r2 
        
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
        This is a function to train different models at once.
    """

    logging.info("Starting Evalution of models:")
    model_train_metrics = []
    model_test_metrics = []
    
    best_model = "Not Found"
    best_r2 = 0

    for model_name, model_object in models.items():
        logging.info(f"Training:{model_name}")
        
        param = params[model_name]
        gs = GridSearchCV(model_object, param, cv=3)# Finding best hyperparameter
        gs.fit(X_train, y_train)

        # model_object.fit(X_train,y_train)
        y_train_pred = gs.predict(X_train)
        y_test_pred = gs.predict(X_test)

        model_train_metrics.append([model_name] + list(evaluate_performance(y_train, y_train_pred)))
        model_test_metrics.append([model_name] + list(evaluate_performance(y_test, y_test_pred)))

        if best_r2<model_test_metrics[-1][-1]:
            best_r2 = model_test_metrics[-1][-1]
            best_model_name = model_name
            best_model = gs.best_estimator_
    
    # Can apply threshold over metrics for best model
    # if best_r2<0.6:
    #   raise CustomException("No best model found")

    logging.info(f"All Models trained and best model is {best_model_name} with r2 = {best_r2}")
    return model_train_metrics, model_test_metrics, best_model










        


