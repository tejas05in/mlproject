import os , sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting training and test input data')
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'XGBoost': XGBRegressor(),
                'Ada Boost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'CatBoosting': CatBoostRegressor(verbose=False),   
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            #To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            #To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score <0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model name is {best_model_name} and r2_score is {best_model_score}")
            save_object (
                obj=best_model,
                file_path=self.model_trainer_config.trained_model_file_path
            )
            
            predicted = best_model.predict(X_test)
            r2_sqaure = r2_score(y_test,predicted)
            return r2_sqaure
            
        
        except Exception as e:
            logging.info("Error occurred while initializing model trainer")
            raise CustomException(e,sys)
