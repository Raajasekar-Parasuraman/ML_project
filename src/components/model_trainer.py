import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,model_evaluate


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Spliting training and test input data')
            Xtrain,ytrain,Xtest,ytest=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models={
                "Linear_regression":LinearRegression(),
                "Decision_tree":DecisionTreeRegressor(),
                "Random_forest":RandomForestRegressor(),
                "Ada_Boost":AdaBoostRegressor(),
                "Gradient_Boost":GradientBoostingRegressor(),
                "XGB":XGBRegressor()
                }
            
            model_trainer_report:dict=model_evaluate(Xtrain,ytrain,Xtest,ytest,models)

            model_trainer_report_df=pd.DataFrame(model_trainer_report.values(),index=model_trainer_report.keys(),columns=['train_score','test_score'])
            
            print(model_trainer_report_df)
            
            logging.info('Models builts and evaluated')


        except Exception as e:
            raise CustomException(e,sys)   

