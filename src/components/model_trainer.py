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
            logging.info('Models builts and evaluated')
            model_trainer_report_df=pd.DataFrame(model_trainer_report.values(),index=model_trainer_report.keys(),columns=['train_score','test_score'])
            

            model_trainer_report_df['fit']=model_trainer_report_df['train_score']-model_trainer_report_df['test_score']
            model_trainer_report_df['fit']=model_trainer_report_df['fit'].apply(lambda x: 'good_fit' if x<0.1 else 'over_fit')
            best_model_score=max(model_trainer_report_df[(model_trainer_report_df['fit']=='good_fit')]['train_score'])

            best_model_name=model_trainer_report_df[model_trainer_report_df['train_score']==best_model_score].index[0]

            best_model_obj=models[best_model_name]

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model_obj)
            
            print(best_model_name,model_trainer_report_df,sep='\n')
            # print(model_trainer_report_df)
            logging.info('Best model selected')


        except Exception as e:
            raise CustomException(e,sys)   

