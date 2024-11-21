import sys
import os
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.utils import save_object,evaluate_model_with_smote
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE



@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info("take the x_train,y_train,x_tets,y_test data")
            x_train,y_train,x_test,y_test=(
                 train_array[:,:-1],
                 train_array[:,-1],
                 test_array[:,:-1],
                 test_array[:,-1]


            )

            logging.info("take the models")
            models={
                "LogisticRegression":LogisticRegression(),
                "SVC":SVC(),
                "RandomForestClassifier":RandomForestClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "DecisionTreeClassifier":DecisionTreeClassifier()
            }

            smote = SMOTE(random_state=42)
            x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)


            model_report:dict = evaluate_model_with_smote(x_train_resampled, y_train_resampled, x_test, y_test, models)

            
            logging.info("best score calculated")
            best_score=max(sorted(model_report.values()))
            
            
            best_name=list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]
            
            logging.info("find the best model")
            best_model=models[best_name]

            print(best_model)
            
            logging.info("check the relation")
            if best_score<0.6:
                raise CustomException("Not found a good model")
            logging.info("we get a good model  both on training and testing data")

            save_object(
                file_path=self.trainer_config.model_file_path,
                obj=best_model
            )
            
            logging.info("Make the prediction")
            prediction=best_model.predict(x_test)
            print(prediction)

            logging.info("calculate accuracy score")
            score=accuracy_score(y_test,prediction)
            print("testing score is",score)

            return score
        
        except Exception as e:
            raise CustomException(e,sys)

        
           