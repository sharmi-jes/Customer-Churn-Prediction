import sys 
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=("artifacts/model.pkl")
            preprocessor_path=("artifacts/preprocessor.pkl")

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.fit_transform(features)
            prediction=model.predict(data_scaled)
            print(prediction)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
        
# numerical_cols=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    #    'IsActiveMember', 'EstimatedSalary']

        # categorical_cols=['Geography', 'Gender']
class CustomData:
    def __init__(self,CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,
                 Geography,Gender):
        self.CreditScore=CreditScore
        self.Age=Age
        self.Tenure=Tenure
        self.Balance=Balance
        self.NumOfProducts=NumOfProducts
        self.HasCrCard=HasCrCard
        self.IsActiveMember=IsActiveMember
        self.EstimatedSalary=EstimatedSalary
        self.Geography=Geography
        self.Gender=Gender

    def get_data_as_dataframe(self):
        try:
            input_data={
                "CreditScore":[self.CreditScore],
                "Age":[self.Age],
                "Tenure":[self.Tenure],
                "Balance":[self.Balance],
                "NumOfProducts":[self.NumOfProducts],
                "HasCrCard":[self.HasCrCard],
                "IsActiveMember":[self.IsActiveMember],
                "EstimatedSalary":[self.EstimatedSalary],
                "Geography":[self.Geography],
                "Gender":[self.Gender],



            }
            return  pd.DataFrame(input_data)
        
        except Exception as e:
            raise CustomException(e,sys)


        

        

