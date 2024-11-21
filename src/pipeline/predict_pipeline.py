import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Preprocess input features
            data_scaled = preprocessor.transform(features)

            # Perform prediction
            prediction = model.predict(data_scaled)
            logging.info(f"Prediction successful: {prediction}")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
                 Geography, Gender):
        self.CreditScore = CreditScore
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary
        self.Geography = Geography
        self.Gender = Gender

    def get_data_as_dataframe(self):
        try:
            input_data = {
                "CreditScore": [self.CreditScore],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
                "NumOfProducts": [self.NumOfProducts],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
            }
            return pd.DataFrame(input_data)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Starting prediction pipeline...")

    # Example input using the CustomData class
    input_data = CustomData(
        CreditScore=750,
        Age=30,
        Tenure=5,
        Balance=50000,
        NumOfProducts=2,
        HasCrCard=1,
        IsActiveMember=1,
        EstimatedSalary=60000,
        Geography="France",
        Gender="Male"
    )

    # Convert input to DataFrame
    features = input_data.get_data_as_dataframe()

    # Initialize prediction pipeline
    pipeline = PredictPipeline()

    # Perform prediction
    try:
        prediction = pipeline.predict(features)
        logging.info(f"Prediction result: {prediction}")
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")

