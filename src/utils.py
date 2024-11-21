import sys
import os
from src.exception import CustomException
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        


def evaluate_model_with_smote(x_train, y_train, x_test, y_test, models):
    """
    Evaluate multiple models on training and testing data after applying SMOTE on the training data.
    
    Parameters:
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.
        models: Dictionary of model names and their corresponding instantiated models.

    Returns:
        Dictionary with model names as keys and their test accuracy scores as values.
    """
    try:
        report = {}
        
        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=42)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

        for model_name, model in models.items():
            # Train the model on the resampled training data
            model.fit(x_train_resampled, y_train_resampled)

            # Predictions on training and testing data
            y_train_pred = model.predict(x_train_resampled)
            y_test_pred = model.predict(x_test)

            # Calculate accuracy scores
            r2_score_train = accuracy_score(y_train_resampled, y_train_pred)
            r2_score_test = accuracy_score(y_test, y_test_pred)

            # Save the test accuracy score in the report
            report[model_name] = r2_score_test

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)