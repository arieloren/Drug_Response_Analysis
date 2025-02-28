import os
os.chdir("..")  # Moves one directory up
from utils.config import base_path
import joblib  
import pandas as pd
import numpy as np
import xgboost as xgb

def train_xgboost_classifier(X_train, y_train, model_name="xgboost_model.pkl"):
    """
    Trains an XGBoost classifier and saves it in base_path.

    Parameters:
    X_train (pd.DataFrame or np.array): Feature matrix for training.
    y_train (pd.Series or np.array): Target variable for training.
    model_name (str): Name of the model file to save.

    Returns:
    xgb.XGBClassifier: The trained XGBoost model.
    """

    model = xgb.XGBClassifier(
        n_estimators=100,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    model_path = os.path.join(base_path, "models", model_name)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    #Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model
