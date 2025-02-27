import pandas as pd
import numpy as np
import xgboost as xgb

def train_xgboost_classifier(X_train, y_train):
    """
    Trains an XGBoost classifier on the given dataset.

    Parameters:
    X_train (pd.DataFrame or np.array): Feature matrix for training.
    y_train (pd.Series or np.array): Target variable for training.

    Returns:
    xgb.XGBClassifier: The trained XGBoost model.
    """
    
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model
