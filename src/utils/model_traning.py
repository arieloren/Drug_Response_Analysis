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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


def train_knn_dt_ensemble(X_train, y_train):
    """
    Trains an ensemble classification model combining:
    - KNN (Local with small k)
    - KNN (Global with large k)
    - Decision Tree (small depth)
    - Uses majority voting for final prediction.

    Parameters:
    X_train (pd.DataFrame or np.array): Feature matrix for training.
    y_train (pd.Series or np.array): Target variable for training.

    Returns:
    VotingClassifier: The trained ensemble model.
    """

    # Define individual models
    knn_local = KNeighborsClassifier(n_neighbors=3)   # Local KNN (small k)
    knn_global = KNeighborsClassifier(n_neighbors=5) # Global KNN (large k)
    decision_tree = DecisionTreeClassifier(max_depth=2, random_state=42) # Small Decision Tree

    # Ensemble using majority voting
    ensemble_model = VotingClassifier(
        estimators=[
            ('knn_local', knn_local),
            ('knn_global', knn_global),
            ('decision_tree', decision_tree)
        ],
        voting='soft'  # Majority vote
    )

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    return ensemble_model
