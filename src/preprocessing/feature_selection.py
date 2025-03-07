# import os
# os.chdir("..")  # Moves one directory up
# import importlib
# importlib.import_module("preprocessing")

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from preprocessing.preprocess import normalize_features

def train_lasso_logistic_regression(X_train, y_train):
    """
    Trains a Lasso-regularized Logistic Regression model with cross-validation.
    """
    lasso = LogisticRegressionCV(Cs=10, penalty="l1", solver="liblinear", cv=5, random_state=42)
    lasso.fit(X_train, y_train)
    return lasso

def get_significant_genes(lasso_model, gene_names):
    """
    Extracts significant genes based on non-zero coefficients from Lasso regression.
    """
    lasso_coeffs = lasso_model.coef_.flatten()  # Get coefficients
    important_genes = pd.DataFrame({"Gene": gene_names, "Coefficient": lasso_coeffs})
    important_genes = important_genes[important_genes["Coefficient"] != 0].sort_values(by="Coefficient", ascending=False)
    
    return important_genes

def getting_best_features(X, y,num_features=10):

    drop_cols = ["disease activity score (das28)", "Gender"] #metadata columns

    X = X.drop(columns=drop_cols, errors="ignore")
    # Step 5: Train Lasso Logistic Regression
    X_scaled = normalize_features(X)
    lasso_model = train_lasso_logistic_regression(X_scaled.values, y)

    # Step 6: Extract significant genes
    significant_genes = get_significant_genes(lasso_model, X.columns)

    top_features = significant_genes.reindex(significant_genes['Coefficient'].abs().sort_values(ascending=False).index).head(num_features)

    return top_features.Gene