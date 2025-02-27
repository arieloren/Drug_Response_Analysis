import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

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

def getting_best_features(X, y,feature_columns,num_features=10):
    # Step 5: Train Lasso Logistic Regression
    lasso_model = train_lasso_logistic_regression(X, y)

    # Step 6: Extract significant genes
    significant_genes = get_significant_genes(lasso_model, feature_columns)

    top_features = significant_genes.reindex(significant_genes['Coefficient'].abs().sort_values(ascending=False).index).head(num_features)

    return top_features.Gene