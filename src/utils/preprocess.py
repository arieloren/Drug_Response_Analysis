import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def concat_metadata_with_gene_expression(metadata, gen_df, filter_nan_target=True):
    """
    Transposes the gene expression DataFrame, merges it with metadata, 
    and optionally drops rows with missing values in the 'Response status' column.
    Parameters:
    - metadata (DataFrame): A DataFrame containing sample metadata with a 'SampleID' column.
    - gen_df (DataFrame): A gene expression DataFrame that includes an 'ID_REF' column.
    - filter_nan_target (bool): If True, drop rows where 'Response status' is NaN.
    
    Returns:
    - merged_df (DataFrame): The merged DataFrame after transposition and filtering.
    """
    # Transpose the gene expression DataFrame and reset index
    df_transposed = gen_df.set_index('ID_REF').T.reset_index()
    # Rename the transposed index column to 'SampleID'
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # Merge the transposed gene expression data with metadata on 'SampleID'
    merged_df = pd.merge(df_transposed, metadata, on='SampleID', how='inner')
    
    # Optionally drop rows where 'Response status' is missing
    if filter_nan_target:
        merged_df = merged_df.dropna(subset=["Response status"])
    
    return merged_df

def prepare_dataset_for_feauture_selection(df):
    """
    Prepares the dataset by:
    - Dropping non-numeric columns.
    - Converting Response status to binary.
    - Handling missing values.
    """
    drop_cols = ["SampleID", "Tissue", "disease state", "protocol", "disease activity score (das28)", "Gender"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert Response status to binary (1 = Responder, 0 = Non-responder)
    df["Response status"] = df["Response status"].map({"Responder": 1, "Non_responder": 0})

    return df

def normalize_features(X):
    """
    Standardizes gene expression values (Z-score normalization).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

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

def evaluate_model(lasso_model, X_test, y_test):
    """
    Evaluates the trained Lasso Logistic Regression model on test data.
    """
    y_pred = lasso_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Responder", "Responder"])
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)