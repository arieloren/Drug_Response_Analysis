import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def prepare_dataset_for_feature_selection(df):
    """
    Prepares the dataset for feature selection by:
    - Converting the "Response status" column to binary values (1 = Responder, 0 = Non-responder).
    - Separating features (X) and target variable (y).
    - Dropping non-numeric and irrelevant columns that are not useful for feature selection.

    Parameters:
    df (pd.DataFrame): The input dataset containing gene expression data and response labels.

    Returns:
    X (pd.DataFrame): The feature matrix containing gene expression values.
    y (pd.Series): The target variable (treatment response) encoded as 0 (Non-responder) or 1 (Responder).
    """
    
    # Convert Response status to binary (1 = Responder, 0 = Non-responder)
    temp_df = df.copy()
    temp_df["Response status"] = temp_df["Response status"].map({"Responder": 1, "Non_responder": 0})
    temp_df["Gender"] = pd.Categorical(temp_df["Gender"])

    # Separate features (X) and target variable (y)
    X = temp_df.drop(columns=["Response status"])
    y = temp_df["Response status"].astype(int)

    # Drop irrelevant columns
    drop_cols = ["SampleID", "Tissue", "disease state",'protocol']
    
    # drop_cols += ["disease activity score (das28)", "Gender"]

    X = X.drop(columns=drop_cols, errors="ignore")
    
    return  X, y

def normalize_features(X):
    scaler = StandardScaler()
    
    # Separate 'Gender' column if it exists
    gender_col = X['Gender'] if 'Gender' in X else None
    
    # Drop 'Gender' before scaling
    X_numeric = X.drop(columns=['Gender'], errors='ignore')

    # Scale only numerical features
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), 
                            columns=X_numeric.columns, 
                            index=X.index)

    # Add 'Gender' column back if it was removed
    if gender_col is not None:
        X_scaled['Gender'] = gender_col.map({"Male": 1, "Female": 0}).astype(int)  # Simple binary encoding
    return X_scaled


