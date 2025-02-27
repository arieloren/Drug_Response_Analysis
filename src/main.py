import os
# os.chdir("..")  # Moves one directory up
import pandas as pd 
import numpy as np 
from utils.config import base_path
from preprocessing.preprocess import concat_metadata_with_gene_expression,prepare_dataset_for_feature_selection,normalize_features
from preprocessing.feature_selection import getting_best_features
from sklearn.model_selection import train_test_split
from models.model_traning import train_xgboost_classifier
from models.model_evaluation import evaluate_model

def main():
    print("🚀 Starting Drug Response Analysis Pipeline...")
    # 1️⃣ Load and Preprocess Data
    print("📊 Loading and preprocessing data...")
    gene_expression = pd.read_csv(base_path/ "gene_expression.csv")
    meta_data = pd.read_csv(base_path/ "meta_data.csv")

    df = concat_metadata_with_gene_expression(meta_data,gene_expression,filter_nan_target=True)
    X,y = prepare_dataset_for_feature_selection(df)

    # Step 1: Split into 80% Train, 20% (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Step 2: Split the remaining 20% into 10% Validation, 10% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    # 2️⃣ Feature Selection
    print("🧬 Selecting top features...")
    selected_features= list(getting_best_features(X_train,y_train))

    # 3️⃣ Train Model
    print("🤖 Training model...")
    selected_features = selected_features + ["disease activity score (das28)", "Gender"]
    X_train,X_val= X_train[selected_features],X_val[selected_features]
    X_train_scaled = normalize_features(X_train)
    X_val_scaled = normalize_features(X_val)
    model = train_xgboost_classifier(X_train_scaled,y_train)

    # 4️⃣ Evaluate Model
    print("📈 Evaluating model...")
    evaluate_model(model,X_train_scaled,X_val_scaled,y_train,y_val)

    # # 5️⃣ Explain Model Predictions
    # print("🔍 Explaining model predictions...")
    # explain_model_predictions(model, X_test_selected)

    # print("✅ Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
