import os
from preprocessing.preprocess import load_and_preprocess_data
from preprocessing.feature_selection import select_top_features
from models.train_model import train_and_save_model
from models.evaluate_model import evaluate_model
from models.explain_model import explain_model_predictions
from utils.config import DATA_PATH, MODEL_PATH

def main():
    print("🚀 Starting Drug Response Analysis Pipeline...")

    # 1️⃣ Load and Preprocess Data
    print("📊 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(os.path.join(DATA_PATH, "gene_expression.csv"),
                                                                os.path.join(DATA_PATH, "meta_data.csv"))

    # 2️⃣ Feature Selection
    print("🧬 Selecting top features...")
    X_train_selected, X_test_selected = select_top_features(X_train, X_test, y_train)

    # 3️⃣ Train Model
    print("🤖 Training model...")
    model = train_and_save_model(X_train_selected, y_train, MODEL_PATH)

    # 4️⃣ Evaluate Model
    print("📈 Evaluating model...")
    evaluate_model(model, X_test_selected, y_test)

    # 5️⃣ Explain Model Predictions
    print("🔍 Explaining model predictions...")
    explain_model_predictions(model, X_test_selected)

    print("✅ Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
