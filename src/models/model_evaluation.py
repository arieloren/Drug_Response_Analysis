from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
from utils.config import base_path  # Ensure base_path is correctly imported

def load_model(model_name="xgboost_model.pkl"):
    """
    Loads a trained model from the base_path/models/ directory.

    Parameters:
    model_name (str): The filename of the saved model.

    Returns:
    The loaded model.
    """

    #  Define model path
    model_path = os.path.join(base_path, "models", model_name)

    #  Check if the file exists before loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load and return the model
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    
    return model

def evaluate_model(model=None, X_train_scaled=None, X_val_scaled=None, y_train=None, y_val=None, generate_report=False):
    """
    Evaluates the trained model by printing accuracy and classification reports.

    Parameters:
    model: Trained classifier (XGBoost or other). If None, it will be loaded from disk.
    X_train_scaled (pd.DataFrame): Scaled training feature matrix.
    X_val_scaled (pd.DataFrame): Scaled validation feature matrix.
    y_train (pd.Series): Training labels.
    y_val (pd.Series): Validation labels.
    generate_report (bool): If True, prints classification reports.
    """

    # Load model only if it is not already provided
    if model is None:
        from models.model_traning import load_model  # Import inside to avoid circular imports
        model = load_model()
        print("Loaded model from disk.")

    y_train_pred = model.predict(X_train_scaled)

    y_val_pred = model.predict(X_val_scaled)

    # Predict on training and validation sets
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Compute and print accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f" Training Accuracy: {train_acc:.4f}")
    print(f" Validation Accuracy: {val_acc:.4f}")
    if generate_report:
        # Generate classification reports
        train_report = classification_report(y_train, y_train_pred, target_names=["Non-Responder", "Responder"])
        val_report = classification_report(y_val, y_val_pred, target_names=["Non-Responder", "Responder"])

        # Print Reports
        print("Classification Report - Training Data:\n", train_report)
        print("\nClassification Report - Validation Data:\n", val_report)








