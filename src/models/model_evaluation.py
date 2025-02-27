from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
from utils.config import base_path  # Ensure base_path is correctly imported
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import numpy as np

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

def plot_confusion_matrix(y_val, y_val_pred):
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    # Define labels
    labels = ["Non-Responder", "Responder"]

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Validation Data")
    plt.show()


class ModelEvaluator:
    """
    A class to evaluate machine learning models with various metrics and visualizations.
    
    This class provides methods to evaluate model performance through accuracy metrics,
    classification reports, confusion matrices, and feature importance plots.
    """
    
    def __init__(self, model=None, X_train_scaled=None, X_val_scaled=None, y_train=None, y_val=None):
        """
        Initialize the ModelEvaluator with model and data.
        
        Parameters:
        model: Trained classifier (XGBoost or other). If None during evaluation, it will be loaded from disk.
        X_train_scaled (pd.DataFrame): Scaled training feature matrix.
        X_val_scaled (pd.DataFrame): Scaled validation feature matrix.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        """
        self.model = model
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_train = y_train
        self.y_val = y_val
        self.y_train_pred = None
        self.y_val_pred = None
        self.train_acc = None
        self.val_acc = None
    
    def load_model_if_needed(self):
        """
        Loads the model from disk if not already provided.
        """
        if self.model is None:
            from models.model_traning import load_model  # Import inside to avoid circular imports
            self.model = load_model()
            print("Loaded model from disk.")
            
    def predict(self):
        """
        Makes predictions on training and validation sets.
        """
        self.load_model_if_needed()
        
        self.y_train_pred = self.model.predict(self.X_train_scaled)
        self.y_val_pred = self.model.predict(self.X_val_scaled)
        
        # Calculate accuracy scores
        from sklearn.metrics import accuracy_score
        self.train_acc = accuracy_score(self.y_train, self.y_train_pred)
        self.val_acc = accuracy_score(self.y_val, self.y_val_pred)
        
    def print_accuracy(self):
        """
        Prints the training and validation accuracy scores.
        """
        if self.train_acc is None or self.val_acc is None:
            self.predict()
            
        print(f" Training Accuracy: {self.train_acc:.4f}")
        print(f" Validation Accuracy: {self.val_acc:.4f}")
        
    def generate_classification_report(self):
        """
        Generates and prints classification reports for training and validation data.
        """
        if self.y_train_pred is None or self.y_val_pred is None:
            self.predict()
            
        from sklearn.metrics import classification_report
        
        # Generate classification reports
        train_report = classification_report(self.y_train, self.y_train_pred, 
                                           target_names=["Non-Responder", "Responder"])
        val_report = classification_report(self.y_val, self.y_val_pred, 
                                         target_names=["Non-Responder", "Responder"])

        # Print Reports
        print("Classification Report - Training Data:\n", train_report)
        print("\nClassification Report - Validation Data:\n", val_report)
        
    def plot_confusion_matrix(self):
        """
        Plots a confusion matrix for the validation data predictions.
        """
        if self.y_val_pred is None:
            self.predict()
            
        # Assuming a plot_confusion_matrix function exists in the global scope
        # You may need to implement or import this function
        plot_confusion_matrix(self.y_val, self.y_val_pred)
        
    def plot_feature_importances(self, top_n=10):
        """
        Plots the feature importances of the model.
        
        Parameters:
        top_n (int): Number of top features to display
        """
        if self.model is None:
            self.load_model_if_needed()
            
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
            
        # Get feature importance values
        feature_importances = self.model.feature_importances_

        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'Feature': self.X_train_scaled.columns,
            'Importance': feature_importances
        })

        # Sort features by importance
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance_df["Importance"][:top_n], 
                   y=feature_importance_df["Feature"][:top_n], 
                   palette="viridis")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Name")
        plt.title("Most Important Features")
        plt.show()

    def analyze_instance_with_shap(self, instance_index):
        """
        Analyzes a specific instance using SHAP values.
        
        Parameters:
        instance_index (int): Index of the instance in validation set to analyze.
        """            
        # Get the instance
        instance = self.X_val_scaled.iloc[[instance_index]]
        actual_label = self.y_val.iloc[instance_index] if self.y_val is not None else "Unknown"

        # Create SHAP explainer and compute SHAP values
        explainer = shap.Explainer(self.model, self.X_train_scaled)
        shap_values = explainer(instance)  # ✅ This returns a shap.Explanation object

        # Make prediction
        prediction = self.model.predict(instance)[0]
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(instance)[0]
            probability_str = f", Probability: {probabilities[1]:.4f}" if len(probabilities) > 1 else ""
        else:
            probability_str = ""

        print(f"Instance #{instance_index}")
        print(f"Actual label: {actual_label}")
        print(f"Predicted label: {prediction}{probability_str}")

        shap.initjs()  # Required for Jupyter (won't affect scripts)
        shap.plots.force(shap_values, matplotlib=True)  # ✅ Force Matplotlib mode
        plt.show()  # ✅ Ensures the force plot is displayed

        # 🔹 Waterfall plot
        shap.plots.waterfall(shap_values[0])
        plt.show()  # ✅ Ensures the waterfall plot is displayed




    # def evaluate(self, print_accuracy=True, generate_report=False, 
    #                 plotting_confusion_matrix=False, plot_feature_importances=False,
    #                 analyze_instance_index=None, top_features=10):
    #     """
    #     Main method to evaluate the model with all selected metrics.
        
    #     Parameters:
    #     print_accuracy (bool): If True, prints accuracy metrics
    #     generate_report (bool): If True, prints classification reports
    #     plotting_confusion_matrix (bool): If True, plots confusion matrix
    #     plot_feature_importances (bool): If True, plots feature importances
    #     top_features (int): Number of top features to display in importance plot
    #     """
    #     # Make predictions first
    #     self.predict()
        
    #     # Run selected evaluation methods
    #     if print_accuracy:
    #         self.print_accuracy()
            
    #     if generate_report:
    #         self.generate_classification_report()
            
    #     if plotting_confusion_matrix:
    #         self.plot_confusion_matrix()
            
    #     if plot_feature_importances:
    #         self.plot_feature_importances(top_n=top_features)


    #     if analyze_instance_index is not None:
    #         self.analyze_instance_with_shap(analyze_instance_index)


