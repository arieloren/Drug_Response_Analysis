import unittest
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from src.models.model_evaluation import ModelEvaluator

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)

        self.X_train = pd.DataFrame(X[:80])
        self.X_val = pd.DataFrame(X[80:])
        self.y_train = pd.Series(y[:80])
        self.y_val = pd.Series(y[80:])

        self.evaluator = ModelEvaluator(
            model=self.model,
            X_train_scaled=self.X_train,
            X_val_scaled=self.X_val,
            y_train=self.y_train,
            y_val=self.y_val
        )

    def test_accuracy_calculation(self):
        self.evaluator.predict()
        self.assertIsNotNone(self.evaluator.val_acc)
        self.assertGreaterEqual(self.evaluator.val_acc, 0.0)
        self.assertLessEqual(self.evaluator.val_acc, 1.0)

    def test_confusion_matrix(self):
        try:
            self.evaluator.plot_confusion_matrix()
        except Exception as e:
            self.fail(f"Confusion matrix plotting failed: {e}")

if __name__ == "__main__":
    unittest.main()
