import unittest
import pandas as pd
import numpy as np
from models.model_evaluation import ModelEvaluator
from xgboost import XGBClassifier
from sklearn.datasets import make_classification

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=10)
        self.model = XGBClassifier().fit(X, y)
        self.evaluator = ModelEvaluator(
            model=self.model,
            X_train_scaled=pd.DataFrame(X[:80]),
            X_val_scaled=pd.DataFrame(X[80:]),
            y_train=pd.Series(y[:80]),
            y_val=pd.Series(y[80:]))
        )

    def test_accuracy_calculation(self):
        self.evaluator.predict()
        self.assertGreaterEqual(self.evaluator.val_acc, 0.0)
        self.assertLessEqual(self.evaluator.val_acc, 1.0)

    def test_confusion_matrix(self):
        # Just test that it runs without errors
        try:
            self.evaluator.plot_confusion_matrix()
        except Exception as e:
            self.fail(f"Confusion matrix plotting failed: {e}")