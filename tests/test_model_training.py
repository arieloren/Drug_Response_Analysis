import unittest
import pandas as pd
import numpy as np
from models.model_training import train_xgboost_classifier
from sklearn.datasets import make_classification

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=10)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)

    def test_xgboost_training(self):
        model = train_xgboost_classifier(self.X, self.y)
        self.assertEqual(model.n_estimators, 100)  # Check default params
        self.assertTrue(hasattr(model, 'fit'))