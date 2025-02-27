import unittest
import pandas as pd
import numpy as np
from preprocessing.feature_selection import getting_best_features
from sklearn.datasets import make_classification

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        X, y = make_classification(n_samples=100, n_features=20, n_informative=5)
        self.X = pd.DataFrame(X, columns=[f'Gene_{i}' for i in range(20)])
        self.y = pd.Series(y)

    def test_feature_selection(self):
        selected = getting_best_features(self.X, self.y, num_features=5)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(gene in self.X.columns for gene in selected))