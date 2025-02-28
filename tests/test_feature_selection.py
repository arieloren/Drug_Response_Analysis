import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.preprocessing.feature_selection import getting_best_features

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=20, n_informative=5, random_state=42)
        self.X = pd.DataFrame(X, columns=[f"Gene_{i}" for i in range(20)])
        self.y = pd.Series(y)

    def test_feature_selection(self):
        selected = getting_best_features(self.X, self.y, num_features=5)
        self.assertEqual(len(selected), 5)
        for gene in selected:
            self.assertIn(gene, self.X.columns)

if __name__ == "__main__":
    unittest.main()
