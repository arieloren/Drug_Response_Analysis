import unittest
import pandas as pd
from src.preprocessing.preprocess import (
    concat_metadata_with_gene_expression,
    prepare_dataset_for_feature_selection,
    normalize_features
)

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta_data = pd.DataFrame({
            'SampleID': ['S1', 'S2'],
            'Response status': ['Responder', 'Non_responder'],
            'Gender': ['Male', 'Female']
        })
        cls.gene_data = pd.DataFrame({
            'ID_REF': ['G1', 'G2'],
            'S1': [1.2, 3.4],
            'S2': [5.6, 7.8]
        })

    def test_concat_metadata(self):
        merged = concat_metadata_with_gene_expression(self.meta_data, self.gene_data)
        self.assertEqual(merged.shape, (2, 5))
        self.assertIn('Response status', merged.columns)

    def test_dataset_preparation(self):
        merged = concat_metadata_with_gene_expression(self.meta_data, self.gene_data)
        X, y = prepare_dataset_for_feature_selection(merged)
        self.assertEqual(y.tolist(), [1, 0])
        self.assertNotIn('SampleID', X.columns)

    def test_normalization(self):
        df = pd.DataFrame({
            'Gene1': [1, 2, 3],
            'Gender': ['Male', 'Female', 'Male']
        })
        normalized = normalize_features(df)
        self.assertAlmostEqual(normalized['Gene1'].mean(), 0.0, places=5)
        self.assertIn('Gender', normalized.columns)

if __name__ == "__main__":
    unittest.main()
