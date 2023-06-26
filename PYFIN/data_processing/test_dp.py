# test_dp.py

import unittest
import pandas as pd
import numpy as np
from PyFin.PYFIN.data_processing.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DataProcessor(numerical_features=['age', 'income'], categorical_features=['gender', 'occupation'])
        self.df = pd.DataFrame({
            'age': [25, 30, 35, np.nan],
            'income': [50000, 60000, np.nan, 80000],
            'gender': ['male', 'female', np.nan, 'male'],
            'occupation': ['engineer', 'doctor', 'lawyer', 'engineer']
        })

    def test_fit_transform(self):
        df_processed = self.processor.fit_transform(self.df)

        # Check that missing values have been filled
        self.assertFalse(df_processed.isnull().any().any())

        # Check that numerical features have been standardized
        self.assertAlmostEqual(df_processed['age'].mean(), 0)
        self.assertAlmostEqual(df_processed['income'].mean(), 0)

        # Check that categorical features have been one-hot encoded
        self.assertTrue(set(['gender_male', 'gender_female', 'gender_missing', 'occupation_engineer', 'occupation_doctor', 'occupation_lawyer']).issubset(df_processed.columns))


    def test_split(self):
        train, test = self.processor.train_test_split(self.df)
        self.assertAlmostEqual(df_processed['age'].mean(), 0)
        self.assertAlmostEqual(df_processed['income'].mean(), 0)
        


if __name__ == '__main__':
    unittest.main()