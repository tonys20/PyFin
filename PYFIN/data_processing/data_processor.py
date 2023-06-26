import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, numerical_features, categorical_features):

        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def fit_transform(self, data, standardize =True):
        data = data.copy()
        for feature in self.numerical_features:
            data[feature].fillna(data[feature].median(), inplace=True)
            data[feature] = (data[feature] - data[feature].mean())/data[feature].std()

        for feature in self.categorical_features:
            data[feature].fillna('missing', inplace =True)
            dummies = pd.get_dummies(data[feature], prefix = feature)
            data = pd.concat([data, dummies], axis = 1)
            data.drop(feature, axis = 1, inplace=True)

    def handle_missing_values(self, strategy="mean"):
        pass

    def detect_outliers(self, method="IQR"):
        pass

    def handle_outliers(self, strategy="drop"):
        pass

    def encode_categorical_variables(self, encoding_strategy="one_hot"):
        pass

    def scale_features(self, scaling_strategy="standardization"):
        pass

    def engineer_features(self, engineering_strategy="polynomial"):
        pass

    def train_test_split(self, data, test_size=0.2, random_state=None):
        data = data.copy()
        test_length = len(data)*test_size
        if random_state == None:
            train = data.iloc[:test_length]
            test = data.iloc[test_length:]

        else:
            range = range(len(data))
            inx = np.random.choice(range, len(data)*test_size, replace=False)
            test = data.loc[idx]
            train = data.iloc[~data.index.isin(idx)]
            
        return train, test 

