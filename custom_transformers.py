import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.bounds = {}
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            low, high = self.bounds[col]
            X[col] = np.clip(X[col], low, high)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['is_senior'] = (X['age'] >= 65).astype(int)
        X['glucose_high_bp'] = X['blood_glucose_level'] * X['hypertension']
        return X