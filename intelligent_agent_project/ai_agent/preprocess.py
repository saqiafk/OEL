from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def make_preprocess_pipeline(numeric_features) -> ColumnTransformer:
    # Only numeric features in this dataset; scale them.
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)], remainder="drop"
    )
    return preprocessor

def train_val_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)