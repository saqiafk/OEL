from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer

@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list
    target_names: list

def load_dataset() -> DatasetBundle:
    """
    Load the Breast Cancer Wisconsin dataset from scikit-learn.
    Returns a DatasetBundle with X (features) and y (labels).
    """
    ds = load_breast_cancer(as_frame=True)
    X = ds.data
    y = ds.target
    return DatasetBundle(
        X=X, y=y, feature_names=list(ds.feature_names), target_names=list(ds.target_names)
    )