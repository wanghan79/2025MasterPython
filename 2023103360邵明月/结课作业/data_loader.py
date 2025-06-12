import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_dataset(self, dataset_type="classification"):
        if dataset_type == "classification":
            X, y = make_classification(
                n_samples=self.config["n_samples"],
                n_features=self.config["n_features"],
                n_classes=self.config["n_classes"],
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=self.config["n_samples"],
                n_features=self.config["n_features"],
                noise=self.config["noise"],
                random_state=42
            )
        return pd.DataFrame(X), pd.Series(y)
    
    def split_data(self, X, y, test_size=0.2):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        return (X.iloc[train_idx], X.iloc[test_idx],
                y.iloc[train_idx], y.iloc[test_idx])
    
    def get_feature_names(self, X):
        return [f"feature_{i}" for i in range(X.shape[1])]
