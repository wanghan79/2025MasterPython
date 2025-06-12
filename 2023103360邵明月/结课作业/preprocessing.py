import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, strategy="standard"):
        self.strategy = strategy
        self.scaler = None
        self.imputer = None
        
    def handle_missing_values(self, X, strategy="mean"):
        self.imputer = SimpleImputer(strategy=strategy)
        return self.imputer.fit_transform(X)
    
    def scale_features(self, X, fit=False):
        if self.strategy == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
            
        if fit:
            self.scaler = scaler.fit(X)
            return self.scaler.transform(X)
        return self.scaler.transform(X)
    
    def encode_categorical(self, X, categorical_cols):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoded = encoder.fit_transform(X[categorical_cols])
        
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        
        numerical_df = X.drop(categorical_cols, axis=1)
        return pd.concat([numerical_df, encoded_df], axis=1)
    
    def feature_selection(self, X, y, method="variance", threshold=0.1):
        if method == "variance":
            variances = X.var(axis=0)
            selected_features = variances[variances > threshold].index
            return X[selected_features]
        
        return X
