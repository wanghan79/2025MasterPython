from sklearn.model_selection import KFold
from models.linear_model import LinearRegression, LogisticRegression
from models.decision_tree import DecisionTree
from utils import log_progress
import numpy as np

class Trainer:
    def __init__(self, model_type="linear", task="regression"):
        self.model_type = model_type
        self.task = task
        self.models = []
        self.kfold = KFold(n_splits=5)
        
    def _init_model(self):
        if self.task == "regression":
            models = {
                "linear": LinearRegression,
                "tree": DecisionTree
            }
        else:
            models = {
                "linear": LogisticRegression,
                "tree": DecisionTree
            }
        return models[self.model_type]()
    
    def train(self, X, y):
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self._init_model()
            model.fit(X_train, y_train)
            
            if self.task == "regression":
                score = model.score(X_val, y_val)
            else:
                score = model.accuracy(X_val, y_val)
                
            fold_scores.append(score)
            self.models.append(model)
            
            log_progress(f"Fold {fold_idx+1} completed. Score: {score:.4f}")
        
        return fold_scores
    
    def predict(self, X):
        predictions = np.zeros((len(self.models), X.shape[0]))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)
        
        if self.task == "regression":
            return np.mean(predictions, axis=0)
        else:
            return np.round(np.mean(predictions, axis=0))
    
    def save_model(self, filepath):
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(self.models, f)
