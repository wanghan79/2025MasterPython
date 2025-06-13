import numpy as np

class Evaluator:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        return true_pos / (true_pos + false_pos + 1e-9)
    
    @staticmethod
    def recall(y_true, y_pred):
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        false_neg = np.sum((y_true == 1) & (y_pred == 0))
        return true_pos / (true_pos + false_neg + 1e-9)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        prec = Evaluator.precision(y_true, y_pred)
        rec = Evaluator.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec + 1e-9)
    
    def get_metrics(self, y_true, y_pred, task="regression"):
        if task == "regression":
            return {
                "MSE": self.mse(y_true, y_pred),
                "MAE": self.mae(y_true, y_pred)
            }
        else:
            return {
                "Accuracy": self.accuracy(y_true, y_pred),
                "Precision": self.precision(y_true, y_pred),
                "Recall": self.recall(y_true, y_pred),
                "F1 Score": self.f1_score(y_true, y_pred)
            }
