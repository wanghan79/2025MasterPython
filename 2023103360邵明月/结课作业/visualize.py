import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def plot_learning_curve(train_scores, val_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label="Training Score")
        plt.plot(val_scores, label="Validation Score")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("learning_curve.png")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes=None):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
    
    @staticmethod
    def plot_feature_importance(importance, feature_names):
        sorted_idx = np.argsort(importance)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance)), importance[sorted_idx])
        plt.yticks(range(len(importance)), feature_names[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance Plot")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
