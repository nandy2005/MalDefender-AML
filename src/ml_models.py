#Machine Learning Models for Malware Detection

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
import lightgbm as lgb
import joblib

class MalwareDetector:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        
    def build_model(self, input_dim=None, **kwargs): 
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 10),
                learning_rate=kwargs.get('learning_rate', 0.1),
                num_leaves=kwargs.get('num_leaves', 31),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        print(f"Built {self.model_type} model")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        print(f"Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
    
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_proba = self.model.predict_proba(X)

        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        elif y_proba.shape[1] == 1:
            y_proba = np.column_stack([1 - y_proba[:, 0], y_proba[:, 0]])
        return y_proba[:, 1]


    
    def evaluate(self, X_test, y_test, return_metrics=False):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy':accuracy_score(y_test, y_pred),
            'precision':precision_score(y_test, y_pred, zero_division=0),
            'recall':recall_score(y_test, y_pred, zero_division=0),
            'f1':f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':roc_auc_score(y_test, y_proba),
            'confusion_matrix':confusion_matrix(y_test, y_pred)
        }
        
        print("\n")
        print(f"Model:{self.model_type.upper()}")
        print("\n")
        print(f"Accuracy:{metrics['accuracy']:.4f}")
        print(f"Precision:{metrics['precision']:.4f}")
        print(f"Recall:{metrics['recall']:.4f}")
        print(f"F1-Score:{metrics['f1']:.4f}")
        print(f"ROC-AUC:{metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\n")
        
        if return_metrics:
            return metrics
        
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath + '.pkl')
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath + '.pkl')
        print(f"Model loaded from {filepath}")


#trial run
"""if __name__ == "__main__":
    train_data = np.load('data/train_processed.npz')
    test_data = np.load('data/test_processed.npz')
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    # Train Random Forest
    rf_detector = MalwareDetector('rf')
    rf_detector.build_model(n_estimators=200, max_depth=20)
    rf_detector.train(X_train, y_train)
    rf_detector.evaluate(X_test, y_test)
    rf_detector.save_model('models/rf_detector')

    #Train lightGBM
    lgb_detector = MalwareDetector('lightgbm')
    lgb_detector.build_model(n_estimators=200, max_depth=10, learning_rate=0.1, num_leaves=31)
    lgb_detector.train(X_train, y_train)
    lgb_detector.evaluate(X_test, y_test)
    lgb_detector.save_model('models/lgb_detector')
    
    print("\nAll models trained and saved!")"""