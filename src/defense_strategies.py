#Defense Strategies Against Adversarial Malware

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

class AdversarialDefense:
    def __init__(self):
        self.models = {}
        self.ensemble = None
        
    def adversarial_training(self, base_model, X_train, y_train,X_adversarial, y_adversarial, adversarial_ratio=0.3):

        print(f"\n{'='*60}")
        print("ADVERSARIAL TRAINING")
        print(f"{'='*60}")
        
        # Determine number of adversarial samples to include
        n_adversarial = int(len(X_train) * adversarial_ratio)
        n_adversarial = min(n_adversarial, len(X_adversarial))
        
        # Sample adversarial examples
        adv_indices = np.random.choice(len(X_adversarial), n_adversarial, replace=False)
        X_adv_sample = X_adversarial[adv_indices]
        y_adv_sample = y_adversarial[adv_indices]
        
        # Combine clean and adversarial data
        X_combined = np.vstack([X_train, X_adv_sample])
        y_combined = np.concatenate([y_train, y_adv_sample])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_combined))
        X_combined = X_combined[shuffle_idx]
        y_combined = y_combined[shuffle_idx]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Adversarial samples added: {n_adversarial}")
        print(f"Combined training set: {len(X_combined)}")
        print(f"Adversarial ratio: {n_adversarial/len(X_combined):.2%}")
        
        
        base_model.train(X_combined, y_combined)
        
        print("Adversarial training complete!")
        
        return base_model
    
    def feature_smoothing(self, X, smoothing_factor=0.1):
        # Apply Gaussian smoothing
        noise=np.random.normal(0, smoothing_factor, X.shape)
        X_smoothed=X + noise
        X_smoothed=np.clip(X_smoothed, 0, 10)
        return X_smoothed
    
    def evaluate_defense(self, model, X_clean, y_clean,X_adversarial, y_adversarial,defense_name="Defense"):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print(f"\n{'='*60}")
        print(f"EVALUATING {defense_name.upper()}")
        print(f"{'='*60}")
        
        # Evaluate on clean samples
        if hasattr(model, 'predict'):
            y_pred_clean = model.predict(X_clean)
        else:
            # For MalwareDetector instances
            y_pred_clean = model.predict(X_clean)
            
        
        acc_clean = accuracy_score(y_clean, y_pred_clean)
        prec_clean = precision_score(y_clean, y_pred_clean, zero_division=0)
        rec_clean = recall_score(y_clean, y_pred_clean, zero_division=0)
        f1_clean = f1_score(y_clean, y_pred_clean, zero_division=0)
        
        # Evaluate on adversarial samples
        if hasattr(model, 'predict'):
            y_pred_adv = model.predict(X_adversarial)
            
        else:
            y_pred_adv = model.predict(X_adversarial)
           
        
        acc_adv = accuracy_score(y_adversarial, y_pred_adv)
        prec_adv = precision_score(y_adversarial, y_pred_adv, zero_division=0)
        rec_adv = recall_score(y_adversarial, y_pred_adv, zero_division=0)
        f1_adv = f1_score(y_adversarial, y_pred_adv, zero_division=0)
        
        # Calculate robustness metrics
        accuracy_drop = acc_clean - acc_adv
        robustness_score = acc_adv / acc_clean if acc_clean > 0 else 0
        
        metrics = {
            'clean_accuracy': acc_clean,
            'clean_precision': prec_clean,
            'clean_recall': rec_clean,
            'clean_f1': f1_clean,
            'adversarial_accuracy': acc_adv,
            'adversarial_precision': prec_adv,
            'adversarial_recall': rec_adv,
            'adversarial_f1': f1_adv,
            'accuracy_drop': accuracy_drop,
            'robustness_score': robustness_score
        }
        
        print(f"\nClean Test Performance:")
        print(f"  Accuracy:{acc_clean:.4f}")
        print(f"  Precision:{prec_clean:.4f}")
        print(f"  Recall:{rec_clean:.4f}")
        print(f"  F1-Score:{f1_clean:.4f}")
        
        print(f"\nAdversarial Test Performance:")
        print(f"  Accuracy:{acc_adv:.4f}")
        print(f"  Precision:{prec_adv:.4f}")
        print(f"  Recall:{rec_adv:.4f}")
        print(f"  F1-Score:{f1_adv:.4f}")
        
        print(f"\nRobustness Metrics:")
        print(f"  Accuracy Drop:{accuracy_drop:.4f}")
        print(f"  Robustness Score:{robustness_score:.4f}")
        print(f"{'='*60}")
        
        return metrics
    