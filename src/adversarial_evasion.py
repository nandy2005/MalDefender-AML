#Adversarial Evasion Techniques for Malware Detection

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class AdversarialEvader:
    def __init__(self, feature_info):
        self.feature_info=feature_info
        self.categories=feature_info['categories']
    
    def _get_feature_range(self, category):
        return self.categories[category]
    
    def header_tampering(self, X, intensity=0.3):
        X_adv=X.copy().astype(float)
        start,end=self._get_feature_range('header')
        noise=np.random.normal(0,intensity, (X_adv.shape[0], end - start))
        X_adv[:, start:end]+=noise
        X_adv[:, start:end] = np.clip(X_adv[:, start:end], 0, 10)
        print(f"Applied header tampering with intensity {intensity}")
        return X_adv
    
    def entropy_manipulation(self, X, target_entropy=7.0):
        X_adv=X.copy()
        start,end=self._get_feature_range('byteentropy')
        current_entropy = X_adv[:, start:end]
        shift=(target_entropy-current_entropy.mean(axis=1, keepdims=True)) * 0.3
        X_adv[:, start:end]+=shift
        X_adv[:, start:end]=np.clip(X_adv[:, start:end], 0, 8)
        print(f"Applied entropy manipulation (target: {target_entropy})")
        return X_adv
    
    def string_obfuscation(self, X, obfuscation_rate=0.5):
        X_adv=X.copy()
        start,end=self._get_feature_range('strings')
        n_features=end-start
        n_modify=int(n_features*obfuscation_rate)
        for i in range(X_adv.shape[0]):
            indices=np.random.choice(range(start,end), n_modify, replace=False)
            X_adv[i,indices]*=np.random.uniform(0,0.3,n_modify)
        print(f"Applied string obfuscation (rate: {obfuscation_rate})")
        return X_adv
    
    def api_substitution(self, X, substitution_rate=0.3):
        X_adv=X.copy()
        start,end=self._get_feature_range('imports')
        n_features=end-start
        n_modify=int(n_features * substitution_rate)
        
        for i in range(X_adv.shape[0]):
            indices = np.random.choice(range(start, end), n_modify, replace=False)
            for idx in indices:
                if X_adv[i, idx]>0.5:
                    X_adv[i, idx]*=0.3
                else:
                    X_adv[i, idx]=min(X_adv[i, idx]*1.5,1.0)
        print(f"Applied API substitution (rate: {substitution_rate})")
        return X_adv
    
    def section_manipulation(self, X, noise_level=0.2):    
        X_adv=X.copy()
        start,end=self._get_feature_range('section')
        
        noise=np.random.normal(0, noise_level, (X_adv.shape[0], end - start))
        X_adv[:, start:end] += noise
        X_adv[:, start:end] = np.clip(X_adv[:, start:end], 0, 10)
        
        print(f"Applied section manipulation (noise: {noise_level})")
        return X_adv
    
    def byte_histogram_smoothing(self, X, smoothing_factor=0.3):
        X_adv=X.copy()
        start,end=self._get_feature_range('bytehistogram')
        uniform = np.ones(end - start) / (end - start)
        for i in range(X_adv.shape[0]):
            current = X_adv[i, start:end]
            X_adv[i, start:end] = (1 - smoothing_factor) * current + smoothing_factor * uniform
            X_adv[i, start:end] /= X_adv[i, start:end].sum()
        
        print(f"Applied byte histogram smoothing (factor: {smoothing_factor})")
        return X_adv
    def combined_evasion(self, X, techniques=['header', 'entropy', 'strings', 'imports']):
        X_adv = X.copy()
        
        print(f"\nApplying combined evasion with techniques: {techniques}")
        
        if 'header' in techniques:
            X_adv = self.header_tampering(X_adv, intensity=0.3)
        
        if 'entropy' in techniques:
            X_adv = self.entropy_manipulation(X_adv, target_entropy=7.0)
        
        if 'strings' in techniques:
            X_adv = self.string_obfuscation(X_adv, obfuscation_rate=0.5)
        
        if 'imports' in techniques:
            X_adv = self.api_substitution(X_adv, substitution_rate=0.3)
        
        if 'section' in techniques:
            X_adv = self.section_manipulation(X_adv, noise_level=0.2)
        
        if 'histogram' in techniques:
            X_adv = self.byte_histogram_smoothing(X_adv, smoothing_factor=0.3)
        
        return X_adv
    
    def evaluate_evasion(self, detector, X_original, y_original, X_adversarial):
        # Predictions on original samples
        y_pred_original=detector.predict(X_original)
        acc_original=accuracy_score(y_original, y_pred_original)
        
        # Predictions on adversarial samples
        y_pred_adversarial=detector.predict(X_adversarial)
        acc_adversarial=accuracy_score(y_original, y_pred_adversarial)
        
        # Evasion rate: samples that were correctly classified but now misclassified
        correctly_classified=(y_pred_original == y_original)
        evaded=(y_pred_adversarial != y_original) & correctly_classified
        evasion_rate=evaded.sum() / correctly_classified.sum() if correctly_classified.sum() > 0 else 0
        
        metrics = {
            'original_accuracy': acc_original,
            'adversarial_accuracy': acc_adversarial,
            'accuracy_drop': acc_original - acc_adversarial,
            'evasion_rate': evasion_rate,
            'samples_evaded': evaded.sum(),
            'total_samples': len(y_original)
        }
        
        print("\n")
        print("EVASION EVALUATION")
        print("\n")
        print(f"Original Accuracy:{metrics['original_accuracy']:.4f}")
        print(f"Adversarial Accuracy:{metrics['adversarial_accuracy']:.4f}")
        print(f"Accuracy Drop:{metrics['accuracy_drop']:.4f}")
        print(f"Evasion Rate:{metrics['evasion_rate']:.4f}")
        print(f"Samples Evaded:{metrics['samples_evaded']}/{metrics['total_samples']}")
        print("\n")
        
        return metrics