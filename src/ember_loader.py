#Ember Loader- loading and preprocessing 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import ember

class EMBERDataLoader:
    def __init__(self, ember_path='../ember_files', version=2):
        self.ember_path = ember_path
        self.version = version
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, subset='train', sample_size=None):
        print(f"Loading EMBER {subset} data...")
        X, y = ember.read_vectorized_features(self.ember_path,subset=subset,feature_version=self.version)
        if sample_size and sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]

        if self.feature_names is None:
            self.feature_names = self._get_feature_names()
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"Label distribution: Benign={np.sum(y==0)}, Malware={np.sum(y==1)}, Unknown={np.sum(y==-1)}")
        
        return X, y
    
    def _get_feature_names(self):
        feature_names = []
        categories = {'bytehistogram': 256,'byteentropy': 256,'strings': 104,'general': 10,'header': 62,'section': 255,'imports': 1280,'exports': 128,'datadirectories': 30}
        
        for cat, count in categories.items():
            feature_names.extend([f"{cat}_{i}" for i in range(count)])
        
        return feature_names
    
    def preprocess(self, X, y, remove_unknown=True, balance_method=None):
        if remove_unknown:
            mask = y != -1
            X = X[mask]
            y = y[mask]
            print(f"After removing unknowns: {len(X)} samples")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if balance_method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X, y = rus.fit_resample(X, y)
            print(f"After undersampling: {len(X)} samples")
        
        print(f"Final label distribution: Benign={np.sum(y==0)}, Malware={np.sum(y==1)}")
        
        return X, y
    
    def normalize(self, X_train, X_test=None):
        X_train_norm = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_norm = self.scaler.transform(X_test)
            return X_train_norm, X_test_norm
        
        return X_train_norm
    
    def get_feature_info(self):
        return {'total_features': len(self.feature_names),'feature_names': self.feature_names,'categories': {'bytehistogram': (0, 256),'byteentropy': (256, 512),'strings': (512, 616),'general': (616, 626),'header': (626, 688),'section': (688, 943),'imports': (943, 2223),'exports': (2223, 2351),'datadirectories': (2351, 2381)}}
    
    def save_processed_data(self, X, y, filepath):
        np.savez_compressed(filepath, X=X, y=y)
        print(f"Saved to {filepath}")
    
    def load_processed_data(self, filepath):
        data = np.load(filepath)
        return data['X'], data['y']


#trial run
"""if __name__ == "__main__":
    loader = EMBERDataLoader(ember_path='../ember_files', version=2)
    X_train, y_train = loader.load_data(subset='train', sample_size=100000)
    X_test, y_test = loader.load_data(subset='test')
    X_train, y_train = loader.preprocess(X_train, y_train, remove_unknown=True, balance_method='undersample')
    X_test, y_test = loader.preprocess(X_test, y_test, remove_unknown=True, balance_method=None)
    X_train_norm, X_test_norm = loader.normalize(X_train, X_test)
    feature_info = loader.get_feature_info()
    print(f"\nTotal features: {feature_info['total_features']}")
    loader.save_processed_data(X_train_norm, y_train, 'train_processed.npz')
    loader.save_processed_data(X_test_norm, y_test, 'test_processed.npz')
    print("\nData preprocessing complete!")"""