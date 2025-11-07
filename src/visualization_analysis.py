#Visualization and Analysis Module
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve)
import shap

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class MalwareVisualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'benign': '#2ecc71',
            'malware': '#e74c3c',
            'adversarial': '#f39c12'
        }
    def plot_confusion_matrices(self, models_dict, X_test, y_test,save_path='confusion_matrices.png'):
        n_models = len(models_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(models_dict.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                       ax=axes[idx], cbar=True,
                       xticklabels=['Benign', 'Malware'],
                       yticklabels=['Benign', 'Malware'])
            
            axes[idx].set_title(f'{name}\n(n={len(y_test)})', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=11)
            axes[idx].set_xlabel('Predicted Label', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrices saved to {save_path}")
    
    def plot_roc_curves(self, models_dict, X_test, y_test,save_path='roc_curves.png'):
        fig, ax = plt.subplots(figsize=(10, 8))
        for name, model in models_dict.items():
            y_proba = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to {save_path}")
    
    def plot_evasion_comparison(self, results_dict, save_path='evasion_comparison.png'):
        techniques = list(results_dict.keys())
        evasion_rates = [results_dict[t]['evasion_rate'] for t in techniques]
        accuracy_drops = [results_dict[t]['accuracy_drop'] for t in techniques]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Evasion rates
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(techniques)))
        bars1 = axes[0].barh(techniques, evasion_rates, color=colors, edgecolor='black')
        axes[0].set_xlabel('Evasion Rate', fontsize=12)
        axes[0].set_title('Evasion Success Rate by Technique', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xlim(0, 1)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars1, evasion_rates):
            width = bar.get_width()
            axes[0].text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{rate:.2%}', va='center', fontsize=10)
        
        # Accuracy drops
        colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(techniques)))
        bars2 = axes[1].barh(techniques, accuracy_drops, color=colors2, edgecolor='black')
        axes[1].set_xlabel('Accuracy Drop', fontsize=12)
        axes[1].set_title('Detection Accuracy Drop by Technique',
                         fontsize=13, fontweight='bold')
        axes[1].set_xlim(0, max(accuracy_drops) * 1.2)
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, drop in zip(bars2, accuracy_drops):
            width = bar.get_width()
            axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{drop:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Evasion comparison saved to {save_path}")
    
    def plot_defense_comparison(self, defense_results, save_path='defense_comparison.png'):
        strategies = list(defense_results.keys())
        clean_acc = [defense_results[s]['clean_accuracy'] for s in strategies]
        adv_acc = [defense_results[s]['adversarial_accuracy'] for s in strategies]
        robustness = [defense_results[s]['robustness_score'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, clean_acc, width, label='Clean Accuracy',
                      color='#3498db', edgecolor='black')
        bars2 = ax.bar(x, adv_acc, width, label='Adversarial Accuracy',
                      color='#e74c3c', edgecolor='black')
        bars3 = ax.bar(x + width, robustness, width, label='Robustness Score',
                      color='#2ecc71', edgecolor='black')
        
        ax.set_xlabel('Defense Strategy', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Defense Strategies Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15, ha='right')
        ax.legend(loc='lower left', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Defense comparison saved to {save_path}")
    
    def shap_feature_importance(self, model, X_train, X_test, feature_names=None, top_n=20,save_path='shap_importance.png'):
        
        # Sample data if too large
        if len(X_train) > 1000:
            X_train_sample = X_train[np.random.choice(len(X_train), 1000, replace=False)]
        else:
            X_train_sample = X_train
        
        if len(X_test) > 500:
            X_test_sample = X_test[np.random.choice(len(X_test), 500, replace=False)]
        else:
            X_test_sample = X_test
        
        # Create explainer
        explainer = shap.TreeExplainer(model, X_train_sample)
        shap_values = explainer.shap_values(X_test_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, 
                         feature_names=feature_names,
                         max_display=top_n, show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP plot saved to {save_path}")
        
        feature_importance = np.abs(shap_values).mean(axis=0)
        if feature_names:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            return importance_df
        
        return feature_importance


