import os
import numpy as np
import json
from datetime import datetime

from ember_loader import EMBERDataLoader
from ml_models import MalwareDetector
from adversarial_evasion import AdversarialEvader
from defense_strategies import AdversarialDefense
from visualization_analysis import MalwareVisualizer
from sklearn.model_selection import train_test_split
class MalwareDetectionPipeline:
    def __init__(self, ember_path='./data', output_dir='./results'):
        self.ember_path=ember_path
        self.output_dir=output_dir
        self.results={}
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/models', exist_ok=True)
        os.makedirs(f'{output_dir}/figures', exist_ok=True)
        os.makedirs(f'{output_dir}/data', exist_ok=True)

    def step1_load_preprocess(self, sample_size=50000):
        print("STEP1: DATA LOADING AND PREPROCESSING\n")
        loader=EMBERDataLoader(self.ember_path, version=2)
        X_train, y_train=loader.load_data('train', sample_size=sample_size)
        X_test, y_test=loader.load_data('test',sample_size=int(0.2*sample_size))
        X_train, y_train=loader.preprocess(X_train,y_train,remove_unknown=True,balance_method='undersample')
        X_test, y_test=loader.preprocess(X_test,y_test,remove_unknown=True,balance_method='undersample')
        X_train, X_test = loader.normalize(X_train, X_test)
        loader.save_processed_data(X_train, y_train, f'{self.output_dir}/data/train_processed.npz')
        loader.save_processed_data(X_test, y_test, f'{self.output_dir}/data/test_processed.npz')
        self.results['data'] = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'train_malware_ratio': float((y_train == 1).mean()),
            'test_malware_ratio': float((y_test == 1).mean())
        }
        print(f"\nData preprocessing complete")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train, y_train, X_test, y_test, loader.get_feature_info()
    
    def step2_train_baseline_models(self, X_train, y_train, X_test, y_test):
        print("STEP2: BASELINE MODEL TRAINING\n")
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train_sub = X_train[:split_idx]
        y_train_sub = y_train[:split_idx]
       
        models={}
        #Random Forest
        print("\nTraining Random Forest")
        rf = MalwareDetector('rf')
        rf.build_model(n_estimators=200, max_depth=20)
        rf.train(X_train_sub, y_train_sub)
        rf_metrics = rf.evaluate(X_test, y_test, return_metrics=True)
        rf.save_model(f'{self.output_dir}/models/rf_detector')
        models['rf']=rf

        #LightGBM
        print("\nTraining LightGBM")
        lgb= MalwareDetector('lightgbm')
        lgb.build_model(n_estimators=200, max_depth=10, learning_rate=0.1, num_leaves=31)
        lgb.train(X_train, y_train)
        lgb_metrics = lgb.evaluate(X_test, y_test, return_metrics=True)
        lgb.save_model(f'{self.output_dir}/models/lgb_baseline')
        models['lgb']=lgb


        self.results['baseline_models'] = {
            'rf': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in rf_metrics.items() if k != 'confusion_matrix'},
            'lgb': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in lgb_metrics.items() if k != 'confusion_matrix'},
        }
        print("Baseline model training done")
        return models

    def step3_adversarial_evasion(self, models, X_test, y_test, feature_info):
        X_malware = X_test[y_test == 1][:3000]
        y_malware = np.ones(len(X_malware))
        lgb_model=models['lgb']
        evader=AdversarialEvader(feature_info)
        evasion_results = {}
        adversarial_samples = {}
        print("\n--- Testing Individual Evasion Techniques ---")
        
        techniques = {
            'Header Tampering': lambda X: evader.header_tampering(X.copy(), 0.3),
            'Entropy Manipulation': lambda X: evader.entropy_manipulation(X.copy(), 7.0),
            'String Obfuscation': lambda X: evader.string_obfuscation(X.copy(), 0.5),
            'API Substitution': lambda X: evader.api_substitution(X.copy(), 0.3),
            'Section Manipulation': lambda X: evader.section_manipulation(X.copy(), 0.2),
            'Byte Histogram Smoothing': lambda X: evader.byte_histogram_smoothing(X.copy(), 0.3)
        }
        
        for name, technique in techniques.items():
            print(f"\n>>> Testing: {name}")
            X_adv = technique(X_malware)
            metrics = evader.evaluate_evasion(lgb_model, X_malware, y_malware, X_adv)
            evasion_results[name] = {
                'original_accuracy': float(metrics['original_accuracy']),
                'adversarial_accuracy': float(metrics['adversarial_accuracy']),
                'accuracy_drop': float(metrics['accuracy_drop']),
                'evasion_rate': float(metrics['evasion_rate']),
                'samples_evaded': int(metrics['samples_evaded']),
                'total_samples': int(metrics['total_samples'])
            }
            adversarial_samples[name] = X_adv
        print("\nTesting Combined Evasion Strategies")
        
        combined_strategies = {
            'Combined-Light': ['header', 'strings'],
            'Combined-Medium': ['header', 'entropy', 'strings'],
            'Combined-Heavy': ['header', 'entropy', 'strings', 'imports'],
            'Combined-Extreme': ['header', 'entropy', 'strings', 'imports', 'section', 'histogram']
        }
        
        for strategy_name, techniques_list in combined_strategies.items():
            print(f"\n>>> Testing: {strategy_name} ({len(techniques_list)} techniques)")
            X_combined = evader.combined_evasion(X_malware.copy(), techniques_list)
            metrics = evader.evaluate_evasion(lgb_model, X_malware, y_malware, X_combined)
            evasion_results[strategy_name] = {
                'original_accuracy': float(metrics['original_accuracy']),
                'adversarial_accuracy': float(metrics['adversarial_accuracy']),
                'accuracy_drop': float(metrics['accuracy_drop']),
                'evasion_rate': float(metrics['evasion_rate']),
                'samples_evaded': int(metrics['samples_evaded']),
                'total_samples': int(metrics['total_samples'])
            }
            adversarial_samples[strategy_name] = X_combined
        
        best_evasion = max(evasion_results.items(), key=lambda x: x[1]['evasion_rate'])
        print(f"\n>>> Best evasion technique: {best_evasion[0]} ({best_evasion[1]['evasion_rate']:.2%})")
        
        X_best_adv = adversarial_samples[best_evasion[0]]
        np.savez_compressed(f'{self.output_dir}/data/adversarial_samples.npz',
                          X_adversarial=X_best_adv,
                          X_original=X_malware,
                          y=y_malware)
        print("\nTesting Evasion Across All Models")
        cross_model_results = {}
        
        for model_name, model in models.items():
            print(f"\n>>> Evaluating {model_name.upper()} against best adversarial samples")
            metrics = evader.evaluate_evasion(model, X_malware, y_malware, X_best_adv)
            cross_model_results[model_name] = {
                'original_accuracy': float(metrics['original_accuracy']),
                'adversarial_accuracy': float(metrics['adversarial_accuracy']),
                'evasion_rate': float(metrics['evasion_rate'])
            }
        
        self.results['evasion'] = {
            'individual_techniques': evasion_results,
            'cross_model_results': cross_model_results,
            'best_technique': best_evasion[0]
        }
        
        print(f"\nAdversarial evasion testing complete")
        
        return X_malware, y_malware, X_best_adv
    
    def step4_defense_strategies(self, X_train, y_train, X_test, y_test, X_malware, y_malware, X_adversarial):
        print("\n STEP 4: DEFENSE STRATEGIES")
        defense = AdversarialDefense()
        X_mal_test = X_malware[:1000]
        y_mal_test = y_malware[:1000]
        X_adv_test = X_adversarial[:1000]
        y_adv_test = np.ones(len(X_adv_test))
        defense_results={}
        def get_balanced_slice(X,y, n_samples=12000):
            idx_class0=np.where(y==0)[0]
            idx_class1=np.where(y==1)[0]
            n0 = min(len(idx_class0), n_samples // 2)
            n1 = min(len(idx_class1), n_samples // 2)
        
            selected_idx = np.concatenate([idx_class0[:n0], idx_class1[:n1]])
            return X[selected_idx], y[selected_idx]
        X_train_bal, y_train_bal = get_balanced_slice(X_train, y_train, n_samples=12000)
        #Random Forest
        print("\nAdversarial Training: Random Forest")
        rf_adv = MalwareDetector('rf')
        rf_adv.build_model(n_estimators=150, max_depth=20)
        rf_adv = defense.adversarial_training(rf_adv,X_train_bal, y_train_bal,X_adversarial[:3000], np.ones(3000),adversarial_ratio=0.3)
        rf_metrics = defense.evaluate_defense(rf_adv,X_mal_test, y_mal_test,X_adv_test, y_adv_test,"Random Forest (Adversarially Trained)")   
        rf_adv.save_model(f'{self.output_dir}/models/rf_adversarial_trained')

        #LightGBM
        print("\n--- Adversarial Training: LightGBM ---")
        lgb_adv = MalwareDetector('lightgbm')
        lgb_adv.build_model(n_estimators=150, max_depth=10)
        lgb_adv = defense.adversarial_training(lgb_adv,X_train_bal, y_train_bal,X_adversarial[:3000], np.ones(3000),adversarial_ratio=0.3)
        lgb_metrics = defense.evaluate_defense(lgb_adv,X_mal_test, y_mal_test,X_adv_test, y_adv_test,"LightGBM (Adversarially Trained)")
        lgb_adv.save_model(f'{self.output_dir}/models/lgb_adversarial_trained')
        print(f"\n Adversarial training completed for RF and LightGBM")
        

    
    def step5_visualization(self, X_test, y_test, X_adversarial, models):
        print("STEP 5: VISUALIZATION AND ANALYSIS")
        viz=MalwareVisualizer()
        print("--- Generating Model Comparison Plots ---")
        model_dict = {
            'Random Forest': models['rf'],
            'LightGBM': models['lgb']
        }
        _, X_vis, _, y_vis = train_test_split(X_test, y_test, test_size=2000, stratify=y_test, random_state=42)
        
        viz.plot_confusion_matrices(model_dict, X_vis, y_vis,
                                   f'{self.output_dir}/figures/confusion_matrices.png')
        
        viz.plot_roc_curves(model_dict, X_vis, y_vis,
                           f'{self.output_dir}/figures/roc_curves.png')
        
        print("--- Generating Evasion Comparison ---")
        viz.plot_evasion_comparison(self.results['evasion']['individual_techniques'],
                                   f'{self.output_dir}/figures/evasion_comparison.png')
        
        
        print("--- Generating SHAP Feature Importance ---")
        viz.shap_feature_importance(
            models['lgb'].model,
            X_test[y_test==1][:1000],
            X_test[y_test==1][1000:1500],
            top_n=25,
            save_path=f'{self.output_dir}/figures/shap_importance.png'
        )
        
        print(f"\nAll visualizations generated")
    def step6_generate_report(self):
        print("STEP 6: REPORT GENERATION")
        with open(f'{self.output_dir}/results_summary.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        with open(f'{self.output_dir}/FINAL_REPORT.md', 'w') as f:
            f.write("# Adversarial Evasion in Static Malware Detection\n")
            f.write("## Manual Evasion Techniques & Defense Strategies\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Dataset Summary
            f.write("## 1. Dataset Summary\n\n")
            f.write(f"- **Training Samples:** {self.results['data']['train_samples']:,}\n")
            f.write(f"- **Test Samples:** {self.results['data']['test_samples']:,}\n")
            f.write(f"- **Features:** {self.results['data']['n_features']}\n")
            f.write(f"- **Train Malware Ratio:** {self.results['data']['train_malware_ratio']:.2%}\n")
            f.write(f"- **Test Malware Ratio:** {self.results['data']['test_malware_ratio']:.2%}\n\n")
            
            # Baseline Models
            f.write("## 2. Baseline Model Performance\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
            f.write("|-------|----------|-----------|--------|----------|----------|\n")
            for model, metrics in self.results['baseline_models'].items():
                f.write(f"| {model.upper()} | {metrics['accuracy']:.4f} | "
                       f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                       f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |\n")
            f.write("\n")
            
            # Evasion Results
            f.write("## 3. Adversarial Evasion Results\n\n")
            f.write("### 3.1 Individual Techniques\n\n")
            f.write("| Technique | Evasion Rate | Accuracy Drop | Samples Evaded |\n")
            f.write("|-----------|--------------|---------------|----------------|\n")
            
            for technique, metrics in self.results['evasion']['individual_techniques'].items():
                f.write(f"| {technique} | {metrics['evasion_rate']:.2%} | "
                       f"{metrics['accuracy_drop']:.4f} | "
                       f"{metrics['samples_evaded']}/{metrics['total_samples']} |\n")
            f.write("\n")
            
            f.write("### 3.2 Cross-Model Evasion\n\n")
            f.write(f"**Best Technique:** {self.results['evasion']['best_technique']}\n\n")
            f.write("| Model | Original Acc | Adversarial Acc | Evasion Rate |\n")
            f.write("|-------|--------------|-----------------|-------------|\n")
            
            for model, metrics in self.results['evasion']['cross_model_results'].items():
                f.write(f"| {model.upper()} | {metrics['original_accuracy']:.4f} | "
                       f"{metrics['adversarial_accuracy']:.4f} | "
                       f"{metrics['evasion_rate']:.2%} |\n")
            f.write("\n")
            
            
            # Key Findings
            f.write("## 4. Key Findings\n\n")
            
            # Best baseline
            best_baseline = max(self.results['baseline_models'].items(), 
                              key=lambda x: x[1]['roc_auc'])
            f.write(f"1. **Best Baseline Model:** {best_baseline[0].upper()} "
                   f"(ROC-AUC: {best_baseline[1]['roc_auc']:.4f})\n\n")
            
            # Most effective evasion
            best_evasion = max(self.results['evasion']['individual_techniques'].items(),
                             key=lambda x: x[1]['evasion_rate'])
            f.write(f"2. **Most Effective Evasion:** {best_evasion[0]} "
                   f"({best_evasion[1]['evasion_rate']:.2%} evasion rate)\n\n")
        
        
        print(f"\nReport generated successfully")
        print(f"   Location: {self.output_dir}/FINAL_REPORT.md")
    
    def run_complete_pipeline(self, sample_size=50000):
        """Execute complete simplified pipeline (No GAN)"""
        try:
            start_time = datetime.now()
            
            # Step 1: Data loading
            X_train, y_train, X_test, y_test, feature_info = self.step1_load_preprocess(sample_size)
            
            # Step 2: Train baseline models
            models = self.step2_train_baseline_models(X_train, y_train, X_test, y_test)
            
            # Step 3: Manual adversarial evasion
            X_malware, y_malware, X_adversarial = self.step3_adversarial_evasion(models, X_test, y_test, feature_info)
            
            # Step 4: Defense strategies
            self.step4_defense_strategies(X_train, y_train, X_test, y_test, X_malware, y_malware, X_adversarial)
            
            # Step 5: Visualization
            self.step5_visualization(X_test, y_test, X_adversarial, models)
            
            # Step 6: Generate report
            self.step6_generate_report()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            print("\n" + "="*70)
            print("PIPELINE EXECUTION COMPLETE")
            print("="*70)
            print(f"\nExecution time: {duration:.2f} minutes")
            print(f"All results saved to: {self.output_dir}/")
            print(f"\nMain outputs:")
            print(f"  - Final Report: {self.output_dir}/FINAL_REPORT.md")
            print(f"  - JSON Results: {self.output_dir}/results_summary.json")
            print(f"  - Figures: {self.output_dir}/figures/")
            print(f"  - Models: {self.output_dir}/models/")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
if __name__ == "__main__":
    pipeline = MalwareDetectionPipeline(
        ember_path='./ember_files',
        output_dir='./malware_detection_results'
    )
    pipeline.run_complete_pipeline(sample_size=100000)