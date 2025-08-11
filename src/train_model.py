#!/usr/bin/env python3
"""
Standalone model training script for fraud detection.
This script can be run from command line or Docker for automated training.
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import optuna

# Import custom preprocessing module
from data_preprocessing import FraudDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class FraudModelTrainer:
    """
    Complete fraud detection model training pipeline.
    """
    
    def __init__(self, data_path: str, output_dir: str = '../models'):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the raw CSV data file
            output_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.processor = FraudDataProcessor()
        self.models = {}
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1, 
                    temporal_split: bool = True, sample_size: int = None) -> None:
        """
        Prepare training, validation, and test datasets.
        """
        logger.info("Starting data preparation...")
        
        if sample_size:
            logger.info(f"Using sample size: {sample_size:,} rows for memory efficiency")
            
            # Load data with sampling
            df = self.processor.load_data(self.data_path)
            if len(df) > sample_size:
                logger.info(f"Sampling {sample_size:,} from {len(df):,} total rows")
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Process the sampled data
            df_clean = self.processor.clean_data(df)
            df_features = self.processor.engineer_features(df_clean)
            
            # Create splits
            self.train_df, self.val_df, self.test_df = self.processor.create_train_test_split(
                df_features, test_size=test_size, val_size=val_size, temporal_split=temporal_split
            )
            
            # Encode and scale
            self.train_df = self.processor.encode_categorical_features(self.train_df, fit=True)
            self.train_df = self.processor.scale_features(self.train_df, fit=True)
            
            self.val_df = self.processor.encode_categorical_features(self.val_df, fit=False)
            self.val_df = self.processor.scale_features(self.val_df, fit=False)
            
            self.test_df = self.processor.encode_categorical_features(self.test_df, fit=False)
            self.test_df = self.processor.scale_features(self.test_df, fit=False)
            
        else:
            # Use full pipeline
            self.train_df, self.val_df, self.test_df = self.processor.process_pipeline(
                file_path=self.data_path,
                test_size=test_size,
                val_size=val_size,
                temporal_split=temporal_split
            )
        
        # Extract features and targets
        self.feature_columns = [col for col in self.train_df.columns 
                               if col not in ['is_fraud', 'is_flagged_fraud']]
        
        self.X_train = self.train_df[self.feature_columns]
        self.y_train = self.train_df['is_fraud']
        
        self.X_val = self.val_df[self.feature_columns]
        self.y_val = self.val_df['is_fraud']
        
        self.X_test = self.test_df[self.feature_columns]
        self.y_test = self.test_df['is_fraud']
        
        logger.info(f"Data preparation completed:")
        logger.info(f"  Training set: {self.X_train.shape}")
        logger.info(f"  Validation set: {self.X_val.shape}")
        logger.info(f"  Test set: {self.X_test.shape}")
        logger.info(f"  Features: {len(self.feature_columns)}")
        logger.info(f"  Training fraud rate: {self.y_train.mean():.4f}")
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, float]:
        """
        Evaluate a trained model and return comprehensive metrics.
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': average_precision_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_cost': int(fp * 1 + fn * 10)  # FP cost=1, FN cost=10
        })
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
        logger.info(f"  F1-Score: {results['f1']:.4f}")
        logger.info(f"  AUC-ROC: {results['auc_roc']:.4f}")
        logger.info(f"  AUC-PR: {results['auc_pr']:.4f}")
        logger.info(f"  Total Cost: {results['total_cost']}")
        
        return results
    
    def train_logistic_regression(self) -> None:
        """
        Train Logistic Regression with SMOTE for class imbalance.
        """
        logger.info("Training Logistic Regression with SMOTE...")
        
        model = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        model.fit(self.X_train, self.y_train)
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = self.evaluate_model(
            model, self.X_val, self.y_val, "Logistic Regression"
        )
    
    def train_random_forest(self) -> None:
        """
        Train Random Forest with class weight balancing.
        """
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = self.evaluate_model(
            model, self.X_val, self.y_val, "Random Forest"
        )
    
    def optimize_xgboost(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using Optuna.
        """
        logger.info(f"Optimizing XGBoost with {n_trials} trials...")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'scale_pos_weight': (self.y_train == 0).sum() / (self.y_train == 1).sum(),
                'random_state': 42,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            
            y_pred_proba = model.predict_proba(self.X_val)[:, 1]
            return roc_auc_score(self.y_val, y_pred_proba)
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best XGBoost AUC-ROC: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_xgboost(self, optimize: bool = True, n_trials: int = 50) -> None:
        """
        Train XGBoost model with optional hyperparameter optimization.
        """
        if optimize:
            best_params = self.optimize_xgboost(n_trials)
        else:
            best_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        # Add fixed parameters
        best_params.update({
            'scale_pos_weight': (self.y_train == 0).sum() / (self.y_train == 1).sum(),
            'random_state': 42,
            'verbosity': 0
        })
        
        logger.info("Training final XGBoost model...")
        model = xgb.XGBClassifier(**best_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['xgboost'] = model
        self.results['xgboost'] = self.evaluate_model(
            model, self.X_val, self.y_val, "XGBoost"
        )
    
    def train_lightgbm(self) -> None:
        """
        Train LightGBM model.
        """
        logger.info("Training LightGBM...")
        
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = self.evaluate_model(
            model, self.X_val, self.y_val, "LightGBM"
        )
    
    def select_best_model(self) -> Tuple[str, Any]:
        """
        Select the best model based on AUC-PR (most important for imbalanced fraud detection).
        """
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['auc_pr'])
        best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} (AUC-PR: {self.results[best_model_name]['auc_pr']:.4f})")
        
        return best_model_name, best_model
    
    def evaluate_on_test_set(self, model_name: str, model) -> Dict[str, float]:
        """
        Evaluate the selected model on the test set.
        """
        logger.info(f"Evaluating {model_name} on test set...")
        
        test_results = self.evaluate_model(model, self.X_test, self.y_test, f"{model_name} (Test)")
        
        return test_results
    
    def save_model_artifacts(self, best_model_name: str, best_model, test_results: Dict) -> None:
        """
        Save all model artifacts for deployment.
        """
        logger.info("Saving model artifacts...")
        
        # Save the best model
        model_path = os.path.join(self.output_dir, 'best_fraud_model.joblib')
        joblib.dump(best_model, model_path)
        
        # Save the data processor
        processor_path = os.path.join(self.output_dir, 'data_processor.joblib')
        joblib.dump(self.processor, processor_path)
        
        # Save feature names
        feature_names_path = os.path.join(self.output_dir, 'feature_names.joblib')
        joblib.dump(self.feature_columns, feature_names_path)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10).to_dict('records')
        elif hasattr(best_model, 'named_steps') and 'classifier' in best_model.named_steps:
            if hasattr(best_model.named_steps['classifier'], 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': np.abs(best_model.named_steps['classifier'].coef_[0])
                }).sort_values('importance', ascending=False).head(10).to_dict('records')
        
        # Create model metadata
        model_metadata = {
            'model_name': best_model_name,
            'model_type': type(best_model).__name__,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_count': len(self.feature_columns),
            'test_performance': test_results,
            'validation_results': self.results,
            'feature_importance': feature_importance,
            'training_data_size': len(self.X_train),
            'validation_data_size': len(self.X_val),
            'test_data_size': len(self.X_test),
            'fraud_rate_train': float(self.y_train.mean()),
            'fraud_rate_test': float(self.y_test.mean())
        }
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'model_metadata.joblib')
        joblib.dump(model_metadata, metadata_path)
        
        logger.info(f"Model artifacts saved to: {self.output_dir}")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Processor: {processor_path}")
        logger.info(f"  - Features: {feature_names_path}")
        logger.info(f"  - Metadata: {metadata_path}")
    
    def train_all_models(self, optimize_xgb: bool = True, xgb_trials: int = 50) -> None:
        """
        Train all models in the pipeline.
        """
        logger.info("Starting comprehensive model training...")
        
        # Train all models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost(optimize=optimize_xgb, n_trials=xgb_trials)
        self.train_lightgbm()
        
        # Select best model
        best_model_name, best_model = self.select_best_model()
        
        # Evaluate on test set
        test_results = self.evaluate_on_test_set(best_model_name, best_model)
        
        # Save artifacts
        self.save_model_artifacts(best_model_name, best_model, test_results)
        
        logger.info("Training pipeline completed successfully!")
        
        # Print summary
        self.print_training_summary(best_model_name, test_results)
    
    def print_training_summary(self, best_model_name: str, test_results: Dict) -> None:
        """
        Print a comprehensive training summary.
        """
        print("\n" + "="*60)
        print("FRAUD DETECTION MODEL TRAINING SUMMARY")
        print("="*60)
        
        print(f"\nDATASET:")
        print(f"  Total transactions: {len(self.X_train) + len(self.X_val) + len(self.X_test):,}")
        print(f"  Training fraud rate: {self.y_train.mean():.4%}")
        print(f"  Features: {len(self.feature_columns)}")
        
        print(f"\nMODELS TRAINED:")
        for model_name, results in self.results.items():
            print(f"  {model_name}: AUC-PR = {results['auc_pr']:.4f}")
        
        print(f"\nBEST MODEL: {best_model_name}")
        print(f"  Test Performance:")
        print(f"    Precision: {test_results['precision']:.4f}")
        print(f"    Recall: {test_results['recall']:.4f}")
        print(f"    F1-Score: {test_results['f1']:.4f}")
        print(f"    AUC-ROC: {test_results['auc_roc']:.4f}")
        print(f"    AUC-PR: {test_results['auc_pr']:.4f}")
        
        print(f"\nBUSINESS IMPACT:")
        print(f"  True Positives: {test_results['true_positives']:,}")
        print(f"  False Negatives: {test_results['false_negatives']:,}")
        print(f"  False Positives: {test_results['false_positives']:,}")
        print(f"  Total Cost: {test_results['total_cost']:,}")
        
        print(f"\nMODEL ARTIFACTS SAVED TO: {self.output_dir}")
        print("="*60)


def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument('--data-path', type=str, required=True,
                       help="Path to the fraud detection CSV file")
    parser.add_argument('--output-dir', type=str, default='../models',
                       help="Directory to save trained models (default: ../models)")
    parser.add_argument('--test-size', type=float, default=0.2,
                       help="Test set proportion (default: 0.2)")
    parser.add_argument('--val-size', type=float, default=0.1,
                       help="Validation set proportion (default: 0.1)")
    parser.add_argument('--no-temporal-split', action='store_true',
                       help="Use random split instead of temporal split")
    parser.add_argument('--no-optimize-xgb', action='store_true',
                       help="Skip XGBoost hyperparameter optimization")
    parser.add_argument('--xgb-trials', type=int, default=50,
                       help="Number of Optuna trials for XGBoost (default: 50)")
    parser.add_argument('--sample-size', type=int, default=None,
                       help="Sample size for large datasets (default: None - use full dataset)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FraudModelTrainer(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    try:
        # Prepare data
        trainer.prepare_data(
            test_size=args.test_size,
            val_size=args.val_size,
            temporal_split=not args.no_temporal_split,
            sample_size=args.sample_size
        )
        
        # Train all models
        trainer.train_all_models(
            optimize_xgb=not args.no_optimize_xgb,
            xgb_trials=args.xgb_trials
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()