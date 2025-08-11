#!/usr/bin/env python3
"""
Model evaluation script for fraud detection.
This script loads trained models and evaluates them on new data.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudModelEvaluator:
    """
    Comprehensive fraud detection model evaluation.
    """
    
    def __init__(self, model_dir: str = '../models'):
        """
        Initialize the evaluator.
        
        Args:
            model_dir: Directory containing saved model artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.processor = None
        self.feature_names = None
        self.metadata = None
        
        self._load_model_artifacts()
    
    def _load_model_artifacts(self) -> None:
        """Load all saved model artifacts."""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'best_fraud_model.joblib')
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            
            # Load data processor
            processor_path = os.path.join(self.model_dir, 'data_processor.joblib')
            self.processor = joblib.load(processor_path)
            logger.info(f"Data processor loaded from: {processor_path}")
            
            # Load feature names
            feature_names_path = os.path.join(self.model_dir, 'feature_names.joblib')
            self.feature_names = joblib.load(feature_names_path)
            logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.joblib')
            self.metadata = joblib.load(metadata_path)
            logger.info(f"Model metadata loaded: {self.metadata['model_name']}")
            
        except FileNotFoundError as e:
            logger.error(f"Model artifact not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def print_model_info(self) -> None:
        """Print information about the loaded model."""
        print("\n" + "="*60)
        print("LOADED MODEL INFORMATION")
        print("="*60)
        
        if self.metadata:
            print(f"Model Name: {self.metadata['model_name']}")
            print(f"Model Type: {self.metadata['model_type']}")
            print(f"Training Date: {self.metadata['training_date']}")
            print(f"Features Count: {self.metadata['features_count']}")
            print(f"Training Data Size: {self.metadata['training_data_size']:,}")
            
            if 'test_performance' in self.metadata:
                test_perf = self.metadata['test_performance']
                print(f"\nOriginal Test Performance:")
                print(f"  Precision: {test_perf['precision']:.4f}")
                print(f"  Recall: {test_perf['recall']:.4f}")
                print(f"  F1-Score: {test_perf['f1']:.4f}")
                print(f"  AUC-ROC: {test_perf['auc_roc']:.4f}")
                print(f"  AUC-PR: {test_perf['auc_pr']:.4f}")
            
            if 'feature_importance' in self.metadata and self.metadata['feature_importance']:
                print(f"\nTop Features:")
                for i, feat in enumerate(self.metadata['feature_importance'][:5], 1):
                    print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
        
        print("="*60)
    
    def preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Preprocess new data using the saved processor.
        
        Args:
            data_path: Path to the CSV file to evaluate
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        logger.info(f"Preprocessing data from: {data_path}")
        
        # Load data
        df = self.processor.load_data(data_path)
        
        # Clean data
        df_clean = self.processor.clean_data(df)
        
        # Engineer features
        df_features = self.processor.engineer_features(df_clean)
        
        # Encode categorical features (using fitted encoders)
        df_encoded = self.processor.encode_categorical_features(df_features, fit=False)
        
        # Scale features (using fitted scaler)
        df_scaled = self.processor.scale_features(df_encoded, fit=False)
        
        logger.info(f"Data preprocessing completed. Shape: {df_scaled.shape}")
        
        return df_scaled
    
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: Optional[np.ndarray] = None,
                             save_plots: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            save_plots: Whether to save evaluation plots
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("Performing comprehensive model evaluation...")
        
        results = {}
        
        # Basic metrics
        results['precision'] = precision_score(y_true, y_pred)
        results['recall'] = recall_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred)
        
        if y_pred_proba is not None:
            results['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            results['auc_pr'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_cost': int(fp * 1 + fn * 10),  # Assumed costs
            'confusion_matrix': cm.tolist()
        })
        
        # Additional metrics
        results['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        results['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Print results
        self._print_evaluation_results(results)
        
        # Create visualizations
        if y_pred_proba is not None:
            self._create_evaluation_plots(y_true, y_pred, y_pred_proba, save_plots)
        
        return results
    
    def _print_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nClassification Metrics:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")
        
        if 'auc_roc' in results:
            print(f"\nAUC Metrics:")
            print(f"  AUC-ROC: {results['auc_roc']:.4f}")
            print(f"  AUC-PR: {results['auc_pr']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {results['true_positives']:,}")
        print(f"  False Negatives: {results['false_negatives']:,}")
        print(f"  True Negatives: {results['true_negatives']:,}")
        print(f"  False Positives: {results['false_positives']:,}")
        
        print(f"\nError Rates:")
        print(f"  False Positive Rate: {results['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {results['false_negative_rate']:.4f}")
        
        print(f"\nBusiness Impact:")
        print(f"  Total Cost (FP:1, FN:10): {results['total_cost']:,}")
        
        # Fraud detection specific insights
        total_frauds = results['true_positives'] + results['false_negatives']
        total_legit = results['true_negatives'] + results['false_positives']
        
        if total_frauds > 0:
            fraud_detection_rate = results['true_positives'] / total_frauds
            print(f"  Fraud Detection Rate: {fraud_detection_rate:.2%}")
            
        if total_legit > 0:
            legitimate_accuracy = results['true_negatives'] / total_legit
            print(f"  Legitimate Transaction Accuracy: {legitimate_accuracy:.2%}")
        
        print("="*60)
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray, save_plots: bool = False) -> None:
        """Create evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.4f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)
        axes[1, 0].plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Prediction Probability Distribution
        axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Non-Fraud', density=True)
        axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Fraud', density=True)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.model_dir, 'evaluation_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to: {plot_path}")
        
        plt.show()
    
    def evaluate_on_file(self, data_path: str, save_plots: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on a CSV file.
        
        Args:
            data_path: Path to the CSV file containing data to evaluate
            save_plots: Whether to save evaluation plots
            
        Returns:
            Evaluation results dictionary
        """
        # Preprocess the data
        df_processed = self.preprocess_data(data_path)
        
        # Extract features and target
        X = df_processed[self.feature_names]
        y = df_processed['is_fraud']
        
        logger.info(f"Evaluating on {len(X)} transactions with {y.mean():.4%} fraud rate")
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Comprehensive evaluation
        results = self.evaluate_comprehensive(
            y_true=y.values,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            save_plots=save_plots
        )
        
        return results
    
    def predict_on_new_data(self, data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on new data and optionally save results.
        
        Args:
            data_path: Path to the CSV file containing new data
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with original data plus predictions
        """
        logger.info(f"Making predictions on new data: {data_path}")
        
        # Load and preprocess data
        df_processed = self.preprocess_data(data_path)
        
        # Make predictions
        X = df_processed[self.feature_names]
        predictions = self.model.predict(X)
        prediction_probas = self.model.predict_proba(X)[:, 1]
        
        # Load original data for output
        original_df = pd.read_csv(data_path)
        
        # Add predictions to original data
        result_df = original_df.copy()
        result_df['predicted_fraud'] = predictions
        result_df['fraud_probability'] = prediction_probas
        
        # Add risk categories
        result_df['risk_category'] = pd.cut(
            prediction_probas,
            bins=[0, 0.1, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        logger.info(f"Predictions completed:")
        logger.info(f"  Total transactions: {len(result_df):,}")
        logger.info(f"  Predicted frauds: {predictions.sum():,} ({predictions.mean():.2%})")
        logger.info(f"  Risk distribution:")
        for risk_level in ['Low', 'Medium', 'High', 'Critical']:
            count = (result_df['risk_category'] == risk_level).sum()
            logger.info(f"    {risk_level}: {count:,} ({count/len(result_df):.1%})")
        
        # Save results if output path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")
        
        return result_df


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Evaluate fraud detection models")
    parser.add_argument('--model-dir', type=str, default='../models',
                       help="Directory containing model artifacts (default: ../models)")
    parser.add_argument('--data-path', type=str, required=True,
                       help="Path to the CSV file to evaluate")
    parser.add_argument('--mode', type=str, choices=['evaluate', 'predict'], default='evaluate',
                       help="Mode: 'evaluate' (with ground truth) or 'predict' (new data)")
    parser.add_argument('--output-path', type=str,
                       help="Path to save predictions (for predict mode)")
    parser.add_argument('--save-plots', action='store_true',
                       help="Save evaluation plots to model directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = FraudModelEvaluator(model_dir=args.model_dir)
        
        # Print model information
        evaluator.print_model_info()
        
        if args.mode == 'evaluate':
            # Evaluation mode (requires ground truth labels)
            results = evaluator.evaluate_on_file(
                data_path=args.data_path,
                save_plots=args.save_plots
            )
            
            # Save evaluation results
            results_path = os.path.join(args.model_dir, 'evaluation_results.joblib')
            joblib.dump(results, results_path)
            logger.info(f"Evaluation results saved to: {results_path}")
            
        else:
            # Prediction mode (new data without ground truth)
            predictions_df = evaluator.predict_on_new_data(
                data_path=args.data_path,
                output_path=args.output_path
            )
            
            if not args.output_path:
                print("\nSample predictions:")
                print(predictions_df[['predicted_fraud', 'fraud_probability', 'risk_category']].head(10))
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()