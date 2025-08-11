"""
Data preprocessing and feature engineering for fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataProcessor:
    """
    Data preprocessing pipeline for fraud detection.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        
    def load_data(self, file_path: str, chunksize: int = 200000) -> pd.DataFrame:
        """
        Load fraud detection dataset.
        
        Args:
            file_path: Path to the CSV file
            chunksize: Size of chunks for reading large files
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            chunk_iter = pd.read_csv(file_path, chunksize=chunksize)
            df = pd.concat(chunk_iter, ignore_index=True)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset and handle missing values.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning")
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
            # Handle missing values based on column type
            for column in missing_values[missing_values > 0].index:
                if df_clean[column].dtype in ['float64', 'int64']:
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                else:
                    df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
        
        # Remove duplicate rows
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_shape - df_clean.shape[0]} duplicate rows")
        
        # Data quality checks
        self._perform_data_quality_checks(df_clean)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _perform_data_quality_checks(self, df: pd.DataFrame) -> None:
        """Perform data quality checks and log warnings."""
        
        # Check for negative amounts
        if (df['amount'] < 0).any():
            logger.warning("Found negative transaction amounts")
        
        # Check for same source and destination accounts
        same_accounts = (df['src_acc'] == df['dst_acc']).sum()
        if same_accounts > 0:
            logger.warning(f"Found {same_accounts} transactions with same source and destination")
        
        # Check balance consistency
        balance_inconsistent = ((df['src_bal'] - df['amount']) != df['src_new_bal']).sum()
        if balance_inconsistent > 0:
            logger.warning(f"Found {balance_inconsistent} transactions with inconsistent balances")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for fraud detection.
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        df_features = df.copy()
        
        # Time-based features
        df_features['day'] = df_features['time_ind'] // 24
        df_features['hour'] = df_features['time_ind'] % 24
        df_features['is_weekend'] = ((df_features['day'] % 7) >= 5).astype(int)
        df_features['is_night'] = ((df_features['hour'] < 6) | (df_features['hour'] >= 22)).astype(int)
        
        # Amount-based features
        df_features['log_amount'] = np.log1p(df_features['amount'])
        df_features['amount_rounded'] = (df_features['amount'] % 1 == 0).astype(int)
        
        # Balance-based features
        df_features['src_balance_change'] = df_features['src_new_bal'] - df_features['src_bal']
        df_features['dst_balance_change'] = df_features['dst_new_bal'] - df_features['dst_bal']
        df_features['src_balance_ratio'] = np.where(df_features['src_bal'] > 0, 
                                                   df_features['amount'] / df_features['src_bal'], 
                                                   0)
        df_features['src_balance_after_ratio'] = np.where(df_features['src_bal'] > 0,
                                                         df_features['src_new_bal'] / df_features['src_bal'],
                                                         0)
        
        # Transaction type features (will be encoded later)
        df_features['is_transfer'] = (df_features['transac_type'] == 'TRANSFER').astype(int)
        df_features['is_cash_out'] = (df_features['transac_type'] == 'CASH_OUT').astype(int)
        df_features['is_payment'] = (df_features['transac_type'] == 'PAYMENT').astype(int)
        
        # Account behavior features (aggregated features)
        df_features = self._create_aggregated_features(df_features)
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features
    
    def _create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features based on account behavior."""
        
        # Source account frequency features
        src_acc_counts = df.groupby('src_acc').size()
        df['src_acc_frequency'] = df['src_acc'].map(src_acc_counts)
        
        # Destination account frequency features
        dst_acc_counts = df.groupby('dst_acc').size()
        df['dst_acc_frequency'] = df['dst_acc'].map(dst_acc_counts)
        
        # Amount statistics by source account
        src_acc_amount_stats = df.groupby('src_acc')['amount'].agg(['mean', 'std', 'max'])
        df['src_acc_avg_amount'] = df['src_acc'].map(src_acc_amount_stats['mean'])
        df['src_acc_std_amount'] = df['src_acc'].map(src_acc_amount_stats['std']).fillna(0)
        df['amount_vs_src_avg'] = df['amount'] / (df['src_acc_avg_amount'] + 1e-8)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with features
            fit: Whether to fit the encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features")
        df_encoded = df.copy()
        
        categorical_columns = ['transac_type']
        
        for column in categorical_columns:
            if column in df_encoded.columns:
                if fit:
                    encoder = LabelEncoder()
                    df_encoded[f'{column}_encoded'] = encoder.fit_transform(df_encoded[column])
                    self.encoders[column] = encoder
                else:
                    if column in self.encoders:
                        # Handle unseen categories
                        try:
                            df_encoded[f'{column}_encoded'] = self.encoders[column].transform(df_encoded[column])
                        except ValueError:
                            logger.warning(f"Unseen categories in {column}, using mode")
                            mode_value = self.encoders[column].transform([df_encoded[column].mode()[0]])[0]
                            df_encoded[f'{column}_encoded'] = mode_value
                
                # Drop original categorical column
                df_encoded = df_encoded.drop(columns=[column])
        
        # Drop non-feature columns
        columns_to_drop = ['src_acc', 'dst_acc', 'time_ind']  # Keep day and hour
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]
        df_encoded = df_encoded.drop(columns=existing_cols_to_drop)
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: DataFrame with encoded features
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling numerical features")
        df_scaled = df.copy()
        
        # Separate target and features
        if 'is_fraud' in df_scaled.columns:
            target_cols = ['is_fraud', 'is_flagged_fraud']
        else:
            target_cols = ['is_flagged_fraud'] if 'is_flagged_fraud' in df_scaled.columns else []
        
        feature_columns = [col for col in df_scaled.columns if col not in target_cols]
        
        if fit:
            scaler = StandardScaler()
            df_scaled[feature_columns] = scaler.fit_transform(df_scaled[feature_columns])
            self.scalers['features'] = scaler
            self.feature_columns = feature_columns
        else:
            if 'features' in self.scalers:
                df_scaled[self.feature_columns] = self.scalers['features'].transform(
                    df_scaled[self.feature_columns]
                )
        
        return df_scaled
    
    def create_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2, 
                               val_size: float = 0.1,
                               temporal_split: bool = True,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            df: Processed dataframe
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            temporal_split: Whether to use temporal split (recommended for fraud detection)
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating train/validation/test splits")
        
        if temporal_split:
            # Sort by day for temporal split
            df_sorted = df.sort_values('day')
            n_total = len(df_sorted)
            
            # Calculate split indices
            test_start_idx = int(n_total * (1 - test_size))
            val_start_idx = int(test_start_idx * (1 - val_size))
            
            train_df = df_sorted.iloc[:val_start_idx]
            val_df = df_sorted.iloc[val_start_idx:test_start_idx]
            test_df = df_sorted.iloc[test_start_idx:]
            
            logger.info("Using temporal split")
        else:
            # Stratified random split
            y = df['is_fraud'] if 'is_fraud' in df.columns else None
            
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, stratify=y, random_state=random_state
            )
            
            y_train_val = train_val_df['is_fraud'] if 'is_fraud' in train_val_df.columns else None
            val_size_adjusted = val_size / (1 - test_size)
            
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size_adjusted, 
                stratify=y_train_val, random_state=random_state
            )
            
            logger.info("Using stratified random split")
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Log fraud distribution in each split
        if 'is_fraud' in df.columns:
            for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
                fraud_rate = split_df['is_fraud'].mean() * 100
                logger.info(f"{split_name} fraud rate: {fraud_rate:.3f}%")
        
        return train_df, val_df, test_df
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        return self.feature_columns if self.feature_columns else []
    
    def process_pipeline(self, file_path: str, 
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        temporal_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to the raw data file
            test_size: Test set proportion
            val_size: Validation set proportion
            temporal_split: Whether to use temporal splitting
            
        Returns:
            Tuple of (train_df, val_df, test_df) - all processed and ready for ML
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Load and clean data
        df = self.load_data(file_path)
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_features = self.engineer_features(df_clean)
        
        # Create splits before encoding (to avoid data leakage)
        train_df, val_df, test_df = self.create_train_test_split(
            df_features, test_size=test_size, val_size=val_size, temporal_split=temporal_split
        )
        
        # Encode and scale features (fit on training data only)
        train_df = self.encode_categorical_features(train_df, fit=True)
        train_df = self.scale_features(train_df, fit=True)
        
        val_df = self.encode_categorical_features(val_df, fit=False)
        val_df = self.scale_features(val_df, fit=False)
        
        test_df = self.encode_categorical_features(test_df, fit=False)
        test_df = self.scale_features(test_df, fit=False)
        
        logger.info("Preprocessing pipeline completed successfully")
        return train_df, val_df, test_df


def main():
    """Example usage of the preprocessing pipeline."""
    processor = FraudDataProcessor()
    
    # Process the data
    train_df, val_df, test_df = processor.process_pipeline(
        file_path="../data/fraud_mock.csv",
        temporal_split=True
    )
    
    # Print summary information
    print(f"Training set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Feature columns: {len(processor.get_feature_names())}")


if __name__ == "__main__":
    main()