#!/usr/bin/env python3
"""
FastAPI service for fraud detection model serving.
Provides REST API endpoints for fraud prediction and result storage.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sqlite3
import json
import logging
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import uvicorn
from contextlib import contextmanager

# Import custom preprocessing
import sys
sys.path.append('/app/src')
from data_preprocessing import FraudDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="REST API for real-time fraud detection using machine learning",
    version="1.0.0"
)

# Database configuration
DB_PATH = "/app/data/fraud_predictions.db"

# Global model variables
model = None
processor = None
feature_names = None
metadata = None

class TransactionInput(BaseModel):
    """Input schema for transaction prediction"""
    time_ind: int = Field(..., description="Time indicator (hours from start)")
    transac_type: str = Field(..., description="Transaction type (PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT)")
    amount: float = Field(..., gt=0, description="Transaction amount")
    src_acc: str = Field(..., description="Source account identifier")
    src_bal: float = Field(..., ge=0, description="Source account balance before transaction")
    src_new_bal: float = Field(..., ge=0, description="Source account balance after transaction")
    dst_acc: str = Field(..., description="Destination account identifier")
    dst_bal: float = Field(..., ge=0, description="Destination account balance before transaction")
    dst_new_bal: float = Field(..., ge=0, description="Destination account balance after transaction")
    is_flagged_fraud: Optional[int] = Field(0, description="System flagged as fraud (0: not flagged, 1: flagged)")

class PredictionResponse(BaseModel):
    """Response schema for fraud prediction"""
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    risk_level: str
    prediction_timestamp: str
    model_name: str
    confidence_score: float

class FraudRecord(BaseModel):
    """Schema for fraud record retrieval"""
    id: int
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    risk_level: str
    prediction_timestamp: str
    model_name: str
    confidence_score: float
    input_data: Dict[str, Any]

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize SQLite database with fraud predictions table"""
    logger.info("Initializing database...")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create fraud predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fraud_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                is_fraud INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                input_data TEXT NOT NULL
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fraud_predictions_timestamp 
            ON fraud_predictions(prediction_timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fraud_predictions_fraud 
            ON fraud_predictions(is_fraud)
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")

def load_model_artifacts():
    """Load trained model and preprocessing components"""
    global model, processor, feature_names, metadata
    
    try:
        model_dir = "/app/models"
        
        # Load model
        model_path = os.path.join(model_dir, "best_fraud_model.joblib")
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load preprocessor
        processor_path = os.path.join(model_dir, "data_processor.joblib")
        processor = joblib.load(processor_path)
        logger.info(f"Preprocessor loaded from {processor_path}")
        
        # Load feature names
        feature_names_path = os.path.join(model_dir, "feature_names.joblib")
        feature_names = joblib.load(feature_names_path)
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "model_metadata.joblib")
        metadata = joblib.load(metadata_path)
        logger.info(f"Model metadata loaded: {metadata['model_name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        return False

def preprocess_transaction(transaction: TransactionInput) -> pd.DataFrame:
    """Preprocess a single transaction for prediction"""
    
    # Convert to DataFrame
    data = {
        'time_ind': [transaction.time_ind],
        'transac_type': [transaction.transac_type],
        'amount': [transaction.amount],
        'src_acc': [transaction.src_acc],
        'src_bal': [transaction.src_bal],
        'src_new_bal': [transaction.src_new_bal],
        'dst_acc': [transaction.dst_acc],
        'dst_bal': [transaction.dst_bal],
        'dst_new_bal': [transaction.dst_new_bal],
        'is_fraud': [0],  # Placeholder
        'is_flagged_fraud': [transaction.is_flagged_fraud]
    }
    
    df = pd.DataFrame(data)
    
    # Apply preprocessing pipeline
    df_clean = processor.clean_data(df)
    df_features = processor.engineer_features(df_clean)
    df_encoded = processor.encode_categorical_features(df_features, fit=False)
    df_scaled = processor.scale_features(df_encoded, fit=False)
    
    # Extract features for prediction
    X = df_scaled[feature_names]
    
    return X

def calculate_risk_level(probability: float) -> str:
    """Calculate risk level based on fraud probability"""
    if probability < 0.1:
        return "Low"
    elif probability < 0.3:
        return "Medium" 
    elif probability < 0.7:
        return "High"
    else:
        return "Critical"

def calculate_confidence_score(probability: float) -> float:
    """Calculate confidence score based on how close probability is to decision boundary"""
    # Higher confidence when probability is close to 0 or 1
    return max(probability, 1 - probability)

def save_prediction(transaction_id: str, prediction: PredictionResponse, input_data: dict):
    """Save prediction to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO fraud_predictions 
                (transaction_id, is_fraud, fraud_probability, risk_level, 
                 prediction_timestamp, model_name, confidence_score, input_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction_id,
                prediction.is_fraud,
                prediction.fraud_probability,
                prediction.risk_level,
                prediction.prediction_timestamp,
                prediction.model_name,
                prediction.confidence_score,
                json.dumps(input_data)
            ))
            
            conn.commit()
            logger.info(f"Prediction saved for transaction {transaction_id}")
            
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        raise HTTPException(status_code=500, detail="Error saving prediction to database")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Fraud Detection API...")
    
    # Initialize database
    init_database()
    
    # Load model artifacts
    if not load_model_artifacts():
        logger.error("Failed to load model artifacts")
        raise RuntimeError("Cannot start API without model artifacts")
    
    logger.info("Fraud Detection API started successfully!")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "model": metadata['model_name'] if metadata else "Unknown",
        "endpoints": ["/predict", "/frauds", "/health", "/model-info"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model is not None,
        "database_accessible": os.path.exists(DB_PATH)
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": metadata['model_name'],
        "model_type": metadata['model_type'],
        "training_date": metadata['training_date'],
        "features_count": metadata['features_count'],
        "test_performance": metadata['test_performance'],
        "feature_importance": metadata.get('feature_importance', [])[:10]  # Top 10 features
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud probability for a given transaction
    """
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate unique transaction ID
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Preprocess transaction
        X = preprocess_transaction(transaction)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        # Calculate derived metrics
        risk_level = calculate_risk_level(probability)
        confidence_score = calculate_confidence_score(probability)
        
        # Create response
        response = PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=int(prediction),
            fraud_probability=float(probability),
            risk_level=risk_level,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=metadata['model_name'],
            confidence_score=confidence_score
        )
        
        # Save to database
        save_prediction(transaction_id, response, transaction.dict())
        
        logger.info(f"Prediction made for {transaction_id}: fraud={prediction}, prob={probability:.4f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/frauds", response_model=List[FraudRecord])
async def get_frauds(
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    fraud_only: Optional[bool] = True
):
    """
    Retrieve fraud predictions from database
    
    Args:
        limit: Maximum number of records to return (default: 100)
        offset: Number of records to skip (default: 0)  
        fraud_only: Return only fraudulent predictions (default: True)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build query based on parameters
            base_query = '''
                SELECT id, transaction_id, is_fraud, fraud_probability, risk_level,
                       prediction_timestamp, model_name, confidence_score, input_data
                FROM fraud_predictions
            '''
            
            if fraud_only:
                query = base_query + " WHERE is_fraud = 1"
            else:
                query = base_query
                
            query += " ORDER BY prediction_timestamp DESC LIMIT ? OFFSET ?"
            
            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()
            
            # Convert to response format
            results = []
            for row in rows:
                try:
                    input_data = json.loads(row['input_data'])
                except:
                    input_data = {}
                
                results.append(FraudRecord(
                    id=row['id'],
                    transaction_id=row['transaction_id'],
                    is_fraud=row['is_fraud'],
                    fraud_probability=row['fraud_probability'],
                    risk_level=row['risk_level'],
                    prediction_timestamp=row['prediction_timestamp'],
                    model_name=row['model_name'],
                    confidence_score=row['confidence_score'],
                    input_data=input_data
                ))
            
            logger.info(f"Retrieved {len(results)} fraud records")
            return results
            
    except Exception as e:
        logger.error(f"Error retrieving frauds: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving fraud records")

@app.get("/frauds/stats")
async def get_fraud_stats():
    """Get fraud detection statistics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) as total FROM fraud_predictions")
            total = cursor.fetchone()['total']
            
            # Fraud predictions
            cursor.execute("SELECT COUNT(*) as frauds FROM fraud_predictions WHERE is_fraud = 1")
            frauds = cursor.fetchone()['frauds']
            
            # Risk level distribution
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count 
                FROM fraud_predictions 
                GROUP BY risk_level
            ''')
            risk_distribution = {row['risk_level']: row['count'] for row in cursor.fetchall()}
            
            # Recent activity (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) as recent 
                FROM fraud_predictions 
                WHERE datetime(prediction_timestamp) >= datetime('now', '-1 day')
            ''')
            recent = cursor.fetchone()['recent']
            
            return {
                "total_predictions": total,
                "fraud_predictions": frauds,
                "fraud_rate": frauds / total if total > 0 else 0,
                "risk_distribution": risk_distribution,
                "recent_predictions_24h": recent
            }
            
    except Exception as e:
        logger.error(f"Error getting fraud stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )