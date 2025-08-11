#!/usr/bin/env python3
"""
Test script for the Fraud Detection API
"""

import requests
import json
import time
import random

# API base URL
API_BASE = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Health Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Test the model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{API_BASE}/model-info")
    print(f"Model Info Status: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        print(f"Model: {info['model_name']}")
        print(f"Training Date: {info['training_date']}")
        print(f"Features: {info['features_count']}")
        print(f"Test AUC-PR: {info['test_performance']['auc_pr']:.4f}")
    print()

def create_sample_transaction(fraud_like=False):
    """Create a sample transaction for testing"""
    if fraud_like:
        # Create a transaction more likely to be fraud
        return {
            "time_ind": random.randint(1, 1000),
            "transac_type": random.choice(["TRANSFER", "CASH_OUT"]),  # Higher fraud types
            "amount": random.uniform(50000, 500000),  # Large amounts
            "src_acc": f"acc{random.randint(100000, 999999)}",
            "src_bal": random.uniform(50000, 100000),
            "src_new_bal": 0,  # Account emptied
            "dst_acc": f"acc{random.randint(100000, 999999)}",
            "dst_bal": random.uniform(0, 1000),
            "dst_new_bal": random.uniform(50000, 500000),
            "is_flagged_fraud": 1
        }
    else:
        # Create a normal transaction
        amount = random.uniform(10, 1000)
        src_bal = random.uniform(1000, 10000)
        return {
            "time_ind": random.randint(1, 1000),
            "transac_type": "PAYMENT",
            "amount": amount,
            "src_acc": f"acc{random.randint(100000, 999999)}",
            "src_bal": src_bal,
            "src_new_bal": src_bal - amount,
            "dst_acc": f"acc{random.randint(100000, 999999)}",
            "dst_bal": random.uniform(0, 1000),
            "dst_new_bal": random.uniform(1000, 2000),
            "is_flagged_fraud": 0
        }

def test_prediction():
    """Test the prediction endpoint"""
    print("Testing prediction endpoint...")
    
    # Test normal transaction
    print("1. Testing normal transaction:")
    normal_txn = create_sample_transaction(fraud_like=False)
    response = requests.post(f"{API_BASE}/predict", json=normal_txn)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Transaction ID: {result['transaction_id']}")
        print(f"   Fraud Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
        print(f"   Probability: {result['fraud_probability']:.4f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence_score']:.4f}")
    else:
        print(f"   Error: {response.status_code} - {response.text}")
    
    print()
    
    # Test fraud-like transaction
    print("2. Testing fraud-like transaction:")
    fraud_txn = create_sample_transaction(fraud_like=True)
    response = requests.post(f"{API_BASE}/predict", json=fraud_txn)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Transaction ID: {result['transaction_id']}")
        print(f"   Fraud Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
        print(f"   Probability: {result['fraud_probability']:.4f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence_score']:.4f}")
    else:
        print(f"   Error: {response.status_code} - {response.text}")
    
    print()

def test_frauds_endpoint():
    """Test the frauds retrieval endpoint"""
    print("Testing frauds endpoint...")
    
    # Get all frauds
    response = requests.get(f"{API_BASE}/frauds?limit=10&fraud_only=false")
    
    if response.status_code == 200:
        frauds = response.json()
        print(f"Retrieved {len(frauds)} predictions")
        
        if frauds:
            print("Recent predictions:")
            for i, fraud in enumerate(frauds[:3], 1):
                print(f"   {i}. ID: {fraud['transaction_id']}")
                print(f"      Fraud: {'YES' if fraud['is_fraud'] else 'NO'}")
                print(f"      Probability: {fraud['fraud_probability']:.4f}")
                print(f"      Risk: {fraud['risk_level']}")
                print(f"      Time: {fraud['prediction_timestamp']}")
                print()
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_stats_endpoint():
    """Test the statistics endpoint"""
    print("Testing statistics endpoint...")
    
    response = requests.get(f"{API_BASE}/frauds/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Fraud Predictions: {stats['fraud_predictions']}")
        print(f"Fraud Rate: {stats['fraud_rate']:.2%}")
        print(f"Risk Distribution: {stats['risk_distribution']}")
        print(f"Recent (24h): {stats['recent_predictions_24h']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print()

def main():
    """Run all API tests"""
    print("=" * 60)
    print("FRAUD DETECTION API TESTS")
    print("=" * 60)
    print()
    
    try:
        # Test basic endpoints
        test_health()
        test_model_info()
        
        # Test predictions (this will create data)
        test_prediction()
        
        # Wait a moment for data to be saved
        time.sleep(1)
        
        # Test data retrieval
        test_frauds_endpoint()
        test_stats_endpoint()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running at http://localhost:8000")
        print("Run: docker-compose --profile api up")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()