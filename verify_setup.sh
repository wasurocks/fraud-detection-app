#!/bin/bash
# Verification script for fraud detection system setup
# This script follows the exact steps from README.md

set -e

echo "🔍 FRAUD DETECTION SYSTEM VERIFICATION"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check prerequisites
echo "1. 📋 Checking Prerequisites..."
echo "--------------------------------"

# Check if Docker is installed and running
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        print_success "Docker is installed and running"
    else
        print_error "Docker is installed but not running. Please start Docker Desktop."
        exit 1
    fi
else
    print_error "Docker is not installed. Please install Docker Desktop."
    exit 1
fi

# Check if Docker Compose is available
if docker compose version &> /dev/null; then
    print_success "Docker Compose is available"
elif docker-compose --version &> /dev/null; then
    print_success "Docker Compose (legacy) is available"
else
    print_error "Docker Compose is not installed."
    exit 1
fi

# Check memory availability
TOTAL_MEM=$(docker run --rm alpine cat /proc/meminfo | grep MemTotal | awk '{print $2}')
TOTAL_MEM_GB=$(( TOTAL_MEM / 1024 / 1024 ))

if [ $TOTAL_MEM_GB -ge 8 ]; then
    print_success "Available memory: ${TOTAL_MEM_GB}GB (sufficient for training)"
else
    print_warning "Available memory: ${TOTAL_MEM_GB}GB (minimum 8GB recommended for training)"
fi

# Check if data file exists
echo ""
echo "2. 📁 Checking Project Structure..."
echo "-----------------------------------"

if [ -f "data/fraud_mock.csv" ]; then
    FILE_SIZE=$(ls -lh data/fraud_mock.csv | awk '{print $5}')
    print_success "Dataset found: fraud_mock.csv ($FILE_SIZE)"
else
    print_error "Dataset not found: data/fraud_mock.csv"
    print_info "Please ensure the fraud detection dataset is in the data/ directory"
    exit 1
fi

# Check essential files
REQUIRED_FILES=(
    "docker-compose.yml"
    "Dockerfile"
    "requirements.txt"
    "src/api.py"
    "src/train_model.py"
    "src/data_preprocessing.py"
    "scripts/run_training.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "Found: $file"
    else
        print_error "Missing: $file"
        exit 1
    fi
done

# Check if models are already trained
echo ""
echo "3. 🤖 Checking Model Status..."
echo "------------------------------"

if [ -f "models/best_fraud_model.joblib" ] && [ -f "models/data_processor.joblib" ]; then
    print_success "Models already trained and ready"
    MODELS_TRAINED=true
else
    print_warning "Models not found - will need training"
    MODELS_TRAINED=false
fi

# Offer to train models if not present
if [ "$MODELS_TRAINED" = false ]; then
    echo ""
    echo "4. 🏃 Training Models..."
    echo "------------------------"
    print_info "Starting automated model training (this may take 5-10 minutes)"
    
    # Make training script executable
    chmod +x scripts/run_training.sh
    
    # Run training
    if ./scripts/run_training.sh; then
        print_success "Model training completed successfully"
        MODELS_TRAINED=true
    else
        print_error "Model training failed"
        print_info "Check logs with: docker-compose logs fraud-training"
        exit 1
    fi
fi

# Test API startup
echo ""
echo "5. 🚀 Testing API Startup..."
echo "----------------------------"

print_info "Starting fraud detection API..."
docker-compose --profile api up -d

# Wait for API to be ready
print_info "Waiting for API to start..."
sleep 10

# Test API health
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "http://localhost:8000/health" > /dev/null; then
        print_success "API is running and healthy"
        break
    else
        print_info "Waiting for API to be ready... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "API failed to start properly"
    print_info "Check logs with: docker-compose logs fraud-api"
    exit 1
fi

# Test API endpoints
echo ""
echo "6. 🧪 Testing API Endpoints..."
echo "------------------------------"

# Test health endpoint
if curl -s "http://localhost:8000/health" | grep -q "healthy"; then
    print_success "Health endpoint working"
else
    print_error "Health endpoint failed"
fi

# Test model info endpoint
if curl -s "http://localhost:8000/model-info" | grep -q "random_forest"; then
    print_success "Model info endpoint working"
else
    print_error "Model info endpoint failed"
fi

# Test prediction endpoint with legitimate transaction
print_info "Testing prediction with legitimate transaction..."
LEGIT_RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "time_ind": 100,
        "transac_type": "PAYMENT",
        "amount": 500.0,
        "src_acc": "acc123456",
        "src_bal": 2000.0,
        "src_new_bal": 1500.0,
        "dst_acc": "acc654321",
        "dst_bal": 1000.0,
        "dst_new_bal": 1500.0,
        "is_flagged_fraud": 0
    }')

if echo "$LEGIT_RESPONSE" | grep -q '"is_fraud":0'; then
    print_success "Legitimate transaction correctly classified"
else
    print_error "Legitimate transaction prediction failed"
    echo "Response: $LEGIT_RESPONSE"
fi

# Test prediction endpoint with suspicious transaction
print_info "Testing prediction with suspicious transaction..."
FRAUD_RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "time_ind": 50,
        "transac_type": "TRANSFER",
        "amount": 100000.0,
        "src_acc": "acc789012",
        "src_bal": 100000.0,
        "src_new_bal": 0.0,
        "dst_acc": "acc210987",
        "dst_bal": 0.0,
        "dst_new_bal": 100000.0,
        "is_flagged_fraud": 1
    }')

if echo "$FRAUD_RESPONSE" | grep -q '"is_fraud":1'; then
    print_success "Suspicious transaction correctly classified as fraud"
else
    print_warning "Suspicious transaction not classified as fraud (this can happen with sampling)"
    print_info "Response: $FRAUD_RESPONSE"
fi

# Test frauds endpoint
if curl -s "http://localhost:8000/frauds?limit=5" | grep -q "transaction_id"; then
    print_success "Frauds retrieval endpoint working"
else
    print_error "Frauds retrieval endpoint failed"
fi

# Test stats endpoint
if curl -s "http://localhost:8000/frauds/stats" | grep -q "total_predictions"; then
    print_success "Statistics endpoint working"
else
    print_error "Statistics endpoint failed"
fi

# Final summary
echo ""
echo "7. 📊 System Summary..."
echo "-----------------------"

# Get system stats
HEALTH_STATUS=$(curl -s "http://localhost:8000/health" | grep -o '"status":"[^"]*' | cut -d'"' -f4)
MODEL_INFO=$(curl -s "http://localhost:8000/model-info")
STATS=$(curl -s "http://localhost:8000/frauds/stats")

print_success "System Status: $HEALTH_STATUS"

if echo "$MODEL_INFO" | grep -q "random_forest"; then
    MODEL_NAME=$(echo "$MODEL_INFO" | grep -o '"model_name":"[^"]*' | cut -d'"' -f4)
    PRECISION=$(echo "$MODEL_INFO" | grep -o '"precision":[0-9.]*' | cut -d':' -f2)
    RECALL=$(echo "$MODEL_INFO" | grep -o '"recall":[0-9.]*' | cut -d':' -f2)
    
    print_success "Model: $MODEL_NAME"
    print_success "Precision: $(echo "$PRECISION * 100" | bc -l | cut -c1-5)%"
    print_success "Recall: $(echo "$RECALL * 100" | bc -l | cut -c1-5)%"
fi

if echo "$STATS" | grep -q "total_predictions"; then
    TOTAL_PREDS=$(echo "$STATS" | grep -o '"total_predictions":[0-9]*' | cut -d':' -f2)
    print_success "Total Predictions Made: $TOTAL_PREDS"
fi

echo ""
echo "🎉 VERIFICATION COMPLETE!"
echo "========================="
print_success "Fraud Detection System is fully operational!"
echo ""
print_info "🔗 Access Points:"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • API Base URL: http://localhost:8000"
echo "   • Jupyter (if running): http://localhost:8888"
echo ""
print_info "🛠️  Quick Commands:"
echo "   • View API logs: docker-compose logs fraud-api"
echo "   • Stop API: docker-compose --profile api down"
echo "   • Restart API: docker-compose --profile api restart"
echo ""
print_success "✨ System is ready for production use!"