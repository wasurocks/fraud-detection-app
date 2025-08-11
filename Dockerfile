# Multi-stage Dockerfile for fraud detection ML pipeline
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and notebooks
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY data/ ./data/

# Create models directory for artifacts
RUN mkdir -p models

# Expose Jupyter port
EXPOSE 8888

# Default command - can be overridden
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
