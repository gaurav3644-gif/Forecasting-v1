#!/bin/bash

echo "========================================"
echo " Sales Forecasting Application"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        echo "Make sure Python 3 is installed"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -d "venv/lib/python*/site-packages/fastapi" ]; then
    echo ""
    echo "Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Start the application
echo ""
echo "========================================"
echo "Starting application..."
echo "========================================"
echo ""
echo "Application will be available at:"
echo "http://localhost:8002"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8002 --reload
