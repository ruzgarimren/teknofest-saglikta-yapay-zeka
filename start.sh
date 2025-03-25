#!/bin/bash

# Check if dataset parameter is provided
if [ -z "$1" ]; then
    echo "Error: Dataset ID is required"
    echo "Usage: ./start.sh <dataset_id> (1, 2, or 3)"
    exit 1
fi

# Validate dataset parameter
if [ "$1" != "1" ] && [ "$1" != "2" ] && [ "$1" != "3" ]; then
    echo "Error: Dataset ID must be 1, 2, or 3"
    exit 1
fi

# Make sure the script is executable
chmod +x start.sh

# Create necessary directories
mkdir -p data
mkdir -p outputs

# Install required packages if needed
pip install -r requirements.txt

# Run the main script with dataset parameter
python3 src/run.py --dataset $1
