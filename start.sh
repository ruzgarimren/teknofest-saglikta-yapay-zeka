#!/bin/bash

# Make sure the script is executable
chmod +x start.sh

# Create necessary directories
mkdir -p data
mkdir -p outputs

# Install required packages if needed
pip install -r requirements.txt

# Run the main script
python3 src/run.py
