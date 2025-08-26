#!/bin/bash

# Run script for DO_DD-scripts application
# This script activates the DODD environment and launches the Streamlit app

# Change to script directory
cd "$(dirname "$0")"

# Activate the DODD environment
echo "Activating DODD environment..."
source DODD/bin/activate

# Launch the application
echo "Starting CT Metal Artifact Characterization app..."
echo "Access at: http://192.168.1.11:4224"
echo ""

# Run streamlit with specific configuration
streamlit run app/main.py \
    --server.address 192.168.1.11 \
    --server.port 4224 \
    --server.headless true \
    --browser.gatherUsageStats false