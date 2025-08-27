#!/bin/bash

# Medical Imaging Streamlit Runner with 3GB Limits
# This script ensures proper configuration loading for large CT volumes

# Change to script directory
cd "$(dirname "$0")"

# Auto-detect network environment
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [[ "$CURRENT_IP" == "172.30."* ]]; then
    SERVER_ADDRESS="172.30.98.21"
    LOCATION="☁️  Cloud Environment"
elif [[ "$CURRENT_IP" == "192.168.1."* ]]; then
    SERVER_ADDRESS="192.168.1.11"
    LOCATION="🏠 Home Network"
else
    SERVER_ADDRESS="localhost"
    LOCATION="🖥️  Local Machine"
fi

# Display configuration info
echo "🏥 Starting Medical CT Imaging Application"
echo "================================================"
echo "📊 Max Upload Size: 3072 MB (3 GB)"
echo "📨 Max Message Size: 3072 MB (3 GB)"
echo "🔧 Configuration: .streamlit/config.toml"
echo "📍 Location: $LOCATION"
echo "🌐 Access URL: http://$SERVER_ADDRESS:8530"
echo "================================================"
echo ""

# Check if config file exists
if [ -f ".streamlit/config.toml" ]; then
    echo "✅ Configuration file found: .streamlit/config.toml"
else
    echo "⚠️  Configuration file missing! Creating default..."
    mkdir -p .streamlit
    cat > .streamlit/config.toml << 'EOF'
[server]
maxUploadSize = 3072
maxMessageSize = 3072
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#ff6b6b"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#ffffff"
EOF
fi

# Activate virtual environment if it exists
if [ -d "DODD" ]; then
    echo "🐍 Activating DODD virtual environment..."
    source DODD/bin/activate
fi

# Kill any existing streamlit processes to avoid port conflicts
echo "🧹 Cleaning up existing streamlit processes..."
pkill -f "streamlit" 2>/dev/null || true
sleep 2

# Start Streamlit with explicit configuration
echo "🚀 Starting fresh application..."
echo ""

# Run from project root so .streamlit/config.toml is found
# Use both config file AND command line parameters for maximum compatibility
streamlit run app/main.py \
    --server.address "$SERVER_ADDRESS" \
    --server.port 8530 \
    --server.maxUploadSize 3072 \
    --server.maxMessageSize 3072 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false