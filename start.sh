#!/bin/bash
# Startup script for Railway deployment
# Reads Railway's PORT env var (or defaults to 8501)
# Creates required data directories before starting

set -e

# Ensure data directories exist (volume mount may be empty)
mkdir -p /data/workspaces

PORT="${PORT:-8501}"
echo "Starting Personal AI Agent on port $PORT..."
echo "Binding to 0.0.0.0:$PORT"

exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none
