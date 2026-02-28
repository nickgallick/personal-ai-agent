#!/bin/bash
# Startup script that reads Railway's PORT env var (or defaults to 8501)
PORT="${PORT:-8501}"
exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true
