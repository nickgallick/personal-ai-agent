# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies + Playwright Chromium dependencies
# We install deps manually to avoid the --with-deps flag which fails on newer Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    # Playwright Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2t64 \
    libxshmfence1 \
    libx11-xcb1 \
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Install Playwright Chromium browser (without --with-deps since we installed deps above)
RUN playwright install chromium

# Copy application code
COPY . .

# Create data and workspace directories
RUN mkdir -p /data /app/workspaces

# Set environment variables
ENV DATABASE_PATH=/data/agent.db
ENV SCHEDULER_DB_PATH=/data/scheduler.db
ENV WORKSPACE_ROOT=/app/workspaces
ENV PYTHONUNBUFFERED=1

# Copy and prepare startup script
COPY start.sh .
RUN chmod +x start.sh

# Expose default port (Railway overrides via $PORT)
EXPOSE 8501

# Start via shell script so $PORT is resolved at runtime
CMD ["./start.sh"]
