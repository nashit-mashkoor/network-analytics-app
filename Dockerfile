FROM python:3.7-slim-bullseye AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pin setuptools and install numpy first (prophet requires numpy during build)
RUN pip install --no-cache-dir pip==23.0.1 setuptools==65.5.0 wheel numpy==1.21.5 Cython==0.29.26

# Install dependencies into virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Production Stage ---
FROM python:3.7-slim-bullseye

# Security: Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopenblas-base \
    libgomp1 \
    dbus \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* \
    && dbus-uuidgen > /etc/machine-id

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appgroup . .

# Create writable directories for Streamlit and Matplotlib
RUN mkdir -p /home/appuser/.streamlit \
    /home/appuser/.config/matplotlib \
    /home/appuser/.cache \
    && chown -R appuser:appgroup /home/appuser

# Security: Set proper permissions
RUN chmod -R 755 /app && \
    find /app -type f -exec chmod 644 {} \;

# Security: Switch to non-root user
USER appuser

# Use virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Set matplotlib config directory
ENV MPLCONFIGDIR="/home/appuser/.config/matplotlib"

# Security: Streamlit production settings
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8502/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
