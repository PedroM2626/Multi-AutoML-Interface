# Use official Python runtime as a parent image
FROM python:3.11-slim

# Runtime defaults for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    JAVA_HOME=/usr/lib/jvm/default-java \
    PATH=$PATH:$JAVA_HOME/bin

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for AutoML frameworks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    python3-dev \
    default-jre \
    default-jdk \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to maximize Docker layer cache reuse
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5000

# Streamlit health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false"]
