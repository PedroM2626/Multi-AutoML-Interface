# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for AutoGluon, FLAML and H2O
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1 \
    python3-dev \
    default-jre \
    default-jdk \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5000

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Command to run the application
CMD ["streamlit", "run", "app.py"]
