# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data if they don't exist
RUN mkdir -p parquets csv_to_dash

# Expose port
EXPOSE 8050

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main_dash.py"]
