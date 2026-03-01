# Use official Python image
FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (required for scapy + networking)
RUN apt-get update && apt-get install -y \
    gcc \
    libpcap-dev \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app:app"]
#app:app → file name : Flask instance name
#--workers 2 → multi-process handling
#Production-grade serving