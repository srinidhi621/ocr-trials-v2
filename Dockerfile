FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, pdf2image, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads, outputs, and inputs
RUN mkdir -p /app/uploads /app/output /app/inputs

# Test imports before deployment (fail fast if something is wrong)
RUN python test_import.py

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use shell form so PORT variable expands correctly
# Add error logging to stdout
CMD gunicorn -b 0.0.0.0:$PORT -w 2 --timeout 600 --threads 2 --access-logfile - --error-logfile - --log-level debug app:app
