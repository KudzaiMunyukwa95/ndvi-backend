FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (CapRover will map this)
EXPOSE 5000

# Run with Gunicorn using preload to initialize GEE once
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "180", "--preload", "gee_ndvi_generator:app"]
