FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (CapRover will map this)
EXPOSE 5000

# Use Gunicorn with Uvicorn workers for better process management
# Preload ensures GEE initializes once before forking workers
CMD ["gunicorn", "gee_ndvi_generator:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "180", "--preload"]
