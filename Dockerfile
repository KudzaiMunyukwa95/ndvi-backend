FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (CapRover will map this)
EXPOSE 5000

# Run with Uvicorn (FastAPI's recommended server)
# Using 4 workers for parallel processing, timeout of 180s
CMD ["uvicorn", "gee_ndvi_generator_fastapi:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "4", "--timeout-keep-alive", "180"]
