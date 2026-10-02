# Production Dockerfile for LunarMatch Web Application
FROM python:3.12-slim

WORKDIR /app

# Install essential system dependencies for GDAL/rasterio and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy build manifest and project files
COPY pyproject.toml README.md ./
COPY src ./src
COPY sample_data ./sample_data
COPY configs ./configs

# Upgrade pip and install package with geo and report extras
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .[geo,report]

# Expose port and configure environment
EXPOSE 8000
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Create and switch to non-root user
RUN useradd -m -u 1000 lunaruser && \
    mkdir -p /app/outputs && \
    chown -R lunaruser:lunaruser /app
USER lunaruser

# Start web application
CMD ["python", "-m", "lunarmatch.web.server"]
