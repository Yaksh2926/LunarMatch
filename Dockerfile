# Multi-stage lightweight Dockerfile for LunarMatch (< 250 MB)
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV and GDAL/rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install package wheels into isolated environment
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --prefix=/install .[geo,report]

FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src ./src

# Set non-root user for security
RUN useradd -m -u 1000 lunaruser && chown -R lunaruser:lunaruser /app
USER lunaruser

ENTRYPOINT ["lunarmatch"]
CMD ["--help"]
