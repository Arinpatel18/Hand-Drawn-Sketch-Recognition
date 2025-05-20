FROM python:3.10-slim

WORKDIR /app

# Copy all files into the container
COPY . .

# Install system dependencies required for packages like PIL and Torch
RUN apt-get update && apt-get install -y \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    Flask \
    gunicorn \
    joblib \
    numpy \
    pandas \
    gdown \
    matplotlib \
    scikit-learn \
    tqdm \
    torch \
    torchvision \
    pillow \
    warnings

# Expose port 8080 for Cloud Run
CMD ["gunicorn", "-b", ":8080", "app:app"]
