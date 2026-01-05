FROM python:3.10-slim

# Prevent Python buffering issues
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies required by OpenCV + DeepFace
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire backend code
COPY . .

# Render exposes PORT env variable
EXPOSE 8000

# ⚠️ SINGLE WORKER ONLY (DeepFace safe)
CMD ["uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
