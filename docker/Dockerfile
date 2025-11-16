FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
    libavutil-dev libswresample-dev libswscale-dev pkg-config \
    build-essential git curl ca-certificates \
    libportaudio2 portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements1.txt .

# Install PyAV first
RUN pip install --no-cache-dir av==12.0.0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements1.txt

COPY . .

RUN mkdir -p /app/chroma_db /app/logs /app/data/user_uploads

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "hybrid_main.py"]
