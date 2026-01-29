FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hugging Face caches (persist via docker-compose volume)
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    DIFFUSERS_CACHE=/app/.cache/huggingface/diffusers \
    # Streamlit defaults for containers
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# System deps:
# - libgl1/libglib2.0-0: common runtime deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create cache dirs
RUN mkdir -p /app/.cache/huggingface/transformers /app/.cache/huggingface/diffusers

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
