FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (models mounted via volume)
COPY src/ ./src/

# Default environment
ENV HOST=0.0.0.0
ENV PORT=7779
ENV CHECKPOINT_PATH=/models/checkpoint_best_ppl_8.65.pth
ENV TOKENIZER_PATH=/models/tokenizer/tinystories_10k
ENV DEVICE=cuda

EXPOSE 7779

CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7779"]
