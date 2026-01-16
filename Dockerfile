FROM python:3.10-slim

# Force rebuild: 2026-01-16-v3
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better caching)
COPY requirements-demo.txt .

# Install CPU-only dependencies (fast, ~30s instead of ~3min)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-demo.txt

# Copy source code
COPY src/ src/
COPY scripts/demo_gradio.py scripts/
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY checkpoints/ checkpoints/
COPY outputs/evaluation_report.json outputs/

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

EXPOSE 7860

CMD ["python", "scripts/demo_gradio.py"]
