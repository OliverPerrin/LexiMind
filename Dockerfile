FROM python:3.12-slim

# Force rebuild: 2026-03-10-v1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better caching)
COPY requirements-demo.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-demo.txt

# Copy source code (demo only needs the gradio script and artifacts)
COPY scripts/demo_gradio.py scripts/
COPY artifacts/ artifacts/

# Copy evaluation metrics if available (demo handles absence gracefully)
RUN mkdir -p outputs
COPY outputs/evaluation_report.jso[n] outputs/

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

EXPOSE 7860

CMD ["python", "scripts/demo_gradio.py"]
