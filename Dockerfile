FROM python:3.12-slim

# Force rebuild: 2026-03-10-v3
WORKDIR /app

# Copy only requirements first (for better caching)
COPY requirements-demo.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-demo.txt

# Copy source code (demo only needs the gradio script and data)
COPY scripts/demo_gradio.py scripts/
COPY data/discovery_dataset.jsonl data/

# Copy evaluation metrics if available (demo handles absence gracefully)
RUN mkdir -p outputs
COPY outputs/evaluation_report.jso[n] outputs/

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV PYTHONUNBUFFERED="1"

EXPOSE 7860

CMD ["python", "scripts/demo_gradio.py"]
