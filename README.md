# LexiMind: A Multi-Task NLP Model

LexiMind is a state-of-the-art Natural Language Processing model designed for complex document understanding. It leverages a modern, pre-trained Transformer architecture to perform three sophisticated tasks simultaneously: text summarization, emotion classification, and topic clustering.

This project is built with industry-standard MLOps practices, including configuration management with Hydra, experiment tracking with MLflow, and containerization with Docker, making it a reproducible and scalable solution.

## Core Features

*   **Abstractive Summarization:** Generates concise, coherent summaries of long-form text.
*   **Emotion Classification:** Identifies the primary emotion (e.g., Joy, Sadness, Anger) conveyed in a document.
*   **Topic Clustering:** Groups documents into thematic clusters based on their content.

## Model Architecture

LexiMind is built on a powerful pre-trained Transformer backbone (such as FLAN-T5), which is fine-tuned for high performance on the specified tasks. To ensure computational efficiency without sacrificing accuracy, the model is trained using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).

The model employs a multi-task learning framework, with a shared encoder-decoder core and distinct output heads for each task. This approach allows the model to learn rich, generalized representations of language, improving performance across all functions. Training is accelerated using Flash Attention and mixed-precision computation.

## Getting Started

### Prerequisites

*   Python 3.10+
*   Poetry for dependency management
*   Docker (for containerized deployment)
*   An NVIDIA GPU with CUDA support (for training and accelerated inference)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LexiMind.git
    cd LexiMind
    ```

2.  **Install dependencies:**
    Poetry will handle the virtual environment and package installation.
    ```bash
    poetry install
    ```

3.  **Download dataset:**
    (Instructions for downloading your specific dataset would go here)
    ```bash
    poetry run python scripts/download_data.py
    ```

4.  **Preprocess data:**
    ```bash
    poetry run python scripts/preprocess_data.py
    ```

## Usage

### Configuration

All training and model parameters are managed via Hydra. Configurations are located in the `configs/` directory. You can easily override parameters from the command line.

### Training

To start the training process with a base configuration:

```bash
poetry run python src/train.py
```

To override a parameter, such as the learning rate:

```bash
poetry run python src/train.py training.learning_rate=5e-5
```

Experiments are automatically tracked with MLflow. You can view results by running `mlflow ui` in your terminal.

### Evaluation

To evaluate a trained model checkpoint against the test set:

```bash
poetry run python src/evaluate.py model_checkpoint=checkpoints/best.pt
```

Evaluation metrics and model outputs will be saved to the `outputs/` directory.

### Inference & Demo

A Gradio demo is available to interact with the trained model. To launch it:

```bash
poetry run python scripts/demo_gradio.py
```

Navigate to the local URL provided to access the web interface for summarization, classification, and clustering.

## Docker

For fully reproducible builds and easy deployment, you can use the provided Dockerfile.

1.  **Build the Docker image:**
    ```bash
    docker build -t leximind .
    ```

2.  **Run the Gradio demo in a container:**
    ```bash
    docker run -p 7860:7860 leximind
    ```

## Project Structure

```
├── configs/            # Hydra configuration files
├── data/               # Raw, processed, and external data
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── scripts/            # Helper scripts (data download, demo, etc.)
├── src/                # Core source code for the model and training
│   ├── data/           # Data loading and preprocessing
│   ├── model/          # Model architecture and components
│   └── training/       # Training and evaluation loops
├── tests/              # Unit and integration tests
├── Dockerfile          # Docker configuration
├── pyproject.toml      # Project metadata and dependencies (for Poetry)
└── README.md
```

## Code Quality

This project enforces high code quality standards using the following tools:

*   **Ruff:** For lightning-fast linting and code formatting.
*   **MyPy:** For static type checking.

These checks are automated on every commit using pre-commit hooks. To set them up, run:

```bash
poetry run pre-commit install
```