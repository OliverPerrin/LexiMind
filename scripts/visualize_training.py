"""
Visualize training metrics from MLflow runs.

Generates plots showing:
- Loss curves (training/validation)
- Task-specific metrics over time
- Learning rate schedule
- Training speed analysis

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.tracking
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def load_training_history() -> dict[str, object] | None:
    """Load training history from JSON if available."""
    history_path = OUTPUTS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            data: dict[str, object] = json.load(f)
            return data
    return None


def get_latest_run():
    """Get the most recent MLflow run."""
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    client = mlflow.tracking.MlflowClient()

    # Get the experiment (LexiMind)
    experiment = client.get_experiment_by_name("LexiMind")
    if not experiment:
        logger.error("No 'LexiMind' experiment found")
        return None

    # Get all runs, sorted by start time
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        logger.error("No runs found in experiment")
        return None

    return runs[0]


def plot_loss_curves(run):
    """Plot training and validation loss over time."""
    client = mlflow.tracking.MlflowClient()

    # Get metrics
    train_loss = client.get_metric_history(run.info.run_id, "train_total_loss")
    val_loss = client.get_metric_history(run.info.run_id, "val_total_loss")

    fig, ax = plt.subplots(figsize=(12, 6))

    if not train_loss:
        # Create placeholder plot
        ax.text(
            0.5,
            0.5,
            "No training data yet\n\nWaiting for first epoch to complete...",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Extract steps and values
        train_steps = [m.step for m in train_loss]
        train_values = [m.value for m in train_loss]

        ax.plot(train_steps, train_values, label="Training Loss", linewidth=2, alpha=0.8)

        if val_loss:
            val_steps = [m.step for m in val_loss]
            val_values = [m.value for m in val_loss]
            ax.plot(val_steps, val_values, label="Validation Loss", linewidth=2, alpha=0.8)

        ax.legend(fontsize=11)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Progress: Total Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUTS_DIR / "training_loss_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Saved loss curve to {output_path}")
    plt.close()


def plot_task_metrics(run):
    """Plot metrics for each task."""
    client = mlflow.tracking.MlflowClient()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task-Specific Training Metrics", fontsize=16, fontweight="bold")

    # Summarization
    ax = axes[0, 0]
    train_sum = client.get_metric_history(run.info.run_id, "train_summarization_loss")
    val_sum = client.get_metric_history(run.info.run_id, "val_summarization_loss")

    if train_sum:
        ax.plot(
            [m.step for m in train_sum], [m.value for m in train_sum], label="Train", linewidth=2
        )
    if val_sum:
        ax.plot([m.step for m in val_sum], [m.value for m in val_sum], label="Val", linewidth=2)
    ax.set_title("Summarization Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Emotion
    ax = axes[0, 1]
    train_emo = client.get_metric_history(run.info.run_id, "train_emotion_loss")
    val_emo = client.get_metric_history(run.info.run_id, "val_emotion_loss")
    train_f1 = client.get_metric_history(run.info.run_id, "train_emotion_f1")
    val_f1 = client.get_metric_history(run.info.run_id, "val_emotion_f1")

    if train_emo:
        ax.plot(
            [m.step for m in train_emo],
            [m.value for m in train_emo],
            label="Train Loss",
            linewidth=2,
        )
    if val_emo:
        ax.plot(
            [m.step for m in val_emo], [m.value for m in val_emo], label="Val Loss", linewidth=2
        )

    ax2 = ax.twinx()
    if train_f1:
        ax2.plot(
            [m.step for m in train_f1],
            [m.value for m in train_f1],
            label="Train F1",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )
    if val_f1:
        ax2.plot(
            [m.step for m in val_f1],
            [m.value for m in val_f1],
            label="Val F1",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

    ax.set_title("Emotion Detection", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("F1 Score")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Topic
    ax = axes[1, 0]
    train_topic = client.get_metric_history(run.info.run_id, "train_topic_loss")
    val_topic = client.get_metric_history(run.info.run_id, "val_topic_loss")
    train_acc = client.get_metric_history(run.info.run_id, "train_topic_accuracy")
    val_acc = client.get_metric_history(run.info.run_id, "val_topic_accuracy")

    if train_topic:
        ax.plot(
            [m.step for m in train_topic],
            [m.value for m in train_topic],
            label="Train Loss",
            linewidth=2,
        )
    if val_topic:
        ax.plot(
            [m.step for m in val_topic], [m.value for m in val_topic], label="Val Loss", linewidth=2
        )

    ax2 = ax.twinx()
    if train_acc:
        ax2.plot(
            [m.step for m in train_acc],
            [m.value for m in train_acc],
            label="Train Acc",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )
    if val_acc:
        ax2.plot(
            [m.step for m in val_acc],
            [m.value for m in val_acc],
            label="Val Acc",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

    ax.set_title("Topic Classification", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    # Get final metrics
    summary_text = "Final Metrics (Last Epoch)\n" + "=" * 35 + "\n\n"

    if val_topic and val_acc:
        summary_text += f"Topic Accuracy: {val_acc[-1].value:.1%}\n"
    if val_emo and val_f1:
        summary_text += f"Emotion F1: {val_f1[-1].value:.1%}\n"
    if val_sum:
        summary_text += f"Summarization Loss: {val_sum[-1].value:.3f}\n"

    ax.text(0.1, 0.5, summary_text, fontsize=12, family="monospace", verticalalignment="center")

    plt.tight_layout()
    output_path = OUTPUTS_DIR / "task_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Saved task metrics to {output_path}")
    plt.close()


def plot_learning_rate(run):
    """Plot learning rate schedule if available."""
    client = mlflow.tracking.MlflowClient()
    lr_metrics = client.get_metric_history(run.info.run_id, "learning_rate")

    fig, ax = plt.subplots(figsize=(12, 5))

    if not lr_metrics:
        # Create placeholder
        ax.text(
            0.5,
            0.5,
            "No learning rate data yet\n\n(Will be logged in future training runs)",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        steps = [m.step for m in lr_metrics]
        values = [m.value for m in lr_metrics]

        ax.plot(steps, values, linewidth=2, color="darkblue")

        # Mark warmup region
        warmup_steps = 1000  # From config
        if warmup_steps < max(steps):
            ax.axvline(warmup_steps, color="red", linestyle="--", alpha=0.5, label="Warmup End")
            ax.legend()

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule (Cosine with Warmup)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUTS_DIR / "learning_rate_schedule.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Saved LR schedule to {output_path}")
    plt.close()


def main():
    """Generate all training visualizations."""
    logger.info("Loading MLflow data...")

    run = get_latest_run()
    if not run:
        logger.error("No training run found. Make sure training has started.")
        return

    logger.info(f"Analyzing run: {run.info.run_id}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations...")

    plot_loss_curves(run)
    plot_task_metrics(run)
    plot_learning_rate(run)

    logger.info("\n" + "=" * 60)
    logger.info("✓ All visualizations saved to outputs/")
    logger.info("=" * 60)
    logger.info("  - training_loss_curve.png")
    logger.info("  - task_metrics.png")
    logger.info("  - learning_rate_schedule.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
