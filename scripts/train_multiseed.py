"""
Multi-seed training wrapper for LexiMind.

Runs training across multiple seeds and aggregates results with mean ± std.
This addresses the single-seed limitation identified in review feedback.

Usage:
    python scripts/train_multiseed.py --seeds 17 42 123 --config training=full
    python scripts/train_multiseed.py --seeds 17 42 123 456 789 --config training=medium
    python scripts/train_multiseed.py --seeds 42 123 456 789 1000 --use-pcgrad --config training=full

Author: Oliver Perrin
Date: February 2026
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Estimated training time per seed on RTX 4070 12GB (hours).
# ~62K summ + 43K emotion + 3.4K topic, batch_size=6, grad_accum=7,
# compile_decoder=false, dynamic padding on all tasks, ROUGE/BLEU gated
# to validation only. Pre-April-2026 builds were ~12h/seed.
ESTIMATED_HOURS_PER_SEED = 8.0


def run_single_seed(
    seed: int,
    config_overrides: str,
    base_dir: Path,
    use_pcgrad: bool = False,
    gradient_conflict_frequency: int = 0,
) -> tuple[Dict, float]:
    """Run training for a single seed and return (history, wall_clock_hours)."""
    seed_dir = base_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/train.py",
        f"seed={seed}",
        f"checkpoint_out={seed_dir}/checkpoints/best.pt",
        f"history_out={seed_dir}/training_history.json",
        f"labels_out={seed_dir}/labels.json",
    ]
    if use_pcgrad:
        cmd.append("training.trainer.use_pcgrad=true")
    if gradient_conflict_frequency > 0:
        cmd.append(f"training.trainer.gradient_conflict_frequency={gradient_conflict_frequency}")
    if config_overrides:
        cmd.extend(config_overrides.split())

    print(f"\n{'=' * 60}")
    print(f"Training seed {seed}")
    if gradient_conflict_frequency > 0:
        print(f"  Gradient-conflict diagnostics: every {gradient_conflict_frequency} steps")
    print(f"{'=' * 60}")
    log_path = seed_dir / "train.log"
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_path} (tail -f to watch live)")

    # Allocate a pty so tqdm (and anything else TTY-sensitive) keeps in-place
    # updates. Raw bytes (including \r carriage returns) are mirrored to both
    # the terminal and the log file. To read the log later without the CR spam:
    #   sed 's/.*\r//g' outputs/multiseed_emnlp/seed_<N>/train.log
    seed_start = time.time()
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        cmd, stdout=slave_fd, stderr=slave_fd, stdin=slave_fd, close_fds=True
    )
    os.close(slave_fd)
    stdout_fd = sys.stdout.fileno()
    with open(log_path, "wb") as log_file:
        try:
            while True:
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    os.write(stdout_fd, data)
                    log_file.write(data)
                    log_file.flush()
                elif process.poll() is not None:
                    break
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass
    process.wait()
    wall_hours = (time.time() - seed_start) / 3600.0

    if process.returncode != 0:
        print(
            f"  WARNING: Seed {seed} training failed (exit code {process.returncode}); "
            f"full traceback in {log_path}"
        )
        return {}, wall_hours

    history_path = seed_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            data: Dict = json.load(f)  # type: ignore[no-any-return]
            return data, wall_hours
    return {}, wall_hours


def run_evaluation(seed: int, base_dir: Path, extra_args: List[str] | None = None) -> Dict:
    """Run evaluation for a single seed and return results."""
    seed_dir = base_dir / f"seed_{seed}"
    checkpoint = seed_dir / "checkpoints" / "best.pt"
    labels = seed_dir / "labels.json"
    output = seed_dir / "evaluation_report.json"

    if not checkpoint.exists():
        print(f"  Skipping eval for seed {seed}: no checkpoint found")
        return {}

    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        f"--checkpoint={checkpoint}",
        f"--labels={labels}",
        f"--output={output}",
        "--split=test",
        "--tune-thresholds",
        "--bootstrap",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  Evaluating seed {seed}...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: Seed {seed} evaluation failed")
        return {}

    if output.exists():
        with open(output) as f:
            data: Dict = json.load(f)  # type: ignore[no-any-return]
            return data
    return {}


def aggregate_results(all_results: Dict[int, Dict]) -> Dict:
    """Aggregate evaluation results across seeds with mean ± std."""
    if not all_results:
        return {}

    # Collect all metric paths
    metric_values: Dict[str, List[float]] = {}
    for _seed, results in all_results.items():
        for task, task_metrics in results.items():
            if not isinstance(task_metrics, dict):
                continue
            for metric_name, value in task_metrics.items():
                if (
                    isinstance(value, (int, float))
                    and metric_name != "num_samples"
                    and metric_name != "num_classes"
                ):
                    key = f"{task}/{metric_name}"
                    metric_values.setdefault(key, []).append(float(value))

    aggregated: Dict[str, Dict[str, float]] = {}
    for key, values in sorted(metric_values.items()):
        arr = np.array(values)
        aggregated[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": len(values),
        }

    return aggregated


def print_summary(aggregated: Dict, seeds: List[int]) -> None:
    """Print human-readable summary of multi-seed results."""
    print(f"\n{'=' * 70}")
    print(f"MULTI-SEED RESULTS SUMMARY ({len(seeds)} seeds: {seeds})")
    print(f"{'=' * 70}")

    # Group by task
    tasks: Dict[str, Dict[str, Dict]] = {}
    for key, stats in aggregated.items():
        task, metric = key.split("/", 1)
        tasks.setdefault(task, {})[metric] = stats

    for task, metrics in sorted(tasks.items()):
        print(f"\n  {task.upper()}:")
        for metric, stats in sorted(metrics.items()):
            mean = stats["mean"]
            std = stats["std"]
            # Format based on metric type
            if "accuracy" in metric:
                print(f"    {metric:25s}: {mean * 100:.1f}% ± {std * 100:.1f}%")
            else:
                print(f"    {metric:25s}: {mean:.4f} ± {std:.4f}")


def generate_latex_table(aggregated: Dict, seeds: List[int]) -> str:
    """Generate a LaTeX-ready table with mean ± std for key metrics.

    Produces a table suitable for direct inclusion in an EMNLP paper.
    """
    # Define the metrics we want in the table, grouped by task
    metric_rows = [
        (
            "Summarization",
            [
                ("ROUGE-1", "summarization/rouge1"),
                ("ROUGE-2", "summarization/rouge2"),
                ("ROUGE-L", "summarization/rougeL"),
            ],
        ),
        (
            "Topic",
            [
                ("Accuracy", "topic/accuracy"),
                ("Macro F1", "topic/macro_f1"),
            ],
        ),
        (
            "Emotion",
            [
                ("Sample-avg F1", "emotion/sample_avg_f1"),
                ("Macro F1", "emotion/macro_f1"),
                ("Micro F1", "emotion/micro_f1"),
            ],
        ),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{LexiMind results ({len(seeds)} seeds)}}",
        r"\label{tab:multiseed}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Score} \\",
        r"\midrule",
    ]

    for task_name, task_metrics in metric_rows:
        lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{task_name}}}}} \\\\")
        for display_name, key in task_metrics:
            if key in aggregated:
                mean = aggregated[key]["mean"]
                std = aggregated[key]["std"]
                if "accuracy" in key:
                    lines.append(
                        f"\\quad {display_name} & ${mean * 100:.1f} \\pm {std * 100:.1f}$ \\\\"
                    )
                else:
                    lines.append(f"\\quad {display_name} & ${mean:.4f} \\pm {std:.4f}$ \\\\")
            else:
                lines.append(f"\\quad {display_name} & -- \\\\")

    # Check for frozen tuned metrics
    frozen_keys = [k for k in aggregated if "frozen_tuned" in k]
    if frozen_keys:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{2}{l}{\textit{Emotion (val-tuned $\tau$)}} \\")
        for key in sorted(frozen_keys):
            metric_name = key.split("/")[-1].replace("frozen_tuned_", "").replace("_", " ").title()
            mean = aggregated[key]["mean"]
            std = aggregated[key]["std"]
            lines.append(f"\\quad {metric_name} & ${mean:.4f} \\pm {std:.4f}$ \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training for LexiMind")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[17, 42, 123], help="Random seeds to train with"
    )
    parser.add_argument(
        "--config", type=str, default="", help="Hydra config overrides (e.g., 'training=full')"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/multiseed"), help="Base output directory"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only aggregate existing results",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation, only aggregate training histories",
    )
    parser.add_argument(
        "--use-pcgrad",
        action="store_true",
        help="Enable PCGrad (gradient surgery) for all seed runs",
    )
    parser.add_argument(
        "--conflict-seed",
        type=int,
        default=None,
        help=(
            "Seed for which to enable gradient-conflict diagnostics "
            "(inter-task cosine similarity every --conflict-frequency steps)"
        ),
    )
    parser.add_argument(
        "--conflict-frequency",
        type=int,
        default=500,
        help="Steps between gradient-conflict diagnostics measurements",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Print time estimate
    n_seeds = len(args.seeds)
    est_hours = n_seeds * ESTIMATED_HOURS_PER_SEED
    pcgrad_note = " (with PCGrad)" if args.use_pcgrad else ""
    print(f"\n{'=' * 60}")
    print(f"MULTI-SEED TRAINING{pcgrad_note}")
    print(f"{'=' * 60}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Number of runs: {n_seeds}")
    print(f"  Estimated time per seed: ~{ESTIMATED_HOURS_PER_SEED:.0f} hours (RTX 4070 12GB)")
    print(f"  Estimated total time: ~{est_hours:.0f} hours ({est_hours / 24:.1f} days)")
    if args.use_pcgrad:
        print("  PCGrad: ENABLED (will propagate to each seed run)")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'=' * 60}")

    # Training phase
    per_seed_hours: Dict[int, float] = {}
    if not args.skip_training:
        train_start = time.time()
        for i, seed in enumerate(args.seeds):
            conflict_freq = args.conflict_frequency if seed == args.conflict_seed else 0
            _, seed_elapsed = run_single_seed(
                seed,
                args.config,
                args.output_dir,
                use_pcgrad=args.use_pcgrad,
                gradient_conflict_frequency=conflict_freq,
            )
            per_seed_hours[seed] = seed_elapsed
            remaining = (n_seeds - i - 1) * seed_elapsed
            print(f"\n  Seed {seed} completed in {seed_elapsed:.1f}h")
            if i < n_seeds - 1:
                print(f"  Estimated remaining: ~{remaining:.1f}h")
        total_train = (time.time() - train_start) / 3600
        print(f"\n  Total training time: {total_train:.1f}h")

        # Persist per-seed wall-clock for the paper's "9.1 ± 0.4h" style reporting.
        wall_path = args.output_dir / "wall_clock.json"
        with open(wall_path, "w") as f:
            json.dump(
                {
                    "per_seed_hours": {str(k): v for k, v in per_seed_hours.items()},
                    "total_hours": sum(per_seed_hours.values()),
                    "mean_hours": (
                        sum(per_seed_hours.values()) / len(per_seed_hours)
                        if per_seed_hours
                        else 0.0
                    ),
                },
                f,
                indent=2,
            )
        print(f"  Wall-clock summary: {wall_path}")

    # Evaluation phase
    all_eval_results: Dict[int, Dict] = {}
    if not args.skip_eval:
        for seed in args.seeds:
            result = run_evaluation(seed, args.output_dir)
            if result:
                all_eval_results[seed] = result

    # Aggregate and save
    if all_eval_results:
        aggregated = aggregate_results(all_eval_results)
        print_summary(aggregated, args.seeds)

        # Generate and print LaTeX table
        latex = generate_latex_table(aggregated, args.seeds)
        print(f"\n{'=' * 70}")
        print("LATEX TABLE (copy-paste into paper)")
        print(f"{'=' * 70}")
        print(latex)

        # Save LaTeX table to file
        latex_path = args.output_dir / "results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\n  LaTeX table saved to: {latex_path}")

        # Save aggregated results
        output_path = args.output_dir / "aggregated_results.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "seeds": args.seeds,
                    "use_pcgrad": args.use_pcgrad,
                    "conflict_seed": args.conflict_seed,
                    "per_seed": {str(k): v for k, v in all_eval_results.items()},
                    "per_seed_hours": {str(k): v for k, v in per_seed_hours.items()},
                    "aggregated": aggregated,
                },
                f,
                indent=2,
            )
        print(f"  Saved to: {output_path}")
    else:
        print("\nNo evaluation results to aggregate.")


if __name__ == "__main__":
    main()
