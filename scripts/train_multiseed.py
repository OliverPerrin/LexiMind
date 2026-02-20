#!/usr/bin/env python3
"""
Multi-seed training wrapper for LexiMind.

Runs training across multiple seeds and aggregates results with mean ± std.
This addresses the single-seed limitation identified in review feedback.

Usage:
    python scripts/train_multiseed.py --seeds 17 42 123 --config training=full
    python scripts/train_multiseed.py --seeds 17 42 123 456 789 --config training=medium

Author: Oliver Perrin
Date: February 2026
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def run_single_seed(seed: int, config_overrides: str, base_dir: Path) -> Dict:
    """Run training for a single seed and return the training history."""
    seed_dir = base_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/train.py",
        f"seed={seed}",
        f"checkpoint_out={seed_dir}/checkpoints/best.pt",
        f"history_out={seed_dir}/training_history.json",
        f"labels_out={seed_dir}/labels.json",
    ]
    if config_overrides:
        cmd.extend(config_overrides.split())

    print(f"\n{'='*60}")
    print(f"Training seed {seed}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: Seed {seed} training failed (exit code {result.returncode})")
        return {}

    history_path = seed_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return {}


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
        sys.executable, "scripts/evaluate.py",
        f"--checkpoint={checkpoint}",
        f"--labels={labels}",
        f"--output={output}",
        "--skip-bertscore",
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
            return json.load(f)
    return {}


def aggregate_results(all_results: Dict[int, Dict]) -> Dict:
    """Aggregate evaluation results across seeds with mean ± std."""
    if not all_results:
        return {}

    # Collect all metric paths
    metric_values: Dict[str, List[float]] = {}
    for seed, results in all_results.items():
        for task, task_metrics in results.items():
            if not isinstance(task_metrics, dict):
                continue
            for metric_name, value in task_metrics.items():
                if isinstance(value, (int, float)) and metric_name != "num_samples" and metric_name != "num_classes":
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
    print(f"\n{'='*70}")
    print(f"MULTI-SEED RESULTS SUMMARY ({len(seeds)} seeds: {seeds})")
    print(f"{'='*70}")

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
                print(f"    {metric:25s}: {mean*100:.1f}% ± {std*100:.1f}%")
            else:
                print(f"    {metric:25s}: {mean:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training for LexiMind")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 123],
                        help="Random seeds to train with")
    parser.add_argument("--config", type=str, default="",
                        help="Hydra config overrides (e.g., 'training=full')")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multiseed"),
                        help="Base output directory")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only aggregate existing results")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only aggregate training histories")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training phase
    if not args.skip_training:
        for seed in args.seeds:
            run_single_seed(seed, args.config, args.output_dir)

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

        # Save aggregated results
        output_path = args.output_dir / "aggregated_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "seeds": args.seeds,
                "per_seed": {str(k): v for k, v in all_eval_results.items()},
                "aggregated": aggregated,
            }, f, indent=2)
        print(f"\n  Saved to: {output_path}")
    else:
        print("\nNo evaluation results to aggregate.")


if __name__ == "__main__":
    main()
