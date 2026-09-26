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
    metric_seeds: Dict[str, List[int]] = {}
    for seed, results in all_results.items():
        for task, task_metrics in results.items():
            if task.startswith("_") or not isinstance(task_metrics, dict):
                continue
            for metric_name, value in task_metrics.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and metric_name != "num_samples"
                    and metric_name != "num_classes"
                ):
                    if not np.isfinite(value):
                        raise ValueError(
                            f"Non-finite report value for seed {seed}: {task}/{metric_name}"
                        )
                    key = f"{task}/{metric_name}"
                    metric_values.setdefault(key, []).append(float(value))
                    metric_seeds.setdefault(key, []).append(seed)

    aggregated = {}
    for key, values in sorted(metric_values.items()):
        arr = np.array(values)
        aggregated[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(values) > 1 else None,
            "std_definition": "sample standard deviation; unavailable with one seed",
            "seeds": metric_seeds[key],
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": len(values),
        }

    return aggregated


def _format_result(stats: Dict, *, percent: bool = False, latex: bool = False) -> str:
    scale = 100 if percent else 1
    mean = stats["mean"] * scale
    spread = stats["std"]
    precision = 1 if percent else 4
    value = f"{mean:.{precision}f}"
    if percent:
        value += r"\%" if latex else "%"
    if spread is not None:
        value += (r" $\pm$ " if latex else " ± ") + f"{spread * scale:.{precision}f}"
    else:
        value += " (spread unmeasured)"
    return value + f"; n={stats['n_seeds']}"


def print_summary(aggregated: Dict, seeds: List[int]) -> None:
    """Each metric reports its actual successful seed count, including partial runs."""
    print(f"\nRESULTS FROM {len(seeds)} REPORTS: {seeds}")
    for key, stats in sorted(aggregated.items()):
        print(f"  {key}: {_format_result(stats, percent='accuracy' in key)}")


def generate_latex_table(aggregated: Dict, seeds: List[int]) -> str:
    """Render observed metrics with per-metric sample counts, never missing-run zeros."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{LexiMind stored report values ({len(seeds)} reports; per-metric seed counts shown)}}",
        r"\begin{tabular}{ll}",
        r"\hline",
        r"Metric & Value \\",
        r"\hline",
    ]
    for key, stats in sorted(aggregated.items()):
        name = key.replace("_", r"\_")
        lines.append(
            name + " & " + _format_result(stats, percent="accuracy" in key, latex=True) + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
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
        help="Skip training; model evaluation still runs unless --skip-eval is also passed",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip model evaluation and evaluation-result aggregation",
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
    failed_training_seeds = set()
    if not args.skip_training:
        train_start = time.time()
        for i, seed in enumerate(args.seeds):
            conflict_freq = args.conflict_frequency if seed == args.conflict_seed else 0
            history, seed_elapsed = run_single_seed(
                seed,
                args.config,
                args.output_dir,
                use_pcgrad=args.use_pcgrad,
                gradient_conflict_frequency=conflict_freq,
            )
            per_seed_hours[seed] = seed_elapsed
            remaining = (n_seeds - i - 1) * seed_elapsed
            if history:
                print(f"\n  Seed {seed} completed in {seed_elapsed:.1f}h")
            else:
                failed_training_seeds.add(seed)
                print(
                    f"\n  Seed {seed} did not produce a completed run; elapsed {seed_elapsed:.1f}h"
                )
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
            if seed in failed_training_seeds:
                print(
                    f"  Skipping seed {seed} evaluation after failed training; old checkpoints are not reused"
                )
                continue
            result = run_evaluation(seed, args.output_dir)
            if result:
                all_eval_results[seed] = result

    # Aggregate and save
    if all_eval_results:
        aggregated = aggregate_results(all_eval_results)
        completed_seeds = list(all_eval_results)
        print_summary(aggregated, completed_seeds)

        # Generate and print LaTeX table
        latex = generate_latex_table(aggregated, completed_seeds)
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
                    "seeds": completed_seeds,
                    "requested_seeds": args.seeds,
                    "missing_seeds": [seed for seed in args.seeds if seed not in all_eval_results],
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
