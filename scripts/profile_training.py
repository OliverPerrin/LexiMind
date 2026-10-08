"""
Profile the production LexiMind training loop on CUDA, MPS or CPU.

Runs a few training steps under torch.profiler to capture:
- CUDA kernel timing, or CPU operator traces on MPS/CPU
- CUDA allocation traces/peak, or sampled MPS driver allocations
- Synchronized diagnostic step timing and observed device memory
- Chrome trace (viewable in chrome://tracing or Perfetto UI)

Outputs:
    outputs/profile/           -- Chrome trace + stacks
    summary.json + stdout     -- Timing, memory basis and operator summaries

Usage:
    python scripts/profile_training.py                   # default: 20 steps
    python scripts/profile_training.py training=default      # explicit reviewed paths required
    PROFILE_STEPS=40 python scripts/profile_training.py   # custom step count
    PROFILE_OUTPUT_DIR=outputs/profile-mps python scripts/profile_training.py training=book_lora device=mps data.processed.topic=/path/to/reviewed/splits

Author: Oliver Perrin
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Dict

import hydra
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import set_seed
from src.data.dataloader import build_task_dataloaders
from src.data.dataset import load_training_datasets, validate_task_directories
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.training.trainer import Trainer, TrainerConfig
from src.training.utils import (
    build_training_model,
    build_training_optimizer,
    prepare_training_runtime,
)


class ProfileLoader:
    """Use the real loader with an exact outer-step budget; Trainer handles cycling."""

    def __init__(self, loader, steps: int):
        if steps < 1 or len(loader) == 0:
            raise ValueError("Profiling requires positive steps and nonempty task loaders")
        self.loader, self.steps, self.dataset = loader, steps, loader.dataset

    def __len__(self):
        return self.steps

    def __iter__(self):
        return iter(self.loader)


def profile_epoch(
    trainer: Trainer, loaders: Dict, steps: int, step_callback: Callable[[], None] | None = None
) -> Dict[str, float]:
    """Measure the production training loop, including its accumulation and metrics."""
    return trainer._run_epoch(
        {task: ProfileLoader(loader, steps) for task, loader in loaders.items()},
        train=True,
        epoch=1,
        step_callback=step_callback,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    profile_steps = int(os.environ.get("PROFILE_STEPS", 20))
    warmup_steps = 3  # let CUDA graphs / torch.compile settle
    active_steps = profile_steps - warmup_steps
    if active_steps <= 3:
        raise ValueError("PROFILE_STEPS must be at least 7 for warmup and an active trace")

    data_cfg = cfg.data
    if data_cfg.get("topic_problem_type", "single_label") != "single_label" and not cfg.get(
        "training", {}
    ).get("book_lora", {}).get("enabled", False):
        raise ValueError(
            "The legacy profiler does not support partial book labels; no profile was started"
        )
    if cfg.get("resume_from"):
        raise ValueError("Profiler starts from the configured base; resume_from is unsupported")
    trainer_cfg = cfg.training.get("trainer", {})
    enabled_tasks = list(trainer_cfg.get("tasks", ["summarization", "emotion", "topic"]))
    validate_task_directories(data_cfg.processed, enabled_tasks)

    set_seed(cfg.seed)
    device, snapshot = prepare_training_runtime(cfg)
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError("Profiler supports CUDA, MPS or CPU")

    print(f"Profiling {profile_steps} steps ({warmup_steps} warmup + {active_steps} active)")
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else str(device)
    print(f"Device: {device_name}")

    # ---------- Setup (mirrors train.py) ----------

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Index only the required prefix; profiling does not read validation/test.
    max_samples = max(200, profile_steps * 10 * 3)
    train_datasets, _ = load_training_datasets(
        data_cfg.processed,
        enabled_tasks,
        max_train_samples=max_samples,
        include_validation=False,
        topic_problem_type=data_cfg.get("topic_problem_type", "single_label"),
    )
    emotion_classes = getattr(train_datasets.get("emotion"), "emotion_classes", [])
    topic_classes = getattr(train_datasets.get("topic"), "topic_classes", [])

    tok_cfg = data_cfg.get("tokenizer", {})
    max_len = int(cfg.training.get("tokenizer_max_length") or tok_cfg.get("max_length", 512))
    tokenizer = Tokenizer(
        TokenizerConfig(
            pretrained_model_name=str(snapshot)
            if snapshot
            else tok_cfg.get("pretrained_model_name", "google/flan-t5-base"),
            max_length=max_len,
        )
    )

    dl_cfg = cfg.training.get("dataloader", {})
    train_loaders = build_task_dataloaders(
        train_datasets,
        tokenizer,
        shuffle=True,
        batch_size=int(dl_cfg.get("batch_size", 8)),
        num_workers=int(dl_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        max_length=max_len,
        classification_max_length=min(256, max_len),
    )

    model, adapter_binding = build_training_model(
        cfg,
        tokenizer,
        num_emotions=len(emotion_classes),
        num_topics=len(topic_classes),
        topic_problem_type=data_cfg.get("topic_problem_type", "single_label"),
        snapshot=snapshot,
    )
    optimizer = build_training_optimizer(model, cfg, device)

    # ---------- Profile loop ----------

    out_dir = Path(os.environ.get("PROFILE_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "profile")))
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        field.name: trainer_cfg[field.name]
        for field in fields(TrainerConfig)
        if field.name in trainer_cfg
    }
    scheduler_cfg = cfg.training.get("scheduler", {})
    settings.update(
        scheduler_type=scheduler_cfg.get("name", "cosine"),
        warmup_steps=int(scheduler_cfg.get("warmup_steps", 500)),
    )
    trainer = Trainer(model, optimizer, TrainerConfig(**settings), device, tokenizer)
    trainer._setup_scheduler(
        {task: ProfileLoader(loader, profile_steps) for task, loader in train_loaders.items()}, 1
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    observed_memory = []

    def memory():
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device)
        if device.type == "mps":
            return torch.mps.driver_allocated_memory()
        return None

    # Warmup outside profiler to let torch.compile finish
    print(f"\nWarmup ({warmup_steps} steps)...")
    profile_epoch(trainer, train_loaders, warmup_steps)
    synchronize()

    # Profile
    print(f"Profiling ({active_steps} steps)...")
    trace_path = str(out_dir / "trace")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_times = []
    started = previous = time.perf_counter()

    def completed_step():
        nonlocal previous
        synchronize()
        now = time.perf_counter()
        step_times.append(now - previous)
        observed_memory.append(memory())
        prof.step()
        previous = time.perf_counter()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=2,
            active=active_steps - 3,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        metrics = profile_epoch(trainer, train_loaders, active_steps, completed_step)

    synchronize()

    elapsed = time.perf_counter() - started
    report = {
        "device": str(device),
        "profile_steps": profile_steps,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "batch_size": int(dl_cfg.get("batch_size", 8)),
        "gradient_accumulation_steps": trainer.config.gradient_accumulation_steps,
        "nominal_effective_batch_size": int(dl_cfg.get("batch_size", 8))
        * trainer.config.gradient_accumulation_steps,
        "optimizer_updates_including_warmup": trainer.global_step,
        "accumulation_boundary": "Warmup and active loops each flush their final partial accumulation window",
        "synchronized_elapsed_seconds": elapsed,
        "step_seconds": step_times,
        "memory_basis": "CUDA allocator peak"
        if device.type == "cuda"
        else "MPS driver samples"
        if device.type == "mps"
        else "unavailable",
        "max_observed_device_bytes": max(
            (value for value in observed_memory if value is not None), default=None
        ),
        "metrics": metrics,
        "test_split_opened": False,
        "adapter_binding": asdict(adapter_binding) if adapter_binding else None,
        "model_config": model._leximind_training_model_config,
        "indexed_train_samples": {task: len(dataset) for task, dataset in train_datasets.items()},
        "trace_scope": "CPU and CUDA kernels"
        if device.type == "cuda"
        else "CPU operations; MPS device timing measured by synchronization"
        if device.type == "mps"
        else "CPU operations",
        "timing_scope": "Diagnostic loop with profiler, per-step synchronization and metrics; not unconstrained throughput",
        "training_dtype": "bfloat16 autocast" if trainer.use_bfloat16 else "float32",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=25
        )
    )
    print(json.dumps(report, indent=2))

    # Export Chrome trace
    chrome_trace = out_dir / "chrome_trace.json"
    # tensorboard_trace_handler already saved this Kineto trace; exporting the
    # same profiler result again fails on recent PyTorch releases.
    saved_traces = list(Path(trace_path).glob("*.pt.trace.json"))
    if not saved_traces:
        raise RuntimeError("Profiler callback did not save an active Chrome trace")
    shutil.copyfile(max(saved_traces, key=lambda path: path.stat().st_mtime_ns), chrome_trace)
    print(f"\nChrome trace: {chrome_trace}")
    print("  Open in: chrome://tracing or https://ui.perfetto.dev")

    # Export stacks for flamegraph
    stacks_path = out_dir / "profiler_stacks.txt"
    prof.export_stacks(
        str(stacks_path), "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    print(f"Profiler stacks: {stacks_path}")
    print(f"  Generate flamegraph: flamegraph.pl {stacks_path} > flamegraph.svg")

    print(f"\nTensorBoard traces: {trace_path}/")
    print(f"  View with: tensorboard --logdir={trace_path}")


if __name__ == "__main__":
    main()
