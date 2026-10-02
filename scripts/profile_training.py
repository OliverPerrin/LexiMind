"""
Profile LexiMind training with PyTorch Profiler.

Runs a few training steps under torch.profiler to capture:
- CUDA kernel timing (per-operator breakdown)
- GPU memory usage (peak allocations, memory timeline)
- CPU/GPU overlap and idle time
- Chrome trace (viewable in chrome://tracing or Perfetto UI)

Outputs:
    outputs/profile/           -- Chrome trace + stacks
    stdout                     -- Summary table of top CUDA operations

Usage:
    python scripts/profile_training.py                   # default: 20 steps
    python scripts/profile_training.py training=default      # explicit reviewed paths required
    PROFILE_STEPS=40 python scripts/profile_training.py   # custom step count

Author: Oliver Perrin
"""

from __future__ import annotations

import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Callable, Dict

import hydra
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_task_dataloaders
from src.data.dataset import load_training_datasets, validate_task_directories
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.models.factory import ModelConfig, build_multitask_model
from src.training.trainer import Trainer, TrainerConfig


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
    if data_cfg.get("topic_problem_type", "single_label") != "single_label":
        raise ValueError(
            "The legacy profiler does not support partial book labels; no profile was started"
        )
    trainer_cfg = cfg.training.get("trainer", {})
    enabled_tasks = list(trainer_cfg.get("tasks", ["summarization", "emotion", "topic"]))
    validate_task_directories(data_cfg.processed, enabled_tasks)

    device = torch.device(cfg.device)
    if device.type != "cuda":
        print("Profiler requires CUDA. Set device=cuda.")
        return

    print(f"Profiling {profile_steps} steps ({warmup_steps} warmup + {active_steps} active)")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # ---------- Setup (mirrors train.py) ----------

    torch.backends.cudnn.benchmark = True
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Index only the required prefix; profiling does not read validation/test.
    max_samples = max(200, profile_steps * 10 * 3)
    train_datasets, _ = load_training_datasets(
        data_cfg.processed, enabled_tasks, max_train_samples=max_samples, include_validation=False
    )
    emotion_classes = getattr(train_datasets.get("emotion"), "emotion_classes", [])
    topic_classes = getattr(train_datasets.get("topic"), "topic_classes", [])

    tok_cfg = data_cfg.get("tokenizer", {})
    max_len = int(cfg.training.get("tokenizer_max_length") or tok_cfg.get("max_length", 512))
    tokenizer = Tokenizer(
        TokenizerConfig(
            pretrained_model_name=tok_cfg.get("pretrained_model_name", "google/flan-t5-base"),
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
        pin_memory=True,
        max_length=max_len,
        classification_max_length=min(256, max_len),
    )

    # Build model
    grad_ckpt = cfg.training.get(
        "gradient_checkpointing", cfg.model.get("gradient_checkpointing", False)
    )
    use_rel_pos = cfg.training.get(
        "use_relative_position_bias", cfg.model.get("use_relative_position_bias", False)
    )

    model_cfg = ModelConfig(
        d_model=cfg.model.d_model,
        vocab_size=getattr(cfg.model, "vocab_size", None),
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        ffn_dim=cfg.model.ffn_dim,
        dropout=cfg.model.dropout,
        use_pretrained=cfg.model.use_pretrained,
        pretrained_model_name=cfg.model.pretrained_model_name,
        activation=getattr(cfg.model, "activation", "gelu"),
        use_relative_position_bias=use_rel_pos,
        gradient_checkpointing=grad_ckpt,
    )

    model = build_multitask_model(
        tokenizer,
        num_emotions=len(emotion_classes),
        num_topics=len(topic_classes),
        config=model_cfg,
    ).to(device)

    # Freeze layers (same as train.py)
    freeze_layers = cfg.training.get("freeze_encoder_layers", 0)
    if freeze_layers > 0:
        if hasattr(model.encoder, "embed_tokens"):
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = False
        if hasattr(model.encoder, "layers"):
            for i, layer in enumerate(model.encoder.layers):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

    # Compile (same as train.py)
    if trainer_cfg.get("use_pcgrad", False):
        torch._functorch.config.donated_buffer = False
    compile_mode = "default" if grad_ckpt else "reduce-overhead"
    compile_dynamic = compile_mode == "default"
    if cfg.training.get("compile_encoder", True):
        model.encoder = torch.compile(model.encoder, mode=compile_mode, dynamic=compile_dynamic)
    if cfg.training.get("compile_decoder", True):
        model.decoder = torch.compile(model.decoder, mode=compile_mode, dynamic=compile_dynamic)

    # Optimizer
    opt_cfg = cfg.training.get("optimizer", {})
    use_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-5)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
        eps=float(opt_cfg.get("eps", 1e-8)),
        betas=tuple(float(value) for value in opt_cfg.get("betas", (0.9, 0.999))),
        fused=use_fused,
    )

    # ---------- Profile loop ----------

    out_dir = PROJECT_ROOT / "outputs" / "profile"
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

    # Warmup outside profiler to let torch.compile finish
    print(f"\nWarmup ({warmup_steps} steps)...")
    profile_epoch(trainer, train_loaders, warmup_steps)
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling ({active_steps} steps)...")
    trace_path = str(out_dir / "trace")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
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
        profile_epoch(trainer, train_loaders, active_steps, prof.step)

    torch.cuda.synchronize()

    # ---------- Summary ----------

    print("\n" + "=" * 80)
    print("TOP CUDA OPERATIONS (by total CUDA time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print("\n" + "=" * 80)
    print("TOP CUDA OPERATIONS (by GPU memory)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))

    # Memory summary
    print("\n" + "=" * 80)
    print("GPU MEMORY SUMMARY")
    print("=" * 80)
    print(torch.cuda.memory_summary(abbreviated=True))

    # Export Chrome trace
    chrome_trace = out_dir / "chrome_trace.json"
    prof.export_chrome_trace(str(chrome_trace))
    print(f"\nChrome trace: {chrome_trace}")
    print("  Open in: chrome://tracing or https://ui.perfetto.dev")

    # Export stacks for flamegraph
    stacks_path = out_dir / "profiler_stacks.txt"
    prof.export_stacks(str(stacks_path), "self_cuda_time_total")
    print(f"CUDA stacks: {stacks_path}")
    print(f"  Generate flamegraph: flamegraph.pl {stacks_path} > flamegraph.svg")

    print(f"\nTensorBoard traces: {trace_path}/")
    print(f"  View with: tensorboard --logdir={trace_path}")


if __name__ == "__main__":
    main()
