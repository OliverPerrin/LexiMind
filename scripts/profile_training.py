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
    python scripts/profile_training.py training=full      # use full config
    PROFILE_STEPS=40 python scripts/profile_training.py   # custom step count

Author: Oliver Perrin
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import (
    build_emotion_dataloader,
    build_summarization_dataloader,
    build_topic_dataloader,
)
from src.data.dataset import (
    EmotionDataset,
    SummarizationDataset,
    TopicDataset,
    load_emotion_jsonl,
    load_summarization_jsonl,
    load_topic_jsonl,
)
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.models.factory import ModelConfig, build_multitask_model


def load_splits(data_dir: Path, loader_fn):
    splits = {}
    for name, aliases in [("train", ["train"]), ("val", ["val", "validation"])]:
        for alias in aliases:
            path = data_dir / f"{alias}.jsonl"
            if path.exists():
                splits[name] = loader_fn(str(path))
                break
    return splits


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    profile_steps = int(os.environ.get("PROFILE_STEPS", 20))
    warmup_steps = 3  # let CUDA graphs / torch.compile settle
    active_steps = profile_steps - warmup_steps

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

    data_cfg = cfg.data
    trainer_cfg = cfg.training.get("trainer", {})

    # Load small subsets -- profiling doesn't need the full dataset
    max_samples = max(200, profile_steps * 10 * 3)
    summ_splits = load_splits(Path(data_cfg.processed.summarization), load_summarization_jsonl)
    emot_splits = load_splits(Path(data_cfg.processed.emotion), load_emotion_jsonl)
    topic_splits = load_splits(Path(data_cfg.processed.topic), load_topic_jsonl)
    for splits in [summ_splits, emot_splits, topic_splits]:
        splits["train"] = splits["train"][:max_samples]

    tok_cfg = data_cfg.get("tokenizer", {})
    max_len = int(cfg.training.get("tokenizer_max_length") or tok_cfg.get("max_length", 512))
    tokenizer = Tokenizer(TokenizerConfig(
        pretrained_model_name=tok_cfg.get("pretrained_model_name", "google/flan-t5-base"),
        max_length=max_len,
    ))

    summ_train = SummarizationDataset(summ_splits["train"])
    emot_train = EmotionDataset(emot_splits["train"])
    topic_train = TopicDataset(topic_splits["train"])

    dl_cfg = cfg.training.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", 8))
    num_workers = int(dl_cfg.get("num_workers", 4))
    classification_max_len = min(256, max_len)

    train_loaders = {
        "summarization": build_summarization_dataloader(
            summ_train, tokenizer, shuffle=True,
            max_source_length=max_len, max_target_length=max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        ),
        "emotion": build_emotion_dataloader(
            emot_train, tokenizer, shuffle=True, max_length=classification_max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        ),
        "topic": build_topic_dataloader(
            topic_train, tokenizer, shuffle=True, max_length=classification_max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        ),
    }

    # Build model
    grad_ckpt = cfg.training.get("gradient_checkpointing", cfg.model.get("gradient_checkpointing", False))
    use_rel_pos = cfg.training.get("use_relative_position_bias", cfg.model.get("use_relative_position_bias", False))

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
        num_emotions=len(emot_train.emotion_classes),
        num_topics=len(topic_train.topic_classes),
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
    compile_mode = "default" if grad_ckpt else "reduce-overhead"
    if cfg.training.get("compile_encoder", True):
        model.encoder = torch.compile(model.encoder, mode=compile_mode)
    if cfg.training.get("compile_decoder", True):
        model.decoder = torch.compile(model.decoder, mode=compile_mode)

    # Optimizer
    opt_cfg = cfg.training.get("optimizer", {})
    use_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-5)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
        fused=use_fused,
    )

    # ---------- Profile loop ----------

    out_dir = PROJECT_ROOT / "outputs" / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    iterators = {task: iter(loader) for task, loader in train_loaders.items()}
    task_names = list(train_loaders.keys())
    accum = int(trainer_cfg.get("gradient_accumulation_steps", 4))
    use_bf16 = torch.cuda.is_bf16_supported()
    task_weights = trainer_cfg.get("task_weights") or {}

    emotion_loss_fn = torch.nn.BCEWithLogitsLoss()
    topic_loss_fn = torch.nn.CrossEntropyLoss()

    def get_batch(task):
        try:
            batch = next(iterators[task])
        except StopIteration:
            iterators[task] = iter(train_loaders[task])
            batch = next(iterators[task])
        return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def training_step(step):
        """One training step across all tasks."""
        for task in task_names:
            batch = get_batch(task)
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                if task == "summarization":
                    inputs = {"src_ids": batch["src_ids"], "tgt_ids": batch["tgt_ids"]}
                    if "src_mask" in batch:
                        inputs["src_mask"] = batch["src_mask"]
                    logits = model.forward("summarization", inputs)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch["labels"].view(-1),
                        ignore_index=-100, label_smoothing=0.1,
                    )
                elif task == "emotion":
                    inputs = {"input_ids": batch["input_ids"]}
                    if "attention_mask" in batch:
                        inputs["attention_mask"] = batch["attention_mask"]
                    logits = model.forward("emotion", inputs)
                    loss = emotion_loss_fn(logits, batch["labels"].float())
                elif task == "topic":
                    inputs = {"input_ids": batch["input_ids"]}
                    if "attention_mask" in batch:
                        inputs["attention_mask"] = batch["attention_mask"]
                    logits = model.forward("topic", inputs)
                    loss = topic_loss_fn(logits, batch["labels"])
                else:
                    continue

            weight = task_weights.get(task, 1.0)
            scaled = (loss * weight) / accum
            scaled.backward()

        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Warmup outside profiler to let torch.compile finish
    print(f"\nWarmup ({warmup_steps} steps)...")
    for s in range(warmup_steps):
        training_step(s)
    optimizer.zero_grad()
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
            wait=1, warmup=2, active=active_steps - 3, repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for s in range(active_steps):
            training_step(warmup_steps + s)
            prof.step()

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
