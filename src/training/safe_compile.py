"""Safe defaults for `torch.compile` to reduce instability in tests and training."""

from __future__ import annotations

from typing import Any, cast

import torch


def _set_attr(obj: object, name: str, value: Any) -> None:
    """Set attribute on dynamic objects only if it exists (keeps static checkers quiet)."""

    target = getattr(obj, name, None)
    if target is not None:
        setattr(obj, name, value)


def compile_model_safe(
    model: torch.nn.Module,
    mode: str = "default",
    dynamic: bool | None = None,
) -> torch.nn.Module:
    """Safely compile model with inductor backend.

    Parameters mirror `torch.compile` but default to conservative settings.
    """

    return cast(
        torch.nn.Module,
        torch.compile(model, backend="inductor", mode=mode, dynamic=dynamic),
    )


def apply_safe_config() -> None:
    """Apply conservative torch._inductor and torch._dynamo settings if present."""

    inductor = getattr(torch, "_inductor", None)
    cfg = getattr(inductor, "config", None) if inductor is not None else None

    if cfg is not None:
        _set_attr(cfg, "epilogue_fusion", False)
        _set_attr(cfg, "coordinate_descent_tuning", False)
        triton_cfg = getattr(cfg, "triton", None)
        if triton_cfg is not None:
            _set_attr(triton_cfg, "cudagraphs", False)
            _set_attr(triton_cfg, "max_autotune_gemm", False)

    dynamo_cfg = getattr(torch, "_dynamo", None)
    if dynamo_cfg is not None:
        dyn_config = getattr(dynamo_cfg, "config", None)
        if dyn_config is not None:
            _set_attr(dyn_config, "suppress_errors", True)
            _set_attr(dyn_config, "cache_size_limit", 64)

    print("✓ Applied safe inductor configuration")
