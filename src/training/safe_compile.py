"""
Safe torch.compile configuration that prevents NaN issues.

Author: Oliver Perrin
Date: December 2025
"""

import torch


def compile_model_safe(
    model: torch.nn.Module,
    mode: str = "default",
) -> torch.nn.Module:
    """
    Compile model with inductor backend and safety guardrails.

    Uses 'default' mode which gives inductor speedups without CUDA graphs.
    CUDA graphs (reduce-overhead mode) don't work with dynamic shapes or
    shared embeddings like in T5.

    Args:
        model: Model to compile
        mode: Compilation mode ("default" recommended, avoid "reduce-overhead")

    Returns:
        Compiled model (or original if compilation fails)
    """
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping compilation")
        return model

    try:
        # Configure for stability
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 64  # Allow more graph variations

        # Disable aggressive optimizations that can cause NaNs
        if hasattr(torch, "_inductor"):
            cfg = torch._inductor.config
            if hasattr(cfg, "epilogue_fusion"):
                cfg.epilogue_fusion = False
            if hasattr(cfg, "coordinate_descent_tuning"):
                cfg.coordinate_descent_tuning = False
            if hasattr(cfg, "force_fuse_int_mm_with_mul"):
                cfg.force_fuse_int_mm_with_mul = False
            # Explicitly disable CUDA graphs
            if hasattr(cfg, "triton"):
                if hasattr(cfg.triton, "cudagraphs"):
                    cfg.triton.cudagraphs = False
                if hasattr(cfg.triton, "max_autotune_gemm"):
                    cfg.triton.max_autotune_gemm = False

        # Compile with inductor (no CUDA graphs)
        compiled = torch.compile(model, mode=mode, fullgraph=False, dynamic=True)
        print(f"✓ Compiled with inductor ({mode} mode)")
        return compiled

    except Exception as e:
        print(f"⚠ Inductor compilation failed: {e}")
        print("  Falling back to aot_eager")
        try:
            return torch.compile(model, backend="aot_eager")
        except Exception:
            print("  Using uncompiled model")
            return model


def apply_safe_config():
    """Apply safe configuration to torch._inductor before any compilation."""
    if hasattr(torch, "_inductor"):
        cfg = torch._inductor.config
        if hasattr(cfg, "epilogue_fusion"):
            cfg.epilogue_fusion = False
        if hasattr(cfg, "coordinate_descent_tuning"):
            cfg.coordinate_descent_tuning = False
        if hasattr(cfg, "triton"):
            if hasattr(cfg.triton, "cudagraphs"):
                cfg.triton.cudagraphs = False
            if hasattr(cfg.triton, "max_autotune_gemm"):
                cfg.triton.max_autotune_gemm = False

    # Dynamo config for stability
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 64
    print("✓ Applied safe inductor configuration")
