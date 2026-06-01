"""
Compute device selection for CETI (Apple Silicon MPS, CUDA, CPU).

Tuned for MacBook M-series Max (128GB): use Metal GPU when available and all CPU
cores for dataloader / BLAS threads.

Environment:
  CETI_DEVICE=auto|mps|cuda|cpu   (default: auto)
  CETI_REQUIRE_MPS=1              exit if MPS unavailable (recommended on Mac for training)
"""

from __future__ import annotations

import os
import platform
import sys


def device_preference() -> str:
    """Read CETI_DEVICE env (auto | mps | cuda | cpu)."""
    return os.environ.get("CETI_DEVICE", "auto").strip().lower()


def mps_available() -> bool:
    import torch

    return bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )


def configure_compute(max_threads: bool = True) -> int:
    """
    Set PyTorch / OpenMP thread counts to available logical cores.

    Returns:
        Number of threads configured.
    """
    import torch

    n = os.cpu_count() or 8
    if max_threads:
        torch.set_num_threads(n)
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    return n


def get_device(prefer: str | None = None) -> "torch.device":
    """
    Select best available device.

    prefer: auto | mps | cuda | cpu (defaults to CETI_DEVICE or auto)
    """
    import torch

    prefer = (prefer or device_preference()).lower()

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps":
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")


def require_mps_for_training() -> "torch.device":
    """
    Return MPS device or exit with instructions (when CETI_REQUIRE_MPS=1 on Darwin).
    """
    device = get_device("mps" if device_preference() in ("auto", "mps") else device_preference())
    require = os.environ.get("CETI_REQUIRE_MPS", "").strip() in ("1", "true", "yes")

    if platform.system() == "Darwin" and require and device.type != "mps":
        print(
            "ERROR: CETI_REQUIRE_MPS=1 but MPS is not available.\n"
            "  Install PyTorch with Metal support, e.g.:\n"
            "    pip install torch torchvision\n"
            "  Verify: python ceti/scripts/verify_mps.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if platform.system() == "Darwin" and device.type == "cpu" and device_preference() in ("auto", "mps"):
        print(
            "WARNING: Running on CPU — training will be very slow on Mac.\n"
            "  Set CETI_DEVICE=mps and fix PyTorch Metal support.\n"
            "  See: bash ceti/scripts/setup_mac_mps.sh\n",
            file=sys.stderr,
        )

    return device


def device_name(device: "torch.device") -> str:
    import torch

    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        chip = platform.processor() or "Apple Silicon"
        mem_gb = os.environ.get("CETI_UNIFIED_MEMORY_GB", "")
        suffix = f", {mem_gb}GB unified" if mem_gb else ""
        return f"MPS / Metal ({chip}{suffix})"
    return f"CPU ({os.cpu_count()} cores)"


def configure_mps_for_training() -> None:
    """
    Reduce MPS OOM / 'Killed: 9' on Mac during ViT-L teacher–student training.

    Call once before training (not needed for inference-only).
    """
    if platform.system() != "Darwin":
        return
    # Allow PyTorch to grow MPS allocations under memory pressure (PyTorch 2.x)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def optimal_dataloader_workers(
    requested: int | None = None,
    device: "torch.device | None" = None,
) -> int:
    """
    DataLoader workers for the current platform.

    macOS + MPS: default **0** (multiprocessing duplicates RAM → 'Killed: 9').
    Override: export CETI_DATALOADER_WORKERS=4
    """
    import torch

    if os.environ.get("CETI_DATALOADER_WORKERS", "").strip().isdigit():
        return int(os.environ["CETI_DATALOADER_WORKERS"])

    n = os.cpu_count() or 8
    dev = device
    if dev is None:
        try:
            dev = get_device()
        except Exception:
            dev = None

    if platform.system() == "Darwin" and dev is not None and dev.type == "mps":
        return 0

    if requested is not None:
        return max(0, min(requested, n))
    if platform.system() == "Darwin":
        return 0
    return min(12, max(4, n // 2))


def pin_memory_for_device(device: "torch.device") -> bool:
    return device.type == "cuda"


def empty_cache(device: "torch.device") -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def autocast_device_type(device: "torch.device") -> str:
    """Device type string for torch.autocast (mps uses 'mps' on recent PyTorch)."""
    if device.type == "mps":
        return "mps"
    if device.type == "cuda":
        return "cuda"
    return "cpu"
