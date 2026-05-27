import torch
from typing import Optional

def detect_hw() -> dict:
    """
    Detects available GPU hardware and returns key properties.
    Returns a dict with keys: vendor, name, arch, runtime, vram_gb.
    Falls back to CPU info if no GPU is available.
    """
    if not torch.cuda.is_available():
        return {
            "vendor": "CPU",
            "name": "CPU",
            "arch": "N/A",
            "runtime": "N/A",
            "vram_gb": 0.0,
        }

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)

    vendor   = _detect_vendor(props.name)
    arch     = get_gpu_arch()
    runtime  = _get_runtime_version(vendor)
    vram_gb  = round(props.total_memory / (1024 ** 3), 2)

    return {
        "vendor":   vendor,
        "name":     props.name,
        "arch":     arch,
        "runtime":  runtime,
        "vram_gb":  vram_gb,
    }


def get_gpu_arch() -> str:
    """
    Returns the GPU architecture string.
    - NVIDIA: SM capability string, e.g. 'sm_86'
    - AMD:    GFX architecture string, e.g. 'gfx942'  (via torch.version.hip or fallback)
    - Unknown: 'unknown'
    """
    if not torch.cuda.is_available():
        return "unknown"

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)

    # ── NVIDIA ──────────────────────────────────────────────────────────────
    if _detect_vendor(props.name) == "NVIDIA":
        major = props.major
        minor = props.minor
        return f"sm_{major}{minor}"

    # ── AMD (ROCm / HIP) ────────────────────────────────────────────────────
    if _detect_vendor(props.name) == "AMD":
        return _get_amd_arch(props)

    return "unknown"


# ── Private helpers ──────────────────────────────────────────────────────────

def _detect_vendor(device_name: str) -> str:
    """Infers vendor from the device name string."""
    name_lower = device_name.lower()
    if any(kw in name_lower for kw in ("nvidia", "tesla", "quadro", "geforce", "rtx", "gtx", "a100", "h100", "l40")):
        return "NVIDIA"
    if any(kw in name_lower for kw in ("amd", "radeon", "instinct", "vega", "navi", "mi300", "mi250")):
        return "AMD"
    if any(kw in name_lower for kw in ("intel", "arc", "xe")):
        return "Intel"
    return "Unknown"


def _get_amd_arch(props) -> str:
    """
    Attempts to retrieve AMD GFX architecture.
    PyTorch/ROCm exposes gcn_arch_name when available.
    Falls back to a best-effort major/minor mapping.
    """
    # ROCm builds of PyTorch may expose gcn_arch_name
    if hasattr(props, "gcn_arch_name") and props.gcn_arch_name:
        return props.gcn_arch_name  # e.g. "gfx942"

    # Fallback: build a rough string from compute capability
    major = getattr(props, "major", 0)
    minor = getattr(props, "minor", 0)
    return f"gfx{major}{minor}0"   # approximate, e.g. gfx900


def _get_runtime_version(vendor: str) -> str:
    """
    Returns the relevant runtime version string for the detected vendor.
    - NVIDIA → CUDA version
    - AMD    → ROCm / HIP version
    - Other  → 'N/A'
    """
    if vendor == "NVIDIA":
        cuda_ver = torch.version.cuda
        return f"CUDA {cuda_ver}" if cuda_ver else "CUDA (version unknown)"

    if vendor == "AMD":
        hip_ver = getattr(torch.version, "hip", None)
        return f"ROCm/HIP {hip_ver}" if hip_ver else "ROCm (version unknown)"

    return "N/A"