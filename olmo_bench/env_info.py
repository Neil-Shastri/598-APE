import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

REPO = Path(__file__).resolve().parent.parent


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_query():
    return _run(["nvidia-smi", "--format=csv,noheader",
                 "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu"])


def git_info():
    head = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "--no-optional-locks", "status", "--porcelain"])
    return {"git_hash": head or "uncommitted", "git_dirty": bool(status)}


def triton_version():
    try:
        import triton
        return triton.__version__
    except ImportError:
        return None


def collect(model, device, dtype, model_id, revision):
    device = torch.device(device)
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "transformers": transformers.__version__,
        "triton": triton_version(),
        "model_id": model_id,
        "revision": revision,
        "dtype": str(dtype).removeprefix("torch."),
        "attn_implementation": model.config._attn_implementation,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "nvidia_smi": gpu_query(),
    }
    info.update(git_info())
    return info
