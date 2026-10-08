"""Describe the machine a benchmark ran on, so results are never orphaned from it."""

import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from typing import Any

_PACKAGES = (
    "numpy",
    "opencv-python-headless",
    "torch",
    "ultralytics",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "openvino",
    "tensorrt-cu12",
    "fastapi",
    "uvicorn",
)


def cpu_name() -> str:
    """Marketing name of the CPU, e.g. "Intel(R) Core(TM) i9-14900HX"."""
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        return out.stdout.strip() or platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def gpu_info() -> dict[str, Any] | None:
    """Name, memory and driver of GPU 0 via NVML, or None without an NVIDIA GPU."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            "name": str(pynvml.nvmlDeviceGetName(handle)),
            "memory_gb": round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1e9, 1),
            "driver": str(pynvml.nvmlSystemGetDriverVersion()),
        }
    finally:
        pynvml.nvmlShutdown()


def collect() -> dict[str, Any]:
    """Everything needed to interpret (or reproduce) a result."""
    import psutil

    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue

    battery = psutil.sensors_battery()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True) or os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu": gpu_info(),
        # Laptops throttle on battery; a result taken unplugged is not comparable.
        "on_ac_power": None if battery is None else bool(battery.power_plugged),
        "python": platform.python_version(),
        "packages": versions,
        "git_commit": commit or None,
    }


def hardware_tag(info: dict[str, Any]) -> str:
    """Filesystem-safe id like ``i9-14900hx_rtx-4070-laptop``."""

    def slug(text: str) -> str:
        text = re.sub(
            r"\((r|tm)\)|intel|amd|nvidia|geforce|core|processor|gpu|cpu", " ", text.lower()
        )
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")

    parts = [slug(info["cpu"])]
    if info.get("gpu"):
        parts.append(slug(info["gpu"]["name"]))
    return "_".join(p for p in parts if p)
