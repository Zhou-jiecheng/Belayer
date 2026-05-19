#!/usr/bin/env python3
"""
Realtime GPU SM activity monitor using NVML directly via ctypes.

This does not shell out to `nvidia-smi`.

Notes:
- NVML "gpu utilization" is the percentage of time over the recent sample window
  during which one or more kernels were executing on the GPU. In practice this is
  the closest widely-available proxy for SM activity without using CUPTI/DCGM.
- Sampling granularity is controlled by the driver/NVML implementation, typically
  somewhere between ~1s and ~1/6s windows.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0


class NvmlError(RuntimeError):
    pass


class NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


@dataclass
class GpuSample:
    index: int
    name: str
    uuid: str
    sm_util: int
    mem_util: int
    mem_used_mb: int
    mem_total_mb: int
    temperature_c: int
    power_w: float | None


class Nvml:
    def __init__(self) -> None:
        lib_names = (
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            "nvml.dll",
        )
        last_error: Exception | None = None
        self.lib = None
        for lib_name in lib_names:
            try:
                self.lib = ctypes.CDLL(lib_name)
                break
            except OSError as exc:
                last_error = exc
        if self.lib is None:
            raise NvmlError(f"Failed to load NVML library: {last_error}")

        self._bind()

    def _bind(self) -> None:
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlShutdown.restype = ctypes.c_int
        self.lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetName.restype = ctypes.c_int
        self.lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self.lib.nvmlDeviceGetUUID.restype = ctypes.c_int
        self.lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self.lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self.lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvmlUtilization)]
        self.lib.nvmlDeviceGetTemperature.restype = ctypes.c_int
        self.lib.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self.lib.nvmlErrorString.argtypes = [ctypes.c_int]

    def check(self, code: int, func: str) -> None:
        if code == NVML_SUCCESS:
            return
        message = self.lib.nvmlErrorString(code).decode("utf-8", errors="replace")
        raise NvmlError(f"{func} failed: {message} (code={code})")

    def init(self) -> None:
        self.check(self.lib.nvmlInit_v2(), "nvmlInit_v2")

    def shutdown(self) -> None:
        self.check(self.lib.nvmlShutdown(), "nvmlShutdown")

    def device_count(self) -> int:
        count = ctypes.c_uint()
        self.check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
        return int(count.value)

    def handle(self, index: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self.check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        return handle

    def name(self, handle: ctypes.c_void_p) -> str:
        buf = ctypes.create_string_buffer(96)
        self.check(self.lib.nvmlDeviceGetName(handle, buf, ctypes.sizeof(buf)), "nvmlDeviceGetName")
        return buf.value.decode("utf-8", errors="replace")

    def uuid(self, handle: ctypes.c_void_p) -> str:
        buf = ctypes.create_string_buffer(96)
        self.check(self.lib.nvmlDeviceGetUUID(handle, buf, ctypes.sizeof(buf)), "nvmlDeviceGetUUID")
        return buf.value.decode("utf-8", errors="replace")

    def utilization(self, handle: ctypes.c_void_p) -> tuple[int, int]:
        util = NvmlUtilization()
        self.check(
            self.lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)),
            "nvmlDeviceGetUtilizationRates",
        )
        return int(util.gpu), int(util.memory)

    def temperature(self, handle: ctypes.c_void_p) -> int:
        temp = ctypes.c_uint()
        self.check(
            self.lib.nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU, ctypes.byref(temp)),
            "nvmlDeviceGetTemperature",
        )
        return int(temp.value)

    def power_w(self, handle: ctypes.c_void_p) -> float | None:
        power_mw = ctypes.c_uint()
        code = self.lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power_mw))
        if code != NVML_SUCCESS:
            return None
        return power_mw.value / 1000.0

    def memory_mb(self, handle: ctypes.c_void_p) -> tuple[int, int]:
        class NvmlMemory(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong),
            ]

        mem = NvmlMemory()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)), "nvmlDeviceGetMemoryInfo")
        return int(mem.used // (1024 * 1024)), int(mem.total // (1024 * 1024))


def parse_gpu_indices(raw: str | None, device_count: int) -> list[int]:
    if not raw:
        return list(range(device_count))
    indices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= device_count:
            raise ValueError(f"GPU index out of range: {idx}, device_count={device_count}")
        indices.append(idx)
    return indices


def collect_samples(nvml: Nvml, gpu_indices: Iterable[int]) -> list[GpuSample]:
    samples = []
    for idx in gpu_indices:
        handle = nvml.handle(idx)
        sm_util, mem_util = nvml.utilization(handle)
        mem_used_mb, mem_total_mb = nvml.memory_mb(handle)
        samples.append(
            GpuSample(
                index=idx,
                name=nvml.name(handle),
                uuid=nvml.uuid(handle),
                sm_util=sm_util,
                mem_util=mem_util,
                mem_used_mb=mem_used_mb,
                mem_total_mb=mem_total_mb,
                temperature_c=nvml.temperature(handle),
                power_w=nvml.power_w(handle),
            )
        )
    return samples


def format_table(samples: list[GpuSample], timestamp: str) -> str:
    lines = [
        f"[{timestamp}]",
        " idx  sm%  mem%  mem_used/total(MB)  tempC  powerW  name",
    ]
    for sample in samples:
        power = "-" if sample.power_w is None else f"{sample.power_w:6.1f}"
        lines.append(
            f" {sample.index:>3}  {sample.sm_util:>3}  {sample.mem_util:>4}  "
            f"{sample.mem_used_mb:>7}/{sample.mem_total_mb:<7}  "
            f"{sample.temperature_c:>5}  {power:>6}  {sample.name}"
        )
    return "\n".join(lines)


def format_json(samples: list[GpuSample], timestamp: str) -> str:
    payload = {
        "timestamp": timestamp,
        "gpus": [
            {
                "index": sample.index,
                "name": sample.name,
                "uuid": sample.uuid,
                "sm_util": sample.sm_util,
                "mem_util": sample.mem_util,
                "mem_used_mb": sample.mem_used_mb,
                "mem_total_mb": sample.mem_total_mb,
                "temperature_c": sample.temperature_c,
                "power_w": sample.power_w,
            }
            for sample in samples
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def format_sm_one_line(samples: list[GpuSample]) -> str:
    return " ".join(f"rank{sample.index}={sample.sm_util}%" for sample in samples)


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime GPU SM activity monitor via NVML")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU indices, default: all")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print one sample and exit")
    parser.add_argument("--json", action="store_true", help="Print each sample as JSON")
    parser.add_argument(
        "--sm-only-one-line",
        action="store_true",
        help="Print a single line with only per-rank SM utilization",
    )
    args = parser.parse_args()

    stopped = False

    def handle_signal(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    nvml = Nvml()
    nvml.init()
    try:
        device_count = nvml.device_count()
        gpu_indices = parse_gpu_indices(args.gpus, device_count)

        while not stopped:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            samples = collect_samples(nvml, gpu_indices)
            if args.json:
                print(format_json(samples, timestamp), flush=True)
            elif args.sm_only_one_line:
                print(format_sm_one_line(samples), flush=True)
            else:
                print(format_table(samples, timestamp), flush=True)
                if not args.once:
                    print(flush=True)

            if args.once:
                break
            time.sleep(max(0.05, args.interval))
    finally:
        nvml.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
