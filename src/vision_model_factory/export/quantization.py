"""INT8 quantization candidate evaluation and benchmarking utilities."""

import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import onnxruntime as ort

from vision_model_factory.contracts.models import SampleRecord


def validate_calibration_samples(samples: List[SampleRecord]) -> None:
    """Ensure calibration data comes strictly from the train split and never val, test, or audit."""
    for s in samples:
        if s.split != "train":
            raise ValueError(
                f"Calibration forbidden on non-train split. Sample '{s.sample_id}' has split '{s.split}'"
            )


def benchmark_onnx_model(
    onnx_path: Union[str, Path],
    input_shape: tuple = (1, 3, 640, 640),
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, Any]:
    """
    Measure target execution provider latency metrics (p50, p95) on CPU.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy})

    latencies_ms: List[float] = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    mean = float(np.mean(latencies_ms))

    return {
        "provider": "CPUExecutionProvider",
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "latency_ms_p50": round(p50, 2),
        "latency_ms_p95": round(p95, 2),
        "latency_ms_mean": round(mean, 2),
    }
