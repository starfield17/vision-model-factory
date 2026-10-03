"""Quantization candidate support utilities.

v1 scope: calibration-source guard plus latency and peak-memory measurement of an ONNX
session. INT8 candidate *production* is not implemented in this repository yet, so
nothing here claims to create a quantized graph; `TargetSpec.precision = "int8"` can only
be published for a graph that actually carries integer quantization nodes, which the
export layer verifies from the graph itself.
"""

import resource
import sys
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from vision_model_factory.contracts.models import SampleRecord

# Peak resident set size is reported in kilobytes by getrusage on POSIX.
_RUSAGE_KB_TO_BYTES = 1024


class CalibrationSourceError(ValueError):
    """Raised when calibration data would leak evaluation data into a model."""


def validate_calibration_samples(samples: List[SampleRecord]) -> None:
    """Ensure calibration data comes strictly from the train split.

    Calibration on val, test or audit data would tune the quantized model on the very
    samples later used to measure it, which invalidates every published quality number.
    """
    offenders = [s for s in samples if s.split != "train"]
    if offenders:
        sample = offenders[0]
        raise CalibrationSourceError(
            f"Calibration forbidden on non-train split. Sample '{sample.sample_id}' has "
            f"split '{sample.split}' ({len(offenders)} offending sample(s) total)."
        )


def benchmark_inference_session(
    session: "ort.InferenceSession",
    input_tensor: np.ndarray,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, float]:
    """Measure per-run latency percentiles on an already-created session."""
    if benchmark_runs < 1:
        raise ValueError(f"benchmark_runs must be >= 1, got {benchmark_runs}")
    input_name = session.get_inputs()[0].name

    for _ in range(warmup_runs):
        session.run(None, {input_name: input_tensor})

    latencies_ms: List[float] = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_tensor})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "latency_ms_p50": round(float(np.percentile(latencies_ms, 50)), 3),
        "latency_ms_p95": round(float(np.percentile(latencies_ms, 95)), 3),
        "latency_ms_mean": round(float(np.mean(latencies_ms)), 3),
    }


def measure_peak_memory_bytes(session: "ort.InferenceSession", input_tensor: np.ndarray) -> int:
    """Peak memory of this process across the inference burst, in bytes.

    Uses peak resident set size, which is the deployment-relevant number: it includes
    arena allocations that instantaneous RSS after a single run would hide. A platform
    without ru_maxrss support raises rather than reporting a made-up figure.
    """
    input_name = session.get_inputs()[0].name
    session.run(None, {input_name: input_tensor})
    peak = _peak_rss_bytes()
    if peak <= 0:
        raise RuntimeError("Peak memory measurement returned no usable value.")
    return int(peak)


def _peak_rss_bytes() -> int:
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS reports bytes; Linux reports kilobytes.
        return int(peak_kb)
    return int(peak_kb) * _RUSAGE_KB_TO_BYTES


__all__ = [
    "CalibrationSourceError",
    "benchmark_inference_session",
    "measure_peak_memory_bytes",
    "validate_calibration_samples",
]
