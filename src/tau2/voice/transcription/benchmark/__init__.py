"""
Transcription benchmark module for evaluating speech-to-text quality.

This module provides tools for creating benchmarks and calculating various
metrics including WER, Quality Score, Significant WER, and Input Coverage.
"""

from tau2.voice.transcription.benchmark.metrics import (
    BaseMetric,
    InputCoverageMetric,
    InputCoverageResult,
    MetricCalculator,
    MetricConfig,
    MetricResult,
    NormalizationHelper,
    QualityMetric,
    QualityResult,
    SignificantWERMetric,
    SignificantWERResult,
    TranscriptPair,
    WERMetric,
    WERResult,
    process_benchmark_directory,
)

__all__ = [
    "BaseMetric",
    "InputCoverageMetric",
    "InputCoverageResult",
    "MetricCalculator",
    "MetricConfig",
    "MetricResult",
    "NormalizationHelper",
    "QualityMetric",
    "QualityResult",
    "SignificantWERMetric",
    "SignificantWERResult",
    "TranscriptPair",
    "WERMetric",
    "WERResult",
    "process_benchmark_directory",
]
