"""
Transcription benchmark metrics calculation module.

This module provides classes and functions for calculating various metrics
on transcription benchmarks including WER, Quality Score, Significant WER,
and Input Coverage.
"""

import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from jiwer import process_words, wer
from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.message import UserMessage
from tau2.utils.llm_utils import generate
from tau2.voice.transcription.benchmark.prompts import (
    EXTRACT_INPUTS_PROMPT,
    NORMALIZE_FOR_WER_PROMPT,
    SCORE_TRANSCRIPT_PROMPT,
    SIGNIFICANT_WORD_ERRORS_PROMPT,
)

# Constants
DEFAULT_LLM_MODEL = "gpt-4.1"
DEFAULT_MAX_WORKERS = 8


class MetricResult(BaseModel):
    """Base class for metric results."""

    pass


class WERResult(MetricResult):
    """Word Error Rate calculation result."""

    wer: float = Field(..., description="Word error rate score")
    normalized_gold: Optional[str] = Field(
        None, description="Normalized gold transcript"
    )
    normalized_llm: Optional[str] = Field(None, description="Normalized LLM transcript")


class QualityResult(MetricResult):
    """Quality score calculation result."""

    quality_score: Optional[int] = Field(None, description="Quality score (0-3)")


class SignificantWERResult(MetricResult):
    """Significant Word Error Rate calculation result."""

    significant_error_rate: float = Field(..., description="Rate of significant errors")
    significant_error_count: int = Field(
        ..., description="Number of significant errors"
    )
    total_words: int = Field(..., description="Total words in reference")
    error_details: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed error information"
    )
    normalized_gold: Optional[str] = Field(
        None, description="Normalized gold transcript"
    )
    normalized_llm: Optional[str] = Field(None, description="Normalized LLM transcript")


class InputCoverageResult(MetricResult):
    """Input coverage calculation result."""

    extracted_inputs: List[Dict[str, str]] = Field(
        default_factory=list, description="Extracted inputs"
    )
    input_total: int = Field(0, description="Total number of inputs")
    input_matched: int = Field(0, description="Number of matched inputs")
    input_coverage: float = Field(0.0, description="Coverage percentage")
    hit_values: List[str] = Field(
        default_factory=list, description="Successfully matched values"
    )
    missed_values: List[str] = Field(default_factory=list, description="Missed values")


class MetricConfig(BaseModel):
    """Configuration for metric calculation."""

    normalize: bool = Field(True, description="Whether to normalize transcripts")
    model: str = Field(DEFAULT_LLM_MODEL, description="LLM model to use")
    max_workers: int = Field(
        DEFAULT_MAX_WORKERS, description="Maximum concurrent workers"
    )
    recompute: bool = Field(
        False, description="Force recomputation of existing metrics"
    )


class TranscriptPair(BaseModel):
    """A pair of gold and predicted transcripts."""

    gold: str = Field("", description="Gold/reference transcript")
    predicted: str = Field("", description="Predicted/hypothesis transcript")


class BaseMetric(ABC):
    """Base class for all metrics."""

    name: str
    base_columns: List[str]
    normalized_columns: List[str] = []

    @abstractmethod
    def calculate(
        self,
        transcript_pair: TranscriptPair,
        config: MetricConfig,
        cached_normalized: Optional[Tuple[str, str]] = None,
    ) -> MetricResult:
        """Calculate the metric for a single transcript pair."""
        raise NotImplementedError

    def get_required_columns(self, normalize: bool) -> List[str]:
        """Get list of required columns for this metric."""
        columns = self.base_columns.copy()
        if normalize and self.normalized_columns:
            columns.extend(self.normalized_columns)
        return columns

    def check_exists(self, df: pd.DataFrame, normalize: bool) -> bool:
        """Check if metric already exists in dataframe."""
        return all(col in df.columns for col in self.get_required_columns(normalize))


class NormalizationHelper:
    """Helper class for text normalization operations."""

    @staticmethod
    def normalize_pair(
        gold: str, predicted: str, model: str = DEFAULT_LLM_MODEL
    ) -> Tuple[Optional[str], Optional[str]]:
        """Normalize a transcript pair using LLM."""
        if not gold and not predicted:
            return "", ""

        prompt = NORMALIZE_FOR_WER_PROMPT.format(
            expected_transcript=gold, actual_transcript=predicted
        )

        try:
            messages = [UserMessage(role="user", content=prompt)]
            response = generate(
                model=model, messages=messages, call_name="normalize_for_wer"
            )
            result = response.content

            if isinstance(result, str):
                import json

                result = json.loads(result)

            normalized_gold = result.get("normalized_expected")
            normalized_pred = result.get("normalized_actual")

            if normalized_gold is not None and normalized_pred is not None:
                return normalized_gold, normalized_pred
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")

        return None, None

    @staticmethod
    def normalize_for_input_match(text: str) -> str:
        """Normalize text for input matching (lowercase, remove punctuation/spaces)."""
        if not text:
            return ""
        text = text.lower()
        for char in ".,()-/@_ ":
            text = text.replace(char, "")
        return text


class WERMetric(BaseMetric):
    """Word Error Rate metric implementation."""

    name = "wer"
    base_columns = ["wer"]
    normalized_columns = ["normalized_gold", "normalized_llm"]

    def calculate(
        self,
        transcript_pair: TranscriptPair,
        config: MetricConfig,
        cached_normalized: Optional[Tuple[str, str]] = None,
    ) -> WERResult:
        """Calculate WER for a transcript pair."""
        gold = transcript_pair.gold
        predicted = transcript_pair.predicted

        # Handle empty strings
        if not gold and not predicted:
            return WERResult(wer=0.0, normalized_gold="", normalized_llm="")
        elif not gold:
            return WERResult(wer=1.0, normalized_gold="", normalized_llm=predicted)
        elif not predicted:
            return WERResult(wer=1.0, normalized_gold=gold, normalized_llm="")

        # Get normalized texts
        if cached_normalized:
            normalized_ref, normalized_hyp = cached_normalized
        elif config.normalize:
            normalized_ref, normalized_hyp = NormalizationHelper.normalize_pair(
                gold, predicted, config.model
            )
            if normalized_ref is None or normalized_hyp is None:
                logger.warning("Normalization failed, using original transcripts")
                normalized_ref, normalized_hyp = gold, predicted
        else:
            normalized_ref, normalized_hyp = gold, predicted

        # Calculate WER
        try:
            error_rate = wer(normalized_ref, normalized_hyp)
            return WERResult(
                wer=error_rate,
                normalized_gold=normalized_ref,
                normalized_llm=normalized_hyp,
            )
        except Exception as e:
            logger.warning(f"Error calculating WER: {e}")
            return WERResult(
                wer=-1.0, normalized_gold=normalized_ref, normalized_llm=normalized_hyp
            )


class QualityMetric(BaseMetric):
    """Quality score metric implementation."""

    name = "quality"
    base_columns = ["quality_score"]

    def calculate(
        self,
        transcript_pair: TranscriptPair,
        config: MetricConfig,
        cached_normalized: Optional[Tuple[str, str]] = None,
    ) -> QualityResult:
        """Calculate quality score for a transcript pair."""
        prompt = SCORE_TRANSCRIPT_PROMPT.format(
            gold_transcript=transcript_pair.gold,
            llm_transcript=transcript_pair.predicted,
        )

        try:
            messages = [UserMessage(role="user", content=prompt)]
            response = generate(
                model=config.model,
                messages=messages,
                call_name="score_transcript_quality",
            )

            result = (
                json.loads(response.content)
                if isinstance(response.content, str)
                else response.content
            )
            score = result.get("score")

            if isinstance(score, int) and 0 <= score <= 3:
                return QualityResult(quality_score=score)
        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")

        return QualityResult(quality_score=None)


class SignificantWERMetric(BaseMetric):
    """Significant Word Error Rate metric implementation."""

    name = "significant_wer"
    base_columns = [
        "significant_error_rate",
        "significant_error_count",
        "total_words",
        "error_details",
    ]
    normalized_columns = ["normalized_gold", "normalized_llm"]

    def _extract_word_errors(
        self, ref_text: str, hyp_text: str
    ) -> List[Dict[str, Any]]:
        """Extract word-level errors using jiwer."""
        errors = []

        try:
            result = process_words(ref_text, hyp_text)
            ref_words = ref_text.split()
            hyp_words = hyp_text.split()

            if result.alignments and len(result.alignments) > 0:
                for chunk in result.alignments[0]:
                    if chunk.type == "substitute":
                        ref_word = (
                            ref_words[chunk.ref_start_idx]
                            if chunk.ref_start_idx < len(ref_words)
                            else ""
                        )
                        hyp_word = (
                            hyp_words[chunk.hyp_start_idx]
                            if chunk.hyp_start_idx < len(hyp_words)
                            else ""
                        )
                        errors.append(
                            {
                                "error": f"Substitution: '{ref_word}' to '{hyp_word}' at position {chunk.ref_start_idx}",
                                "type": "substitution",
                                "position": chunk.ref_start_idx,
                                "ref_word": ref_word,
                                "hyp_word": hyp_word,
                            }
                        )
                    elif chunk.type == "delete":
                        ref_word = (
                            ref_words[chunk.ref_start_idx]
                            if chunk.ref_start_idx < len(ref_words)
                            else ""
                        )
                        errors.append(
                            {
                                "error": f"Deletion: '{ref_word}' at position {chunk.ref_start_idx}",
                                "type": "deletion",
                                "position": chunk.ref_start_idx,
                                "ref_word": ref_word,
                            }
                        )
                    elif chunk.type == "insert":
                        hyp_word = (
                            hyp_words[chunk.hyp_start_idx]
                            if chunk.hyp_start_idx < len(hyp_words)
                            else ""
                        )
                        errors.append(
                            {
                                "error": f"Insertion: '{hyp_word}' at position {chunk.ref_start_idx}",
                                "type": "insertion",
                                "position": chunk.ref_start_idx,
                                "hyp_word": hyp_word,
                            }
                        )
        except Exception as e:
            logger.warning(f"Word alignment failed: {e}")

        return errors

    def _score_errors(
        self, errors: List[Dict[str, Any]], ref_text: str, hyp_text: str, model: str
    ) -> List[Dict[str, Any]]:
        """Score word-level errors using LLM."""
        if not errors:
            return []

        error_descriptions = [{"error": err["error"], "score": None} for err in errors]
        errors_str = json.dumps(error_descriptions, indent=2)

        prompt = SIGNIFICANT_WORD_ERRORS_PROMPT.format(
            expected_transcript=ref_text, actual_transcript=hyp_text, errors=errors_str
        )

        try:
            messages = [UserMessage(role="user", content=prompt)]
            response = generate(
                model=model, messages=messages, call_name="score_word_errors"
            )
            result = (
                json.loads(response.content)
                if isinstance(response.content, str)
                else response.content
            )

            scored_errors = []
            if result and "scores" in result:
                llm_scores = result["scores"]
                for i, err in enumerate(errors):
                    score = llm_scores[i].get("score", 2) if i < len(llm_scores) else 2
                    err_with_score = err.copy()
                    err_with_score["score"] = score
                    scored_errors.append(err_with_score)
            else:
                scored_errors = [dict(err, score=2) for err in errors]

            return scored_errors
        except Exception as e:
            logger.warning(f"Error scoring failed: {e}")
            return [dict(err, score=2) for err in errors]

    def calculate(
        self,
        transcript_pair: TranscriptPair,
        config: MetricConfig,
        cached_normalized: Optional[Tuple[str, str]] = None,
    ) -> SignificantWERResult:
        """Calculate Significant WER for a transcript pair."""
        gold = transcript_pair.gold
        predicted = transcript_pair.predicted

        # Handle empty reference
        if not gold:
            return SignificantWERResult(
                significant_error_rate=0.0,
                significant_error_count=0,
                total_words=0,
                error_details=[],
                normalized_gold=None,
                normalized_llm=predicted,
            )

        # Get normalized texts
        if cached_normalized:
            normalized_ref, normalized_hyp = cached_normalized
        elif config.normalize:
            normalized_ref, normalized_hyp = NormalizationHelper.normalize_pair(
                gold, predicted, config.model
            )
            if normalized_ref is None or normalized_hyp is None:
                logger.warning("Normalization failed, using original transcripts")
                normalized_ref, normalized_hyp = gold, predicted
        else:
            normalized_ref, normalized_hyp = gold, predicted

        # Calculate metrics
        ref_words = normalized_ref.split()
        total_words = len(ref_words)

        if total_words == 0:
            return SignificantWERResult(
                significant_error_rate=0.0,
                significant_error_count=0,
                total_words=0,
                error_details=[],
                normalized_gold=normalized_ref,
                normalized_llm=normalized_hyp,
            )

        # Get and score errors
        errors = self._extract_word_errors(normalized_ref, normalized_hyp)
        if not errors:
            return SignificantWERResult(
                significant_error_rate=0.0,
                significant_error_count=0,
                total_words=total_words,
                error_details=[],
                normalized_gold=normalized_ref,
                normalized_llm=normalized_hyp,
            )

        scored_errors = self._score_errors(
            errors, normalized_ref, normalized_hyp, config.model
        )
        significant_errors_count = sum(
            1 for err in scored_errors if err.get("score") == 1
        )
        significant_error_rate = significant_errors_count / total_words

        return SignificantWERResult(
            significant_error_rate=significant_error_rate,
            significant_error_count=significant_errors_count,
            total_words=total_words,
            error_details=scored_errors,
            normalized_gold=normalized_ref,
            normalized_llm=normalized_hyp,
        )


class InputCoverageMetric(BaseMetric):
    """Input coverage metric implementation."""

    name = "input_coverage"
    base_columns = [
        "extracted_inputs",
        "input_total",
        "input_matched",
        "input_coverage",
        "hit_values",
        "missed_values",
    ]

    def _extract_inputs(
        self, transcript: str, model: str
    ) -> Optional[List[Dict[str, str]]]:
        """Extract input values from transcript using LLM."""
        if not transcript:
            return []

        prompt = EXTRACT_INPUTS_PROMPT.format(transcript=transcript)

        try:
            messages = [UserMessage(role="user", content=prompt)]
            response = generate(
                model=model, messages=messages, call_name="extract_inputs"
            )

            import json

            result = (
                json.loads(response.content)
                if isinstance(response.content, str)
                else response.content
            )

            if result and "inputs" in result:
                valid_types = [
                    "name",
                    "email",
                    "phone",
                    "street",
                    "city",
                    "zip",
                    "id",
                    "dob",
                    "other",
                ]
                inputs = result["inputs"]

                valid_inputs = [
                    inp
                    for inp in inputs
                    if isinstance(inp, dict)
                    and "type" in inp
                    and "value" in inp
                    and inp["type"] in valid_types
                ]
                return valid_inputs
        except Exception as e:
            logger.warning(f"Input extraction failed: {e}")

        return None

    def calculate(
        self,
        transcript_pair: TranscriptPair,
        config: MetricConfig,
        cached_normalized: Optional[Tuple[str, str]] = None,
    ) -> InputCoverageResult:
        """Calculate input coverage for a transcript pair."""
        # Extract inputs from gold transcript
        inputs = self._extract_inputs(transcript_pair.gold, config.model)

        if inputs is None:
            return InputCoverageResult()

        if not inputs:
            return InputCoverageResult(extracted_inputs=[])

        # Match inputs in predicted transcript
        norm_pred = NormalizationHelper.normalize_for_input_match(
            transcript_pair.predicted
        )
        total = len(inputs)
        hit_values = []
        missed_values = []

        for inp in inputs:
            value = inp.get("value", "")
            norm_value = NormalizationHelper.normalize_for_input_match(value)

            if norm_value and norm_value in norm_pred:
                hit_values.append(value)
            else:
                missed_values.append(value)

        matched = len(hit_values)
        coverage = matched / total if total > 0 else 0.0

        return InputCoverageResult(
            extracted_inputs=inputs,
            input_total=total,
            input_matched=matched,
            input_coverage=coverage,
            hit_values=hit_values,
            missed_values=missed_values,
        )


class MetricCalculator:
    """Main class for calculating transcription metrics."""

    def __init__(self):
        """Initialize metric calculator with available metrics."""
        self.metrics: Dict[str, BaseMetric] = {
            "wer": WERMetric(),
            "quality": QualityMetric(),
            "significant_wer": SignificantWERMetric(),
            "input_coverage": InputCoverageMetric(),
        }

    def process_dataframe(
        self, df: pd.DataFrame, metrics: List[str], config: MetricConfig
    ) -> pd.DataFrame:
        """Process a dataframe and calculate requested metrics."""
        # Load existing metrics if not recomputing
        if not config.recompute:
            df = self._load_existing_metrics(
                df, Path(".") / "transcription_metrics.csv"
            )

        # Check if normalized columns exist
        has_normalized = (
            "normalized_gold" in df.columns and "normalized_llm" in df.columns
        )

        # Add normalized columns first if needed (before calculating metrics)
        if config.normalize and not has_normalized:
            df = self._add_normalized_columns(df, config)
            has_normalized = True

        # Process each metric
        for metric_name in metrics:
            if metric_name not in self.metrics:
                logger.warning(f"Unknown metric: {metric_name}")
                continue

            metric = self.metrics[metric_name]

            # Skip if already exists
            if metric.check_exists(df, config.normalize):
                logger.info(f"Metric {metric_name} already exists, skipping...")
                continue

            logger.info(f"Calculating {metric_name}...")

            # Calculate metric for all rows
            results = self._calculate_metric_batch(df, metric, config, has_normalized)

            # Update dataframe with results
            df = self._update_dataframe(df, metric, results, config)

            # Log statistics
            self._log_statistics(metric_name, results)

        return df

    def _calculate_metric_batch(
        self,
        df: pd.DataFrame,
        metric: BaseMetric,
        config: MetricConfig,
        has_normalized: bool,
    ) -> List[MetricResult]:
        """Calculate metric for all rows in batch with concurrency."""

        def process_row(idx, row):
            transcript_pair = TranscriptPair(
                gold=str(row["gold_transcript"])
                if pd.notna(row["gold_transcript"])
                else "",
                predicted=str(row["llm_transcript"])
                if pd.notna(row["llm_transcript"])
                else "",
            )

            # Use cached normalized texts if available
            cached_normalized = None
            if (
                has_normalized
                and config.normalize
                and metric.name in ["wer", "significant_wer"]
            ):
                norm_gold = row.get("normalized_gold", None)
                norm_llm = row.get("normalized_llm", None)

                if pd.notna(norm_gold) and pd.notna(norm_llm):
                    cached_normalized = (str(norm_gold), str(norm_llm))

            result = metric.calculate(transcript_pair, config, cached_normalized)
            return idx, result

        # Process rows concurrently
        results = [None] * len(df)

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []
            for idx, row in df.iterrows():
                future = executor.submit(process_row, idx, row)
                futures.append(future)

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _update_dataframe(
        self,
        df: pd.DataFrame,
        metric: BaseMetric,
        results: List[MetricResult],
        config: MetricConfig,
    ) -> pd.DataFrame:
        """Update dataframe with metric results."""
        import json

        if metric.name == "wer":
            df["wer"] = [r.wer for r in results]
            if config.normalize:
                df["normalized_gold"] = [r.normalized_gold for r in results]
                df["normalized_llm"] = [r.normalized_llm for r in results]

        elif metric.name == "quality":
            df["quality_score"] = [r.quality_score for r in results]

        elif metric.name == "significant_wer":
            df["significant_error_rate"] = [r.significant_error_rate for r in results]
            df["significant_error_count"] = [r.significant_error_count for r in results]
            df["total_words"] = [r.total_words for r in results]
            df["error_details"] = [
                json.dumps(r.error_details) if r.error_details else "" for r in results
            ]
            if config.normalize:
                df["normalized_gold"] = [r.normalized_gold for r in results]
                df["normalized_llm"] = [r.normalized_llm for r in results]

        elif metric.name == "input_coverage":
            df["extracted_inputs"] = [
                json.dumps(r.extracted_inputs) if r.extracted_inputs else ""
                for r in results
            ]
            df["input_total"] = [r.input_total for r in results]
            df["input_matched"] = [r.input_matched for r in results]
            df["input_coverage"] = [r.input_coverage for r in results]
            df["hit_values"] = [json.dumps(r.hit_values) for r in results]
            df["missed_values"] = [json.dumps(r.missed_values) for r in results]

        return df

    def _log_statistics(self, metric_name: str, results: List[MetricResult]) -> None:
        """Log statistics for calculated metrics."""
        if metric_name == "wer":
            valid_wer = [r.wer for r in results if r.wer >= 0]
            if valid_wer:
                logger.info(
                    f"WER - Average: {sum(valid_wer) / len(valid_wer):.4f}, "
                    f"Min: {min(valid_wer):.4f}, Max: {max(valid_wer):.4f}"
                )

        elif metric_name == "quality":
            valid_scores = [
                r.quality_score for r in results if r.quality_score is not None
            ]
            if valid_scores:
                logger.info(
                    f"Quality - Average: {sum(valid_scores) / len(valid_scores):.2f}, "
                    f"Distribution: {dict((i, valid_scores.count(i)) for i in range(4))}"
                )

        elif metric_name == "significant_wer":
            valid_rates = [
                (r.significant_error_rate, r.total_words)
                for r in results
                if r.total_words > 0
            ]
            if valid_rates:
                avg_rate = sum(rate for rate, _ in valid_rates) / len(valid_rates)
                logger.info(
                    f"Significant WER - Average Significant Error Rate: {avg_rate:.4f}"
                )

        elif metric_name == "input_coverage":
            valid_coverage = [
                (r.input_coverage, r.input_total) for r in results if r.input_total > 0
            ]
            if valid_coverage:
                avg_coverage = sum(cov for cov, _ in valid_coverage) / len(
                    valid_coverage
                )
                logger.info(f"Input Coverage - Average: {avg_coverage:.2%}")

    def _load_existing_metrics(
        self, df: pd.DataFrame, metrics_path: Path
    ) -> pd.DataFrame:
        """Load existing metrics from file and merge with current dataframe."""
        if not metrics_path.exists():
            return df

        logger.info(f"Found existing metrics file: {metrics_path}")
        existing_df = pd.read_csv(metrics_path)

        # Ensure we have the same rows
        if len(existing_df) != len(df):
            logger.warning(
                "Existing metrics file has different number of rows, ignoring cached metrics"
            )
            return df

        # Get list of existing metric columns
        existing_columns = set(existing_df.columns) - set(df.columns)
        logger.info(f"Existing metric columns: {existing_columns}")

        # Merge existing metrics
        for col in existing_columns:
            df[col] = existing_df[col]

        return df

    def _add_normalized_columns(
        self, df: pd.DataFrame, config: MetricConfig
    ) -> pd.DataFrame:
        """Add normalized text columns if they don't exist."""
        if "normalized_gold" in df.columns and "normalized_llm" in df.columns:
            return df

        logger.info("Calculating normalized texts...")

        def normalize_row(idx, row):
            gold = (
                str(row["gold_transcript"]) if pd.notna(row["gold_transcript"]) else ""
            )
            llm = str(row["llm_transcript"]) if pd.notna(row["llm_transcript"]) else ""
            return idx, NormalizationHelper.normalize_pair(gold, llm, config.model)

        # Process rows concurrently
        results = [None] * len(df)

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []
            for idx, row in df.iterrows():
                future = executor.submit(normalize_row, idx, row)
                futures.append(future)

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        # Extract normalized texts
        normalized_golds = []
        normalized_llms = []

        for result in results:
            if result:
                norm_gold, norm_llm = result
                normalized_golds.append(norm_gold if norm_gold is not None else "")
                normalized_llms.append(norm_llm if norm_llm is not None else "")
            else:
                normalized_golds.append("")
                normalized_llms.append("")

        df["normalized_gold"] = normalized_golds
        df["normalized_llm"] = normalized_llms

        return df


def process_benchmark_directory(
    benchmark_dir: Path, metrics: List[str], config: MetricConfig
) -> Optional[pd.DataFrame]:
    """Process a single benchmark directory and calculate metrics."""
    # Load data
    csv_path = benchmark_dir / "transcription_data.csv"
    if not csv_path.exists():
        logger.warning(f"No transcription_data.csv found in {benchmark_dir}")
        return None

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Create metric calculator and process
    calculator = MetricCalculator()
    df = calculator.process_dataframe(df, metrics, config)

    # Save updated metrics
    metrics_path = benchmark_dir / "transcription_metrics.csv"
    df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")

    return df
