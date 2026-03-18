from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BasePostprocessor(ABC):
    """Abstract base class for postprocessors that refine or re-rank retrieval results."""

    def __init__(self, **params):
        """Initialize the postprocessor with arbitrary keyword parameters.

        Args:
            **params: Configuration parameters for the postprocessor.
        """
        self.params = params

    @abstractmethod
    def process(
        self,
        results: List[Tuple[str, float]],
        input_data: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """Process a single list of retrieval results.

        Args:
            results: A list of (document, score) tuples to postprocess.
            input_data: The original input data (e.g. query) associated with the results.
            state: Shared pipeline state that may be read or updated.

        Returns:
            A postprocessed list of (document, score) tuples.
        """
        pass

    def process_batch(
        self,
        results_list: List[List[Tuple[str, float]]],
        input_data_list: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> List[List[Tuple[str, float]]]:
        """Process a batch of retrieval result lists.

        Default implementation applies :meth:`process` to each item independently.

        Args:
            results_list: A list of result lists, one per input.
            input_data_list: A list of input data dicts corresponding to each result list.
            state: Shared pipeline state that may be read or updated.

        Returns:
            A list of postprocessed result lists.
        """
        return [
            self.process(results, input_data, state)
            for results, input_data in zip(results_list, input_data_list)
        ]
