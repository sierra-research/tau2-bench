from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseInputPreprocessor(ABC):
    """Abstract base class for input preprocessors that transform queries before retrieval."""

    def __init__(self, **params):
        """Initialize the input preprocessor with arbitrary keyword parameters.

        Args:
            **params: Configuration parameters for the preprocessor.
        """
        self.params = params

    @abstractmethod
    def process(
        self, input_data: Dict[str, Any], state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single input query.

        Args:
            input_data: A dict containing the raw query and any associated metadata.
            state: Shared pipeline state that may be read or updated.

        Returns:
            A (possibly transformed) input data dict to be passed downstream.
        """
        pass

    def process_batch(
        self, input_data_list: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a batch of input queries.

        Default implementation applies :meth:`process` to each input independently.

        Args:
            input_data_list: A list of input data dicts, one per query.
            state: Shared pipeline state that may be read or updated.

        Returns:
            A list of processed input data dicts.
        """
        return [self.process(input_data, state) for input_data in input_data_list]
