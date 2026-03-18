"""OpenRouter embedder supporting multiple embedding models.

OpenRouter provides access to various embedding models including Qwen3 embeddings
through a unified API.
"""

import os
import time
from typing import List, Optional

import numpy as np
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from tau2.knowledge.embedders.base import BaseEmbedder

# From https://huggingface.co/Qwen/Qwen3-Embedding-8B#vllm-usage
DEFAULT_QWEN_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class OpenRouterEmbedder(BaseEmbedder):
    """Embedder using OpenRouter's embedding models.

    For Qwen models, queries should be prefixed with an instruction following Qwen's format:
        'Instruct: {task_description}\\nQuery:{query}'
    Documents should NOT be prefixed.

    Use `query_instruction` parameter to set the instruction for query embedding.
    Set `query_instruction=None` or empty string for document embedding.
    """

    # Model configurations with dimensions
    # See: https://openrouter.ai/models?order=newest&q=embedding
    SUPPORTED_MODELS = {
        # "qwen3-embedding-0.6b": {  # Not available on OpenRouter as of Jan 2026
        #     "api_string": "qwen/qwen3-embedding-0.6b",
        #     "dimension": 1024,
        #     "size": "0.6B",
        # },
        "qwen3-embedding-4b": {
            "api_string": "qwen/qwen3-embedding-4b",
            "dimension": 2560,
            "size": "4B",
            "requires_instruction": True,
        },
        "qwen3-embedding-8b": {
            "api_string": "qwen/qwen3-embedding-8b",
            "dimension": 4096,
            "size": "8B",
            "requires_instruction": True,
        },
        "gemini-embedding-001": {
            "api_string": "google/gemini-embedding-001",
            "dimension": 3072,
            "size": None,
            "requires_instruction": False,
        },
    }

    def __init__(
        self,
        model: str = "qwen3-embedding-8b",
        api_key: Optional[str] = None,
        encoding_format: str = "float",
        prefix: str = "",
        query_instruction: Optional[str] = None,
    ):
        self.prefix = prefix
        self._requires_instruction = False

        # Handle both short names and full API strings
        if model in self.SUPPORTED_MODELS:
            self.model_name = model
            self.model_api_string = self.SUPPORTED_MODELS[model]["api_string"]
            self.dimension = self.SUPPORTED_MODELS[model]["dimension"]
            self._requires_instruction = self.SUPPORTED_MODELS[model].get(
                "requires_instruction", False
            )
        else:
            # Check if the provided string matches any API string
            found = False
            for key, config in self.SUPPORTED_MODELS.items():
                if config["api_string"] == model:
                    self.model_name = key
                    self.model_api_string = model
                    self.dimension = config["dimension"]
                    self._requires_instruction = config.get(
                        "requires_instruction", False
                    )
                    found = True
                    break

            if not found:
                # Allow using arbitrary models not in the predefined list
                # This enables using new models as they become available
                self.model_name = model.replace("/", "-")
                self.model_api_string = model
                self.dimension = None  # Unknown dimension for custom models
                # Check if it's a Qwen model by name
                self._requires_instruction = "qwen" in model.lower()

        # Set query instruction for Qwen models
        # If query_instruction is explicitly passed, use it (even if empty string)
        # If not passed (None), use default for Qwen models
        if query_instruction is not None:
            self.query_instruction = query_instruction
        elif self._requires_instruction:
            self.query_instruction = DEFAULT_QWEN_QUERY_INSTRUCTION
        else:
            self.query_instruction = None

        self.encoding_format = encoding_format

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    def _format_text(self, text: str) -> str:
        """Format text for embedding, applying instruction prefix if needed.

        For Qwen models with query_instruction set, formats as:
            'Instruct: {instruction}\\nQuery:{text}'
        Otherwise returns text as-is (with legacy prefix if set).
        """
        if self.query_instruction:
            # Qwen instruction format from documentation
            return f"Instruct: {self.query_instruction}\nQuery:{text}"
        elif self.prefix:
            # Legacy prefix support
            return f"{self.prefix}{text}"
        return text

    def embed(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        if not texts:
            raise ValueError("No text to embed.")

        # Apply formatting (instruction prefix for queries, nothing for documents)
        texts = [self._format_text(t) for t in texts]

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_api_string,
                    encoding_format=self.encoding_format,
                )
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)
            except (APIError, APIConnectionError, RateLimitError) as e:
                last_exception = e
                wait_time = 2**attempt
                time.sleep(wait_time)
            except Exception as e:
                if (
                    "JSONDecodeError" in str(type(e).__name__)
                    or "json" in str(e).lower()
                ):
                    last_exception = e
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(
            f"OpenRouter embedding failed after {max_retries} retries. "
            f"Model: {self.model_api_string}. Last error: {last_exception}"
        )

    def get_name(self) -> str:
        """Return the name of the embedder."""
        base_name = f"openrouter_{self.model_name}"
        if self.query_instruction:
            return f"{base_name}_query"
        return base_name

    def get_dimension(self) -> Optional[int]:
        """Return the embedding dimension (None if unknown)."""
        return self.dimension

    def get_query_instruction(self) -> Optional[str]:
        """Return the current query instruction."""
        return self.query_instruction

    def requires_instruction(self) -> bool:
        """Return whether this model requires instruction prefix for queries."""
        return self._requires_instruction

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Return list of supported model names."""
        return list(cls.SUPPORTED_MODELS.keys())

    @classmethod
    def get_default_query_instruction(cls) -> str:
        """Return the default query instruction for Qwen models."""
        return DEFAULT_QWEN_QUERY_INSTRUCTION
