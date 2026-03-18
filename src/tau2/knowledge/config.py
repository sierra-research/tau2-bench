from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if "document_preprocessors" not in config:
        config["document_preprocessors"] = []
    if not isinstance(config["document_preprocessors"], list):
        raise ValueError("document_preprocessors must be a list")

    for i, dp in enumerate(config["document_preprocessors"]):
        if "type" not in dp:
            raise ValueError(f"document_preprocessor[{i}] must have 'type'")

    if "input_preprocessors" not in config:
        config["input_preprocessors"] = []
    if not isinstance(config["input_preprocessors"], list):
        raise ValueError("input_preprocessors must be a list")

    for i, ip in enumerate(config["input_preprocessors"]):
        if "type" not in ip:
            raise ValueError(f"input_preprocessor[{i}] must have 'type'")

    has_retriever = "retriever" in config
    has_retrievers = "retrievers" in config

    if not has_retriever and not has_retrievers:
        raise ValueError("Config must have 'retriever' or 'retrievers'")

    if has_retriever and has_retrievers:
        raise ValueError("Config cannot have both 'retriever' and 'retrievers'")

    if has_retriever:
        if "type" not in config["retriever"]:
            raise ValueError("retriever must have 'type'")

    if has_retrievers:
        if not isinstance(config["retrievers"], list):
            raise ValueError("retrievers must be a list")
        if len(config["retrievers"]) == 0:
            raise ValueError("retrievers list cannot be empty")
        for i, ret in enumerate(config["retrievers"]):
            if "type" not in ret:
                raise ValueError(f"retrievers[{i}] must have 'type'")

    if "postprocessors" not in config:
        config["postprocessors"] = []
    if not isinstance(config["postprocessors"], list):
        raise ValueError("postprocessors must be a list")

    for i, pp in enumerate(config["postprocessors"]):
        if "type" not in pp:
            raise ValueError(f"postprocessor[{i}] must have 'type'")

    if "tool_name" in config and not isinstance(config["tool_name"], str):
        raise ValueError("tool_name must be a string")

    if "description" in config and not isinstance(config["description"], str):
        raise ValueError("description must be a string")

    if "parameters" in config:
        if not isinstance(config["parameters"], dict):
            raise ValueError(
                "parameters must be a dict mapping param_name to description"
            )
        for param_name, param_desc in config["parameters"].items():
            if not isinstance(param_name, str) or not isinstance(param_desc, str):
                raise ValueError(
                    f"parameters must map string keys to string descriptions"
                )


def get_default_config(
    embedder_type: str = "openai",
    embedder_model: str = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    if embedder_type == "full_kb":
        return {
            "document_preprocessors": [],
            "input_preprocessors": [],
            "retriever": {"type": "full_kb", "params": {}},
            "postprocessors": [],
        }

    embedder_params = {}
    if embedder_model:
        embedder_params["model"] = embedder_model

    config = {
        "document_preprocessors": [
            {
                "type": "embedding_indexer",
                "params": {
                    "embedder_type": embedder_type,
                    "embedder_params": embedder_params,
                    "state_key": "doc_embeddings",
                },
            }
        ],
        "input_preprocessors": [
            {
                "type": "embedding_encoder",
                "params": {
                    "embedder_type": embedder_type,
                    "embedder_params": embedder_params,
                    "input_key": "query",
                    "output_key": "query_embedding",
                },
            }
        ],
        "retriever": {
            "type": "cosine",
            "params": {
                "embedding_key": "query_embedding",
                "index_key": "doc_embeddings",
                "top_k": top_k,
            },
        },
        "postprocessors": [{"type": "identity", "params": {}}],
    }

    return config
