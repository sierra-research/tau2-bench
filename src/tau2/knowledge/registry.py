from typing import Any, Dict, Type

DOCUMENT_PREPROCESSORS: Dict[str, Type] = {}
INPUT_PREPROCESSORS: Dict[str, Type] = {}
RETRIEVERS: Dict[str, Type] = {}
POSTPROCESSORS: Dict[str, Type] = {}


def register_document_preprocessor(name: str):
    def decorator(cls):
        DOCUMENT_PREPROCESSORS[name] = cls
        return cls

    return decorator


def register_input_preprocessor(name: str):
    def decorator(cls):
        INPUT_PREPROCESSORS[name] = cls
        return cls

    return decorator


def register_retriever(name: str):
    def decorator(cls):
        RETRIEVERS[name] = cls
        return cls

    return decorator


def register_postprocessor(name: str):
    def decorator(cls):
        POSTPROCESSORS[name] = cls
        return cls

    return decorator


def get_document_preprocessor(name: str, params: Dict[str, Any]):
    if name not in DOCUMENT_PREPROCESSORS:
        available = list(DOCUMENT_PREPROCESSORS.keys())
        raise ValueError(
            f"Unknown document_preprocessor: {name}. Available: {available}"
        )
    return DOCUMENT_PREPROCESSORS[name](**params)


def get_input_preprocessor(name: str, params: Dict[str, Any]):
    if name not in INPUT_PREPROCESSORS:
        available = list(INPUT_PREPROCESSORS.keys())
        raise ValueError(f"Unknown input_preprocessor: {name}. Available: {available}")
    return INPUT_PREPROCESSORS[name](**params)


def get_retriever(name: str, params: Dict[str, Any]):
    if name not in RETRIEVERS:
        available = list(RETRIEVERS.keys())
        raise ValueError(f"Unknown retriever: {name}. Available: {available}")
    return RETRIEVERS[name](**params)


def get_postprocessor(name: str, params: Dict[str, Any]):
    if name not in POSTPROCESSORS:
        available = list(POSTPROCESSORS.keys())
        raise ValueError(f"Unknown postprocessor: {name}. Available: {available}")
    return POSTPROCESSORS[name](**params)
