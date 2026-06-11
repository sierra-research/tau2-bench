# Copyright Sierra
"""Discovery/loading of language packs.

Each language lives in its own subpackage under
``tau2/multilingual/languages/<lang>/`` whose ``__init__`` (or ``pack``
module) calls ``register_language_pack`` at import time. This loader imports
every subpackage exactly once; it is invoked lazily by the registry getters,
so simply installing a pack directory makes it available everywhere.
"""

import importlib
import pkgutil

from loguru import logger

_loaded = False


def load_language_packs() -> None:
    """Import all language subpackages (idempotent)."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    import tau2.multilingual.languages as languages_pkg

    for module_info in pkgutil.iter_modules(languages_pkg.__path__):
        module_name = f"{languages_pkg.__name__}.{module_info.name}"
        try:
            importlib.import_module(module_name)
        except Exception:
            logger.exception(f"Failed to load language pack module '{module_name}'")
            raise
