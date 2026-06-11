# Copyright Sierra
"""Hindi language pack.

Self-registers the Hindi ``LanguagePack`` at import time. The loader
(``tau2.multilingual.loader``) imports this subpackage during discovery, which
is enough to make the pack, its personas, and their voice ids available
everywhere — no shared-code edits required.
"""

from tau2.multilingual.languages.hi.pack import HINDI_PACK, register

register()

__all__ = ["HINDI_PACK", "register"]
