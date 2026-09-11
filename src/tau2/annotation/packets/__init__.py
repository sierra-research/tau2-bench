# Copyright Sierra
"""HTML annotation packets: standalone conversation-review pages + audio.

``tau2 annotate packets`` builds a directory of per-simulation review pages
(transcript, audio player, task/policy/reward context, one of three annotation
forms) with an index page and a provenance manifest. Annotators work fully
offline in a browser; their filled CSVs come back through ``tau2 annotate
ingest`` via exact header match against the form row models.
"""

from tau2.annotation.packets.builder import PacketBuildOptions, build_packet
from tau2.annotation.packets.forms import FormType, ingest_browser_csv

__all__ = [
    "FormType",
    "PacketBuildOptions",
    "build_packet",
    "ingest_browser_csv",
]
