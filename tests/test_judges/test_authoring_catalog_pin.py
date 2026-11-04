"""The authoring catalog's recorded revision is the content digest of its bytes."""

import hashlib
from pathlib import Path

import tau2.judges.authoring_catalog as catalog


def test_source_revision_is_catalog_content_digest():
    path = Path(catalog.__file__).with_name("nativeness_authoring_catalog.json")
    assert catalog.SOURCE_REVISION == hashlib.sha256(path.read_bytes()).hexdigest()
