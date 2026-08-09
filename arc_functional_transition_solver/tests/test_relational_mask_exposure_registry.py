from __future__ import annotations

import pytest

from afts_arc.experiment_safety import canonical_sha256
from scripts.freeze_relational_mask_exposure_registry import (
    _validate_scan_identity,
)


def test_exposure_registry_recomputes_scan_content_id() -> None:
    content = {
        "schema": "fixture",
        "query_gold_read": False,
        "scanned_family_count": 1,
        "families": [{"family_id": "family-a"}],
    }
    scan = {"scan_id": canonical_sha256(content), **content}
    _validate_scan_identity(scan)

    scan["families"] = [{"family_id": "family-b"}]
    with pytest.raises(ValueError, match="content ID mismatch"):
        _validate_scan_identity(scan)
