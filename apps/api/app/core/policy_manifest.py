from __future__ import annotations

import json
from typing import Any, Dict

from app.core.storage import upload_text


def build_policy_manifest(policy_id: str, checkpoint_uri: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "policy_id": policy_id,
        "checkpoint_uri": checkpoint_uri,
        "metadata": metadata,
    }


def upload_policy_manifest(policy_id: str, manifest: Dict[str, Any]) -> str:
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    key = f"policies/{policy_id}/manifest.json"
    return upload_text(key, payload, content_type="application/json")
