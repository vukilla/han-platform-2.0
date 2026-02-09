from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from app.core.storage import upload_text


def compute_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(dataset_id: str, version: int, clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "version": version,
        "clip_count": len(clips),
        "clips": clips,
    }


def upload_manifest(dataset_id: str, manifest: Dict[str, Any]) -> str:
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    key = f"datasets/{dataset_id}/manifest.json"
    return upload_text(key, payload, content_type="application/json")
