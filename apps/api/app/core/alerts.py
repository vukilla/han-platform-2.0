import json
import urllib.request

from app.core.config import get_settings


def send_alert(event: str, payload: dict) -> None:
    settings = get_settings()
    if not settings.alert_webhook_url:
        return
    body = json.dumps({"event": event, "payload": payload}).encode("utf-8")
    request = urllib.request.Request(
        settings.alert_webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=5)
    except Exception:
        return
