# Security & Abuse Controls (MVP)

## Rate limiting
- Simple per-IP rate limiter middleware in API (in-memory, dev-only).
- Production should move to Redis-backed limiter.

## Upload safeguards
- Pre-signed uploads are bucket-scoped and time-limited.
- Future: add MIME validation and malware scan in worker.

## Privacy notes
- Video demos may contain PII; treat raw uploads as sensitive.
- Avoid logging raw content; store only URIs + metadata.

## TODO for production
- AV scanning on upload.
- AuthN/AuthZ hardening (JWT rotation, scoped API keys).
- Audit log of admin validation actions.
