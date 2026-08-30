"""Privacy: street video is personal data, so faces and plates are redacted at source.

Redaction runs inside the processing loop, before any frame is written or served, so the
raw imagery never leaves the pipeline. What downstream sees is already de-identified: the
safety signal (boxes, risk, tracks) is preserved while the identifying pixels are blurred.
"""
from .redact import REDACTION_POLICY, RedactionPolicy, redact_frame

__all__ = ["REDACTION_POLICY", "RedactionPolicy", "redact_frame"]
