"""Reversible masking of Indian personal identifiers before text leaves our infrastructure.

PAN, Aadhaar, mobile numbers and e-mails become placeholders such as [PAN_1]. The model works with the
placeholders and the real values are put back into its answer, so the user still sees their own data.
"""

from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PAN", re.compile(r"\b[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"(?<![\d-])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?![\d-])")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")),
    ("MOBILE", re.compile(r"(?<!\d)(\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")),
]


class Redactor:
    def __init__(self) -> None:
        self._by_value: dict[str, str] = {}
        self._by_token: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    @property
    def count(self) -> int:
        return len(self._by_token)

    def redact(self, text: str) -> str:
        for kind, pattern in PATTERNS:
            text = pattern.sub(lambda m, k=kind: self._token(k, m.group(0)), text)
        return text

    def restore(self, text: str) -> str:
        if not self._by_token:
            return text
        return re.sub(r"\[(PAN|AADHAAR|EMAIL|MOBILE)_\d+\]", lambda m: self._by_token.get(m.group(0), m.group(0)), text)

    def _token(self, kind: str, value: str) -> str:
        if value in self._by_value:
            return self._by_value[value]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        token = f"[{kind}_{self._counts[kind]}]"
        self._by_value[value] = token
        self._by_token[token] = value
        return token
