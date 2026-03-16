"""Parse time strings and delta minutes from model output."""

import re
from typing import List, Optional, Tuple

TIME_RE = re.compile(r"\b(0?[1-9]|1[0-2])\s*[:h]\s*([0-5]\d)\b", re.IGNORECASE)
TIME_HMS_RE = re.compile(
    r"\b(0?[1-9]|1[0-2])\s*:\s*([0-5]\d)\s*:\s*([0-5]\d)\b",
    re.IGNORECASE,
)
DELTA_RE = re.compile(r"\b(-?\d{1,4})\s*(?:min|mins|minutes|m)\b", re.IGNORECASE)
DELTA_FALLBACK_RE = re.compile(r"\b(?:delta|difference|diff)\b\s*[:=]?\s*(-?\d{1,4})\b", re.IGNORECASE)


def parse_hhmm(text: str) -> Optional[int]:
    """Return minutes since midnight parsed from the first HH:MM match."""
    match = TIME_RE.search(text or "")
    if not match:
        return None
    hh = int(match.group(1))
    mm = int(match.group(2))
    if hh == 12:
        hh = 0
    return hh * 60 + mm


def parse_hhmmss(text: str) -> Optional[int]:
    """Return seconds on a 12-hour cycle parsed from the first HH:MM:SS match."""
    match = TIME_HMS_RE.search(text or "")
    if not match:
        return None
    hh = int(match.group(1))
    mm = int(match.group(2))
    ss = int(match.group(3))
    if hh == 12:
        hh = 0
    return hh * 3600 + mm * 60 + ss


def parse_hhmm_all(text: str) -> List[int]:
    """Return all parsed HH:MM matches as minutes since midnight."""
    minutes = []
    for match in TIME_RE.finditer(text or ""):
        hh = int(match.group(1))
        mm = int(match.group(2))
        if hh == 12:
            hh = 0
        minutes.append(hh * 60 + mm)
    return minutes


def parse_delta_minutes(text: str) -> Optional[int]:
    """Return delta minutes parsed from explicit delta mentions or time pairs."""
    if not text:
        return None
    match = DELTA_RE.search(text)
    if match:
        return int(match.group(1))
    match = DELTA_FALLBACK_RE.search(text)
    if match:
        return int(match.group(1))
    times = parse_hhmm_all(text)
    if len(times) >= 2:
        return times[1] - times[0]
    return None
