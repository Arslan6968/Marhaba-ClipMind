"""Pipeline state, cancellation tokens, progress callbacks, and checkpoint helpers."""
import json
from pathlib import Path
import re
import threading
from typing import Callable, Optional, Dict, Any, Tuple
import psutil


class CancellationToken:
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self):
        with self._lock:
            self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


ProgressCallback = Callable[[str, float, str], None]
"""Signature: callback(stage_name, progress_0_to_1, status_message)"""


def current_rss_mb() -> float:
    try:
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def parse_clock(text: str) -> float:
    """Parses '1:23:45' or '12:30' or '90' into seconds."""
    parts = [float(p) for p in re.split(r":", text.strip()) if p.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"Invalid timestamp: {text}")


def parse_section_bounds(section: Optional[str], total_dur: float) -> Tuple[float, float]:
    if not section:
        return 0.0, total_dur
    a, _, b = section.replace(" ", "").partition("-")
    lo = max(0.0, parse_clock(a)) if a else 0.0
    hi = min(total_dur, parse_clock(b)) if b else total_dur
    if hi <= lo:
        raise ValueError(f"Invalid section bounds '{section}' for duration {total_dur:.1f}s")
    return lo, hi


def save_checkpoint(state_path: Path, stage: str, data: Dict[str, Any]) -> None:
    try:
        cur = {}
        if state_path.is_file():
            cur = json.loads(state_path.read_text(encoding="utf-8"))
        cur["last_stage"] = stage
        cur.setdefault("stages", {})[stage] = data
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    except Exception:
        pass
