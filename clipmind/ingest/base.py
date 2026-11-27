"""Abstract base for video source adapters."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable
from ..schemas import SourceInfo

class SourceAdapter(ABC):
    @abstractmethod
    def can_handle(self, url_or_path: str) -> bool:
        """Return True if this adapter supports the given input."""
        pass

    @abstractmethod
    def probe(self, url_or_path: str) -> SourceInfo:
        """Inspect and return source metadata without full download."""
        pass

    @abstractmethod
    def fetch_media(self, url_or_path: str, dest_dir: Path, audio_only: bool = False,
                    progress_cb: Optional[Callable[[float, str], None]] = None) -> Path:
        """Download or copy the media file to dest_dir."""
        pass
