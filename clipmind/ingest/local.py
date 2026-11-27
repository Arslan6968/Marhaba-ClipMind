"""Local file ingestion adapter (zero-copy read-in-place for large SSD efficiency)."""
from pathlib import Path
from typing import Optional, Callable

from .base import SourceAdapter
from ..schemas import SourceInfo
from ..media.probe import probe_media, MediaError

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".flac", ".m4a"}


class LocalFileAdapter(SourceAdapter):
    def can_handle(self, url_or_path: str) -> bool:
        if url_or_path.startswith(("http://", "https://")):
            return False
        p = Path(url_or_path)
        return p.is_file() or p.suffix.lower() in ALLOWED_EXTENSIONS

    def probe(self, url_or_path: str) -> SourceInfo:
        path = Path(url_or_path).expanduser().resolve()
        if not path.is_file():
            return SourceInfo(
                url_or_path=url_or_path,
                source_type="local_file",
                platform="local",
                is_supported=False,
                error_message=f"File not found: {url_or_path}",
            )

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return SourceInfo(
                url_or_path=str(path),
                source_type="local_file",
                platform="local",
                is_supported=False,
                error_message=f"Unsupported file format: {path.suffix}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        try:
            meta = probe_media(path)
            if not meta.get("has_video") and not meta.get("has_audio"):
                return SourceInfo(
                    url_or_path=str(path),
                    source_type="local_file",
                    platform="local",
                    is_supported=False,
                    error_message="Media file contains neither video nor audio streams.",
                )
            return SourceInfo(
                url_or_path=str(path),
                source_type="local_file",
                platform="local",
                is_supported=True,
                title=path.stem,
                duration=meta["duration"],
                width=meta["width"],
                height=meta["height"],
                fps=meta["fps"],
                has_audio=meta["has_audio"],
            )
        except MediaError as e:
            return SourceInfo(
                url_or_path=str(path),
                source_type="local_file",
                platform="local",
                is_supported=False,
                error_message=f"Corrupt or unreadable media file: {e}",
            )

    def fetch_media(self, url_or_path: str, dest_dir: Path, audio_only: bool = False,
                    progress_cb: Optional[Callable[[float, str], None]] = None) -> Path:
        """Uses the local file in-place instead of wasting SSD space copying multi-GB videos."""
        src = Path(url_or_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Local media file not found: {src}")
        if progress_cb:
            progress_cb(1.0, f"Using local file in-place: {src.name}")
        return src
