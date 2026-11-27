"""Universal Ingestion Manager for Marhaba ClipMind."""
from .base import SourceAdapter
from .local import LocalFileAdapter
from .ytdlp import YtDlpAdapter, url_cache_dir
from .url import DirectUrlAdapter
from .discovery import discover_channel_videos, remember_clipped, load_clipped_history
from .trends import analyze_youtube_trends, fetch_youtube_suggestions
from ..schemas import SourceInfo

_ADAPTERS = [
    LocalFileAdapter(),
    YtDlpAdapter(),
    DirectUrlAdapter(),
]


def get_adapter(url_or_path: str) -> SourceAdapter | None:
    for adapter in _ADAPTERS:
        if adapter.can_handle(url_or_path):
            return adapter
    return None


def probe_source(url_or_path: str) -> SourceInfo:
    adapter = get_adapter(url_or_path)
    if not adapter:
        return SourceInfo(
            url_or_path=url_or_path,
            source_type="unknown",
            platform="unknown",
            is_supported=False,
            error_message="Unsupported input format. Provide a local file path, YouTube/Twitch/Kick/Vimeo URL, or direct media link.",
        )
    return adapter.probe(url_or_path)
