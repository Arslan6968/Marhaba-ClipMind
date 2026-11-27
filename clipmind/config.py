"""Configuration and runtime environment for Marhaba ClipMind v3.
Optimized for local-first execution on 8 GB RAM CPU-first machines.
"""
import os
import shutil
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT_DIR.parent
MODELS_DIR = Path(os.getenv("CLIPMIND_MODELS_DIR", str(ROOT_DIR / "models")))
FONTS_DIR = ROOT_DIR / "fonts"
DATA_DIR = Path(os.getenv("CLIPMIND_DATA_DIR", str(Path.home() / ".clipmind")))
CACHE_DIR = DATA_DIR / "cache"

WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
}

ASPECT_DIMENSIONS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}


class Settings:
    def __init__(self):
        self.workspace_dir: Path = WORKSPACE_DIR
        self.models_dir: Path = MODELS_DIR
        self.fonts_dir: Path = FONTS_DIR
        self.data_dir: Path = DATA_DIR
        self.cache_dir: Path = CACHE_DIR

        self.default_whisper_size: str = os.getenv("CLIPMIND_WHISPER_SIZE", "base")
        self.whisper_model_path: str = os.getenv("CLIPMIND_WHISPER_PATH", str(MODELS_DIR / "whisper-base"))
        self.yunet_model_path: str = os.getenv("CLIPMIND_YUNET_PATH", str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"))
        self.llm_model_path: str = os.getenv("CLIPMIND_LLM_PATH", str(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))

        # Audio & Chunking
        self.audio_sample_rate: int = 16000
        self.chunk_duration_sec: int = 300  # 5-minute chunks
        self.max_auto_blocks_long_video: int = 12  # up to 60 min of hottest content on >1.5h streams

        # Clip Duration Constraints
        self.min_clip_sec: float = 15.0
        self.max_clip_sec: float = 90.0
        self.target_clip_sec: float = 45.0

        # Memory & Hardware Guards (8 GB RAM target)
        self.min_free_ram_mb: int = 600
        self.llm_min_free_ram_mb: int = 1600
        self.cpu_threads: int = max(1, min(4, os.cpu_count() or 4))

        # Video Resolution & Formats
        self.target_width: int = 1080
        self.target_height: int = 1920
        self.default_caption_font: str = "Noto Nastaliq Urdu, Segoe UI, Inter, Arial"
        self.caption_fonts_dir: Path = FONTS_DIR

        # Tool Paths
        self.ffmpeg_bin: str = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_bin: str = shutil.which("ffprobe") or "ffprobe"
        self.ytdlp_bin: str = shutil.which("yt-dlp") or "yt-dlp"

    def resolve_whisper_model(self, size: str | None = None) -> str:
        """Resolves a Whisper model size ('tiny'|'base'|'small') or local path."""
        chosen = size or self.default_whisper_size
        if Path(chosen).exists():
            return str(chosen)
        candidates = [
            self.models_dir / f"whisper-{chosen}",
            self.workspace_dir.parent / "marhaba-clipmind" / "backend" / "models" / f"whisper-{chosen}",
            self.data_dir / "models" / f"whisper-{chosen}",
        ]
        for c in candidates:
            if c.is_dir() and (c / "model.bin").is_file():
                return str(c)
        if chosen == "base" and Path(self.whisper_model_path).exists():
            return self.whisper_model_path
        return WHISPER_REPOS.get(chosen, chosen)

    def resolve_yunet_model(self) -> Path | None:
        candidates = [
            Path(self.yunet_model_path),
            self.models_dir / "face_detection_yunet_2023mar.onnx",
            self.workspace_dir.parent / "marhaba-clipmind" / "backend" / "models" / "face_detection_yunet_2023mar.onnx",
            Path.home() / ".beyondgtaclips" / "models" / "face_yunet.onnx",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def dimensions_for(self, aspect_ratio: str, src_w: int = 1920, src_h: int = 1080) -> Tuple[int, int]:
        if aspect_ratio in ASPECT_DIMENSIONS:
            return ASPECT_DIMENSIONS[aspect_ratio]
        # original
        w = max(2, src_w - src_w % 2)
        h = max(2, src_h - src_h % 2)
        return w, h


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
