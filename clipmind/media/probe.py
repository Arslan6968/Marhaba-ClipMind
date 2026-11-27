"""FFprobe wrapper for media metadata and validation."""
import json
from pathlib import Path
import subprocess
from typing import Dict, Any

from ..config import get_settings

class MediaError(Exception):
    pass

def probe_media(path: Path) -> Dict[str, Any]:
    """Inspects a media file using ffprobe and extracts duration, dimensions, fps, audio presence."""
    settings = get_settings()
    path = Path(path).resolve()
    if not path.is_file():
        raise MediaError(f"File not found: {path}")

    cmd = [
        settings.ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration,size,bit_rate",
        "-show_streams",
        "-of", "json",
        str(path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise MediaError(f"Failed to probe media file: {e}")

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    dur = 0.0
    if "duration" in fmt:
        try:
            dur = float(fmt["duration"])
        except (ValueError, TypeError):
            pass
    if dur <= 0.0 and v_stream and "duration" in v_stream:
        try:
            dur = float(v_stream["duration"])
        except (ValueError, TypeError):
            pass
    if dur <= 0.0 and a_stream and "duration" in a_stream:
        try:
            dur = float(a_stream["duration"])
        except (ValueError, TypeError):
            pass

    width = int(v_stream.get("width", 0)) if v_stream else 0
    height = int(v_stream.get("height", 0)) if v_stream else 0

    fps = 30.0
    if v_stream and "r_frame_rate" in v_stream:
        try:
            num, den = v_stream["r_frame_rate"].split("/")
            if float(den) > 0:
                fps = float(num) / float(den)
        except Exception:
            pass

    return {
        "duration": round(dur, 2),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "has_audio": a_stream is not None,
        "has_video": v_stream is not None,
        "size_bytes": int(fmt.get("size", 0)),
        "audio_codec": a_stream.get("codec_name") if a_stream else None,
        "video_codec": v_stream.get("codec_name") if v_stream else None,
    }
