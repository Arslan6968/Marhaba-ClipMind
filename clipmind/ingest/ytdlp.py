"""Platform source adapter via yt-dlp with audio-first & targeted section downloads.
Respects platform limits, DRM, and access restrictions; never bypasses authentication or paywalls.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Optional, Callable, Dict, Any, Tuple, List
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np

from .base import SourceAdapter
from ..schemas import SourceInfo
from ..config import get_settings

PLATFORM_DOMAINS = {
    "youtube": {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"},
    "twitch": {"twitch.tv", "www.twitch.tv", "clips.twitch.tv", "m.twitch.tv"},
    "vimeo": {"vimeo.com", "www.vimeo.com", "player.vimeo.com"},
    "tiktok": {"tiktok.com", "www.tiktok.com", "vm.tiktok.com"},
    "facebook": {"facebook.com", "www.facebook.com", "fb.watch"},
    "x": {"x.com", "twitter.com", "www.twitter.com"},
    "kick": {"kick.com", "www.kick.com"},
}


def ytdlp_base_args(url: str = "") -> List[str]:
    """Builds standard yt-dlp flags including JS runtime detection for YouTube."""
    args = ["--no-playlist", "--no-warnings"]
    if "youtu" in url.lower() and not shutil.which("deno"):
        for rt in ("node", "bun"):
            p = shutil.which(rt)
            if p:
                args += ["--js-runtimes", f"{rt}:{p}"]
                break
    return args


def friendly_ytdlp_error(stderr_text: str, url: str = "") -> str:
    low = (stderr_text or "").lower()
    if "not a bot" in low or "429" in low:
        return "YouTube anti-bot / rate-limit gate triggered (Sign in to confirm you're not a bot). Respecting platform gate; use local downloads or wait before retrying."
    if "confirm your age" in low or "sign in to confirm" in low or "age-restricted" in low:
        return "This video is age-restricted and requires platform login (skipped; never bypasses access controls)."
    if "subscriber" in low or "sub-only" in low or "only available to" in low or "members-only" in low:
        return "This video is restricted to subscribers/members (skipped; access controls respected)."
    if "drm" in low or "protected" in low:
        return "This stream uses DRM protection and cannot be downloaded."
    if "private video" in low or "unavailable" in low:
        return "Video is private, deleted, or unavailable in this region."
    if "kick" in url.lower() and ("403" in low or "forbidden" in low):
        return "Kick blocked automated access for this VOD (403 Forbidden). Download locally or pick another stream."
    clean_lines = [l.strip() for l in (stderr_text or "").splitlines() if l.strip()]
    return clean_lines[-1] if clean_lines else "Unknown platform download error."


def url_cache_dir(url: str) -> Path:
    """Stable per-URL cache directory so audio, transcripts, and metadata survive across runs."""
    settings = get_settings()
    h = hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", url.split("://")[-1])[:36].strip("_")
    d = settings.cache_dir / f"{slug}_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _finished_files(folder: Path, stem: str) -> List[Path]:
    return [
        p for p in folder.glob(f"{stem}.*")
        if not p.name.endswith((".part", ".ytdl", ".json", ".tmp", ".npz")) and ".part-" not in p.name
    ]


class YtDlpAdapter(SourceAdapter):
    def can_handle(self, url_or_path: str) -> bool:
        if not url_or_path.startswith(("http://", "https://")):
            return False
        host = (urlparse(url_or_path).hostname or "").lower()
        return any(host in domains for domains in PLATFORM_DOMAINS.values())

    def _get_platform_name(self, url: str) -> str:
        host = (urlparse(url).hostname or "").lower()
        for plat, domains in PLATFORM_DOMAINS.items():
            if host in domains:
                return plat
        return "web_video"

    def probe_full_json(self, url: str, refresh: bool = False) -> Dict[str, Any]:
        """Fetches full yt-dlp -J metadata with local disk caching."""
        cache_dir = url_cache_dir(url)
        meta_file = cache_dir / "ytdlp_meta.json"
        if meta_file.is_file() and not refresh:
            try:
                cached = json.loads(meta_file.read_text(encoding="utf-8"))
                # Refresh if it was marked live and >5 min old
                if not (cached.get("live_status") == "is_live" and time.time() - cached.get("_cached_at", 0) > 300):
                    return cached
            except Exception:
                pass

        settings = get_settings()
        cmd = [settings.ytdlp_bin, "-J", "--skip-download", *ytdlp_base_args(url), url]
        last_err = None
        for attempt in range(2):
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                if res.returncode == 0 and res.stdout.strip():
                    data = json.loads(res.stdout)
                    data["_cached_at"] = time.time()
                    try:
                        meta_file.write_text(json.dumps(data), encoding="utf-8")
                    except Exception:
                        pass
                    return data
                last_err = RuntimeError(friendly_ytdlp_error(res.stderr, url))
            except subprocess.TimeoutExpired as e:
                last_err = e
        raise last_err

    def probe(self, url_or_path: str) -> SourceInfo:
        platform = self._get_platform_name(url_or_path)
        try:
            data = self.probe_full_json(url_or_path)
            dur = float(data.get("duration") or 0.0)
            title = data.get("title") or "Untitled Video"
            channel = data.get("channel") or data.get("uploader")
            channel_url = data.get("channel_url") or data.get("uploader_url")
            views = data.get("view_count")
            live_status = data.get("live_status")
            is_live = (live_status == "is_live") or bool(data.get("is_live"))

            lang = data.get("language")
            if not lang:
                auto_caps = data.get("automatic_captions") or {}
                lang = next((k[:-5] for k in auto_caps if k.endswith("-orig")), None)

            if is_live:
                return SourceInfo(
                    url_or_path=url_or_path,
                    source_type="platform_url",
                    platform=platform,
                    is_supported=False,
                    title=title,
                    channel_name=channel,
                    is_live=True,
                    live_status=live_status,
                    error_message="This stream is currently LIVE. Platforms only allow section clipping after the stream finishes.",
                )

            return SourceInfo(
                url_or_path=url_or_path,
                source_type="platform_url",
                platform=platform,
                is_supported=True,
                title=title,
                duration=round(dur, 2),
                width=int(data.get("width") or 1920),
                height=int(data.get("height") or 1080),
                fps=float(data.get("fps") or 30.0),
                has_audio=True,
                channel_name=channel,
                channel_url=channel_url,
                view_count=views,
                is_live=False,
                live_status=live_status,
                language=lang,
            )
        except subprocess.TimeoutExpired:
            return SourceInfo(
                url_or_path=url_or_path,
                source_type="platform_url",
                platform=platform,
                is_supported=False,
                error_message="Platform metadata lookup timed out after 60 seconds.",
            )
        except Exception as e:
            return SourceInfo(
                url_or_path=url_or_path,
                source_type="platform_url",
                platform=platform,
                is_supported=False,
                error_message=str(e),
            )

    def fetch_audio_only(self, url: str, progress_cb: Optional[Callable[[float, str], None]] = None) -> Path:
        """Downloads audio-only stream into cache_dir so multi-hour videos avoid downloading GBs up front."""
        cache_dir = url_cache_dir(url)
        existing = _finished_files(cache_dir, "audio")
        if existing and existing[0].stat().st_size > 10_000:
            if progress_cb:
                progress_cb(1.0, f"Reusing cached audio ({existing[0].stat().st_size / 1e6:.1f} MB)")
            return existing[0]

        for old in cache_dir.glob("audio.*"):
            old.unlink(missing_ok=True)

        if progress_cb:
            progress_cb(0.15, "Downloading audio track (video sections fetched later only for approved clips)...")

        settings = get_settings()
        cmd = [
            settings.ytdlp_bin,
            *ytdlp_base_args(url),
            "-f", "ba/b",
            "-N", "4",
            "-o", str(cache_dir / "audio.%(ext)s"),
            url,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
        if res.returncode != 0:
            raise RuntimeError(friendly_ytdlp_error(res.stderr, url))

        got = _finished_files(cache_dir, "audio")
        if not got:
            raise RuntimeError("Audio download completed but no audio file appeared.")
        if progress_cb:
            progress_cb(1.0, f"Audio ready ({got[0].stat().st_size / 1e6:.1f} MB)")
        return got[0]

    def fetch_video_section(self, url: str, start: float, end: float, dest_path: Path, max_height: int = 1080) -> Path:
        """Downloads ONLY [start, end] seconds of video in up to max_height MP4."""
        if dest_path.is_file() and dest_path.stat().st_size > 40_000:
            return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        for old in dest_path.parent.glob(f"{dest_path.stem}.*"):
            old.unlink(missing_ok=True)

        settings = get_settings()
        s_pad = max(0.0, start)
        e_pad = max(s_pad + 2.0, end)
        fmt = (
            f"bv*[height<={max_height}][vcodec^=avc1]+ba/"
            f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b"
        )
        cmd = [
            settings.ytdlp_bin,
            *ytdlp_base_args(url),
            "-f", fmt,
            "--merge-output-format", "mp4",
            "--download-sections", f"*{s_pad:.2f}-{e_pad:.2f}",
            "-o", str(dest_path.with_suffix(".%(ext)s")),
            url,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=900)
        if res.returncode != 0:
            raise RuntimeError(friendly_ytdlp_error(res.stderr, url))

        got = _finished_files(dest_path.parent, dest_path.stem)
        if not got:
            raise RuntimeError(f"Section download failed to produce video for {s_pad:.1f}-{e_pad:.1f}s")
        if got[0] != dest_path:
            got[0].replace(dest_path)
        return dest_path

    def fetch_storyboard(self, url: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Downloads YouTube preview storyboard (~1 MB for hours of video) using OpenCV (no Pillow required).
        Returns (times, grayscale_frames[n, h, w]) or None.
        """
        if "youtu" not in url.lower():
            return None
        cache_dir = url_cache_dir(url)
        sb_cache = cache_dir / "storyboard.npz"
        if sb_cache.is_file():
            try:
                loaded = np.load(sb_cache)
                return loaded["times"], loaded["frames"]
            except Exception:
                pass

        try:
            full = self.probe_full_json(url)
        except Exception:
            return None

        boards = [
            f for f in (full.get("formats") or [])
            if str(f.get("format_id", "")).startswith("sb") and f.get("fragments") and (f.get("width") or 0) >= 60
        ]
        if not boards:
            return None

        sb = min(boards, key=lambda f: f.get("width") or 999)
        rows, cols = int(sb.get("rows") or 1), int(sb.get("columns") or 1)
        tw, th = int(sb["width"]), int(sb["height"])
        step = 1.0 / float(sb.get("fps") or 0.1)
        frags = sb["fragments"]

        def _fetch_tile_sheet(frag: dict) -> Optional[np.ndarray]:
            for _ in range(2):
                try:
                    with httpx.Client(timeout=20.0) as client:
                        resp = client.get(frag["url"], headers={"User-Agent": "Mozilla/5.0"})
                        if resp.status_code == 200:
                            arr = np.frombuffer(resp.content, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                            return img
                except Exception:
                    continue
            return None

        with ThreadPoolExecutor(max_workers=6) as pool:
            sheets = list(pool.map(_fetch_tile_sheet, frags))

        tiles, times, t = [], [], 0.0
        duration = float(full.get("duration") or 0)
        for frag, sheet in zip(frags, sheets):
            n_here = max(1, int(round(float(frag.get("duration") or step * rows * cols) / step)))
            for k in range(min(n_here, rows * cols)):
                if sheet is not None:
                    r, c = divmod(k, cols)
                    y0, x0 = r * th, c * tw
                    if y0 + th <= sheet.shape[0] and x0 + tw <= sheet.shape[1]:
                        tiles.append(sheet[y0:y0 + th, x0:x0 + tw].astype(np.float32))
                        times.append(t)
                t += step

        if len(tiles) < 8:
            return None

        times_arr = np.array(times, dtype=np.float32)
        frames_arr = np.stack(tiles)
        keep = times_arr <= (duration or times_arr[-1]) + step
        times_arr, frames_arr = times_arr[keep], frames_arr[keep]
        try:
            np.savez_compressed(sb_cache, times=times_arr, frames=frames_arr)
        except Exception:
            pass
        return times_arr, frames_arr

    @staticmethod
    def compute_screen_activity(board: Optional[Tuple[np.ndarray, np.ndarray]], total_sec: float) -> Optional[Dict[str, np.ndarray]]:
        """Per-second visual activity score from storyboard (low = static menus/waiting rooms)."""
        if not board:
            return None
        times, frames = board
        if len(frames) < 3 or total_sec <= 1:
            return None
        diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
        diff = np.concatenate([[diff[0]], diff])
        secs = int(np.ceil(total_sec))
        raw = np.interp(np.arange(secs), times, diff).astype(np.float32)
        k = min(7, len(diff))
        smooth_d = np.convolve(np.pad(diff, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:len(diff)]
        smooth = np.interp(np.arange(secs), times, smooth_d).astype(np.float32)
        return {"raw": raw, "smooth": smooth}

    def fetch_live_chat_speed(self, url: str) -> Optional[List[int]]:
        """Extracts chat message frequency per 10s bucket from YouTube replay VODs, or None."""
        if "youtu" not in url.lower():
            return None
        cache_dir = url_cache_dir(url)
        cached_chat = cache_dir / "chat_speed.json"
        if cached_chat.is_file():
            try:
                return json.loads(cached_chat.read_text(encoding="utf-8"))
            except Exception:
                pass

        try:
            meta = self.probe_full_json(url)
            if meta.get("live_status") not in ("was_live", "post_live"):
                return None
        except Exception:
            return None

        settings = get_settings()
        cmd = [
            settings.ytdlp_bin,
            *ytdlp_base_args(url),
            "--skip-download",
            "--write-subs",
            "--sub-langs", "live_chat",
            "-o", str(cache_dir / "chat"),
            url,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        except Exception:
            pass

        files = sorted(cache_dir.glob("chat*.live_chat.json*"))
        if not files:
            cached_chat.write_text("null", encoding="utf-8")
            return None

        counts: Dict[int, int] = {}
        with open(files[0], encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    item = json.loads(line)["replayChatItemAction"]
                    bucket = int(item.get("videoOffsetTimeMsec", 0)) // 10000
                    counts[bucket] = counts.get(bucket, 0) + len(item.get("actions") or [1])
                except Exception:
                    continue

        for f in files:
            f.unlink(missing_ok=True)

        if not counts:
            cached_chat.write_text("null", encoding="utf-8")
            return None

        rate = [0] * (max(counts) + 1)
        for k, n in counts.items():
            rate[k] = n
        cached_chat.write_text(json.dumps(rate), encoding="utf-8")
        return rate

    def fetch_media(self, url_or_path: str, dest_dir: Path, audio_only: bool = False,
                    progress_cb: Optional[Callable[[float, str], None]] = None) -> Path:
        if audio_only:
            return self.fetch_audio_only(url_or_path, progress_cb=progress_cb)

        settings = get_settings()
        cache_dir = url_cache_dir(url_or_path)
        existing = _finished_files(cache_dir, "source")
        if existing and existing[0].stat().st_size > 50_000:
            if progress_cb:
                progress_cb(1.0, f"Reusing cached video: {existing[0].name}")
            return existing[0]

        if progress_cb:
            progress_cb(0.1, f"Downloading full video for {url_or_path}...")

        cmd = [
            settings.ytdlp_bin,
            *ytdlp_base_args(url_or_path),
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
            "--merge-output-format", "mp4",
            "-o", str(cache_dir / "source.%(ext)s"),
            url_or_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
        if res.returncode != 0:
            raise RuntimeError(friendly_ytdlp_error(res.stderr, url_or_path))

        candidates = _finished_files(cache_dir, "source")
        if not candidates:
            raise RuntimeError("Downloaded file not found in cache directory.")
        if progress_cb:
            progress_cb(1.0, f"Completed download: {candidates[0].name}")
        return candidates[0]
