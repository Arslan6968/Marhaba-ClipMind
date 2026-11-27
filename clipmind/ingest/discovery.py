"""Workflow B: Max-Level Creator and Channel Discovery.
Supports channel URLs, @handles, creator names, topic filtering, time windows, and heat/views/latest sorting
without automatically downloading an entire channel.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time
from typing import List, Optional, Dict, Any

from ..schemas import ChannelVideo
from ..config import get_settings
from .ytdlp import ytdlp_base_args

JUNK_TITLE_WORDS = ("trailer", "teaser", "official audio", "#shorts", "shorts compilation")
WINDOW_HOURS = {"24h": 24, "48h": 48, "week": 168, "month": 720, "all": None}


def _history_path() -> Path:
    return get_settings().data_dir / "clipped_history.json"


def load_clipped_history() -> Dict[str, Any]:
    p = _history_path()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def remember_clipped(url: str, title: str = "") -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_clipped_history()
    data[url] = {"title": title, "at": time.time()}
    try:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _format_views(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n)}"


def _format_age(hours: float) -> str:
    if hours < 1:
        return f"{max(1, int(hours * 60))}m"
    if hours < 48:
        return f"{int(hours)}h"
    return f"{int(hours / 24)}d"


def _compute_heat(item: Dict[str, Any], sort_mode: str = "hot", topic_rel: float = 1.0) -> float:
    now = time.time()
    posted = float(item.get("posted_timestamp") or 0.0)
    age_h = max(0.5, (now - posted) / 3600.0) if posted > 0 else None
    views = float(item.get("views") or 0)
    likes = float(item.get("likes") or 0)
    per_hour = views / age_h if age_h else 0.0

    why = []
    if views > 0:
        why.append(f"{_format_views(views)} views")
    if age_h is not None:
        why.append(f"{_format_views(per_hour)}/h")
        why.append(f"{_format_age(age_h)} ago")
    elif item.get("upload_date"):
        why.append(str(item["upload_date"]))

    if item.get("is_live"):
        why.insert(0, "LIVE")

    if sort_mode == "latest":
        score = posted if posted > 0 else (views * 0.001)
    elif sort_mode == "views":
        score = views
    else:
        # 'hot': blend velocity (views/hr) and total reach
        score = (max(per_hour, 1.0) ** 0.55) * (max(views, 1.0) ** 0.45) if age_h else max(views, 1.0)
        if views > 0 and likes > 0:
            ratio = likes / views
            score *= 1.0 + min(0.4, ratio * 8.0)

    if item.get("is_live"):
        score *= 1.2
    score *= (0.4 + 0.6 * topic_rel)

    item["reason"] = " · ".join(why) or "Recent upload"
    return round(score, 2)


def _enrich_video(url: str) -> Optional[Dict[str, Any]]:
    settings = get_settings()
    cmd = [settings.ytdlp_bin, "-J", "--skip-download", *ytdlp_base_args(url), url]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
        if res.returncode != 0 or not res.stdout.strip():
            return None
        full = json.loads(res.stdout)
        if full.get("live_status") == "is_upcoming":
            return None
        auto_caps = full.get("automatic_captions") or {}
        lang = full.get("language") or next((k[:-5] for k in auto_caps if k.endswith("-orig")), None)
        return {
            "views": int(full.get("view_count") or 0),
            "likes": int(full.get("like_count") or 0),
            "duration": float(full.get("duration") or 0.0),
            "posted_timestamp": float(full.get("release_timestamp") or full.get("timestamp") or 0.0),
            "upload_date": str(full.get("upload_date") or ""),
            "is_live": full.get("live_status") == "is_live" or bool(full.get("is_live")),
            "channel_name": full.get("channel") or full.get("uploader") or "",
            "language": lang,
        }
    except Exception:
        return None


def _build_target_urls(creator_or_url: str, limit: int, topic: Optional[str]) -> List[str]:
    q = creator_or_url.strip()
    if q.startswith(("http://", "https://")):
        base = q.rstrip("/")
        if "twitch.tv" in base.lower() and "/videos" not in base.lower():
            return [f"{base}/videos?filter=archives&sort=time"]
        if "kick.com" in base.lower() and "/videos" not in base.lower():
            return [f"{base}/videos"]
        if not any(base.endswith(sfx) for sfx in ("/videos", "/streams", "/shorts", "/playlists")):
            return [f"{base}/videos", f"{base}/streams"]
        return [base]
    if q.startswith("@"):
        handle = q
        return [f"https://www.youtube.com/{handle}/videos", f"https://www.youtube.com/{handle}/streams"]

    search_q = f"{q} {topic}".strip() if topic else q
    fetch_n = min(25, max(12, limit * 3))
    return [f"ytsearch{fetch_n}:{search_q}"]


def discover_channel_videos(
    creator_or_url: str,
    limit: int = 8,
    topic: Optional[str] = None,
    within: str = "all",
    sort_by: str = "hot",
    include_shorts: bool = False,
    include_clipped: bool = False,
    enrich: bool = True,
) -> List[ChannelVideo]:
    """Discovers videos/streams for a creator name, @handle, or channel URL."""
    settings = get_settings()
    targets = _build_target_urls(creator_or_url, limit, topic)
    done_urls = set() if include_clipped else set(load_clipped_history().keys())

    raw_entries: List[Dict[str, Any]] = []
    seen_ids = set()

    for url in targets:
        cmd = [
            settings.ytdlp_bin,
            "--flat-playlist",
            "--playlist-end", str(min(30, max(12, limit * 3))),
            "-J",
            *ytdlp_base_args(url),
            url,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
            if res.returncode != 0 or not res.stdout.strip():
                continue
            data = json.loads(res.stdout)
            entries = data.get("entries") or []
            channel_fallback = data.get("channel") or data.get("uploader") or data.get("title") or ""
            for e in entries:
                if not e or not e.get("id"):
                    continue
                vid_id = str(e["id"])
                if vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)
                vid_url = e.get("url") or e.get("webpage_url") or f"https://www.youtube.com/watch?v={vid_id}"
                if not vid_url.startswith("http"):
                    vid_url = f"https://www.youtube.com/watch?v={vid_id}"
                if vid_url in done_urls:
                    continue

                title = (e.get("title") or "").strip()
                low_title = title.lower()
                dur = float(e.get("duration") or 0.0)
                is_live = bool(e.get("is_live") or e.get("live_status") == "is_live")

                if not include_shorts:
                    if "/shorts/" in vid_url or (0 < dur < 75 and not is_live):
                        continue
                    if any(jw in low_title for jw in JUNK_TITLE_WORDS):
                        continue

                raw_entries.append({
                    "video_id": vid_id,
                    "title": title,
                    "url": vid_url,
                    "channel_name": e.get("channel") or e.get("uploader") or channel_fallback,
                    "duration": round(dur, 2),
                    "views": int(e.get("view_count") or 0),
                    "likes": 0,
                    "upload_date": str(e.get("upload_date") or ""),
                    "posted_timestamp": float(e.get("release_timestamp") or e.get("timestamp") or 0.0),
                    "is_live": is_live,
                    "language": None,
                })
        except Exception:
            continue

    if not raw_entries:
        return []

    # Topic scoring using semantic matcher if topic is specified
    if topic:
        from ..understand.intent import calculate_topic_relevance
        scored_entries = []
        for item in raw_entries:
            rel = calculate_topic_relevance(item["title"], topic)
            if rel >= 0.40:
                item["topic_relevance"] = rel
                scored_entries.append(item)
        # If semantic filter left nothing, keep loose token match fallback
        raw_entries = scored_entries or raw_entries

    # Enrich only top entries missing key metadata (avoids extra -J bursts that trigger platform anti-bot checks)
    if enrich:
        to_check = raw_entries[:min(len(raw_entries), max(8, limit * 2))]
        need_idx = [
            i for i, it in enumerate(to_check)
            if (it.get("posted_timestamp", 0.0) == 0.0 and not it.get("upload_date")) or it.get("duration", 0.0) == 0.0
        ]
        if need_idx:
            with ThreadPoolExecutor(max_workers=2) as pool:
                enriched_vals = list(pool.map(lambda idx: _enrich_video(to_check[idx]["url"]), need_idx))
            drop_indices = set()
            for idx, extra in zip(need_idx, enriched_vals):
                if extra is None and sort_by == "latest":
                    drop_indices.add(idx)
                elif extra:
                    to_check[idx].update({k: v for k, v in extra.items() if v not in (None, "", 0) or k == "is_live"})
            raw_entries = [it for i, it in enumerate(to_check) if i not in drop_indices]
        else:
            raw_entries = to_check

    # Time window filter
    max_age_h = WINDOW_HOURS.get(within)
    if max_age_h is not None:
        now = time.time()
        filtered = []
        for item in raw_entries:
            posted = float(item.get("posted_timestamp") or 0.0)
            if posted == 0.0 or (now - posted) <= max_age_h * 3600.0:
                filtered.append(item)
        raw_entries = filtered

    results: List[ChannelVideo] = []
    for item in raw_entries:
        rel = float(item.get("topic_relevance", 1.0))
        heat = _compute_heat(item, sort_mode=sort_by, topic_rel=rel)
        results.append(ChannelVideo(
            video_id=item["video_id"],
            title=item["title"],
            url=item["url"],
            channel_name=item.get("channel_name", ""),
            duration=float(item.get("duration", 0.0)),
            views=int(item.get("views", 0)),
            likes=int(item.get("likes", 0)),
            upload_date=str(item.get("upload_date", "")),
            posted_timestamp=float(item.get("posted_timestamp", 0.0)),
            is_live=bool(item.get("is_live", False)),
            heat_score=heat,
            topic_relevance=rel,
            language=item.get("language"),
            reason=item.get("reason", ""),
        ))

    results.sort(key=lambda v: -v.heat_score)
    return results[:limit]
