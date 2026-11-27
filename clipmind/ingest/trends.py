"""YouTube Search Autocomplete & Trend Demand Analyzer (100% public, zero API keys).
Identifies what viewers are actively searching and watching so ClipMind prioritizes
high-demand videos and extracts the exact high-demand segments as separate clips.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import re
import subprocess
from typing import List, Dict, Any, Optional

import httpx

from ..config import get_settings
from ..schemas import ChannelVideo
from .ytdlp import ytdlp_base_args

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "how",
    "who", "which", "2026", "2027", "2025", "pakistan", "punjab", "video", "videos", "full",
    "complete", "details", "update", "updates", "today", "latest", "new", "online", "all",
    "mein", "hai", "hain", "kya", "kaise", "kab", "aur", "ki", "ka", "ke", "se", "ko",
}


def fetch_youtube_suggestions(seed_query: str, max_results: int = 10) -> List[str]:
    """Fetches real-time YouTube search autocomplete queries for a seed topic."""
    for client_id in ("firefox", "chrome"):
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    "https://suggestqueries.google.com/complete/search",
                    params={"client": client_id, "ds": "yt", "q": seed_query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list):
                        got = [str(x).strip() for x in payload[1][:max_results] if str(x).strip()]
                        if got:
                            return got
        except Exception:
            continue
    return []


def fetch_youtube_top_videos(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetches top-viewed YouTube search results for a query via yt-dlp --flat-playlist."""
    settings = get_settings()
    import urllib.parse
    # sp=CAMSAhAB sorts YouTube search by view count for type=video
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}&sp=CAMSAhAB"
    cmd = [
        settings.ytdlp_bin,
        "--flat-playlist",
        "--playlist-end", str(limit),
        "-J",
        *ytdlp_base_args(url),
        url,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=35)
        if res.returncode != 0 or not res.stdout.strip():
            return []
        data = json.loads(res.stdout)
        out = []
        for e in (data.get("entries") or []):
            if not e or not e.get("id"):
                continue
            out.append({
                "id": e["id"],
                "title": (e.get("title") or "").strip(),
                "views": int(e.get("view_count") or 0),
                "duration": float(e.get("duration") or 0.0),
                "channel": e.get("channel") or e.get("uploader") or "",
            })
        return out
    except Exception:
        return []


def extract_seed_topics(videos: List[ChannelVideo], max_seeds: int = 4) -> List[str]:
    """Extracts concise 2-4 word search seeds (without narrow year ranges) for YouTube autocomplete."""
    seeds: List[str] = []
    for v in videos:
        head = re.split(r"[|?:–—]", v.title)[0].strip()
        # Strip narrow tokens like 2026-27 or Session so autocomplete yields broad student searches
        head = re.sub(r"\b(?:20\d{2}(?:-\d{2})?|session|pakistan|last year)\b", "", head, flags=re.I)
        words = [w for w in head.split() if len(w) >= 2]
        if len(words) >= 2:
            seed = " ".join(words[:3])
            low = seed.lower()
            if low not in {s.lower() for s in seeds}:
                seeds.append(seed)
        if len(seeds) >= max_seeds:
            break
    return seeds


def analyze_youtube_trends(seed_queries: List[str], videos: Optional[List[ChannelVideo]] = None) -> Dict[str, Any]:
    """Analyzes YouTube search autocomplete + top-viewed videos to rank high-demand subtopics."""
    if not seed_queries and videos:
        seed_queries = extract_seed_topics(videos)
    if not seed_queries:
        return {"seeds": [], "suggestions": [], "hot_keywords": [], "top_videos": []}

    all_suggestions: List[str] = []
    top_yt_videos: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        sug_futs = [pool.submit(fetch_youtube_suggestions, q, 8) for q in seed_queries[:4]]
        vid_futs = [pool.submit(fetch_youtube_top_videos, q, 8) for q in seed_queries[:3]]
        for f in sug_futs:
            all_suggestions.extend(f.result() or [])
        for f in vid_futs:
            top_yt_videos.extend(f.result() or [])

    # Deduplicate suggestions preserving order
    seen_s = set()
    uniq_suggestions = []
    for s in all_suggestions:
        low = s.lower()
        if low not in seen_s:
            seen_s.add(low)
            uniq_suggestions.append(s)

    # Extract high-demand keywords weighted by search suggestion frequency + video view counts
    kw_scores: Counter = Counter()
    seed_tokens = {w.lower() for q in seed_queries for w in re.findall(r"[A-Za-z0-9]{2,}", q)}

    for s in uniq_suggestions:
        toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}", s) if t.lower() not in STOP_WORDS]
        for t in toks:
            kw_scores[t] += 3.0 if t not in seed_tokens else 0.8
        for a, b in zip(toks, toks[1:]):
            phrase = f"{a} {b}"
            kw_scores[phrase] += 4.5

    for tv in top_yt_videos:
        weight = max(1.0, (tv["views"] / 1000.0) ** 0.35)
        toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}", tv["title"]) if t.lower() not in STOP_WORDS]
        for t in set(toks):
            kw_scores[t] += weight * (1.2 if t not in seed_tokens else 0.4)
        for a, b in set(zip(toks, toks[1:])):
            kw_scores[f"{a} {b}"] += weight * 1.5

    # Clean redundant unigrams already covered by top bigrams
    top_raw = [k for k, _ in kw_scores.most_common(30)]
    hot_keywords = []
    for k in top_raw:
        if " " not in k and any(k in b.split() for b in top_raw[:14] if " " in b and kw_scores[b] >= kw_scores[k] * 0.7):
            continue
        hot_keywords.append({"term": k, "demand_score": round(kw_scores[k], 1)})
        if len(hot_keywords) >= 14:
            break

    # Score channel videos against discovered demand
    ranked_channel_videos = []
    if videos:
        for v in videos:
            low_t = v.title.lower()
            matched = [hk["term"] for hk in hot_keywords if hk["term"] in low_t]
            demand_boost = sum(hk["demand_score"] for hk in hot_keywords if hk["term"] in low_t)
            combined = round(v.heat_score + demand_boost * 1.5, 2)
            ranked_channel_videos.append({
                "video": v,
                "combined_demand_score": combined,
                "matched_trends": matched[:6],
            })
        ranked_channel_videos.sort(key=lambda x: -x["combined_demand_score"])

    return {
        "seeds": seed_queries,
        "suggestions": uniq_suggestions[:20],
        "hot_keywords": hot_keywords,
        "top_market_videos": top_yt_videos[:10],
        "ranked_channel_videos": ranked_channel_videos,
    }
