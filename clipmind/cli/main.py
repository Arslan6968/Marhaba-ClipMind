"""Command-Line Interface for Marhaba ClipMind v3."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil

from ..config import get_settings
from ..ingest import discover_channel_videos, probe_source, analyze_youtube_trends
from ..understand.intent import parse_user_query, generate_adaptive_questions
from ..understand.brains import NeedAgentInput, LocalLlamaBrain
from ..pipeline.runner import ClipMindPipeline, PipelineError, remake_clips
from ..production.presets import EDITING_PRESETS

AGENT_EXIT_CODE = 3


def _check_ffmpeg_libass(ffmpeg_bin: str) -> bool:
    try:
        res = subprocess.run([ffmpeg_bin, "-filters"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
        return "subtitles" in res.stdout
    except Exception:
        return False


def cmd_doctor(args):
    """Checks dependencies, FFmpeg libass/RTL support, models, RAM, and JS runtimes."""
    settings = get_settings()
    print("=== Marhaba ClipMind v3 Doctor ===")

    ffmpeg_path = shutil.which(settings.ffmpeg_bin)
    ffmpeg_ok = bool(ffmpeg_path)
    libass_ok = _check_ffmpeg_libass(settings.ffmpeg_bin) if ffmpeg_ok else False
    print(f"[{'PASS' if ffmpeg_ok else 'FAIL'}] FFmpeg: {ffmpeg_path or settings.ffmpeg_bin} (libass subtitles: {'YES' if libass_ok else 'NO'})")

    ytdlp_path = shutil.which(settings.ytdlp_bin)
    ytdlp_ok = bool(ytdlp_path)
    print(f"[{'PASS' if ytdlp_ok else 'FAIL'}] yt-dlp: {ytdlp_path or settings.ytdlp_bin}")

    js_rt = next((rt for rt in ("deno", "node", "bun") if shutil.which(rt)), None)
    print(f"[{'PASS' if js_rt else 'WARN'}] YouTube JS Runtime: {js_rt or 'None found (install deno or node for reliable YouTube extraction)'}")

    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    free_gb = mem.available / (1024**3)
    ram_ok = free_gb >= 0.60
    print(f"[{'PASS' if ram_ok else 'WARN'}] RAM: {free_gb:.1f} GB available / {total_gb:.1f} GB total (CPU threads: {settings.cpu_threads})")

    for sz in ("tiny", "base", "small"):
        resolved = settings.resolve_whisper_model(sz)
        exists = Path(resolved).is_dir()
        mark = "INSTALLED" if exists else "on-demand"
        print(f"[INFO] Whisper '{sz}': {mark} ({resolved})")

    yunet = settings.resolve_yunet_model()
    print(f"[{'PASS' if yunet else 'WARN'}] YuNet Face Model: {yunet or 'Not found (Haar fallback active)'}")

    fonts_dir = Path(settings.fonts_dir)
    urdu_ok = any(fonts_dir.glob("*NotoNastaliq*.ttf")) or any(fonts_dir.glob("*gulzar*.ttf"))
    print(f"[{'PASS' if urdu_ok else 'WARN'}] Urdu RTL & Latin Fonts: {fonts_dir}")

    llm_ok, llm_msg = LocalLlamaBrain().available()
    print(f"[{'PASS' if llm_ok else 'INFO'}] Optional Local LLM (llama.cpp): {llm_msg}")

    if ffmpeg_ok and ytdlp_ok and libass_ok:
        print("\nSTATUS: READY. Marhaba ClipMind is primed for universal local-first clipping.")
        return 0
    print("\nSTATUS: ACTION NEEDED. Verify missing critical components above.")
    return 1


def cmd_styles(args):
    """Displays the 6 genuinely distinct editing presets."""
    print("=== Marhaba ClipMind Editing Presets ===")
    for k, p in EDITING_PRESETS.items():
        print(f"\n[{k.upper()}] - {p['label']}")
        print(f"  Description : {p['description']}")
        print(f"  Layout      : {p['preferred_layout']} (fallback: {p['fallback_layout']})")
        print(f"  Punch Zooms : {p.get('punch_ins', False)} | Hook Card: {p.get('hook_card', False)} | Fade: {p.get('fade', False)}")


def cmd_probe(args):
    """Probes source metadata and returns adaptive questions if user intent is incomplete."""
    info = probe_source(args.url_or_path)
    if not info.is_supported:
        if args.json:
            print(json.dumps({"supported": False, "error": info.error_message}, indent=2))
        else:
            print(f"[UNSUPPORTED] {info.error_message}")
        return 1

    questions = generate_adaptive_questions(info.model_dump(), None, args.intent or "")
    if args.json:
        print(json.dumps({
            "supported": True,
            "source": info.model_dump(),
            "adaptive_questions": questions,
        }, indent=2))
        return 0

    dur_str = f"{int(info.duration // 60)}:{int(info.duration % 60):02d}" if info.duration else "Unknown"
    print(f"\nTitle    : {info.title}")
    print(f"Platform : {info.platform} | Duration: {dur_str} | Resolution: {info.width}x{info.height}")
    if info.channel_name:
        print(f"Creator  : {info.channel_name}")
    if questions:
        print("\nSuggested Clarification Questions:")
        for idx, q in enumerate(questions, 1):
            print(f"  {idx}. {q['question']}")
            for opt in q["options"]:
                print(f"     - {opt}")
    else:
        print("\nIntent is clear — ready to run `clip` directly!")
    return 0


def cmd_find(args):
    """Workflow B: Discovers creator videos/streams with topic, age, and heat filtering."""
    videos = discover_channel_videos(
        args.creator,
        limit=args.count,
        topic=args.topic,
        within=args.within,
        sort_by=args.sort,
        include_shorts=args.include_shorts,
        include_clipped=args.all,
    )

    if args.json:
        print(json.dumps([v.model_dump() for v in videos], indent=2, ensure_ascii=False))
        return 0

    print(f"\nDiscovered {len(videos)} video(s) for '{args.creator}' (topic: {args.topic or 'all'}, sort: {args.sort}):")
    if not videos:
        print("  No matching videos found. Try expanding --within all or adjusting spelling.")
        return 0

    print("-" * 80)
    for idx, v in enumerate(videos, 1):
        dur_str = f"{int(v.duration // 60)}:{int(v.duration % 60):02d}" if v.duration > 0 else "LIVE/Stream"
        print(f"{idx}. [{dur_str}] {v.title}")
        print(f"   Creator: {v.channel_name or args.creator} | {v.reason}")
        print(f"   URL: {v.url}")
    print("-" * 80)
    print("Run `python clipmind.py clip <URL> --intent \"<your objective>\"` to process a chosen video.")
    return 0


def cmd_trend(args):
    """Analyzes YouTube search autocomplete + top-viewed videos to rank high-demand channel videos and topics."""
    target = args.target
    is_channel = target.startswith(("http://", "https://", "@")) or args.channel
    videos = discover_channel_videos(target, limit=args.count, topic=args.topic, enrich=True) if is_channel else []
    seeds = [args.topic] if args.topic else []
    if not seeds and not is_channel:
        seeds = [target]

    report = analyze_youtube_trends(seeds, videos=videos if videos else None)

    if args.json:
        payload = {
            "seeds": report["seeds"],
            "suggestions": report["suggestions"],
            "hot_keywords": report["hot_keywords"],
            "ranked_videos": [
                {
                    "title": r["video"].title,
                    "url": r["video"].url,
                    "duration": r["video"].duration,
                    "views": r["video"].views,
                    "combined_demand_score": r["combined_demand_score"],
                    "matched_trends": r["matched_trends"],
                }
                for r in report.get("ranked_channel_videos", [])
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"\n=== YouTube Trend & Search Demand Analysis ===")
    print(f"Seeds: {', '.join(report['seeds'])}")
    if report["suggestions"]:
        print("\nTop YouTube Search Autocomplete Queries:")
        for s in report["suggestions"][:10]:
            print(f"  - {s}")
    if report["hot_keywords"]:
        print("\nHigh-Demand Keywords:")
        print("  " + " | ".join(f"{hk['term']} ({hk['demand_score']})" for hk in report["hot_keywords"][:10]))
    if report.get("ranked_channel_videos"):
        print("\nChannel Videos Ranked by Search Demand + Velocity:")
        for i, r in enumerate(report["ranked_channel_videos"][:8], 1):
            v = r["video"]
            print(f"  {i}. [{r['combined_demand_score']}] {v.title} ({v.reason})")
            print(f"     {v.url}")
    return 0


def cmd_clip(args):
    """Workflow A: Runs the full clipping pipeline on a single video URL or local file."""
    intent = parse_user_query(args.intent or "")
    if args.topic:
        intent.topic = args.topic
    if args.speaker:
        intent.speaker = args.speaker
    if args.preset:
        intent.preset = args.preset
    if args.duration:
        intent.target_duration = args.duration
    if args.max_clips:
        intent.max_clips = args.max_clips
    if args.aspect:
        intent.aspect_ratio = args.aspect
    if args.caption_lang:
        intent.caption_lang = args.caption_lang
    if args.language:
        intent.language = args.language
    if args.whisper_model:
        intent.whisper_model = args.whisper_model
    if args.brain:
        intent.brain = args.brain
    if args.section:
        intent.section = args.section
    if args.sfx:
        intent.sfx = True
    if args.no_normalize:
        intent.normalize_audio = False
    if args.fast:
        intent.fast_encode = True

    output_dir = Path(args.out) if args.out else None
    if not args.json:
        print("\n=== Starting Marhaba ClipMind Pipeline ===")
        print(f"Source    : {args.url_or_path}")
        print(f"Objective : {intent.objective} | Topic: {intent.topic or 'Auto'} | Speaker: {intent.speaker or 'Auto'}")
        print(f"Format    : {intent.aspect_ratio} | Duration: {intent.target_duration} | Brain: {intent.brain}")
        print("-" * 65)

    def progress_callback(stage, pct, msg):
        if args.json:
            return
        pct_int = int(pct * 100)
        bar = "=" * (pct_int // 4) + "-" * (25 - (pct_int // 4))
        print(f"[{pct_int:3d}%] [{bar}] [{stage.upper()}] {msg}", flush=True)

    pipeline = ClipMindPipeline()
    try:
        res = pipeline.run(
            source_input=args.url_or_path,
            user_intent=intent,
            output_dir=output_dir,
            progress_cb=progress_callback,
        )
    except NeedAgentInput as need:
        if args.json:
            print(json.dumps({
                "status": "awaiting_agent",
                "task_file": str(need.task_file),
                "prompts": [{"prompt": str(p), "answer": str(a)} for p, a in need.pairs],
            }, indent=2))
        else:
            print(f"\n[AGENT MODE PAUSED] Story verification prompts ready at: {need.task_file}")
            print("Read STORY_TASK.md, inspect contact sheets, write .answer.json files, and re-run this exact command.")
        return AGENT_EXIT_CODE
    except PipelineError as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}, indent=2))
        else:
            print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        payload = dict(res)
        payload["rendered_clips"] = [c.model_dump() for c in res.get("rendered_clips", [])]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("\n" + "=" * 65)
    print(f"COMPLETED! Generated {res['clips_count']} verified clip(s).")
    if res["clips_count"] > 0:
        print(f"Output Directory   : {res['output_dir']}")
        print(f"Publishing Manifest: {res['manifest_file']}")
        print(f"Posting Guide      : {res['posting_pack']}")
        print(f"Peak RAM           : {res.get('peak_rss_mb', 0):.0f} MB | Elapsed: {res.get('processing_time_sec', 0):.1f}s")
    else:
        print(f"Note: {res.get('message', 'No clips met the quality gate.')}")
    print("=" * 65)
    return 0


def cmd_remake(args):
    """Re-renders finished clips in an output folder without re-downloading or re-transcribing."""
    try:
        res = remake_clips(
            output_dir=Path(args.folder),
            clip_indices=args.clip,
            preset=args.preset,
            hook=args.hook,
            title=args.title,
            caption_lang=args.caption_lang,
            aspect_ratio=args.aspect,
            sfx=True if args.sfx else None,
        )
        print(f"Remade {res['remade_count']} clip(s) successfully in {res['output_dir']}")
        return 0
    except PipelineError as e:
        print(f"[ERROR] Remake failed: {e}", file=sys.stderr)
        return 1


def cmd_prune(args):
    """Deletes cached media/transcripts older than N days."""
    cache_dir = get_settings().cache_dir
    if not cache_dir.is_dir():
        print("Cache directory is empty.")
        return 0
    cutoff = time.time() - args.days * 86400
    removed = 0
    for child in cache_dir.iterdir():
        try:
            if child.stat().st_mtime < cutoff:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed += 1
        except Exception:
            pass
    print(f"Pruned {removed} cached item(s) older than {args.days} day(s).")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="clipmind", description="Marhaba ClipMind v3: Universal AI Video Clipping Engine")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("doctor", help="Check dependencies, models, FFmpeg libass, and RAM")
    sub.add_parser("styles", help="List the 6 professional editing presets")

    prb = sub.add_parser("probe", help="Inspect URL/file metadata and suggest adaptive questions")
    prb.add_argument("url_or_path", help="Video URL or local file path")
    prb.add_argument("--intent", help="Optional initial user instruction")
    prb.add_argument("--json", action="store_true", help="Output JSON")

    f = sub.add_parser("find", help="Discover creator videos and streams (Workflow B)")
    f.add_argument("creator", help="Creator name, @handle, or channel URL")
    f.add_argument("--topic", help="Semantic topic filter")
    f.add_argument("--within", choices=["24h", "48h", "week", "month", "all"], default="all")
    f.add_argument("--sort", choices=["hot", "views", "latest"], default="hot")
    f.add_argument("--count", type=int, default=6)
    f.add_argument("--include-shorts", action="store_true")
    f.add_argument("--all", action="store_true", help="Include previously clipped URLs")
    f.add_argument("--json", action="store_true")

    tr = sub.add_parser("trend", help="Analyze YouTube search autocomplete & trend demand for a creator or topic")
    tr.add_argument("target", help="Creator URL, @handle, or topic query")
    tr.add_argument("--topic", help="Optional seed topic")
    tr.add_argument("--channel", action="store_true", help="Treat target as creator name search")
    tr.add_argument("--count", type=int, default=8)
    tr.add_argument("--json", action="store_true")

    c = sub.add_parser("clip", help="Analyze video and produce verified clips (Workflow A)")
    c.add_argument("url_or_path", help="Video URL or local file path")
    c.add_argument("--intent", help="Natural language objective")
    c.add_argument("--topic", help="Specific topic focus")
    c.add_argument("--speaker", help="Target speaker or person name inside video")
    c.add_argument("--preset", choices=list(EDITING_PRESETS.keys()))
    c.add_argument("--aspect", choices=["9:16", "16:9", "1:1", "original"], default="9:16")
    c.add_argument("--duration", choices=["short", "medium", "long", "any"], default="medium")
    c.add_argument("--max-clips", type=int, default=4)
    c.add_argument("--caption-lang", choices=["original", "roman_urdu", "none"], default="original")
    c.add_argument("--language", help="Force speech language code (e.g., en, ur)")
    c.add_argument("--whisper-model", choices=["tiny", "base", "small"], default="base")
    c.add_argument("--brain", choices=["auto", "agent", "local", "rules", "none"], default="auto")
    c.add_argument("--section", help="Only process a section, e.g. '1:00:00-2:30:00'")
    c.add_argument("--sfx", action="store_true", help="Add subtle whoosh/pop sound design")
    c.add_argument("--no-normalize", action="store_true", help="Skip -14 LUFS loudness normalization")
    c.add_argument("--fast", action="store_true", help="Use hardware h264 encoder if available")
    c.add_argument("--out", help="Custom output directory")
    c.add_argument("--json", action="store_true")

    r = sub.add_parser("remake", help="Re-render clips in a finished output directory without re-downloading")
    r.add_argument("folder", help="Output folder containing manifest.json")
    r.add_argument("--clip", type=int, action="append", help="Specific 1-based clip index (repeatable)")
    r.add_argument("--preset", choices=list(EDITING_PRESETS.keys()))
    r.add_argument("--aspect", choices=["9:16", "16:9", "1:1", "original"])
    r.add_argument("--caption-lang", choices=["original", "roman_urdu", "none"])
    r.add_argument("--hook", help="New hook text (*stars* highlight accent words)")
    r.add_argument("--title", help="New title")
    r.add_argument("--sfx", action="store_true")

    pr = sub.add_parser("prune", help="Delete cached downloads older than N days")
    pr.add_argument("--days", type=int, default=7)

    args = parser.parse_args()
    if args.subcommand == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.subcommand == "styles":
        sys.exit(cmd_styles(args))
    elif args.subcommand == "probe":
        sys.exit(cmd_probe(args))
    elif args.subcommand == "find":
        sys.exit(cmd_find(args))
    elif args.subcommand == "trend":
        sys.exit(cmd_trend(args))
    elif args.subcommand == "clip":
        sys.exit(cmd_clip(args))
    elif args.subcommand == "remake":
        sys.exit(cmd_remake(args))
    elif args.subcommand == "prune":
        sys.exit(cmd_prune(args))


if __name__ == "__main__":
    main()
