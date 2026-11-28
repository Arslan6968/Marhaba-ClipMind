"""Max-Level Video Rendering Engine with Smart Visual Framing, Audio Polish,
Hardware Encoding Fallback, and Strict Post-Render Decode Verification.
"""
from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from ..schemas import ClipCandidate, RenderedClip, ContentClassification, UserIntent, WordToken
from ..config import get_settings
from .presets import get_preset
from .captions import generate_ass_subtitles
from .layout import build_ffmpeg_filtergraph
from .audio_fx import build_sfx_track
from .publishing import generate_publishing_package
from ..media.probe import probe_media, MediaError
from ..media.face import analyze_clip_visuals
from ..media.video import create_rendered_check_strip


@lru_cache(maxsize=1)
def detect_video_encoder(fast: bool = False) -> str:
    settings = get_settings()
    if not fast:
        return "libx264"
    for enc in ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox"):
        probe_cmd = [
            settings.ffmpeg_bin, "-v", "error", "-nostdin",
            "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
            "-c:v", enc, "-f", "null", "-",
        ]
        if subprocess.run(probe_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return enc
    return "libx264"


def compute_punch_windows(spike_db: Optional[List[float]], start: float, end: float, max_punches: int = 3) -> List[Tuple[float, float]]:
    """Finds up to 3 loudest spike seconds (>= 5 dB jump, >= 6s apart) for dynamic punch-in zooms."""
    if not spike_db or end - start < 5.0:
        return []
    a = max(0, int(start))
    b = min(len(spike_db), int(end))
    if b <= a:
        return []
    seg = np.asarray(spike_db[a:b], dtype=np.float32)
    out: List[float] = []
    for idx in np.argsort(-seg):
        if seg[idx] < 5.0 or len(out) >= max_punches:
            break
        t_rel = (a + int(idx)) - start
        if t_rel < 1.2 or t_rel > (end - start) - 1.6:
            continue
        if all(abs(t_rel - prev) >= 6.0 for prev in out):
            out.append(t_rel)
    return [(round(max(0.0, t - 0.15), 2), round(t + 1.35, 2)) for t in sorted(out)]


def verify_export(path: Path, expected_w: int, expected_h: int, expected_duration: float,
                  expect_audio: bool, tol: float = 0.85) -> Dict[str, Any]:
    """Verifies file existence, exact dimensions, codecs, duration tolerance, and full FFmpeg decode."""
    settings = get_settings()
    checks: Dict[str, Any] = {}
    errors: List[str] = []

    if not path.is_file() or path.stat().st_size < 1000:
        return {"ok": False, "errors": ["Output file missing or empty"], "checks": {}}

    try:
        m = probe_media(path)
    except MediaError as e:
        return {"ok": False, "errors": [str(e)], "checks": {}}

    checks.update({
        "width": m["width"],
        "height": m["height"],
        "duration": round(m["duration"], 2),
        "video_codec": m["video_codec"],
        "audio_codec": m["audio_codec"],
        "size_bytes": path.stat().st_size,
    })

    if (m["width"], m["height"]) != (expected_w, expected_h):
        errors.append(f"Dimensions {m['width']}x{m['height']} != expected {expected_w}x{expected_h}")
    if m["video_codec"] != "h264":
        errors.append(f"Video codec '{m['video_codec']}' is not h264")
    if abs(m["duration"] - expected_duration) > tol:
        errors.append(f"Duration {m['duration']:.2f}s differs from expected {expected_duration:.2f}s")
    if expect_audio and m["audio_codec"] != "aac":
        errors.append(f"Audio track missing or codec '{m['audio_codec']}' is not aac")

    dec = subprocess.run(
        [settings.ffmpeg_bin, "-v", "error", "-nostdin", "-xerror", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=300,
    )
    checks["decodes"] = (dec.returncode == 0)
    if dec.returncode != 0:
        errors.append(f"Decode verification failed: {dec.stderr.strip()[:200]}")

    return {"ok": not errors, "errors": errors, "checks": checks}


def render_clip(
    clip: ClipCandidate,
    video_path: Path,
    output_dir: Path,
    genre_info: ContentClassification,
    intent: UserIntent,
    cam_box: Optional[List[float]] = None,
    face_center: Optional[Tuple[float, float]] = None,
    spike_db: Optional[List[float]] = None,
    creator_name: Optional[str] = None,
    source_offset_in_file: float = 0.0,
    progress_cb: Optional[Any] = None,
) -> RenderedClip:
    """Renders a verified clip candidate to finished MP4 with smart framing, Urdu/English ASS captions, and audio polish."""
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = output_dir / f"{clip.id}.mp4"
    ass_path = output_dir / f"{clip.id}.ass"
    sfx_wav = output_dir / f"{clip.id}_sfx.wav"

    meta = probe_media(video_path)
    src_w = meta["width"] or 1920
    src_h = meta["height"] or 1080
    has_audio = bool(meta["has_audio"])

    target_w, target_h = settings.dimensions_for(intent.aspect_ratio, src_w, src_h)
    preset_name = intent.preset or genre_info.recommended_preset or "clean"
    preset = get_preset(preset_name)

    # Local seek inside video_path (supports both full local file & trimmed section file)
    local_start = max(0.0, source_offset_in_file if source_offset_in_file > 0 or video_path.name.startswith("section_") else clip.start)
    if video_path.name.startswith("section_"):
        local_start = max(0.0, source_offset_in_file)
    else:
        local_start = max(0.0, clip.start)

    # Per-clip deep visual framing analysis
    clip_visuals = analyze_clip_visuals(video_path, local_start, clip.duration, src_w, src_h)
    eff_cam = clip_visuals.get("cam_box") or cam_box
    eff_center = clip_visuals.get("face_center") or face_center
    speaker_track = clip_visuals.get("speaker_track_norm")

    # Select layout mode respecting preset, aspect ratio, and visual evidence
    if intent.aspect_ratio != "9:16":
        layout_mode = intent.aspect_ratio
    else:
        pref = preset.get("preferred_layout", "fit_blur")
        if pref == "facecam_stacked":
            layout_mode = clip_visuals["mode"] if eff_cam else preset.get("fallback_layout", "fit_blur")
        elif pref == "subject_crop":
            if clip_visuals["mode"] in ("facecam_stacked", "facecam_overlay"):
                layout_mode = clip_visuals["mode"]
            elif speaker_track or eff_center:
                layout_mode = "subject_crop"
            else:
                layout_mode = preset.get("fallback_layout", "fit_blur")
        elif pref == "hook_card_5_4":
            layout_mode = "hook_card_5_4"
        else:
            layout_mode = pref

    # Punch-in windows for dynamic/gaming presets
    punches = compute_punch_windows(spike_db, clip.start, clip.end) if preset.get("punch_ins") else []

    # Prepare word tokens (preserving real Whisper word timings!)
    words = list(clip.words) if clip.words else []
    if not words and clip.transcript_text.strip():
        toks = clip.transcript_text.split()
        step = max(0.05, clip.duration / len(toks))
        words = [
            WordToken(w=t, s=round(clip.start + i * step, 2), e=round(clip.start + (i + 1) * step, 2), p=0.8)
            for i, t in enumerate(toks)
        ]

    # Hook card placement & caption safe margin adjustments per layout
    hook_card_text = None
    hook_y = None
    margin_v_override = None
    if preset.get("hook_card"):
        hook_card_text = clip.hook_text or clip.title
    if layout_mode == "hook_card_5_4":
        hook_y = 210
        margin_v_override = 330
    elif layout_mode == "facecam_stacked":
        margin_v_override = 360

    if intent.caption_lang != "none" and words:
        # Derive a crisp per-clip topic tag matching what THIS clip actually discusses
        topic_badge = genre_info.genre.replace("_", " ")
        if intent.topic:
            low_tx = clip.transcript_text.lower()
            from ..understand.urdu import transliterate_urdu_to_roman
            probe_tx = f"{low_tx} {transliterate_urdu_to_roman(low_tx).lower()}"
            t_words = [w for w in intent.topic.split() if len(w) >= 3]
            matched_t = [w for w in t_words if w.lower() in probe_tx]
            chosen_words = (t_words[:2] + [w for w in matched_t if w not in t_words[:2]])[:3]
            topic_badge = " ".join(chosen_words) if chosen_words else " ".join(t_words[:3])
        ass_content = generate_ass_subtitles(
            words=words,
            clip_start=clip.start,
            clip_end=clip.end,
            preset_name=preset_name,
            hook_card_text=hook_card_text,
            width=target_w,
            height=target_h,
            caption_lang=intent.caption_lang,
            hook_y=hook_y,
            margin_v_override=margin_v_override,
            creator_name=creator_name,
            topic_label=topic_badge,
            motion_graphics=(preset_name != "cinematic"),
        )
        ass_path.write_text(ass_content, encoding="utf-8")
        use_ass_path: Optional[str] = str(ass_path)
    else:
        use_ass_path = None

    p_color = {
        "educational": "0xF59E0B",
        "social_media": "0xFF3B3B",
        "gaming": "0x1E90FF",
        "podcast": "0x10B981",
        "islamic": "0x10B981",
        "cinematic": "0xE2E8F0",
        "dynamic": "0xFFD400",
        "clean": "0xE2E8F0",
    }.get(preset_name, "0x0BB9F5")

    filter_complex, v_label = build_ffmpeg_filtergraph(
        layout_mode=layout_mode,
        src_width=src_w,
        src_height=src_h,
        target_width=target_w,
        target_height=target_h,
        cam_box=eff_cam,
        face_center=eff_center,
        speaker_track_norm=speaker_track,
        clip_duration=clip.duration,
        ass_path=use_ass_path,
        fonts_dir=settings.caption_fonts_dir,
        punch_windows=punches,
        fade=bool(preset.get("fade")),
        aspect_ratio=intent.aspect_ratio,
        progress_bar=(preset_name not in ("clean", "cinematic")),
        progress_color=p_color,
    )

    # Optional SFX track
    sfx_path = None
    if intent.sfx and has_audio:
        sfx_path = build_sfx_track(sfx_wav, clip.duration, punches, include_hook_whoosh=bool(hook_card_text or punches))

    if progress_cb:
        progress_cb(0.3, f"Encoding {clip.id} ({clip.duration:.1f}s, {preset_name}, {layout_mode})...")

    inputs = ["-ss", f"{local_start:.3f}", "-t", f"{clip.duration:.3f}", "-i", str(video_path)]
    audio_chain = ""
    if has_audio:
        audio_chain = ";[0:a]aresample=48000"
        if preset.get("fade") and clip.duration > 2.0:
            fade_out_st = max(0.4, clip.duration - 0.5)
            audio_chain += f",afade=t=in:st=0:d=0.3,afade=t=out:st={fade_out_st:.2f}:d=0.5"
        if sfx_path:
            inputs += ["-i", str(sfx_path)]
            audio_chain += "[a0];[1:a]aresample=48000[a1];[a0][a1]amix=inputs=2:duration=first:normalize=0"
        if intent.normalize_audio:
            audio_chain += ",loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
        audio_chain += "[aout]"

    full_graph = filter_complex + audio_chain
    enc = detect_video_encoder(intent.fast_encode)
    enc_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p"] if enc == "libx264" else [
        "-c:v", enc, "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M", "-pix_fmt", "yuv420p"
    ]

    cmd = [
        settings.ffmpeg_bin, "-y", "-v", "error", "-nostdin",
        *inputs,
        "-filter_complex", full_graph,
        "-map", v_label,
    ]
    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += [*enc_args, "-movflags", "+faststart", str(out_mp4)]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1800)
    if res.returncode != 0 and use_ass_path:
        # Clean fallback without subtitle filter if filter failed
        fg_no_sub, v_lbl_fb = build_ffmpeg_filtergraph(
            layout_mode=layout_mode, src_width=src_w, src_height=src_h,
            target_width=target_w, target_height=target_h, cam_box=eff_cam,
            face_center=eff_center, speaker_track_norm=speaker_track,
            clip_duration=clip.duration, ass_path=None, punch_windows=punches,
            fade=bool(preset.get("fade")), aspect_ratio=intent.aspect_ratio,
        )
        cmd_fb = [
            settings.ffmpeg_bin, "-y", "-v", "error", "-nostdin",
            "-ss", f"{local_start:.3f}", "-t", f"{clip.duration:.3f}", "-i", str(video_path),
            "-filter_complex", fg_no_sub + audio_chain, "-map", v_lbl_fb,
        ]
        if has_audio:
            cmd_fb += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        cmd_fb += ["-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)]
        res_fb = subprocess.run(cmd_fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1800)
        if res_fb.returncode != 0:
            raise RuntimeError(f"FFmpeg render failed for {clip.id}: {res.stderr.strip()} | Fallback: {res_fb.stderr.strip()}")
    elif res.returncode != 0:
        raise RuntimeError(f"FFmpeg render failed for {clip.id}: {res.stderr.strip()}")

    # Clean temporary ASS and SFX files
    ass_path.unlink(missing_ok=True)
    sfx_wav.unlink(missing_ok=True)

    # Strict Export Verification (never claim exported unless verified!)
    ver = verify_export(out_mp4, target_w, target_h, clip.duration, expect_audio=has_audio)
    if not ver["ok"]:
        raise RuntimeError(f"Export verification failed for {clip.id}: {'; '.join(ver['errors'])}")

    # Generate final post-render check strip (showing actual hook, captions & layout)
    check_strip_path = output_dir / f"check_{clip.id}.jpg"
    create_rendered_check_strip(out_mp4, clip.duration, check_strip_path)

    publishing = generate_publishing_package(clip, genre_info, creator_name=creator_name, caption_lang=intent.caption_lang)

    if progress_cb:
        progress_cb(1.0, f"Verified export ready: {out_mp4.name}")

    return RenderedClip(
        clip_id=clip.id,
        output_path=str(out_mp4),
        check_strip_path=str(check_strip_path) if check_strip_path.is_file() else None,
        story_sheet_path=clip.story_sheet_path,
        duration=clip.duration,
        preset=preset_name,
        aspect_ratio=intent.aspect_ratio,
        layout_mode=layout_mode,
        dimensions=clip.dimensions,
        publishing=publishing,
        verification=ver,
        render_meta={
            "source_file": str(video_path),
            "local_start": local_start,
            "clip": clip.model_dump(),
            "genre_info": genre_info.model_dump(),
            "intent": intent.model_dump(),
            "cam_box": eff_cam,
            "face_center": eff_center,
            "creator_name": creator_name,
        },
    )
