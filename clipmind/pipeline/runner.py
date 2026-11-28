"""Universal ClipMind Pipeline Runner & Remake Engine.
Audio-first remote ingestion, chunked transcription caching, sequential 8 GB RAM execution,
strict export verification, and instant `remake` re-rendering.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Dict, Any, List, Optional

import numpy as np

from ..config import get_settings
from ..schemas import (
    UserIntent,
    SourceInfo,
    Transcript,
    ContentClassification,
    ClipCandidate,
    RenderedClip,
)
from ..ingest import get_adapter, probe_source, remember_clipped, url_cache_dir
from ..ingest.ytdlp import YtDlpAdapter
from ..media.probe import probe_media
from ..media.audio import (
    extract_audio_chunks,
    calculate_loudness_profile,
    compute_db_spikes,
    sync_clip_to_audio,
)
from ..media.video import detect_scene_cuts, extract_sampled_frames, create_composite_check_strip
from ..media.face import FaceTracker
from ..understand.asr import SpeechTranscriber
from ..understand.classifier import classify_content
from ..understand.urdu import build_domain_asr_prompt
from ..understand.brains import resolve_brain_mode
from ..moments.discoverer import discover_moments
from ..production.render import render_clip
from .state import (
    CancellationToken,
    ProgressCallback,
    current_rss_mb,
    parse_section_bounds,
    save_checkpoint,
)


class PipelineError(Exception):
    pass


def _select_hot_chunks(chunks: List[Dict[str, Any]], spike_db: List[float],
                       chat_speed: Optional[List[int]], screen_activity: Optional[Dict[str, Any]],
                       max_chunks: int) -> List[Dict[str, Any]]:
    """On multi-hour streams, selects the hottest chunks combining dB spikes, chat bursts, and visual activity."""
    if len(chunks) <= max_chunks:
        return chunks

    chat_base = float(np.median([c for c in chat_speed if c > 0])) if chat_speed and any(chat_speed) else 0.0
    raw_act = screen_activity.get("smooth") if screen_activity else None

    scored = []
    for c in chunks:
        s0, s1 = int(c["offset"]), int(c["offset"] + c["duration"])
        seg = np.sort(np.asarray(spike_db[s0:min(len(spike_db), s1)], dtype=np.float32))[-30:]
        score = float(seg.mean()) if len(seg) else 0.0

        if chat_base > 0 and chat_speed:
            b0, b1 = int(s0 // 10), int(s1 // 10) + 1
            chat_seg = chat_speed[b0:min(len(chat_speed), b1)]
            if chat_seg:
                score += min(10.0, (max(chat_seg) / chat_base) * 1.5)

        if raw_act is not None and len(raw_act) > 0:
            act_seg = raw_act[s0:min(len(raw_act), s1)]
            if len(act_seg) and float(np.median(act_seg)) < 1.5:
                score *= 0.20

        scored.append((score, c))

    top = [c for _, c in sorted(scored, key=lambda x: -x[0])[:max_chunks]]
    return sorted(top, key=lambda c: c["offset"])


def _write_posting_and_manifest(
    out_dir: Path,
    job_id: str,
    source_input: str,
    source_info: SourceInfo,
    duration: float,
    genre_info: ContentClassification,
    intent: UserIntent,
    rendered_clips: List[RenderedClip],
    rejected_clips: List[Dict[str, Any]],
    brain_mode: str,
    brain_note: str,
    language: str,
    peak_rss_mb: float,
    elapsed_sec: float,
) -> Optional[Path]:
    strip_paths = [Path(rc.check_strip_path) for rc in rendered_clips if rc.check_strip_path]
    composite_strip = create_composite_check_strip(strip_paths, out_dir / "check_all.jpg") if strip_paths else None

    meta_dir = out_dir / ".clip_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    slim_clips = []
    for rc in rendered_clips:
        meta_rel = f".clip_meta/{rc.clip_id}.json"
        if rc.render_meta:
            (meta_dir / f"{rc.clip_id}.json").write_text(
                json.dumps(rc.render_meta, ensure_ascii=False), encoding="utf-8"
            )
        entry = rc.model_dump(exclude={"render_meta"})
        entry["render_meta_file"] = meta_rel
        slim_clips.append(entry)

    manifest = {
        "job_id": job_id,
        "source": source_input,
        "source_title": source_info.title,
        "channel_name": source_info.channel_name,
        "duration_sec": duration,
        "language": language,
        "genre": genre_info.genre,
        "genre_confidence": genre_info.confidence,
        "recommended_preset": genre_info.recommended_preset,
        "brain_mode": brain_mode,
        "brain_note": brain_note,
        "intent": intent.model_dump(),
        "composite_check_strip": str(composite_strip) if composite_strip else None,
        "rendered_clips": slim_clips,
        "rejected_candidates": rejected_clips,
        "peak_rss_mb": peak_rss_mb,
        "processing_time_sec": round(elapsed_sec, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Publishing Pack — {source_info.title}",
        f"**Category:** {genre_info.genre.replace('_', ' ').title()} | **Language:** `{language}` | **Clips Generated:** {len(rendered_clips)}",
        f"**Analysis Mode:** `{brain_mode}` ({brain_note}) | **Processing Time:** {elapsed_sec:.1f}s | **Peak RAM:** {peak_rss_mb:.0f} MB",
        f"**Visual Check Strip:** `{composite_strip.name}`" if composite_strip else "",
        "",
        "---",
    ]
    for idx, rc in enumerate(rendered_clips, 1):
        pub = rc.publishing
        lines.extend([
            f"## Clip {idx}: {pub.title}",
            f"- **File:** `{Path(rc.output_path).name}` ({rc.duration:.1f}s | {rc.aspect_ratio} | layout: `{rc.layout_mode}`)",
            f"- **Preset:** {rc.preset.title()} | **Quality Score:** {pub.quality_score:.1f}/10",
            f"- **Verified:** {rc.verification.get('ok', False)} ({rc.verification.get('checks', {}).get('width')}x{rc.verification.get('checks', {}).get('height')})",
            f"- **Selection Rationale:** {pub.rationale}",
            "",
            "### Suggested Caption:",
            "```text",
            pub.caption,
            "```",
            "",
            f"### Hashtags: {' '.join(pub.hashtags)}",
            "",
            "### A/B Hook Variations:",
            *[f"{i}. {h}" for i, h in enumerate(pub.ab_hooks, 1)],
            "",
            "---",
        ])

    if rejected_clips:
        lines.append("## Filtered Candidates (Honest Quality Gate)")
        for rc in rejected_clips:
            lines.append(f"- `{rc['id']}` ({rc['start']:.1f}s-{rc['end']:.1f}s, score {rc['score']}): {rc['reason']}")

    (out_dir / "POSTING.md").write_text("\n".join(lines), encoding="utf-8")
    return composite_strip


class ClipMindPipeline:
    def __init__(self, work_root: Optional[Path] = None):
        self.work_root = work_root or Path.cwd() / "clipmind_workspace"

    def run(
        self,
        source_input: str,
        user_intent: Optional[UserIntent] = None,
        output_dir: Optional[Path] = None,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_token: Optional[CancellationToken] = None,
        keep_temp: bool = False,
    ) -> Dict[str, Any]:
        """Executes the universal clipping pipeline end-to-end."""
        start_time = time.time()
        settings = get_settings()
        intent = user_intent or UserIntent()
        cancel = cancel_token or CancellationToken()
        peak_rss = current_rss_mb()

        def report(stage: str, pct: float, msg: str):
            nonlocal peak_rss
            peak_rss = max(peak_rss, current_rss_mb())
            if progress_cb:
                progress_cb(stage, pct, msg)

        # 1. Ingestion & Validation
        report("probe", 0.04, f"Inspecting source: {source_input}...")
        source_info = probe_source(source_input)
        if not source_info.is_supported:
            raise PipelineError(f"Source cannot be processed: {source_info.error_message}")

        src_hash = hashlib.sha1(source_input.encode("utf-8")).hexdigest()[:8]
        safe_title = "".join(c for c in source_info.title if c.isalnum() or c in (" ", "_", "-")).strip()[:40] or "clip_job"
        job_id = f"{safe_title}_{src_hash}"

        job_dir = self.work_root / job_id
        temp_dir = job_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_dir = output_dir or (job_dir / "clips")
        out_dir.mkdir(parents=True, exist_ok=True)
        state_file = job_dir / "job_state.json"

        save_checkpoint(state_file, "probe", source_info.model_dump())
        if cancel.is_cancelled:
            return {"status": "cancelled", "job_id": job_id}

        # 2. Acquire Media (Audio-First for platform URLs; zero-copy for local files)
        adapter = get_adapter(source_input)
        if not adapter:
            raise PipelineError("No compatible source adapter found.")

        is_remote_platform = isinstance(adapter, YtDlpAdapter)
        cache_dir = url_cache_dir(source_input) if is_remote_platform else temp_dir

        if is_remote_platform:
            report("download", 0.10, "Downloading audio track & platform signals (video sections fetched later)...")
            audio_source_path = adapter.fetch_audio_only(
                source_input,
                progress_cb=lambda p, m: report("download", 0.10 + p * 0.10, m),
            )
            video_source_path: Optional[Path] = None
            board = adapter.fetch_storyboard(source_input)
            chat_speed = adapter.fetch_live_chat_speed(source_input)
        else:
            report("download", 0.10, "Preparing local/direct media source...")
            media_path = adapter.fetch_media(
                source_input,
                dest_dir=temp_dir,
                audio_only=False,
                progress_cb=lambda p, m: report("download", 0.10 + p * 0.10, m),
            )
            audio_source_path = media_path
            video_source_path = media_path
            board = None
            chat_speed = None

        media_meta = probe_media(audio_source_path)
        duration = media_meta["duration"] or source_info.duration
        if not media_meta.get("has_audio"):
            report("complete", 1.0, "Media contains no audio track.")
            return {
                "status": "completed",
                "job_id": job_id,
                "clips_count": 0,
                "message": "No audio track found in this media; speech-driven discovery requires audio.",
                "rendered_clips": [],
            }

        lo, hi = parse_section_bounds(intent.section, duration)
        screen_activity = YtDlpAdapter.compute_screen_activity(board, duration) if board else None
        save_checkpoint(state_file, "ingest", {"duration": duration, "section": [lo, hi]})
        if cancel.is_cancelled:
            return {"status": "cancelled", "job_id": job_id}

        # 3. Audio Chunking & Loudness Profile
        report("audio", 0.25, "Chunking audio (5-min segments) & streaming loudness profile...")
        chunks = extract_audio_chunks(
            audio_source_path,
            cache_dir / "audio_chunks",
            chunk_sec=settings.chunk_duration_sec,
            start_at=lo,
            end_at=hi,
        )
        energy_profile = calculate_loudness_profile(audio_source_path)
        spike_db = compute_db_spikes(energy_profile)
        chunks_to_transcribe = _select_hot_chunks(
            chunks, spike_db, chat_speed, screen_activity, settings.max_auto_blocks_long_video
        )
        save_checkpoint(state_file, "audio", {"total_chunks": len(chunks), "transcribed_chunks": len(chunks_to_transcribe)})

        if cancel.is_cancelled:
            return {"status": "cancelled", "job_id": job_id}

        # 4. Sequential Chunked Speech Transcription (with per-chunk JSON cache)
        report("transcribe", 0.35, f"Transcribing {len(chunks_to_transcribe)} chunk(s) with faster-whisper ({intent.whisper_model})...")
        transcriber = SpeechTranscriber(model_size_or_path=intent.whisper_model)
        eff_lang = intent.language or (source_info.language if source_info.language and len(source_info.language) <= 3 else None)
        asr_prompt = build_domain_asr_prompt(
            title=source_info.title,
            topic=intent.topic,
            channel_name=source_info.channel_name,
            language=eff_lang,
        )
        transcript = transcriber.transcribe_chunks(
            chunks_to_transcribe,
            language=eff_lang,
            cache_dir=cache_dir / "asr_cache",
            progress_cb=lambda p, m: report("transcribe", 0.35 + p * 0.20, m),
            initial_prompt=asr_prompt,
        )
        save_checkpoint(state_file, "transcribe", {"language": transcript.language, "segments": len(transcript.segments)})

        if not transcript.segments:
            report("complete", 1.0, "No speech detected in video.")
            return {
                "status": "completed",
                "job_id": job_id,
                "clips_count": 0,
                "message": "No spoken words were found in this media.",
                "rendered_clips": [],
            }

        if cancel.is_cancelled:
            return {"status": "cancelled", "job_id": job_id}

        # 5. Scene Cuts, Sampled Frames & Face Presence
        if video_source_path and media_meta.get("has_video", True):
            report("analyze", 0.58, "Detecting scene cuts and sampling visual frames...")
            scene_cuts = detect_scene_cuts(video_source_path)
            sampled_frames = extract_sampled_frames(video_source_path, temp_dir / "frames", step_sec=4.0)
            report("faces", 0.66, "Scanning sampled frames for faces and speaker framing...")
            face_tracker = FaceTracker()
            face_info = face_tracker.analyze_frame_faces([f["path"] for f in sampled_frames[:25]])
        else:
            scene_cuts = []
            sampled_frames = []
            face_info = {"has_faces": False, "is_facecam": False, "cam_box": None, "center": None}

        # 6. Content Genre Classification
        report("classify", 0.70, "Classifying content category and selecting optimal preset...")
        genre_info = classify_content(
            transcript=transcript,
            cuts=scene_cuts,
            duration=duration,
            has_facecam=bool(face_info.get("is_facecam")),
        )
        save_checkpoint(state_file, "classify", genre_info.model_dump())

        if cancel.is_cancelled:
            return {"status": "cancelled", "job_id": job_id}

        # 7. Intelligent Moment Discovery & Story Verification
        brain_mode, brain_note = resolve_brain_mode(intent.brain)
        report("discover", 0.75, f"Discovering moments & verifying story completeness ({brain_mode})...")
        rejected_clips: List[Dict[str, Any]] = []
        candidate_clips = discover_moments(
            transcript=transcript,
            energy_profile=energy_profile,
            scene_cuts=scene_cuts,
            frames=sampled_frames,
            has_faces=bool(face_info.get("has_faces")),
            intent=intent,
            video_path=video_source_path,
            output_dir=out_dir,
            chat_speed=chat_speed,
            screen_activity=screen_activity,
            work_dir=temp_dir,
            rejected_out=rejected_clips,
        )

        if not candidate_clips:
            report("complete", 1.0, "No moments passed the context and quality verification gate.")
            return {
                "status": "completed",
                "job_id": job_id,
                "clips_count": 0,
                "genre": genre_info.genre,
                "rejected_candidates": rejected_clips,
                "message": "No moments met the quality and story completeness threshold.",
                "rendered_clips": [],
            }

        # 8. Targeted Video Section Download (for remote URLs) + Smart Render & Verification
        report("render", 0.82, f"Producing {len(candidate_clips)} verified clip(s)...")
        rendered_clips: List[RenderedClip] = []
        sections_cache = out_dir / ".sections"
        sections_cache.mkdir(parents=True, exist_ok=True)

        for idx, clip in enumerate(candidate_clips):
            if cancel.is_cancelled:
                break
            step_pct = 0.82 + (idx / len(candidate_clips)) * 0.15

            clip_src_path = video_source_path
            local_offset = clip.start

            if is_remote_platform:
                lead = 7.0
                tail = 6.0
                dl_start = max(0.0, clip.start - lead)
                dl_end = min(duration, clip.end + tail)
                sec_file = sections_cache / f"section_{clip.id}_{int(dl_start)}_{int(dl_end)}.mp4"
                report("render", step_pct, f"Downloading video slice for {clip.id} ({dl_start:.0f}s-{dl_end:.0f}s)...")
                clip_src_path = adapter.fetch_video_section(source_input, dl_start, dl_end, sec_file)
                synced_at, sync_conf = sync_clip_to_audio(clip_src_path, audio_source_path, dl_start)
                local_offset = max(0.0, clip.start - synced_at) if sync_conf >= 0.45 else max(0.0, clip.start - dl_start)

            report("render", step_pct + 0.02, f"Rendering & verifying {clip.id}: '{clip.title}'...")
            rendered = render_clip(
                clip=clip,
                video_path=clip_src_path,
                output_dir=out_dir,
                genre_info=genre_info,
                intent=intent,
                cam_box=face_info.get("cam_box"),
                face_center=face_info.get("center"),
                spike_db=spike_db,
                creator_name=source_info.channel_name,
                source_offset_in_file=local_offset,
            )
            rendered_clips.append(rendered)

        # 9. Write manifest.json & POSTING.md
        elapsed = time.time() - start_time
        peak_rss = max(peak_rss, current_rss_mb())
        composite_strip = _write_posting_and_manifest(
            out_dir=out_dir,
            job_id=job_id,
            source_input=source_input,
            source_info=source_info,
            duration=duration,
            genre_info=genre_info,
            intent=intent,
            rendered_clips=rendered_clips,
            rejected_clips=rejected_clips,
            brain_mode=brain_mode,
            brain_note=brain_note,
            language=transcript.language,
            peak_rss_mb=peak_rss,
            elapsed_sec=elapsed,
        )

        if is_remote_platform and rendered_clips:
            remember_clipped(source_input, source_info.title)

        if not keep_temp:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        report("complete", 1.0, f"Completed! Generated {len(rendered_clips)} verified clip(s) in {out_dir}")
        return {
            "status": "completed",
            "job_id": job_id,
            "output_dir": str(out_dir),
            "clips_count": len(rendered_clips),
            "language": transcript.language,
            "genre": genre_info.genre,
            "brain_mode": brain_mode,
            "rendered_clips": rendered_clips,
            "rejected_candidates": rejected_clips,
            "composite_check_strip": str(composite_strip) if composite_strip else None,
            "manifest_file": str(out_dir / "manifest.json"),
            "posting_pack": str(out_dir / "POSTING.md"),
            "peak_rss_mb": peak_rss,
            "processing_time_sec": round(elapsed, 2),
        }


def remake_clips(
    output_dir: Path,
    clip_indices: Optional[List[int]] = None,
    preset: Optional[str] = None,
    hook: Optional[str] = None,
    title: Optional[str] = None,
    caption_lang: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    sfx: Optional[bool] = None,
) -> Dict[str, Any]:
    """Re-renders selected clips from an existing output folder without re-downloading or re-transcribing."""
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise PipelineError(f"manifest.json not found in {output_dir}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rendered_list = data.get("rendered_clips") or []
    if not rendered_list:
        raise PipelineError("No clips recorded in manifest.json to remake.")

    targets = set(clip_indices) if clip_indices else set(range(1, len(rendered_list) + 1))
    updated_rendered: List[RenderedClip] = []

    for idx, rc_dict in enumerate(rendered_list, 1):
        rc = RenderedClip.model_validate(rc_dict)
        rm = dict(rc.render_meta or {})
        if not rm and rc_dict.get("render_meta_file"):
            meta_p = Path(output_dir) / rc_dict["render_meta_file"]
            if meta_p.is_file():
                rm = json.loads(meta_p.read_text(encoding="utf-8"))
        rc.render_meta = rm
        if idx not in targets:
            updated_rendered.append(rc)
            continue

        src_file = Path(rm.get("source_file", ""))
        if not src_file.is_file():
            raise PipelineError(f"Cached source video missing for clip {idx}: {src_file}")

        clip = ClipCandidate.model_validate(rm["clip"])
        genre_info = ContentClassification.model_validate(rm["genre_info"])
        intent = UserIntent.model_validate(rm["intent"])

        if preset:
            intent.preset = preset
        if aspect_ratio:
            intent.aspect_ratio = aspect_ratio
        if caption_lang:
            intent.caption_lang = caption_lang
        if sfx is not None:
            intent.sfx = sfx
        if hook:
            clip.hook_text = hook
        if title:
            clip.title = title

        new_rc = render_clip(
            clip=clip,
            video_path=src_file,
            output_dir=Path(output_dir),
            genre_info=genre_info,
            intent=intent,
            cam_box=rm.get("cam_box"),
            face_center=tuple(rm["face_center"]) if rm.get("face_center") else None,
            creator_name=rm.get("creator_name"),
            source_offset_in_file=float(rm.get("local_start", clip.start)),
        )
        updated_rendered.append(new_rc)

    source_info = SourceInfo(
        url_or_path=data.get("source", ""),
        source_type="remake",
        platform="local",
        is_supported=True,
        title=data.get("source_title", "Video"),
        channel_name=data.get("channel_name"),
    )
    comp = _write_posting_and_manifest(
        out_dir=Path(output_dir),
        job_id=data.get("job_id", "remake"),
        source_input=data.get("source", ""),
        source_info=source_info,
        duration=float(data.get("duration_sec", 0.0)),
        genre_info=ContentClassification.model_validate(updated_rendered[0].render_meta["genre_info"]),
        intent=UserIntent.model_validate(updated_rendered[0].render_meta["intent"]),
        rendered_clips=updated_rendered,
        rejected_clips=data.get("rejected_candidates", []),
        brain_mode=data.get("brain_mode", "rules"),
        brain_note=data.get("brain_note", "Remade from cached render metadata"),
        language=data.get("language", "en"),
        peak_rss_mb=current_rss_mb(),
        elapsed_sec=0.0,
    )
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "remade_count": len(targets),
        "composite_check_strip": str(comp) if comp else None,
    }
