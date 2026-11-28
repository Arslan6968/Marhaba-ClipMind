"""Universal Moment Discovery & Quality Gate Coordinator.
Combines acoustic energy spikes, semantic topic/speaker signals, chat bursts, and Q&A anchors,
evaluates variable-length spans across 9 dimensions, deduplicates, and verifies context & story completeness.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..schemas import Transcript, UserIntent, ClipCandidate
from .units import build_thought_units, annotate_units
from .scoring import evaluate_span
from .dedupe import deduplicate_candidates
from .verify import verify_and_adjust_clip
from ..understand.brains import (
    LocalLlamaBrain,
    NeedAgentInput,
    extract_json_object,
    is_grounded,
    resolve_brain_mode,
    write_agent_story_prompts,
)
from ..config import get_settings


def _load_agent_answers(pairs: List[Tuple[Path, Path]]) -> Optional[Dict[str, Dict[str, Any]]]:
    if pairs:
        batch_file = pairs[0][1].parent / "STORY_ANSWERS.json"
        if batch_file.is_file():
            try:
                raw = batch_file.read_text(encoding="utf-8")
                obj = extract_json_object(raw) or json.loads(raw)
                if isinstance(obj, dict) and obj:
                    return obj
            except Exception:
                pass

    answers = {}
    for _, a_file in pairs:
        if not a_file.is_file():
            return None
        try:
            raw = a_file.read_text(encoding="utf-8")
            obj = extract_json_object(raw) or json.loads(raw)
            cid = a_file.name.split(".")[0]
            answers[cid] = obj
        except Exception:
            return None
    return answers


def discover_moments(
    transcript: Transcript,
    energy_profile: List[float],
    scene_cuts: List[float],
    frames: List[Dict[str, Any]],
    has_faces: bool,
    intent: UserIntent,
    video_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    chat_speed: Optional[List[int]] = None,
    screen_activity: Optional[Dict[str, Any]] = None,
    work_dir: Optional[Path] = None,
    rejected_out: Optional[List[Dict[str, Any]]] = None,
    genre: str = "general",
) -> List[ClipCandidate]:
    """Discovers, scores, deduplicates, and verifies complete narrative clips."""
    settings = get_settings()
    units = build_thought_units(transcript)
    if len(units) < 2:
        return []

    annotate_units(
        units,
        energy_profile=energy_profile,
        scene_cuts=scene_cuts,
        intent=intent,
        chat_speed=chat_speed,
        screen_activity=screen_activity,
    )

    interests = sorted(u["interest"] for u in units)
    threshold = max(0.24, interests[int(len(interests) * 0.62)])

    anchors = []
    for i, u in enumerate(units):
        is_topic_hit = bool(intent.topic and u.get("topic_rel", 0.0) >= 0.55)
        if u["interest"] >= threshold or is_topic_hit:
            window = [units[j]["interest"] for j in range(max(0, i - 2), min(len(units), i + 3))]
            if u["interest"] >= max(window) or is_topic_hit:
                anchors.append(i)

    anchors = sorted(set(anchors), key=lambda a: -units[a]["interest"])[:45]
    if not anchors:
        # Fallback: pick top 3 peaks so non-empty speech still evaluates honestly
        anchors = sorted(range(len(units)), key=lambda a: -units[a]["interest"])[:min(5, len(units))]

    total_speech_dur = units[-1]["end"] - units[0]["start"]
    min_dur = min(settings.min_clip_sec, max(6.0, total_speech_dur * 0.70))
    max_dur = settings.max_clip_sec
    if intent.target_duration == "short":
        min_dur = min(10.0, min_dur)
        max_dur = min(35.0, max_dur)
    elif intent.target_duration == "long":
        min_dur = max(40.0, min_dur)

    raw_candidates = []
    for a in anchors:
        best_span = None
        for back in range(0, 7):
            for fwd in range(0, 6):
                s = a - back
                e = a + fwd
                if s < 0 or e >= len(units):
                    continue
                if any(units[j + 1]["start"] - units[j]["end"] > 4.0 for j in range(s, e)):
                    continue
                span_dur = units[e]["end"] - units[s]["start"]
                if span_dur < min_dur or span_dur > max_dur:
                    continue

                res = evaluate_span(units, s, e, a, frames, scene_cuts, has_faces, intent, genre=genre)
                if best_span is None or res["overall_score"] > best_span["overall_score"]:
                    res["anchor_idx"] = a
                    best_span = res

        if best_span:
            raw_candidates.append(best_span)

    if not raw_candidates:
        return []

    deduped = deduplicate_candidates(raw_candidates, max_overlap=0.12, max_text_sim=0.52)
    # Timeline spread: when >= 3 clips requested on videos > 4 min, promote chapter diversity
    if intent.max_clips >= 3 and total_speech_dur > 240.0 and len(deduped) > intent.max_clips:
        for c in deduped:
            c["_bucket"] = int((c["start"] / max(1.0, total_speech_dur)) * 3)
        diverse_pool = []
        seen_buckets = set()
        for c in deduped:
            if c["_bucket"] not in seen_buckets:
                diverse_pool.append(c)
                seen_buckets.add(c["_bucket"])
        for c in deduped:
            if c not in diverse_pool:
                diverse_pool.append(c)
        deduped = diverse_pool
    pool = deduped[:max(intent.max_clips * 2, 8)]

    brain_mode, _ = resolve_brain_mode(intent.brain)
    llm_brain = LocalLlamaBrain() if brain_mode == "local" else None
    is_agent = (brain_mode == "agent" and work_dir is not None)

    verified_clips: List[ClipCandidate] = []
    try:
        for idx, cand in enumerate(pool):
            cand["id"] = f"clip_{idx + 1}"
            clip = verify_and_adjust_clip(
                cand,
                units,
                video_path=video_path,
                intent=intent,
                output_dir=work_dir or output_dir,
                llm_brain=llm_brain,
                generate_story_sheet=is_agent,
            )

            if not clip.verified or clip.overall_score < 4.6:
                reason = clip.rejection_reason or f"Quality score ({clip.overall_score:.1f}/10) below quality gate (4.6)."
                if rejected_out is not None:
                    rejected_out.append({
                        "id": clip.id,
                        "start": clip.start,
                        "end": clip.end,
                        "score": clip.overall_score,
                        "reason": reason,
                    })
                continue

            verified_clips.append(clip)
            if not is_agent and len(verified_clips) >= intent.max_clips:
                break
    finally:
        if llm_brain is not None:
            llm_brain.unload()

    # In --brain agent mode, write STORY_TASK.md + prompt files and wait for host-agent review if unanswered
    if is_agent and verified_clips and work_dir is not None:
        top_for_agent = verified_clips[:intent.max_clips + 2]
        cand_dicts = [
            {
                "id": c.id,
                "start": c.start,
                "end": c.end,
                "transcript_text": c.transcript_text,
                "story_sheet_path": c.story_sheet_path,
            }
            for c in top_for_agent
        ]
        task_file, pairs = write_agent_story_prompts(work_dir, cand_dicts, intent.objective, intent.topic)
        answers = _load_agent_answers(pairs)
        if answers is None:
            raise NeedAgentInput(task_file, pairs)

        refined: List[ClipCandidate] = []
        for c in top_for_agent:
            ans = answers.get(c.id)
            if not ans:
                refined.append(c)
                continue
            verdict = str(ans.get("verdict") or "keep").lower()
            if verdict == "drop":
                if rejected_out is not None:
                    rejected_out.append({
                        "id": c.id,
                        "start": c.start,
                        "end": c.end,
                        "score": c.overall_score,
                        "reason": f"Agent story check dropped clip: {ans.get('reason', 'Not suitable')}",
                    })
                continue
            try:
                ns = float(ans.get("start", c.start))
                ne = float(ans.get("end", c.end))
                if ne - ns >= 6.0 and ne - ns <= 95.0:
                    c.start, c.end = round(ns, 2), round(ne, 2)
                    c.duration = round(c.end - c.start, 2)
                    c.boundary_adjusted = True
            except (TypeError, ValueError):
                pass
            if ans.get("title") and is_grounded(str(ans["title"]), c.transcript_text, 0.3):
                c.title = str(ans["title"])[:70]
            if ans.get("hook") and is_grounded(str(ans["hook"]), c.transcript_text, 0.25):
                c.hook_text = str(ans["hook"])[:140]
            if ans.get("reason"):
                c.rationale = str(ans["reason"])
            c.picked_by = "agent"
            refined.append(c)
            if len(refined) >= intent.max_clips:
                break
        return refined

    return verified_clips[:intent.max_clips]
