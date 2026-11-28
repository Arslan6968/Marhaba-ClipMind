"""Dedicated Story & Context Verification Engine.
Validates transcript continuity, snaps setup/payoff boundaries, trims filler starts,
verifies hook grounding & user objective match, generates contact sheets, and preserves word timings.
"""
from pathlib import Path
import re
from typing import Dict, Any, List, Optional

from ..schemas import ClipCandidate, UserIntent, WordToken, EvaluationDimensions
from ..media.video import create_check_strip, create_story_sheets
from ..understand.brains import is_grounded, LocalLlamaBrain

DEPENDENT_OPENERS = {
    "and", "but", "so", "because", "or", "then", "also", "which",
    "aur", "lekin", "toh", "kyunki", "magar", "phir", "اور", "لیکن", "تو", "کیونکہ", "مگر", "پھر"
}
PRONOUN_OPENERS = {
    "he", "she", "they", "them", "it", "this", "that", "these", "those",
    "yeh", "woh", "ye", "wo", "is", "us", "یہ", "وہ", "اس", "ان"
}
FILLER_WORDS = {"um", "uh", "erm", "uhh", "umm", "yeah", "well", "anyway", "acha", "اچھا"}
END_PUNCT = {".", "!", "?", "۔", "؟", "…"}


def verify_and_adjust_clip(
    candidate: Dict[str, Any],
    units: List[Dict[str, Any]],
    video_path: Optional[Path],
    intent: UserIntent,
    output_dir: Optional[Path] = None,
    llm_brain: Optional[LocalLlamaBrain] = None,
    generate_story_sheet: bool = False,
) -> ClipCandidate:
    """Re-examines transcript, boundaries, visual sheets, and objective match before rendering."""
    s_time = float(candidate["start"])
    e_time = float(candidate["end"])
    notes: List[str] = []
    adjusted = False

    # Locate bounding units
    s_idx = 0
    e_idx = len(units) - 1
    for i, u in enumerate(units):
        if u["start"] <= s_time <= u["end"] or u["start"] >= s_time:
            s_idx = i
            break
    for i in range(len(units) - 1, -1, -1):
        if units[i]["start"] <= e_time <= units[i]["end"] or units[i]["end"] <= e_time:
            e_idx = i
            break

    # 1. Beginning Setup Check: Snap backward if starting on dependent opener, pronoun, or Arabic Ayat boundary
    first_u = units[s_idx]
    first_tok = first_u["first_token"]
    if (first_tok in DEPENDENT_OPENERS or first_tok in PRONOUN_OPENERS or first_u.get("has_arabic_ayat")) and s_idx > 0:
        prev_u = units[s_idx - 1]
        gap = first_u["start"] - prev_u["end"]
        new_dur = e_time - prev_u["start"]
        if (prev_u.get("has_arabic_ayat") or first_tok in DEPENDENT_OPENERS or first_tok in PRONOUN_OPENERS) and gap <= 3.0 and (first_u["start"] - prev_u["start"]) <= 9.0 and new_dur <= 90.0:
            s_time = max(0.0, prev_u["start"] - 0.15)
            s_idx -= 1
            adjusted = True
            notes.append(f"Snapped start back to include setup: '{prev_u['text'][:40]}…'")

    # 1b. Trim isolated leading filler word (e.g. "Um," "Uh,") if remaining span stays valid
    first_words = units[s_idx].get("words", [])
    if len(first_words) >= 3:
        w0_clean = re.sub(r"[^\w]", "", first_words[0].w.lower())
        if w0_clean in FILLER_WORDS and (e_time - first_words[1].s) >= 8.0:
            s_time = max(0.0, first_words[1].s - 0.10)
            adjusted = True
            notes.append(f"Trimmed leading filler word '{first_words[0].w}'")

    # 2. Ending Punchline & Payoff Check: Extend forward up to 5s if next unit completes reaction/answer
    last_u = units[e_idx]
    if e_idx + 1 < len(units):
        nxt_u = units[e_idx + 1]
        gap = nxt_u["start"] - last_u["end"]
        last_char = last_u["text"].rstrip()[-1:] if last_u["text"].strip() else ""
        is_incomplete = last_char not in END_PUNCT
        is_reaction = nxt_u["duration"] <= 5.0 and (
            is_incomplete or nxt_u["expr_score"] >= 0.35 or nxt_u.get("has_laugh") or nxt_u["text"].endswith(("!", "?", "۔"))
        )
        if is_reaction and gap <= 1.6 and (nxt_u["end"] + 0.35 - s_time) <= 92.0:
            e_time = nxt_u["end"] + 0.35
            e_idx += 1
            adjusted = True
            notes.append(f"Extended end to include payoff/reaction: '{nxt_u['text'][:40]}'")

    # 2b. Never cut mid-Ayat at clip end: extend to finish Ayat recitation cleanly
    if e_idx + 1 < len(units) and units[e_idx].get("has_arabic_ayat") and units[e_idx + 1].get("has_arabic_ayat"):
        while e_idx + 1 < len(units) and units[e_idx + 1].get("has_arabic_ayat") and (units[e_idx + 1]["end"] - s_time) <= 95.0:
            e_idx += 1
            e_time = units[e_idx]["end"] + 0.35
            adjusted = True
            notes.append("Extended end to keep Arabic Ayat recitation complete.")

    # 2c. Trim excessive trailing silence cleanly (> 0.6s after last spoken word)
    if units[e_idx].get("words"):
        last_w_end = units[e_idx]["words"][-1].e
        if e_time - last_w_end > 0.65:
            e_time = last_w_end + 0.30
            adjusted = True

    verified_span = units[s_idx:e_idx + 1]
    span_words: List[WordToken] = [
        w for u in verified_span for w in u.get("words", [])
        if w.e >= s_time - 0.05 and w.s <= e_time + 0.05
    ]
    full_text = " ".join(w.w for w in span_words) if span_words else " ".join(u["text"] for u in verified_span)
    dur = round(max(0.0, e_time - s_time), 2)

    dims: EvaluationDimensions = candidate["dimensions"]
    overall = float(candidate["overall_score"])
    picked_by = "rules"
    verified = True
    rejection_reason: Optional[str] = None

    # 3. Intent & Completeness Verification Gate
    if intent.topic and candidate.get("topic_relevance", 1.0) < 0.25:
        verified = False
        rejection_reason = f"Does not meaningfully cover requested topic '{intent.topic}'."
    elif intent.speaker and candidate.get("speaker_relevance", 1.0) < 0.25:
        verified = False
        rejection_reason = f"Requested speaker '{intent.speaker}' not sufficiently matched."
    elif intent.objective == "educational" and dims.content_value < 4.0:
        verified = False
        rejection_reason = "Insufficient educational density or clarity."
    elif intent.objective == "funny" and dims.emotional_value < 3.4:
        verified = False
        rejection_reason = "Insufficient humor, reaction energy, or comedic pacing."
    elif dur < 6.0:
        verified = False
        rejection_reason = "Clip duration too short (< 6s) for standalone context."
    elif dims.standalone_clarity < 3.8 and dims.context_completeness < 4.5:
        verified = False
        rejection_reason = "Fails standalone context completeness check."

    # 4. Optional Local LLM Story Review (never crashes, falls back cleanly)
    title_text = candidate.get("title") or f"Moment at {int(s_time // 60)}:{int(s_time % 60):02d}"
    hook_u = candidate.get("hook_unit", verified_span[0])
    peak_u = candidate.get("peak_unit", verified_span[-1])
    hook_text = hook_u["text"][:130]
    rationale = candidate.get("rationale") or "Selected for coherent thoughts, strong hook, and natural pause boundaries."

    if verified and llm_brain is not None:
        try:
            review = llm_brain.review_candidate(full_text, intent.objective, intent.topic)
            if review is not None:
                picked_by = "local_llm"
                dims = EvaluationDimensions(
                    hook_strength=round(0.5 * dims.hook_strength + 0.5 * review.hook, 2),
                    content_value=round(0.5 * dims.content_value + 0.5 * review.value, 2),
                    emotional_value=round(0.5 * dims.emotional_value + 0.5 * review.entertainment, 2),
                    standalone_clarity=round(0.5 * dims.standalone_clarity + 0.5 * review.standalone, 2),
                    context_completeness=dims.context_completeness if review.complete else min(dims.context_completeness, 4.5),
                    audio_quality=dims.audio_quality,
                    visual_suitability=dims.visual_suitability,
                    boundary_quality=dims.boundary_quality if review.complete else min(dims.boundary_quality, 5.0),
                    engagement_potential=round(0.5 * dims.engagement_potential + 0.5 * review.engagement, 2),
                )
                overall = round(0.5 * overall + 0.5 * ((review.hook + review.value + review.entertainment + review.standalone + review.engagement) / 5.0), 2)
                if review.standalone <= 3.0 and not review.complete:
                    verified = False
                    rejection_reason = f"Local AI reviewer vetoed incomplete clip: {review.reason}"
                if review.title and is_grounded(review.title, full_text):
                    title_text = review.title[:70]
                if review.hook_line and is_grounded(review.hook_line, full_text):
                    hook_text = review.hook_line[:130]
                if review.reason:
                    rationale = review.reason
        except Exception:
            notes.append("Local LLM unavailable; verified via deterministic rules.")

    # 5. Grounding sanity check on hook
    if not is_grounded(hook_text, full_text, min_overlap=0.35):
        hook_text = verified_span[0]["text"][:130]

    # 6. Pre-render visual story check strip / 4x4 story sheet
    cid = str(candidate.get("id", "clip_1"))
    check_strip_path = None
    story_sheet_path = None
    if video_path and video_path.is_file() and output_dir:
        if generate_story_sheet:
            sheets = create_story_sheets(
                video_path,
                max(0.0, s_time - 4.0),
                e_time + 4.0,
                output_dir / f"{cid}_story",
                every_sec=2.0,
            )
            if sheets:
                story_sheet_path = str(sheets[0])
        else:
            strip_file = output_dir / f"check_{cid}.jpg"
            hook_t = candidate.get("hook_time", (s_time + e_time) / 2.0)
            res = create_check_strip(video_path, s_time, hook_t, e_time, strip_file)
            if res:
                check_strip_path = str(res)

    setup_part = "Starts cleanly on a complete thought." if s_idx == candidate.get("anchor_idx", s_idx) else "Includes earlier setup sentence for complete context."
    ab_hooks = candidate.get("ab_hooks") or [hook_text, verified_span[0]["text"][:130]]

    return ClipCandidate(
        id=cid,
        start=round(s_time, 2),
        end=round(e_time, 2),
        duration=dur,
        hook_time=round(candidate.get("hook_time", s_time), 2),
        trigger_time=candidate.get("trigger_time"),
        payoff_time=candidate.get("payoff_time"),
        dimensions=dims,
        overall_score=overall,
        title=title_text,
        headline=candidate.get("headline") or title_text,
        hook_text=hook_text,
        ab_hooks=ab_hooks,
        event_text=peak_u["text"][:140],
        context_text=setup_part,
        payoff_text=units[e_idx]["text"][:130],
        rationale=rationale,
        transcript_text=full_text,
        words=span_words,
        boundary_adjusted=adjusted,
        adjustment_notes=notes,
        story_sheet_path=story_sheet_path,
        check_strip_path=check_strip_path,
        picked_by=picked_by,
        verified=verified,
        rejection_reason=rejection_reason,
    )
