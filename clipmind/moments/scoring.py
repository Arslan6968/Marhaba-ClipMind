"""9-Dimensional Evaluation and Intent-Weighted Scoring Engine."""
import re
import statistics
from typing import List, Dict, Any

from ..schemas import EvaluationDimensions, UserIntent
from ..understand.intent import get_intent_weights, calculate_topic_relevance, calculate_speaker_relevance

DEPENDENT_OPENERS = {
    "and", "but", "so", "because", "or", "then", "also", "which",
    "aur", "lekin", "toh", "kyunki", "magar", "phir", "اور", "لیکن", "تو", "کیونکہ", "مگر", "پھر"
}
PRONOUN_OPENERS = {
    "he", "she", "they", "them", "it", "this", "that", "these", "those",
    "yeh", "woh", "ye", "wo", "is", "us", "یہ", "وہ", "اس", "ان"
}
CURIOSITY_TOKENS = {
    "secret", "truth", "never", "mistake", "best", "worst", "why", "how", "reason",
    "shocking", "insane", "hidden", "actually", "problem", "solution", "exact", "keyframe",
    "raz", "sach", "galti", "kyun", "kaise", "asliyat", "راز", "سچ", "غلطی"
}


def _clamp(val: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, round(val, 2)))


def _clean_title(text: str, max_words: int = 8) -> str:
    norm = re.sub(r"\s*/\s*", " & ", text or "")
    cleaned = re.sub(r"[^\w\s&?؟!%]", "", norm, flags=re.U).strip()
    words = cleaned.split()
    return " ".join(words[:max_words]) if words else "Key Highlight"


def evaluate_span(units: List[Dict[str, Any]], start_idx: int, end_idx: int, anchor_idx: int,
                  frames: List[Dict[str, Any]], cuts: List[float], has_faces: bool,
                  intent: UserIntent, genre: str = "general") -> Dict[str, Any]:
    """Computes scores across all 9 dimensions for a candidate thought span [start_idx, end_idx]."""
    span = units[start_idx:end_idx + 1]
    clip_start = max(0.0, span[0]["start"] - 0.15)
    clip_end = span[-1]["end"] + 0.35
    duration = clip_end - clip_start

    first_unit = span[0]
    last_unit = span[-1]
    tokens = [t for u in span for t in u["tokens"]]
    full_text = " ".join(u["text"] for u in span)
    span_words = [w for u in span for w in u.get("words", [])]

    # 1. Hook Strength (0-10)
    early_units = [u for u in span if u["start"] - clip_start <= 6.5]
    early_hook_unit = max(early_units, key=lambda u: u["interest"]) if early_units else first_unit

    hook_score = 4.0 + 2.5 * early_hook_unit["interest"]
    if first_unit["is_question"]:
        hook_score += 2.0
    if first_unit["first_token"] not in DEPENDENT_OPENERS and first_unit["first_token"] not in PRONOUN_OPENERS:
        hook_score += 1.2
    if first_unit.get("is_filler_start"):
        hook_score -= 1.2
    if any(t in CURIOSITY_TOKENS for t in first_unit["tokens"]):
        hook_score += 1.8
    hook_score = _clamp(hook_score)

    # 2. Content Value (0-10)
    wps = len(tokens) / max(1.0, duration)
    wps_score = 1.0 - min(1.0, abs(wps - 2.4) / 1.6)
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    has_qa = any(u["is_qa_answer"] for u in span) or (first_unit["is_question"] and len(span) >= 2)
    fact_hits = sum(u.get("fact_count", 0) for u in span)
    content_score = 2.0 + 3.2 * wps_score + 2.8 * unique_ratio + (1.4 if has_qa else 0.0) + min(1.6, fact_hits * 0.5)

    topic_rel = calculate_topic_relevance(full_text, intent.topic)
    speaker_rel = calculate_speaker_relevance(full_text, intent.speaker)
    if intent.topic:
        content_score = content_score * 0.35 + (topic_rel * 10.0) * 0.65
    content_score = _clamp(content_score)

    # 3. Emotional / Entertainment Value (0-10)
    max_expr = max(u["expr_score"] for u in span)
    max_z = max(u["energy_z"] for u in span)
    max_spike = max(u.get("spike_db", 0.0) for u in span)
    has_laughter = any(u.get("has_laugh") for u in span)
    max_chat = max(u.get("chat_mult", 1.0) for u in span)

    acoustic_boost = max(max(0.0, min(1.0, (max_z + 1.0) / 4.0)), min(1.0, max_spike / 10.0))
    chat_boost = min(1.5, max(0.0, (max_chat - 1.5) * 0.5))
    emotional_score = 2.0 + 3.8 * max_expr + 2.2 * acoustic_boost + (1.6 if has_laughter else 0.0) + chat_boost
    emotional_score = _clamp(emotional_score)

    # 4. Standalone Clarity (0-10)
    standalone_score = 7.2
    if first_unit["first_token"] in PRONOUN_OPENERS and len(span) < 3:
        standalone_score -= 3.0
    if first_unit["first_token"] in DEPENDENT_OPENERS:
        standalone_score -= 2.5
    if first_unit.get("is_filler_start"):
        standalone_score -= 1.0
    avg_conf = statistics.mean(u["confidence"] for u in span) if span else 0.7
    standalone_score += (avg_conf - 0.7) * 4.0
    if has_qa:
        standalone_score += 1.4
    standalone_score = _clamp(standalone_score)

    # 5. Context Completeness (0-10)
    has_setup = start_idx < anchor_idx
    has_payoff = end_idx >= anchor_idx
    has_beat = any(u.get("dramatic_beat") for u in span)
    context_score = 3.8 + (2.8 if has_setup else 1.0) + (2.8 if has_payoff else 1.0) + (0.6 if has_beat else 0.0)
    context_score = _clamp(context_score)

    # 6. Audio Quality (0-10)
    audio_score = _clamp(avg_conf * 10.0)

    # 7. Visual Suitability (0-10)
    near_frames = [f for f in frames if clip_start - 5.0 <= f["t"] <= clip_end + 5.0]
    visual_score = 6.0
    if near_frames:
        med_sharp = statistics.median(f["sharp"] for f in near_frames)
        med_bright = statistics.median(f["bright"] for f in near_frames)
        if med_sharp >= 45.0:
            visual_score += 2.0
        elif med_sharp < 12.0:
            visual_score -= 2.0
        if 35.0 <= med_bright <= 225.0:
            visual_score += 1.0
        else:
            visual_score -= 2.0
    if has_faces:
        visual_score += 1.0
    visual_score = _clamp(visual_score)

    # 8. Boundary Quality (0-10)
    b_score = 5.8
    if first_unit["first_token"] not in DEPENDENT_OPENERS and not first_unit.get("is_filler_start"):
        b_score += 2.2
    if last_unit["text"] and last_unit["text"][-1] in ".?!؟۔…":
        b_score += 2.0
    boundary_score = _clamp(b_score)

    # 9. Engagement Potential (0-10)
    dur_fit = 1.0 if 20.0 <= duration <= 52.0 else (0.78 if duration <= 75.0 else 0.55)
    curiosity_hits = sum(1 for t in tokens if t in CURIOSITY_TOKENS)
    energy_contrast = min(1.0, max(0.0, (max_z - min(u["energy_z"] for u in span)) / 3.0))
    eng_score = 3.0 + 2.5 * dur_fit + 2.0 * min(1.0, curiosity_hits * 0.25) + 2.5 * energy_contrast
    eng_score = _clamp(eng_score)

    dims = EvaluationDimensions(
        hook_strength=hook_score,
        content_value=content_score,
        emotional_value=emotional_score,
        standalone_clarity=standalone_score,
        context_completeness=context_score,
        audio_quality=audio_score,
        visual_suitability=visual_score,
        boundary_quality=boundary_score,
        engagement_potential=eng_score,
    )

    weights = get_intent_weights(intent)
    overall = sum(getattr(dims, k) * w for k, w in weights.items())

    # Topic & speaker signals + intro/outro wrap-up guard
    if intent.topic and topic_rel < 0.30:
        overall *= 0.45
    if intent.speaker and speaker_rel < 0.35:
        overall *= 0.65

    low_full = full_text.lower()
    outro_markers = ("allah hafiz", "فی امان اللہ", "پیمانی اللہ", "subscribe", "سبسکرائب", "bell icon", "اپنا بہت سارا خیال", "امید کرتا ہوں کہ آج کی")
    if sum(1 for m in outro_markers if m in low_full) >= 1 and (clip_end > 60.0):
        overall *= 0.62  # prefer substantive body explanations over end-of-video channel plugs
    # Intro agenda / question-list teaser guard: prefer substantive answer segments with real facts/steps
    q_count = sum(1 for u in span if u.get("is_question"))
    has_teaser = any(u.get("is_teaser") for u in span)
    if clip_start < 65.0 and (has_teaser or (q_count >= 2 and fact_hits == 0)):
        overall *= 0.68

    # Genre-adaptive intelligence tuning inside the single universal engine
    has_ayat_and_expl = any(u.get("has_arabic_ayat") for u in span) and any(not u.get("has_arabic_ayat") for u in span)
    if genre == "islamic":
        if has_setup and has_payoff:
            overall *= 1.08
        if has_ayat_and_expl:
            overall *= 1.12  # reward complete Ayat recitation + explanation/lesson
    elif genre == "gaming" and (max_spike >= 5.0 or max_chat >= 1.8):
        overall *= 1.10
    elif genre == "influencing" and hook_score >= 7.0 and 18.0 <= duration <= 45.0:
        overall *= 1.08
    elif genre == "movie_shot" and (has_beat or any(u.get("has_cut") for u in span)):
        overall *= 1.10

    non_ayat_units = [u for u in span if not u.get("has_arabic_ayat")]
    peak_u = max(non_ayat_units or span, key=lambda u: u["interest"])
    hook_pref = next((u for u in non_ayat_units if u["start"] - clip_start <= 10.0), early_hook_unit)
    title_text = _clean_title(peak_u["text"], max_words=7)
    hook_line = hook_pref["text"][:130]
    ab_hooks = [
        hook_line,
        first_unit["text"][:130] if first_unit["text"] != hook_line else last_unit["text"][:130],
    ]

    return {
        "start": round(clip_start, 2),
        "end": round(clip_end, 2),
        "duration": round(duration, 2),
        "hook_time": round(early_hook_unit["start"], 2),
        "trigger_time": round(first_unit["start"], 2),
        "payoff_time": round(peak_u["end"], 2),
        "dimensions": dims,
        "overall_score": round(overall, 2),
        "hook_unit": early_hook_unit,
        "peak_unit": peak_u,
        "title": title_text,
        "headline": _clean_title(first_unit["text"], max_words=10),
        "ab_hooks": ab_hooks,
        "topic_relevance": topic_rel,
        "speaker_relevance": speaker_rel,
        "tokens": len(tokens),
        "transcript_text": full_text,
        "words": span_words,
        "avg_visual_activity": round(statistics.mean(u.get("visual_activity", 10.0) for u in span), 2),
    }
