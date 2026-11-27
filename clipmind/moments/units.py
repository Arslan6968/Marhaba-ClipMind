"""Complete-thought unit grouping and multi-signal acoustic/semantic annotation."""
import math
import re
import statistics
from typing import List, Dict, Any, Optional

from ..schemas import Transcript, UserIntent
from ..media.audio import compute_db_spikes
from ..understand.intent import calculate_topic_relevance, calculate_speaker_relevance

END_PUNCT = ".?!؟۔…"
LAUGH_PATTERN = re.compile(r"(?:\bha(?:ha)+h?\b|\bhehe+\b|\blol\b|\blmao\b|ہا ?ہا|😂|🤣)", re.I)
FILLER_OPENERS = {
    "um", "uh", "erm", "uhh", "umm", "anyway", "yeah", "well", "okay", "alright",
    "acha", "matlab", "toh", "اچھا", "مطلب"
}
EXPRESSIVE_WORDS = {
    "wow", "omg", "god", "insane", "crazy", "unbelievable", "incredible", "amazing", "no way", "holy",
    "seriously", "never", "worst", "best", "impossible", "finally", "shocking", "secret", "mistake",
    "truth", "wait", "listen", "remember", "important", "critical", "exact", "problem", "solution",
    "واہ", "کمال", "اوہ", "یار", "حیرت", "ناممکن", "بہترین", "زبردست", "خطرناک", "سچ",
    "راز", "غلطی", "اہم", "ضروری", "مسئلہ", "حل",
    "wah", "kamaal", "kamal", "yaar", "oye", "ajeeb", "zabardast", "hairan", "zaroori"
}
QUESTION_WORDS = {
    "what", "why", "how", "who", "when", "where", "which",
    "کیا", "کیوں", "کیسے", "کون", "کب", "کہاں", "kya", "kyun", "kyon", "kaise", "kaun", "kab", "kahan"
}
TEASER_PHRASES = (
    "aaj ki video", "today in this video", "today we will", "we will discuss", "stay tuned",
    "watch till end", "آج کی وڈیو", "آج کی ویڈیو", "ڈسکس کریں", "پوری ویڈیو", "end tak",
)


def _clean_token(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower(), flags=re.U)


def build_thought_units(transcript: Transcript) -> List[Dict[str, Any]]:
    """Groups words into complete-thought units aligned to punctuation and natural speech pauses."""
    words = transcript.words
    segments = transcript.segments
    if not words and segments:
        words = [w for s in segments for w in s.words]
    if not words:
        return []

    seg_ends = [s.end for s in segments]
    turn_starts = {round(s.start, 2) for s in segments if s.speaker_turn}
    units = []
    current_tokens = []

    for i, w in enumerate(words):
        current_tokens.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.s - w.e) if nxt else 99.0
        has_end_punct = bool(w.w) and w.w[-1] in END_PUNCT
        near_seg_end = any(abs(w.e - e) < 0.35 for e in seg_ends)
        unit_dur = current_tokens[-1].e - current_tokens[0].s
        next_is_turn = bool(nxt and round(nxt.s, 2) in turn_starts)

        if nxt is None or has_end_punct or gap >= 0.8 or next_is_turn or (near_seg_end and gap >= 0.35) or unit_dur > 16.0:
            text = " ".join(t.w for t in current_tokens).strip()
            first_tok = _clean_token(current_tokens[0].w)
            is_q = text.endswith(("?", "؟")) or (first_tok in QUESTION_WORDS and len(current_tokens) >= 3)
            avg_p = sum(t.p for t in current_tokens) / len(current_tokens)
            u_start = round(current_tokens[0].s, 2)

            units.append({
                "index": len(units),
                "start": u_start,
                "end": round(current_tokens[-1].e, 2),
                "duration": round(current_tokens[-1].e - current_tokens[0].s, 2),
                "text": text,
                "first_token": first_tok,
                "is_filler_start": first_tok in FILLER_OPENERS,
                "is_question": is_q,
                "speaker_turn": u_start in turn_starts,
                "confidence": round(avg_p, 2),
                "tokens": [_clean_token(t.w) for t in current_tokens if _clean_token(t.w)],
                "words": list(current_tokens),
            })
            current_tokens = []

    return units


def annotate_units(units: List[Dict[str, Any]], energy_profile: List[float], scene_cuts: List[float],
                   intent: Optional[UserIntent] = None, chat_speed: Optional[List[int]] = None,
                   screen_activity: Optional[Dict[str, Any]] = None) -> None:
    """Enriches thought units with rolling dB spikes, chat bursts, speech rate, topic & speaker scores."""
    if not units:
        return

    spike_db = compute_db_spikes(energy_profile) if energy_profile else []
    energies = [e for e in energy_profile if e > -70.0] or [-40.0]
    med_energy = statistics.median(energies)
    std_energy = statistics.pstdev(energies) or 5.0

    wps_list = [len(u["tokens"]) / max(0.5, u["duration"]) for u in units] or [2.0]
    med_wps = statistics.median(wps_list)
    std_wps = statistics.pstdev(wps_list) or 1.0

    chat_base = float(statistics.median([c for c in chat_speed if c > 0])) if chat_speed and any(chat_speed) else 0.0

    for idx, u in enumerate(units):
        s_int = max(0, int(u["start"]))
        e_int = max(s_int + 1, min(len(energy_profile), int(math.ceil(u["end"])))) if energy_profile else s_int + 1
        slice_e = energy_profile[s_int:e_int] if energy_profile else [med_energy]
        slice_e = slice_e or [med_energy]
        mean_e = sum(slice_e) / len(slice_e)
        peak_e = max(slice_e)

        z = (0.5 * mean_e + 0.5 * peak_e - med_energy) / std_energy
        u["energy_z"] = max(-3.0, min(5.0, round(z, 2)))

        if spike_db:
            sp_slice = spike_db[s_int:min(len(spike_db), e_int)]
            u["spike_db"] = max(sp_slice) if sp_slice else 0.0
        else:
            u["spike_db"] = 0.0

        text = u["text"]
        caps = sum(1 for w in text.split() if len(w) >= 3 and w.isupper() and w.isalpha())
        lex_matches = sum(1 for t in u["tokens"] if t in EXPRESSIVE_WORDS)
        has_laugh = bool(LAUGH_PATTERN.search(text))
        u["has_laugh"] = has_laugh
        u["is_expressive"] = has_laugh or lex_matches > 0 or caps > 0
        expr_val = min(1.0, 0.4 * has_laugh + 0.28 * min(2, lex_matches) + 0.18 * min(2, caps) + 0.18 * text.count("!"))
        u["expr_score"] = round(expr_val, 2)

        wps = len(u["tokens"]) / max(0.5, u["duration"])
        u["wps"] = round(wps, 2)
        u["has_cut"] = any(u["start"] <= c <= u["end"] for c in scene_cuts)

        # Chat burst signal (if VOD replay chat is present)
        u["chat_mult"] = 1.0
        if chat_base > 0 and chat_speed:
            b0 = max(0, int(u["start"] // 10))
            b1 = min(len(chat_speed), int(u["end"] // 10) + 2)
            seg_c = chat_speed[b0:b1]
            if seg_c:
                u["chat_mult"] = round(max(seg_c) / chat_base, 2)

        # Storyboard visual activity signal
        u["visual_activity"] = 10.0
        if screen_activity is not None and len(screen_activity.get("raw", [])) > 0:
            raw_act = screen_activity["raw"]
            seg_a = raw_act[s_int:min(len(raw_act), e_int)]
            if len(seg_a):
                u["visual_activity"] = round(float(sum(seg_a) / len(seg_a)), 2)

        prev = units[idx - 1] if idx > 0 else None
        gap_before = (u["start"] - prev["end"]) if prev else 0.0
        u["is_qa_answer"] = bool(prev and prev["is_question"] and gap_before < 3.5)
        u["dramatic_beat"] = bool(prev and 0.9 <= gap_before <= 2.6 and (u["is_expressive"] or u["spike_db"] >= 4.0))

        # Intent semantic signals
        topic_rel = calculate_topic_relevance(text, intent.topic) if (intent and intent.topic) else 0.0
        spk_rel = calculate_speaker_relevance(text, intent.speaker) if (intent and intent.speaker) else 0.0
        u["topic_rel"] = topic_rel
        u["speaker_rel"] = spk_rel

        norm_spike = min(1.0, u["spike_db"] / 10.0)
        norm_z = max(0.0, min(1.0, (u["energy_z"] + 1.0) / 4.0))
        norm_acoustic = max(norm_spike, norm_z)
        norm_wps = max(0.0, min(1.0, (u["wps"] - (med_wps - std_wps)) / (2.0 * std_wps)))
        chat_bonus = min(0.25, max(0.0, (u["chat_mult"] - 1.5) * 0.08))

        base_interest = (
            0.30 * norm_acoustic +
            0.26 * u["expr_score"] +
            0.14 * (1.0 if u["is_question"] else 0.0) +
            0.12 * (1.0 if u["is_qa_answer"] else 0.0) +
            0.08 * norm_wps +
            0.06 * (1.0 if u["dramatic_beat"] else 0.0) +
            chat_bonus
        )

        # Boost interest when user requested specific objective/topic/speaker
        if intent and intent.topic:
            base_interest = 0.45 * base_interest + 0.55 * topic_rel
        if intent and intent.speaker:
            base_interest = 0.70 * base_interest + 0.30 * spk_rel
        if intent and intent.objective in ("educational", "powerful") and len(u["tokens"]) >= 8:
            # Reward dense, clear explanations even at calm vocal volume
            base_interest = max(base_interest, 0.32 + 0.35 * min(1.0, len(set(u["tokens"])) / 14.0))

        fact_hits = len(re.findall(r"\b\d+(?:\.\d+)?%?", text))
        u["fact_count"] = fact_hits
        u["is_teaser"] = any(tp in text.lower() for tp in TEASER_PHRASES)
        if fact_hits > 0 and intent and intent.objective in ("educational", "powerful", "general"):
            base_interest = min(1.0, base_interest + min(0.18, fact_hits * 0.06))

        u["interest"] = round(min(1.0, base_interest), 3)
