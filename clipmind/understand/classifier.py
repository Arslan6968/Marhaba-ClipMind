"""Universal Content Classification Engine (12 Categories + Optimal Editing Preset Recommendation).
Infers content category from transcript semantics, dialogue turns, cut pacing, and visual structure.
Guides analysis and styling without locking the universal engine into rigid category silos.
"""
import re
from typing import List, Dict

from ..schemas import Transcript, ContentClassification

GENRE_LEXICON: Dict[str, set] = {
    "gaming": {
        "game", "gameplay", "kill", "killed", "died", "boss", "weapon", "level", "mission", "stream",
        "round", "glitch", "cheat", "speedrun", "gta", "minecraft", "fortnite", "valorant", "pubg",
        "cod", "match", "respawn", "spawn", "loot", "quest", "playthrough", "headshot", "noob", "pro",
        "mar gya", "khel", "goli", "dushman", "stage", "کھیل", "گیم"
    },
    "podcast": {
        "podcast", "episode", "guest", "welcome back", "welcome to", "host", "my guest",
        "show", "sponsor", "listener", "listeners", "conversation", "topic", "discussion",
        "guftagu", "khush amdeed", "mehman", "silsila", "پوڈکاسٹ", "گفتگو", "مہمان"
    },
    "interview": {
        "tell us", "how did you", "what made you", "your journey", "advice for", "in your career",
        "when did you start", "do you think", "what is your opinion", "apka tajruba", "ap bataiye",
        "kya waja thi", "apka mashwara", "انٹرویو", "تجربہ"
    },
    "tutorial": {
        "step by step", "click", "select", "install", "download", "settings", "timeline", "keyframe",
        "shortcut", "export", "import", "tab", "panel", "button", "software", "menu", "tool", "proxy",
        "tutorial", "how to", "demonstration", "screen", "drag", "drop", "render", "course", "class",
        "tareeqa", "بٹن", "کلک", "سیٹنگز", "ایڈیٹنگ", "ٹائم لائن", "کورس", "طریقہ"
    },
    "lecture_education": {
        "lecture", "concept", "example", "formula", "definition", "chapter", "learn", "study",
        "students", "professor", "teacher", "theorem", "explanation", "understand", "method",
        "science", "math", "code", "algorithm", "theory", "research", "history", "physics",
        "admission", "merit", "eligibility", "stipend", "seats", "scholarship", "university",
        "college", "medical", "nursing", "exam", "marks", "percentage", "closing", "criteria",
        "sabaq", "parhai", "samjhein", "misal", "سبق", "مثال", "سمجھیں", "علم", "داخلہ", "میرٹ", "تعلیم", "طلباء"
    },
    "speech_presentation": {
        "ladies and gentlemen", "audience", "stage", "keynote", "presentation", "slide", "today i want",
        "thank you for having me", "conference", "summit", "vision", "future", "believe", "dream",
        "khawateen", "hazrat", "تقرير", "خطاب"
    },
    "news_discussion": {
        "breaking", "news", "report", "headline", "government", "economy", "policy", "election",
        "crisis", "update", "press", "journalist", "official", "statement", "analysis",
        "khabar", "siyasi", "حکومت", "خبر", "معیشت", "رپورٹ"
    },
    "debate_conversation": {
        "disagree", "agree", "argument", "point", "debate", "counter", "wrong", "prove",
        "evidence", "fact", "actually", "listen to me", "hold on", "wait a second",
        "ikhtilaf", "bahas", "دلیل", "بحث", "اختلاف"
    },
    "documentary": {
        "century", "empire", "civilization", "discovered", "ancient", "documentary", "era",
        "expedition", "mystery", "archive", "origins", "documented", "historian",
        "tareekh", "تاریخ", "تحقیق"
    },
    "vlog": {
        "morning", "routine", "travel", "vlog", "today we are", "going to", "hotel", "flight",
        "food", "camera", "day in my life", "guys", "arrived", "explore",
        "safar", "سفر", "ولاگ"
    },
    "entertainment": {
        "reaction", "react", "drama", "scandal", "look at this", "insane", "crazy", "hilarious",
        "prank", "challenge", "exposed", "comedy", "funny", "meme", "roast",
        "dekho", "ye dekhein", "hairat", "مذاق", "دیکھو", "کمال"
    },
}

PRESET_MAP = {
    "podcast": "podcast",
    "interview": "podcast",
    "debate_conversation": "podcast",
    "lecture_education": "educational",
    "tutorial": "educational",
    "speech_presentation": "clean",
    "documentary": "clean",
    "news_discussion": "clean",
    "gaming": "gaming",
    "vlog": "dynamic",
    "entertainment": "dynamic",
    "general": "social_media",
}


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text, flags=re.U) if len(w) >= 2]


def classify_content(transcript: Transcript, cuts: List[float], duration: float,
                     has_facecam: bool = False, speaker_count: int = 1) -> ContentClassification:
    """Classifies video across 12 general categories and recommends the best editing preset."""
    full_text = transcript.full_text.lower()
    tokens = set(_tokenize(full_text))

    scores: Dict[str, float] = {k: 0.0 for k in GENRE_LEXICON}
    scores["general"] = 1.0

    # 1. Single-word + phrase lexical matching
    for genre, keywords in GENRE_LEXICON.items():
        for kw in keywords:
            if " " in kw:
                if kw in full_text:
                    scores[genre] += 2.5
            elif kw in tokens:
                scores[genre] += 2.0

    # 2. Structural dialogue & turn-taking signals
    segments = transcript.segments
    q_count = sum(1 for s in segments if s.text.strip().endswith(("?", "؟")))
    turn_count = sum(1 for s in segments if s.speaker_turn)
    q_ratio = q_count / max(1, len(segments))
    turn_ratio = turn_count / max(1, len(segments))

    if q_ratio > 0.16:
        scores["interview"] += 3.5
        scores["podcast"] += 2.0
    elif q_ratio > 0.08:
        scores["interview"] += 1.5
        scores["podcast"] += 1.5

    if speaker_count >= 2 or turn_ratio > 0.25:
        scores["podcast"] += 2.5
        scores["debate_conversation"] += 2.0
        scores["interview"] += 1.5

    # 3. Visual pacing & facecam signals
    dur_min = max(0.5, duration / 60.0)
    cuts_per_min = len(cuts) / dur_min
    if cuts_per_min > 12.0:
        scores["gaming"] += 2.0
        scores["entertainment"] += 2.5
        scores["vlog"] += 1.5
    elif cuts_per_min < 3.0:
        scores["lecture_education"] += 1.5
        scores["podcast"] += 1.5
        scores["speech_presentation"] += 1.5

    if has_facecam:
        scores["gaming"] += 4.5
        scores["entertainment"] += 1.5

    top_genre = max(scores, key=scores.get)
    top_score = scores[top_genre]
    total_score = sum(scores.values())
    confidence = min(1.0, round(top_score / max(1.0, total_score), 2))

    if top_score < 2.0:
        top_genre = "general"
        confidence = 0.5

    rec_preset = PRESET_MAP.get(top_genre, "clean")
    rationale = (
        f"Classified as '{top_genre.replace('_', ' ')}' ({confidence:.0%} confidence) "
        f"from vocabulary cues, dialogue turns ({turn_count}), question rate ({q_ratio:.0%}), "
        f"and visual pacing ({cuts_per_min:.1f} cuts/min)."
    )

    return ContentClassification(
        genre=top_genre,
        confidence=confidence,
        recommended_preset=rec_preset,
        genre_scores={k: round(v, 2) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:5]},
        rationale=rationale,
    )
