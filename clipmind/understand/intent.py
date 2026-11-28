"""Max-Level User Intent System, Offline Semantic Topic Matcher & Adaptive Questioning.
Understands natural language objectives, speaker targets, bilingual English/Urdu/Roman-Urdu topics,
morphological stems, character n-grams, and dynamically rebalances 9D scoring weights.
"""
import math
import re
from typing import Dict, List, Optional, Any, Set

from ..schemas import UserIntent, DEFAULT_WEIGHTS, ContentClassification
from .urdu import is_urdu_text, transliterate_urdu_to_roman

OBJECTIVE_PATTERNS = {
    "funny": [r"\bfunn(y|iest)\b", r"\blaugh", r"\bcomedy\b", r"\bjoke\b", r"\bhilarious\b", r"\bfail\b", r"مذاق", r"لطیفہ", r"ہنسی"],
    "educational": [r"\beducat(ion|ional)?\b", r"\blearn", r"\btutorial\b", r"\bexplain", r"\bconcept\b", r"\bhow to\b", r"\btakeaways?\b", r"\bstep by step\b", r"سبق", r"سمجھا", r"سیکھ"],
    "powerful": [r"\bpowerful\b", r"\binspir", r"\bmotivat", r"\bquote\b", r"\bgreatest\b", r"\bbest statement\b", r"\btruth\b", r"طاقتور", r"جذبہ"],
    "controversial": [r"\bcontrovers", r"\bdebate\b", r"\bfight\b", r"\bargument\b", r"\bclash\b", r"\bdrama\b", r"\bdisagree\b", r"بحث", r"جھگڑا"],
    "emotional": [r"\bemotion(al)?\b", r"\bsad\b", r"\bcry\b", r"\bheart\b", r"\btouching\b", r"\bdeep\b", r"جذباتی", r"رونا", r"دل"],
    "story": [r"\bstory\b", r"\bstories\b", r"\btale\b", r"\bincident\b", r"\bwhat happened\b", r"کہانی", r"واقعہ"],
    "discussion": [r"\bdiscussion\b", r"\bopinion\b", r"\binterview\b", r"\btalk\b", r"\bconversation\b", r"گفتگو"],
    "islamic": [r"\bislamic\b", r"\bbayan\b", r"\btafseer\b", r"\bhadith\b", r"\bquran\b", r"\bdua\b", r"\bseerah\b", r"\bsunnah\b", r"\breminder\b", r"بیان", r"قرآن", r"حدیث", r"تفسیر", r"دعا"],
    "gaming": [r"\bgaming\b", r"\bclutch\b", r"\bheadshot\b", r"\bgameplay\b", r"\brace\b", r"\bwipe\b", r"\bboss fight\b", r"گیم", r"کلچ"],
    "movie_shot": [r"\bmovie\b", r"\bfilm\b", r"\bcinema(tic)?\b", r"\bscene\b", r"\bmovie shot\b", r"\bdrama\b", r"فلم", r"سین", r"ڈرامہ"],
}

# Bilingual English + Urdu + Roman-Urdu Concept Graph
TOPIC_CONCEPTS: List[Set[str]] = [
    {"ai", "artificial intelligence", "machine learning", "deep learning", "neural", "model", "gpt", "llm", "agent", "robot", "automation", "algorithm", "openai", "claude", "gemini", "مصنوعی ذہانت", "اے آئی", "آٹومیشن"},
    {"finance", "money", "market", "stocks", "invest", "investment", "revenue", "profit", "capital", "economy", "wealth", "income", "salary", "rich", "paisa", "sarmaya", "munafa", "پیسہ", "سرمایہ", "منافع", "معیشت"},
    {"business", "startup", "founder", "company", "entrepreneur", "customer", "sales", "strategy", "brand", "saas", "agency", "karobar", "tijarat", "کاروبار", "تجارت", "کسٹمر"},
    {"crypto", "bitcoin", "ethereum", "blockchain", "btc", "eth", "token", "wallet", "coin", "cryptocurrency", "defi", "web3"},
    {"gaming", "game", "gameplay", "stream", "mission", "level", "boss", "weapon", "glitch", "speedrun", "headshot", "khel", "کھیل", "گیم"},
    {"gta", "grand theft auto", "rockstar", "vice city", "los santos", "lucia", "jason", "police", "heist", "chase", "npc"},
    {"health", "diet", "food", "sleep", "workout", "exercise", "fitness", "gym", "body", "brain", "dopamine", "calories", "mental", "sehat", "صحت", "ورزش", "نیند"},
    {"education", "study", "school", "university", "college", "exam", "student", "teacher", "professor", "degree", "parhai", "sabaq", "تعلیم", "پڑھائی", "سبق"},
    {"admission", "admissions", "merit", "closing merit", "merit list", "eligibility", "criteria", "stipend", "seats", "last date", "apply", "nursing", "bs nursing", "mbbs", "bds", "fsc", "mdcat", "medical", "govt", "private", "fee", "scholarship", "داخلہ", "داخلا", "میرٹ", "سیٹ", "سیٹس", "اپلائے", "اللیجبلٹی", "کرایٹریا", "اسٹائپنڈ", "وظیفہ", "فیس"},
    {"editing", "video editing", "davinci", "resolve", "premiere", "cut", "timeline", "keyframe", "color grading", "render", "proxy", "fps", "transitions", "ایڈیٹنگ", "ٹائم لائن", "کی فریم"},
    {"programming", "coding", "code", "software", "developer", "python", "javascript", "api", "bug", "database", "server", "app", "website", "سافٹ ویئر", "کوڈنگ", "ڈویلپر"},
    {"productivity", "discipline", "habit", "habits", "focus", "procrastination", "time management", "routine", "consistency", "goal", "goals", "توجہ", "عادت", "نظم"},
    {"mindset", "psychology", "fear", "confidence", "anxiety", "stress", "happiness", "success", "failure", "mind", "ego", "خوف", "اعتماد", "کامیابی", "ناکاامی"},
    {"career", "job", "resume", "interview", "freelance", "freelancing", "skills", "hiring", "promotion", "remote", "نوکری", "اسکلز", "فری لانسنگ"},
    {"marketing", "social media", "youtube", "tiktok", "instagram", "reels", "shorts", "algorithm", "views", "thumbnail", "hook", "viral", "content", "یوٹیوب", "ویوز", "تھمب نیل"},
    {"politics", "government", "election", "law", "policy", "country", "pakistan", "india", "america", "war", "peace", "حکومت", "سیاست", "الیکشن", "قانون"},
    {"religion", "islam", "islamic", "quran", "faith", "god", "allah", "prayer", "spirituality", "soul", "deen", "bayan", "tafseer", "hadith", "sunnah", "sabr", "tawakkul", "rizq", "dua", "ramzan", "jummah", "اسلام", "قرآن", "دین", "نماز", "بیان", "تفسیر", "حدیث", "صبر", "دعا", "رزق"},
    {"influencing", "influencer", "creator", "personal brand", "pov", "storytelling", "lifestyle", "routine", "advice", "habits", "self improvement", "عادت", "کامیابی", "اعتماد"},
    {"movie", "cinema", "film", "movie shot", "scene", "cinematography", "actor", "director", "dialogue", "drama", "thriller", "twist", "ending", "فلم", "سین", "ڈرامہ", "کردار"},
    {"relationships", "marriage", "family", "love", "friendship", "parenting", "parents", "partner", "shadi", "rishta", "شادی", "خاندان", "محبت"},
    {"science", "physics", "space", "universe", "quantum", "biology", "evolution", "climate", "energy", "earth", "سائنس", "کائنات", "زمین"},
]

STOP_TOKENS = {
    "the", "and", "for", "that", "this", "with", "from", "they", "have", "what", "when", "where", "how",
    "why", "who", "which", "about", "into", "your", "their", "there", "been", "were", "will", "would",
    "could", "should", "just", "like", "very", "more", "most", "some", "such", "only", "also", "than",
    "then", "them", "these", "those", "because", "while", "being", "each", "make", "made", "does",
}


def _stem(word: str) -> str:
    w = word.lower()
    if len(w) <= 4 or not w.isascii():
        return w
    for sfx in ("ization", "ational", "fulness", "iveness", "ousness", "ation", "ments", "ment", "ness", "able", "ible", "ing", "ies", "ied", "ers", "ed", "es", "ly", "s"):
        if w.endswith(sfx) and len(w) - len(sfx) >= 3:
            return w[:-len(sfx)]
    return w


def _trigrams(w: str) -> Set[str]:
    if len(w) < 3:
        return {w}
    return {w[i:i + 3] for i in range(len(w) - 2)}


def parse_user_query(query: str) -> UserIntent:
    """Extracts objective, topic, speaker, duration, preset, aspect ratio, and language from natural instructions."""
    query_clean = (query or "").strip()
    if not query_clean:
        return UserIntent()

    detected_obj = "best_overall"
    for obj, patterns in OBJECTIVE_PATTERNS.items():
        if any(re.search(pat, query_clean, re.I) for pat in patterns):
            detected_obj = obj
            break

    topic = None
    topic_match = re.search(
        r"(?:about|discussing|discusses|topic|regarding|related to|on)\s+['\"]?([^'\",.?!\n]+?)['\"]?(?=\s+(?:in|with|under|from|for)\b|\s+and\s+(?:make|keep|use)\b|$|[,.?!])",
        query_clean,
        re.I,
    )
    if topic_match:
        cand = topic_match.group(1).strip()
        if len(cand) >= 2 and cand.lower() not in {"it", "this", "that", "them", "everything", "clips", "moments", "video", "the video"}:
            topic = cand

    speaker = None
    spk_match = re.search(r"(?:featuring|speaker|when|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:speaks|talks|is|appears|discusses)", query_clean)
    if not spk_match:
        spk_match = re.search(r"(?:featuring|speaker)\s+['\"]?([A-Za-z][A-Za-z\s]{2,25})['\"]?", query_clean, re.I)
    if spk_match:
        speaker = spk_match.group(1).strip()

    target_dur = "medium"
    if re.search(r"\b(short|quick|brief|under 30|bite[- ]sized)\b", query_clean, re.I):
        target_dur = "short"
    elif re.search(r"\b(long|deep|full story|extended|60[- ]90)\b", query_clean, re.I):
        target_dur = "long"

    aspect = "9:16"
    if re.search(r"\b(landscape|horizontal|16:9|wide|widescreen)\b", query_clean, re.I):
        aspect = "16:9"
    elif re.search(r"\b(square|1:1)\b", query_clean, re.I):
        aspect = "1:1"
    elif re.search(r"\b(original aspect|keep aspect|original format)\b", query_clean, re.I):
        aspect = "original"

    caption_lang = "roman_urdu" if re.search(r"\broman\s*urdu\b", query_clean, re.I) else "original"
    if re.search(r"\bno captions|without captions|no subtitles\b", query_clean, re.I):
        caption_lang = "none"

    preset = None
    for p in ["clean", "dynamic", "podcast", "gaming", "educational", "social_media", "islamic", "cinematic"]:
        if re.search(rf"\b{p}\b", query_clean, re.I):
            preset = p
            break
    if preset is None:
        if re.search(r"\b(bayan|tafseer|hadith|quran|deen)\b", query_clean, re.I):
            preset = "islamic"
        elif re.search(r"\b(movie\s*shot|cinema|film\s*scene)\b", query_clean, re.I):
            preset = "cinematic"

    return UserIntent(
        objective=detected_obj,
        topic=topic,
        speaker=speaker,
        target_duration=target_dur,
        preset=preset,
        aspect_ratio=aspect,
        caption_lang=caption_lang,
    )


def get_intent_weights(intent: UserIntent) -> Dict[str, float]:
    """Dynamically re-weights the 9 evaluation dimensions according to the user's objective."""
    weights = dict(DEFAULT_WEIGHTS)
    obj = intent.objective.lower()

    if obj == "funny":
        weights["emotional_value"] = 0.28
        weights["hook_strength"] = 0.22
        weights["engagement_potential"] = 0.16
        weights["content_value"] = 0.08
    elif obj == "educational":
        weights["content_value"] = 0.30
        weights["standalone_clarity"] = 0.22
        weights["context_completeness"] = 0.18
        weights["emotional_value"] = 0.06
    elif obj == "powerful":
        weights["hook_strength"] = 0.26
        weights["content_value"] = 0.24
        weights["engagement_potential"] = 0.18
        weights["emotional_value"] = 0.12
    elif obj == "controversial":
        weights["emotional_value"] = 0.24
        weights["engagement_potential"] = 0.22
        weights["hook_strength"] = 0.20
        weights["content_value"] = 0.12
    elif obj == "story":
        weights["context_completeness"] = 0.28
        weights["standalone_clarity"] = 0.24
        weights["boundary_quality"] = 0.16
        weights["hook_strength"] = 0.12
    elif obj == "emotional":
        weights["emotional_value"] = 0.32
        weights["engagement_potential"] = 0.20
        weights["standalone_clarity"] = 0.16
    elif obj == "discussion":
        weights["content_value"] = 0.22
        weights["context_completeness"] = 0.22
        weights["standalone_clarity"] = 0.18
    elif obj == "islamic":
        weights["context_completeness"] = 0.28
        weights["standalone_clarity"] = 0.24
        weights["content_value"] = 0.24
        weights["boundary_quality"] = 0.12
        weights["emotional_value"] = 0.04
    elif obj == "gaming":
        weights["emotional_value"] = 0.28
        weights["visual_suitability"] = 0.20
        weights["engagement_potential"] = 0.20
        weights["hook_strength"] = 0.18
    elif obj == "movie_shot":
        weights["visual_suitability"] = 0.24
        weights["emotional_value"] = 0.22
        weights["context_completeness"] = 0.20
        weights["boundary_quality"] = 0.16

    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in weights.items()}


def _expand_topic_terms(topic_clean: str) -> Set[str]:
    """Expands a topic string using bilingual concept clusters + stems."""
    topic_tokens = {w for w in re.findall(r"\w+", topic_clean, flags=re.U) if len(w) >= 2 and w not in STOP_TOKENS}
    expanded: Set[str] = set(topic_tokens) | {topic_clean}
    for cluster in TOPIC_CONCEPTS:
        # Match if topic phrase or any meaningful token overlaps a concept cluster
        if topic_clean in cluster or any(t in cluster for t in topic_tokens) or any(_stem(t) in {_stem(c) for c in cluster if " " not in c} for t in topic_tokens):
            expanded.update(cluster)
    return expanded


def calculate_topic_relevance(text: str, topic: Optional[str]) -> float:
    """Computes semantic relevance score [0.0, 1.0] without exact keyword dependency."""
    if not topic or not topic.strip():
        return 1.0

    text_lower = (text or "").lower()
    topic_clean = topic.strip().lower()
    if not text_lower:
        return 0.0

    # Include Romanized phonetic representation when transcript contains Urdu script
    if is_urdu_text(text_lower):
        text_lower = f"{text_lower} {transliterate_urdu_to_roman(text_lower).lower()}"

    # 1. Direct phrase match
    if topic_clean in text_lower:
        return 1.0

    # 2. Concept cluster + multi-word phrase match
    expanded = _expand_topic_terms(topic_clean)
    phrase_hits = sum(1 for term in expanded if " " in term and term in text_lower)

    raw_words = [w for w in re.findall(r"\w+", text_lower, flags=re.U) if len(w) >= 2 and w not in STOP_TOKENS]
    if not raw_words:
        return 0.0
    word_set = set(raw_words)
    stem_set = {_stem(w) for w in word_set}

    single_expanded = {t for t in expanded if " " not in t}
    exp_stems = {_stem(t) for t in single_expanded}

    exact_hits = len(word_set & single_expanded)
    stem_hits = len(stem_set & exp_stems)

    # 3. Fuzzy character trigram match for unseen domain vocabulary
    fuzzy_hits = 0.0
    query_tokens = [t for t in re.findall(r"\w+", topic_clean, flags=re.U) if len(t) >= 4 and t not in STOP_TOKENS]
    if query_tokens and (exact_hits + stem_hits + phrase_hits) == 0:
        for qt in query_tokens:
            q_tri = _trigrams(qt)
            best_sim = 0.0
            for w in word_set:
                if len(w) < 4:
                    continue
                w_tri = _trigrams(w)
                sim = len(q_tri & w_tri) / max(1, len(q_tri | w_tri))
                if sim > best_sim:
                    best_sim = sim
            if best_sim >= 0.45:
                fuzzy_hits += best_sim

    total_signal = phrase_hits * 2.0 + exact_hits * 1.1 + max(0, stem_hits - exact_hits) * 0.8 + fuzzy_hits
    if total_signal >= 2.8:
        return 0.96
    if total_signal >= 1.8:
        return 0.84
    if total_signal >= 0.9:
        return 0.62
    if total_signal >= 0.45:
        return 0.38
    return 0.08


def calculate_speaker_relevance(text: str, speaker: Optional[str]) -> float:
    """Checks if a target speaker/creator is mentioned or quoted in the text."""
    if not speaker or not speaker.strip():
        return 1.0
    low = (text or "").lower()
    tokens = [t.lower() for t in re.findall(r"\w+", speaker) if len(t) >= 2]
    if not tokens:
        return 1.0
    if speaker.strip().lower() in low:
        return 1.0
    hits = sum(1 for t in tokens if t in low)
    return min(1.0, hits / len(tokens)) if hits else 0.2


def generate_adaptive_questions(metadata: Dict[str, Any], classification: Optional[ContentClassification],
                                user_query: str) -> List[Dict[str, Any]]:
    """Generates 1-2 targeted questions ONLY when user intent is underspecified."""
    intent = parse_user_query(user_query)

    if intent.objective != "best_overall" or intent.topic is not None or intent.speaker is not None:
        return []

    questions = []
    genre_hint = classification.genre.replace("_", " ") if classification else "general"
    dur = float(metadata.get("duration") or 0)

    questions.append({
        "question": f"What kind of moments should ClipMind extract from this {genre_hint} video?",
        "options": [
            f"(Recommended) Best highlights overall for {genre_hint}",
            "Educational explanations and key takeaways",
            "Funny and entertaining reactions",
            "A specific topic or person (please name it)",
        ],
    })

    if dur == 0 or dur >= 90:
        questions.append({
            "question": "Preferred clip length and format?",
            "options": [
                "(Recommended) 9:16 Vertical Reels/Shorts (30-60s)",
                "Short punchy vertical clips (15-30s)",
                "Longer complete-story vertical clips (60-90s)",
                "Original aspect ratio / 16:9 landscape",
            ],
        })

    return questions
