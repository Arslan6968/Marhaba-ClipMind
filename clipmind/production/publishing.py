"""Content Publishing Package Generator.
Produces grounded titles, descriptions, captions, hashtags, and A/B hooks from actual spoken content.
Never promises virality, guaranteed views, or guaranteed engagement.
"""
import re
from typing import List, Optional

from ..schemas import ClipCandidate, PublishingPackage, ContentClassification
from ..understand.urdu import is_urdu_text, transliterate_urdu_to_roman

GENRE_TAGS = {
    "gaming": ["#gaming", "#clutch", "#gameplay", "#highlights", "#gamer"],
    "islamic": ["#islamic", "#bayan", "#islamicreminder", "#deen", "#quran"],
    "influencing": ["#mindset", "#motivation", "#creator", "#growth", "#perspective"],
    "movie_shot": ["#cinema", "#moviescene", "#film", "#dialogue", "#cinematography"],
    "podcast": ["#podcast", "#interview", "#conversation", "#perspective", "#insights"],
    "interview": ["#interview", "#dialogue", "#storytime", "#discussion", "#insights"],
    "tutorial": ["#tutorial", "#howto", "#editing", "#guide", "#tips"],
    "lecture_education": ["#education", "#learning", "#explainer", "#knowledge", "#study"],
    "speech_presentation": ["#keynote", "#speech", "#presentation", "#leadership", "#ideas"],
    "news_discussion": ["#news", "#analysis", "#discussion", "#update", "#perspective"],
    "debate_conversation": ["#debate", "#discussion", "#perspective", "#conversation", "#opinions"],
    "documentary": ["#documentary", "#history", "#story", "#facts", "#discovery"],
    "vlog": ["#vlog", "#storytime", "#journey", "#daily", "#lifestyle"],
    "entertainment": ["#entertainment", "#reaction", "#comedy", "#moments", "#clips"],
    "general": ["#shorts", "#reels", "#clips", "#video", "#highlights"],
}

STOP_WORDS = {
    "this", "that", "what", "with", "from", "they", "have", "were", "when", "where",
    "which", "there", "their", "about", "because", "would", "could", "should", "just",
    "really", "actually", "going", "think", "know", "said", "says", "something", "thing",
}


LEADING_CONJ = {"but", "and", "so", "because", "or", "then", "لیکن", "اور", "تو", "مگر", "کیونکہ", "پھر"}


GENRE_CTAS = {
    "lecture_education": "Save this breakdown for reference and ask your questions in the comments.",
    "tutorial": "Bookmark this guide and comment which step helped most.",
    "podcast": "Do you agree with this perspective? Share your thoughts below.",
    "interview": "What stood out most from this answer? Let's discuss below.",
    "debate_conversation": "Which side makes the stronger case? Weigh in below.",
    "news_discussion": "What is your take on this update? Share in the comments.",
    "gaming": "Drop your reaction below and share with a squad mate.",
    "islamic": "JazakAllah Khair — save & share this reminder with others.",
    "influencing": "Save this if it resonated — which point hit hardest?",
    "movie_shot": "Which film or scene does this remind you of? Drop it below.",
}


def _clean_title(text: str) -> str:
    norm = re.sub(r"\s*/\s*", " & ", text or "")
    cleaned = re.sub(r"[^\w\s&?؟!%]", "", norm, flags=re.U).strip()
    words = cleaned.split()
    while len(words) > 2 and words[0].lower() in LEADING_CONJ:
        words.pop(0)
    if len(words) > 8:
        return " ".join(words[:8])
    return " ".join(words) or "Key Discussion Moment"


def generate_publishing_package(clip: ClipCandidate, genre_info: ContentClassification,
                                creator_name: Optional[str] = None,
                                caption_lang: str = "original") -> PublishingPackage:
    """Creates a grounded publishing package from spoken words and evaluation scores."""
    genre = genre_info.genre
    tx_for_tags = transliterate_urdu_to_roman(clip.transcript_text) if is_urdu_text(clip.transcript_text) else clip.transcript_text
    tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9]{4,}", f"{clip.title} {tx_for_tags}") if w.lower() not in STOP_WORDS]

    title = _clean_title(clip.title or clip.event_text or clip.hook_text)
    headline = _clean_title(clip.headline or title)

    hook_clean = (clip.hook_text or title).replace("*", "").strip()
    event_clean = (clip.event_text or "").strip()
    if caption_lang == "roman_urdu":
        if is_urdu_text(hook_clean):
            hook_clean = transliterate_urdu_to_roman(hook_clean)
        if is_urdu_text(event_clean):
            event_clean = transliterate_urdu_to_roman(event_clean)
    desc = f"{genre.replace('_', ' ').title()} excerpt: \"{hook_clean}\""
    if event_clean and event_clean != hook_clean:
        desc += f" — {event_clean}"

    cta = GENRE_CTAS.get(genre, "Save this highlight and share your thoughts below.")
    caption = f"{title}\n\n\"{hook_clean}\"\n\n{cta}"

    tags = list(GENRE_TAGS.get(genre, GENRE_TAGS["general"]))
    for t in tokens:
        tag = f"#{t}"
        if tag not in tags and len(tags) < 7:
            tags.append(tag)
    if creator_name:
        ctag = "#" + re.sub(r"[^a-z0-9]", "", creator_name.lower())
        if len(ctag) >= 3 and ctag not in tags:
            tags.append(ctag)

    raw_ab = list(clip.ab_hooks) if clip.ab_hooks else [hook_clean]
    ab_hooks = [
        (transliterate_urdu_to_roman(h) if (caption_lang == "roman_urdu" and is_urdu_text(h)) else h)
        for h in raw_ab
    ]
    if len(ab_hooks) < 2 and event_clean:
        ab_hooks.append(event_clean[:120])

    rationale = (
        f"Score {clip.overall_score:.1f}/10 ({clip.picked_by}). "
        f"Hook {clip.dimensions.hook_strength:.1f}/10, Content {clip.dimensions.content_value:.1f}/10, "
        f"Standalone clarity {clip.dimensions.standalone_clarity:.1f}/10, Context {clip.dimensions.context_completeness:.1f}/10. "
        f"{clip.rationale}"
    ).strip()

    return PublishingPackage(
        title=title,
        headline=headline,
        description=desc[:260],
        caption=caption,
        hashtags=tags[:8],
        category=genre,
        duration_sec=clip.duration,
        quality_score=clip.overall_score,
        rationale=rationale,
        ab_hooks=ab_hooks[:2],
    )
