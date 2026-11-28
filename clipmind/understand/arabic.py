"""Arabic & Qur'anic Ayat Detection and Respectful Caption Management.
Detects Arabic recitation / Qur'anic Ayat & Hadith quotes in spoken transcripts so they are
never burned onto video as potentially mis-transcribed captions, while protecting Ayat boundaries.
"""
import re
from typing import List, Tuple

ARABIC_BLOCK_RE = re.compile(r"[\u0600-\u06FF]")
ARABIC_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670]")
# Letters used in Arabic script that are absent in standard Urdu orthography
ARABIC_ONLY_CHAR_RE = re.compile(r"[\u0623\u0625\u0629\u0649]")  # أ إ ة ى
# Letters used in Urdu that do not occur in Classical Arabic
URDU_ONLY_CHAR_RE = re.compile(r"[\u0679\u067E\u0686\u0688\u0691\u0698\u06AF\u06BA\u06BE\u06C1\u06D2]")

ARABIC_QURAN_WORDS = {
    "بسم", "الله", "لله", "الرحمن", "الرحيم", "الحمد", "رب", "العالمين", "مالك", "يوم", "الدين",
    "إياك", "نعبد", "نستعين", "اهدنا", "الصراط", "المستقيم", "قل", "هو", "أحد", "احد", "الصمد",
    "لم", "يلد", "يولد", "كفوا", "يا", "أيها", "ايها", "الذين", "آمنوا", "امنوا", "إن", "ان",
    "أن", "على", "في", "من", "إلى", "الى", "عن", "الذي", "التي", "قال", "تعالى", "سبحان",
    "سبحانه", "صلى", "عليه", "وسلم", "أعوذ", "اعوذ", "بالله", "الشيطان", "الرجيم", "ربنا",
    "آتنا", "لا", "إله", "اله", "إلا", "الا", "محمد", "رسول", "صدق", "العظيم", "تبارك",
    "الم", "ذلك", "الكتاب", "ريب", "فيه", "هدى", "للمتقين", "أولئك", "اولئك", "وما", "بما",
    "كان", "كانوا", "والله", "يعلم", "تعملون", "آمين", "امين", "عزوجل", "جل", "جلاله",
}


def _strip_tashkeel(text: str) -> str:
    return ARABIC_TASHKEEL_RE.sub("", text or "")


def is_arabic_ayat_text(text: str) -> bool:
    """Returns True when text is predominantly Arabic / Qur'anic Ayat recitation rather than Urdu/English."""
    if not text or not ARABIC_BLOCK_RE.search(text):
        return False

    tashkeel_hits = len(ARABIC_TASHKEEL_RE.findall(text))
    arabic_only_hits = len(ARABIC_ONLY_CHAR_RE.findall(text))
    urdu_only_hits = len(URDU_ONLY_CHAR_RE.findall(text))

    plain = _strip_tashkeel(text)
    tokens = [re.sub(r"[^\w]", "", t, flags=re.U) for t in plain.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return False

    cue_hits = sum(1 for t in tokens if t in ARABIC_QURAN_WORDS)
    urdu_tok_hits = sum(1 for t in tokens if URDU_ONLY_CHAR_RE.search(t))

    # Strong diacritic or Arabic-orthography signal without Urdu letters
    if tashkeel_hits >= 2 and urdu_tok_hits == 0:
        return True
    if arabic_only_hits >= 2 and urdu_tok_hits == 0:
        return True
    # Pure Arabic Qur'an/Dua vocabulary even when Whisper omits diacritics
    if cue_hits >= 2 and urdu_tok_hits == 0:
        return True
    if len(tokens) >= 2 and (cue_hits / len(tokens)) >= 0.55 and urdu_tok_hits <= 1:
        return True
    if urdu_only_hits == 0 and arabic_only_hits >= 1 and cue_hits >= 1:
        return True
    return False


def split_ayat_vs_speech(text: str) -> Tuple[bool, str]:
    """Identifies if a line is an Arabic Ayat; callers use this to manage/suppress caption burning."""
    return is_arabic_ayat_text(text), (text or "").strip()
