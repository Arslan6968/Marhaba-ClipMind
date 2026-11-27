"""Urdu, Code-Switching (Urdu+English), and Roman Urdu Transliteration utilities.
Adapted and unified from Marhaba Captions for clean multilingual video subtitling.
"""
import re
from typing import Dict, Optional

URDU_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
ALLOWED_TEXT_RE = re.compile(r"[^\u0020-\u007E\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u200C-\u200F\u202A-\u202E\n\t]")
LTR_SPAN_RE = re.compile(r"[A-Za-z0-9]+(?:[%./+\-:][A-Za-z0-9]+)*%?|%")


def sanitize_transcript_text(text: str) -> str:
    """Strips stray non-Urdu/non-Latin hallucinated characters (e.g. Hebrew/CJK) that render as tofu boxes."""
    if not text:
        return text
    cleaned = ALLOWED_TEXT_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()

CODE_SWITCHING_PROMPT = (
    "اردو اور English میں گفتگو۔ Pakistani Urdu code-switching conversation with English words: "
    "Software, Application, Computer, Settings, Export, Button, Click, AI, Video, MP4, Audio, "
    "Project, Studio, Timeline, Screen, YouTube, Online, Keyframe, Render, Edit, Channel."
)

CODE_SWITCHING_MAP: Dict[str, str] = {
    "سافٹ ویئر": "Software", "سافٹوئیر": "Software", "سافٹ ویر": "Software", "سافٹویئر": "Software",
    "ایپلیکیشن": "Application", "اپلیکیشن": "Application", "کمپیوٹر": "Computer",
    "سیٹنگز": "Settings", "سیٹنگ": "Setting", "ایکسپورٹ": "Export", "امپورٹ": "Import",
    "بٹن": "Button", "کلک": "Click", "ڈاؤنلوڈ": "Download", "ڈاؤن لوڈ": "Download",
    "انسٹال": "Install", "ویڈیو": "Video", "آڈیو": "Audio", "پروجیکٹ": "Project",
    "ٹول": "Tool", "فائل": "File", "فولڈر": "Folder", "آن لائن": "Online", "آنلائن": "Online",
    "آف لائن": "Offline", "اے آئی": "AI", "ایم پی فور": "MP4", "اسکرین": "Screen",
    "اسکرین شاٹ": "Screenshot", "ٹائم لائن": "Timeline", "فارمیٹ": "Format", "سلیکٹ": "Select",
    "ایڈٹ": "Edit", "ایڈیٹر": "Editor", "ایڈیٹنگ": "Editing", "اسٹیپ": "Step", "سسٹم": "System",
    "پاورفل": "Powerful", "ڈویلپ": "Develop", "ڈویلپر": "Developer", "ڈویلپمنٹ": "Development",
    "ٹیمپلیٹ": "Template", "ٹیمپلیٹس": "Templates", "موبائل": "Mobile", "انٹرنیٹ": "Internet",
    "ویب سائٹ": "Website", "یوٹیوب": "YouTube", "فیس بک": "Facebook", "انسٹاگرام": "Instagram",
    "واٹس ایپ": "WhatsApp", "اکاؤنٹ": "Account", "پاس ورڈ": "Password", "لنک": "Link",
    "میسج": "Message", "ای میل": "Email", "کیمرہ": "Camera", "ریکارڈ": "Record",
    "ریکارڈنگ": "Recording", "مائیک": "Mic", "مائیکروفون": "Microphone", "چیک": "Check",
    "ٹیسٹ": "Test", "اپ ڈیٹ": "Update", "اپڈیٹ": "Update", "فیچر": "Feature", "فیچرز": "Features",
    "ڈیٹا": "Data", "لوکیشن": "Location", "نوٹیفکیشن": "Notification", "سرچ": "Search",
    "شیئر": "Share", "کاپی": "Copy", "پیسٹ": "Paste", "سیو": "Save", "کینسل": "Cancel",
    "اوکے": "OK", "ڈیلیٹ": "Delete", "کوالٹی": "Quality", "ریزولوشن": "Resolution",
    "براؤزر": "Browser", "ونڈوز": "Windows", "اینڈرائیڈ": "Android", "آئی فون": "iPhone",
    "لیپ ٹاپ": "Laptop", "کی بورڈ": "Keyboard", "ماؤس": "Mouse", "ڈسپلے": "Display",
    "گرافکس": "Graphics", "ڈیزائن": "Design", "لوگو": "Logo", "بینر": "Banner",
    "پلیئر": "Player", "چینل": "Channel", "سبسکرائب": "Subscribe", "لائک": "Like",
    "کمنٹ": "Comment", "ویوز": "Views", "فالورز": "Followers", "ہوم": "Home", "پیج": "Page",
    "ٹیکسٹ": "Text", "فونٹ": "Font", "سائز": "Size", "کلر": "Color", "بیک گراؤنڈ": "Background",
    "ایفیکٹ": "Effect", "اینیمیشن": "Animation", "اسٹائل": "Style", "شارٹ کٹ": "Shortcut",
    "کیپشن": "Caption", "کیپشنز": "Captions", "ٹائٹل": "Title", "سب ٹائٹل": "Subtitle",
    "زوم": "Zoom", "اسکرول": "Scroll", "پلے": "Play", "پاز": "Pause", "اسٹاپ": "Stop",
    "اسٹارٹ": "Start", "نیکسٹ": "Next", "بیک": "Back", "رینڈر": "Render", "پروکسی": "Proxy",
    "کی فریم": "Keyframe", "ٹائم": "Time", "کلپ": "Clip", "کلپس": "Clips",
    "اللیجبلٹی": "Eligibility", "کرایٹریا": "Criteria", "کلوزیگ": "Closing", "کلوزنگ": "Closing",
    "میرٹ": "Merit", "لیسٹ": "List", "لسٹ": "List", "اپلائے": "Apply", "اپلای": "Apply",
    "اسٹائپنڈ": "Stipend", "اسٹائپینڈ": "Stipend", "ہواستیپن": "hua Stipend", "اسٹائپن": "Stipend",
    "انفارمیشن": "Information", "کمپلیٹ": "Complete",
    "کمپریٹ": "Complete", "انستیٹوڑ": "Institutes", "اپلکیشنسے": "Applications",
    "بیس نرسین": "BS Nursing", "بیس نرسنگ": "BS Nursing", "دیرسیم": "BS Nursing",
    "بیس": "BS", "بیاسو": "BS", "نرسین": "Nursing", "نرسنگ": "Nursing",
    "گرمن": "Govt", "گورنمنٹ": "Govt", "سکٹر": "Sector", "سیکٹر": "Sector",
    "ادمییشن": "Admission", "ادمیشن": "Admission", "ایڈمیشن": "Admission",
    "اپن": "Open", "اوپن": "Open", "دیٹ": "Date", "ڈیٹ": "Date",
    "میرٹلس": "Merit List", "میرٹ لسٹ": "Merit List", "میرد": "Merit",
    "توٹل": "Total", "ٹوٹل": "Total", "سیٹے": "Seats", "سیٹیں": "Seats", "سیٹس": "Seats", "سیٹوں": "Seaton",
    "میل": "Male", "فیمیل": "Female", "پی میل": "Female",
    "مارنی": "Morning", "مارننگ": "Morning", "ایمنی": "Evening", "ایوننگ": "Evening", "امین": "Evening",
    "داکومنس": "Documents", "ڈاکومنٹس": "Documents", "دگری": "Degree", "ڈگری": "Degree",
    "پرورا": "Program", "پروگرام": "Program", "وڈیوں": "Video", "وڈیو": "Video",
    "دیسکس": "Discuss", "ڈسکس": "Discuss", "بوائز": "Boys", "بائز": "Boys",
    "گلس": "Girls", "گرلز": "Girls", "پلس": "Plus", "نمبر": "Number",
    "مدکل": "Medical", "میڈیکل": "Medical", "داستیر": "Last Year", "جسٹ": "Just", "جست": "Just", "اپلائی": "Apply",
}

ACRONYMS = {
    "ai", "mp4", "api", "ui", "ux", "hd", "4k", "8k", "url", "id", "cpu", "gpu", "ram",
    "pc", "tv", "v1", "v2", "seo", "pdf", "txt", "srt", "ass", "vtt", "os", "ip", "fps",
}

RMAP = {
    "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch",
    "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh",
    "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f",
    "ق": "q", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o", "ہ": "h",
    "ھ": "h", "ء": "", "ی": "i", "ے": "e", "ئ": "i", "ۂ": "h", "ۓ": "e", "َ": "a", "ِ": "i",
    "ُ": "u", "ً": "an", "ّ": "", "ْ": "", "،": ",", "؟": "?", "۔": ".", "ك": "k", "ي": "i",
}

ROMAN_WORDS = {
    "نہیں": "nahi", "ہے": "hai", "ہیں": "hain", "میں": "mein", "آپ": "aap", "کی": "ki", "کا": "ka", "کے": "ke",
    "یہ": "yeh", "وہ": "woh", "اور": "aur", "کیا": "kya", "نہ": "na", "تو": "toh", "ہو": "ho", "گا": "ga",
    "گی": "gi", "گے": "ge", "گئی": "gayi", "گیا": "gaya", "ہوا": "hua", "ہوئی": "hui", "ہوئے": "hue",
    "کر": "kar", "کریں": "karein", "کرتے": "karte", "رہا": "raha", "رہی": "rahi", "رہے": "rahe",
    "لیے": "liye", "لیکن": "lekin", "مگر": "magar", "کیونکہ": "kyunke", "بہت": "bohot", "سب": "sab",
    "اچھا": "acha", "اچھی": "achi", "جی": "ji", "شکریہ": "shukriya", "خدا": "khuda", "حافظ": "hafiz",
    "کیسے": "kaise", "کہاں": "kahan", "کب": "kab", "کون": "kaun", "کیوں": "kyun", "مجھے": "mujhe",
    "تم": "tum", "تمہارا": "tumhara", "میرا": "mera", "میری": "meri", "ہمارا": "hamara", "ان": "un",
    "اس": "is", "انہیں": "unhein", "اپنے": "apne", "اپنی": "apni", "اپنا": "apna", "ساتھ": "saath",
    "صرف": "sirf", "بھی": "bhi", "اب": "ab", "پھر": "phir", "یہاں": "yahan", "وہاں": "wahan",
    "دنیا": "duniya", "زندگی": "zindagi", "دل": "dil", "لوگ": "log", "بات": "baat", "وقت": "waqt",
    "آج": "aaj", "کل": "kal", "ضروری": "zaroori", "زبردست": "zabardast", "پہلے": "pehle", "بعد": "baad",
    "کوئی": "koi", "کچھ": "kuch", "بنا": "bana", "دیکھیں": "dekhein", "دیکھو": "dekho", "سمجھ": "samajh",
    "شروع": "shuru", "ختم": "khatam", "کام": "kaam", "پیسہ": "paisa", "طریقہ": "tareeqa", "مثال": "misal",
    "کس": "kis", "کیلے": "ke liye", "گتنی": "kitni", "کتنی": "kitni", "کتنے": "kitne", "کتھنے": "kitne",
    "کرنے": "karne", "کرتا": "karta", "لگے": "lage", "ہوگا": "hoga", "ہوگے": "honge", "پر": "par",
    "ہم": "hum", "کر": "kar", "جو": "jo", "سکتے": "sakte", "سکتا": "sakta", "انتظار": "intezar",
    "گھڑیاں": "ghariyan", "گڑیا": "ghariyan", "ختموی": "khatam hui", "ادارے": "idaaray", "ایدارے": "idaaray",
    "چکا": "chuka", "اون": "un", "لگبگ": "lagbhag", "چند": "chand", "کم": "kam", "زیادہ": "ziada",
    "ملے": "mile", "میلے": "mile", "ازافا": "izafa", "اضافہ": "izafa", "ایسے": "aise", "اسی": "isi",
    "آیا": "aaya", "یا": "ya", "تھا": "tha", "تھی": "thi", "سمجھا": "samjha", "جائیں": "jayein",
    "کہتے": "kehte", "لہذا": "lihaza", "اگر": "agar", "جن": "jin", "کروں": "karoon", "ہوں": "hoon",
    "کہ": "ke", "بار": "baar", "اوپر": "upar", "نیچے": "neeche", "لوگ": "log", "سے": "se",
    "کو": "ko", "نے": "ne", "جیگے": "jayenge", "دیگے": "denge", "جیے": "jiye",
    "سٹھنے": "students", "پچھنے": "poochte",
}


def is_urdu_text(text: str) -> bool:
    return bool(URDU_CHAR_RE.search(text or ""))


def format_urdu_rtl_ass(text: str) -> str:
    """Wraps Urdu strings with Unicode RLE/PDF & RLM marks so libass lays out words strictly R-to-L."""
    if not text or not is_urdu_text(text):
        return text
    cleaned = text.strip("\u202a\u202b\u202c\u202d\u202e\u200e\u200f ").replace("\u202b", "").replace("\u202c", "").replace("\u200f", "")
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+%", r"\1%", cleaned)
    anchored = LTR_SPAN_RE.sub(lambda m: f"\u200f{m.group(0)}\u200f", cleaned)
    return f"\u202b{anchored}\u202c"


def build_domain_asr_prompt(
    title: Optional[str] = None,
    topic: Optional[str] = None,
    channel_name: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Builds a concise video-specific initial_prompt so Whisper anchors to domain terms instead of generic words."""
    parts = []
    if channel_name:
        parts.append(channel_name.strip())
    if title:
        clean_title = re.sub(r"\s+", " ", title.strip())[:110]
        parts.append(clean_title)
    if topic:
        clean_topic = re.sub(r"\s+", " ", topic.strip())[:80]
        parts.append(clean_topic)
    domain_ctx = " | ".join(p for p in parts if p)
    if language in ("ur", "hi") or (domain_ctx and is_urdu_text(domain_ctx)):
        base = "اردو اور English گفتگو۔ " + (f"موضوع: {domain_ctx}. " if domain_ctx else "")
        return (base + "Merit, Admission, Eligibility, Seats, Apply, Last Date, Percent, Tutorial, System, Video.").strip()[:220]
    if domain_ctx:
        return f"Context: {domain_ctx}. Clear punctuation and technical terminology."[:220]
    return CODE_SWITCHING_PROMPT[:220]


def format_code_switching(text: str) -> str:
    """Normalizes common English technical terms written in Urdu script inside mixed transcripts."""
    if not text:
        return text
    text = sanitize_transcript_text(text)
    if not is_urdu_text(text):
        return text
    result = text
    for urdu_term, english_term in sorted(CODE_SWITCHING_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(rf"(?<!\S){re.escape(urdu_term)}(?!\S)")
        result = pattern.sub(english_term, result)

    def _fmt_en(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower()
        if low in ACRONYMS:
            return word.upper()
        return word

    return re.sub(r"\b[A-Za-z][A-Za-z0-9]*\b", _fmt_en, result)


def transliterate_urdu_to_roman(text: str) -> str:
    """Converts Urdu script into readable Roman Urdu while preserving Latin/English words."""
    if not text:
        return text
    # Normalize common multi-word Whisper Urdu compounds before token transliteration
    norm_text = text
    for pat, rep in (("بیس نرسین", "BS Nursing"), ("بیس نرسنگ", "BS Nursing")):
        norm_text = norm_text.replace(pat, rep)
    tokens = re.split(r"(\s+|[.,!?؟؛۔،;:()\"'…])", norm_text)
    out = []
    for tok in tokens:
        if not tok or not tok.strip():
            out.append(tok)
            continue
        if not is_urdu_text(tok):
            out.append({"؟": "?", "۔": ".", "،": ",", "؛": ";"}.get(tok, tok))
            continue
        if tok in CODE_SWITCHING_MAP:
            out.append(CODE_SWITCHING_MAP[tok])
        elif tok in ROMAN_WORDS:
            out.append(ROMAN_WORDS[tok])
        else:
            mapped = "y" + "".join(RMAP.get(c, c) for c in tok[1:]) if tok.startswith("ی") and len(tok) > 1 else "".join(RMAP.get(c, c) for c in tok)
            out.append(mapped)
    return "".join(out)
