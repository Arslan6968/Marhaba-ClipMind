"""Max-Level ASS Subtitle Generator: Native Urdu RTL, Roman Urdu Transliteration,
Word-Level Karaoke Synchronization, and *Accent* Hook Cards.
"""
import re
from typing import List, Dict, Any, Optional

from ..schemas import WordToken
from ..understand.urdu import is_urdu_text, transliterate_urdu_to_roman, sanitize_transcript_text, format_urdu_rtl_ass
from ..understand.arabic import is_arabic_ayat_text
from .presets import get_preset
from .motion import (
    motion_styles_block,
    build_animated_hook_events,
    build_lower_third_events,
    build_stat_callout_events,
    spring_caption_tag,
)

URDU_REGEX = re.compile(r"[\u0600-\u06FF]")


def _format_time(sec: float) -> str:
    sec = max(0.0, sec)
    cs = int(round(sec * 100))
    h = cs // 360000
    m = (cs // 6000) % 60
    s = (cs // 100) % 60
    c = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _escape_ass(text: str) -> str:
    clean = sanitize_transcript_text(text)
    return clean.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def _format_rich_hook(text: str, accent_bgr: str = "&H003B3BFF&", base_bgr: str = "&H00FFFFFF&") -> str:
    """Converts *accented words* into inline ASS color tags for LTR; preserves unbroken RTL for Urdu."""
    cleaned = _escape_ass(text)
    if is_urdu_text(cleaned):
        return format_urdu_rtl_ass(cleaned.replace("*", ""))
    parts = re.split(r"(\*[^*]+\*)", cleaned)
    out = []
    for p in parts:
        if p.startswith("*") and p.endswith("*") and len(p) > 2:
            inner = p[1:-1].strip()
            out.append(f"{{\\c{accent_bgr}}}{inner}{{\\c{base_bgr}}}")
        elif p:
            out.append(p)
    return "".join(out)


def group_words_into_lines(words: List[WordToken], clip_start: float, clip_end: float,
                           max_words: int = 4, max_chars: int = 28, max_gap: float = 0.7) -> List[Dict[str, Any]]:
    """Groups word-level tokens into readable mobile reading lines."""
    clip_words = [w for w in words if w.e > clip_start and w.s < clip_end]
    groups = []
    current: List[WordToken] = []

    def flush():
        nonlocal current
        if current:
            groups.append({
                "start": max(clip_start, current[0].s),
                "end": min(clip_end, current[-1].e),
                "words": list(current),
            })
            current = []

    for w in clip_words:
        if current:
            gap = w.s - current[-1].e
            char_count = sum(len(x.w) + 1 for x in current) + len(w.w)
            has_punct = current[-1].w.endswith((".", "!", "?", "۔", "؟"))
            if len(current) >= max_words or gap > max_gap or char_count > max_chars or has_punct:
                flush()
        current.append(w)
    flush()
    return groups


def generate_ass_subtitles(
    words: List[WordToken],
    clip_start: float,
    clip_end: float,
    preset_name: str = "clean",
    hook_card_text: Optional[str] = None,
    width: int = 1080,
    height: int = 1920,
    caption_lang: str = "original",
    hook_y: Optional[int] = None,
    margin_v_override: Optional[int] = None,
    creator_name: Optional[str] = None,
    topic_label: Optional[str] = None,
    motion_graphics: bool = True,
) -> str:
    """Produces a complete ASS subtitle script with fontsdir-compatible Urdu RTL, Roman Urdu, karaoke, and motion graphics."""
    preset = get_preset(preset_name)
    cap = preset["captions"]
    k = height / 1920.0
    max_words = int(cap.get("max_words", 4))

    if caption_lang == "none":
        groups: List[Dict[str, Any]] = []
    else:
        raw_groups = group_words_into_lines(words, clip_start, clip_end, max_words=max_words)
        groups = []
        for g in raw_groups:
            orig_line = " ".join(w.w for w in g["words"])
            # Never burn potentially mis-transcribed Arabic / Qur'anic Ayat as captions
            if is_arabic_ayat_text(orig_line):
                continue
            if caption_lang == "roman_urdu":
                rom_words: List[WordToken] = []
                for w in g["words"]:
                    rom_w = transliterate_urdu_to_roman(w.w)
                    if rom_w == "%" and rom_words and re.search(r"\d$", rom_words[-1].w):
                        prev = rom_words[-1]
                        rom_words[-1] = WordToken(w=f"{prev.w}%", s=prev.s, e=max(prev.e, w.e), p=prev.p)
                    else:
                        rom_words.append(WordToken(w=rom_w, s=w.s, e=w.e, p=w.p))
                g = dict(g, words=rom_words)
            groups.append(g)

        if hook_card_text and is_arabic_ayat_text(hook_card_text):
            hook_card_text = None
        elif hook_card_text and caption_lang == "roman_urdu" and is_urdu_text(hook_card_text):
            hook_card_text = transliterate_urdu_to_roman(hook_card_text)

    all_text = " ".join(w.w for g in groups for w in g["words"])
    is_urdu = bool(URDU_REGEX.search(all_text))

    # Noto Nastaliq Urdu has tall vertical line-box metrics, so scale up 1.22x for mobile readability
    font_size = int(cap["size"] * k * (1.22 if is_urdu else 1.0))
    bold_flag = 0 if is_urdu else cap["bold"]
    font_family = "Noto Nastaliq Urdu" if is_urdu else "Inter"
    hook_font = "Noto Nastaliq Urdu" if is_urdu_text(hook_card_text or "") else "Inter"

    margin_v = int((margin_v_override if margin_v_override is not None else cap.get("margin_v", 360)) * k)
    if height <= 1080:  # landscape 16:9 or 1:1 square
        margin_v = int(height * 0.08)
        font_size = max(36, int(font_size * 0.85))
    margin_lr = int(90 * k)

    h_y = hook_y if hook_y is not None else int(150 * k)
    hook_size = int(46 * k * (1.18 if is_urdu_text(hook_card_text or "") else 1.0))

    # ASS colors are &HAABBGGRR (00 = opaque, FF = transparent)
    active_c = cap.get("active", "&H0000FFFF")
    inactive_c = cap.get("inactive", "&H00FFFFFF")
    border_style = int(cap.get("border_style", 1))
    outline_px = int(cap["outline"] * k) if border_style == 1 else int(14 * k)
    outline_c = "&H00000000" if border_style == 1 else "&H200F172A"
    back_c = "&H64000000"

    hook_border = 1 if motion_graphics else 3
    hook_outline = int(2 * k) if motion_graphics else int(16 * k)
    hook_outline_c = "&H80000000" if motion_graphics else "&H200F172A"
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Default,{font_family},{font_size},{active_c},{inactive_c},{outline_c},{back_c},{bold_flag},0,0,0,100,100,0,0,{border_style},{outline_px},{int(cap['shadow'] * k)},2,{margin_lr},{margin_lr},{margin_v},1",
        f"Style: HookCard,{hook_font},{hook_size},&H00FFFFFF,&H00FFFFFF,{hook_outline_c},{back_c},1,0,0,0,100,100,0,0,{hook_border},{hook_outline},1,8,{margin_lr},{margin_lr},{h_y},1",
        *(motion_styles_block(width, height, accent_bgr=active_c).splitlines() if motion_graphics else []),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    clip_dur = max(1.0, clip_end - clip_start)
    if hook_card_text:
        if motion_graphics:
            lines.extend(build_animated_hook_events(
                hook_card_text,
                duration=clip_dur,
                width=width,
                height=height,
                hook_y=h_y,
                accent_bgr="&H003B3BFF&",
                stay_full=bool(preset.get("hook_card_frame")),
            ))
        else:
            card_end = max(2.0, min(clip_dur, 6.5 if not preset.get("hook_card_frame") else clip_dur))
            rich_hook = _format_rich_hook(hook_card_text, accent_bgr="&H003B3BFF&", base_bgr="&H00FFFFFF&")
            lines.append(
                f"Dialogue: 1,{_format_time(0.0)},{_format_time(card_end)},HookCard,,0,0,0,,{rich_hook}"
            )

    if motion_graphics and (creator_name or topic_label):
        lt_y = int(height * 0.66) if height > 1080 else int(height * 0.72)
        lines.extend(build_lower_third_events(
            creator_name=creator_name,
            topic_label=topic_label,
            duration=clip_dur,
            width=width,
            height=height,
            y_pos=lt_y,
        ))
    if motion_graphics and words:
        st_y = int(height * 0.66) if height > 1080 else int(height * 0.72)
        lines.extend(build_stat_callout_events(
            words=words,
            clip_start=clip_start,
            clip_end=clip_end,
            width=width,
            height=height,
            y_pos=st_y,
        ))

    use_karaoke = bool(cap.get("pop", False))
    spring_tag = spring_caption_tag(motion_graphics and use_karaoke)
    for g in groups:
        rel_start = max(0.0, g["start"] - clip_start)
        rel_end = max(rel_start + 0.35, g["end"] - clip_start)
        line_has_urdu = any(is_urdu_text(w.w) for w in g["words"])
        if line_has_urdu:
            # Never split Urdu lines with mid-line {\k} tags: libass breaks BiDi shaping and renders L-to-R.
            raw_line = _escape_ass(" ".join(w.w for w in g["words"]))
            line_text = (spring_tag if motion_graphics else "") + format_urdu_rtl_ass(raw_line)
        elif use_karaoke:
            text_parts = []
            for w in g["words"]:
                w_dur_cs = max(8, int(round(max(0.08, w.e - w.s) * 100)))
                clean_w = _escape_ass(w.w)
                text_parts.append(f"{{\\k{w_dur_cs}}}{clean_w}")
            line_text = spring_tag + " ".join(text_parts)
        else:
            line_text = _escape_ass(" ".join(w.w for w in g["words"]))

        lines.append(
            f"Dialogue: 0,{_format_time(rel_start)},{_format_time(rel_end)},Default,,0,0,0,,{line_text}"
        )

    return "\n".join(lines)
