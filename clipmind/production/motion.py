"""Professional Motion Graphics Engine for Marhaba ClipMind.
Generates crisp ASS vector panels, animated slide-in Hook Cards, animated Lower-Third Creator/Topic pills,
spring-bounce caption transitions, and smooth FFmpeg progress bar overlays without Pillow or RTL breakage.
"""
import re
from typing import Optional, List, Sequence

from ..understand.urdu import is_urdu_text, sanitize_transcript_text, format_urdu_rtl_ass


def _rounded_rect_path(w: int, h: int, r: int = 14) -> str:
    """Builds a smooth ASS cubic-bezier rounded-rectangle vector path."""
    rr = max(2, min(r, w // 3, h // 3))
    return (
        f"m {rr} 0 l {w - rr} 0 "
        f"b {w} 0 {w} 0 {w} {rr} "
        f"l {w} {h - rr} "
        f"b {w} {h} {w} {h} {w - rr} {h} "
        f"l {rr} {h} "
        f"b 0 {h} 0 {h} 0 {h - rr} "
        f"l 0 {rr} "
        f"b 0 0 0 0 {rr} 0"
    )


def _fmt_t(sec: float) -> str:
    sec = max(0.0, sec)
    cs = int(round(sec * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def _esc(text: str) -> str:
    clean = sanitize_transcript_text(text)
    return clean.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def _format_rich_accent(text: str, accent_bgr: str = "&H003B3BFF&", base_bgr: str = "&H00FFFFFF&") -> str:
    cleaned = _esc(text)
    if is_urdu_text(cleaned):
        # Inline ASS color/override tags split libass BiDi runs into L-to-R; render cohesive R-to-L Urdu instead.
        return format_urdu_rtl_ass(cleaned.replace("*", ""))
    parts = re.split(r"(\*[^*]+\*)", cleaned)
    out = []
    for p in parts:
        if p.startswith("*") and p.endswith("*") and len(p) > 2:
            out.append(f"{{\\c{accent_bgr}}}{p[1:-1].strip()}{{\\c{base_bgr}}}")
        elif p:
            out.append(p)
    return "".join(out)


def motion_styles_block(width: int, height: int, accent_bgr: str = "&H000BB9F5") -> str:
    """Returns ASS Style definitions for animated Hook Banner, Lower-Third Pill, Stat Badges, and Vector Shapes."""
    k = height / 1920.0
    pill_size = max(24, int(34 * k))
    tag_size = max(22, int(26 * k))
    ur_tag_size = max(22, int(28 * k))
    stat_size = max(24, int(30 * k))
    lr = int(75 * k)
    return "\n".join([
        f"Style: MotionVec,Inter,20,{accent_bgr},{accent_bgr},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        f"Style: LowerThirdBg,Inter,20,&H200F172A,&H200F172A,&H4894A3B8,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1",
        f"Style: CreatorPill,Inter,{pill_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,7,{lr},{lr},0,1",
        f"Style: TopicTag,Inter,{tag_size},{accent_bgr},{accent_bgr},&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,1,1,7,{lr},{lr},0,1",
        f"Style: TopicTagUrdu,Noto Nastaliq Urdu,{ur_tag_size},{accent_bgr},{accent_bgr},&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,1,7,{lr},{lr},0,1",
        f"Style: StatValue,Inter,{stat_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,7,{lr},{lr},0,1",
    ])


def build_animated_hook_events(
    hook_text: str,
    duration: float,
    width: int = 1080,
    height: int = 1920,
    hook_y: int = 155,
    accent_bgr: str = "&H003B3BFF&",
    stay_full: bool = False,
) -> list[str]:
    """Generates an animated slide-down + fade Hook Card with an expanding accent underline."""
    if not hook_text or not hook_text.strip():
        return []
    k = height / 1920.0
    end_t = duration if stay_full else min(duration, 6.2)
    cx = width // 2
    y_target = hook_y
    y_start = max(10, hook_y - int(42 * k))
    # Keep hook concise so it renders cleanly inside 1-2 lines without colliding with the accent bar
    words_h = hook_text.strip().split()
    concise_hook = " ".join(words_h[:13]) + ("..." if len(words_h) > 13 else "")
    rich = _format_rich_accent(concise_hook, accent_bgr=accent_bgr, base_bgr="&H00FFFFFF&")

    plain_hook = concise_hook.replace("*", "")
    is_ur = is_urdu_text(plain_hook)
    is_two_lines = len(plain_hook) > (54 if is_ur else 38)
    char_px = (16 if is_ur else 23) * k
    card_w = int(width * 0.86) if is_two_lines else min(int(width * 0.88), max(int(width * 0.48), int(len(plain_hook) * char_px) + int(84 * k)))
    card_h = int((128 if is_ur and is_two_lines else (120 if is_two_lines else (78 if is_ur else 72))) * k)
    card_x = (width - card_w) // 2
    card_y_t = max(16, y_target - int(10 * k))
    card_y_s = max(6, y_start - int(10 * k))
    card_vec = _rounded_rect_path(card_w, card_h, r=max(10, int(14 * k)))

    events = [
        # Glassmorphic rounded backdrop sliding cleanly with text
        f"Dialogue: 2,{_fmt_t(0.0)},{_fmt_t(end_t)},LowerThirdBg,,0,0,0,,"
        f"{{\\an7\\move({card_x},{card_y_s},{card_x},{card_y_t},0,280)\\fad(180,260)\\alpha&H22&\\p1}}{card_vec}{{\\p0}}",
        # Animated Hook Card text sliding into place cleanly over the rounded plate
        f"Dialogue: 3,{_fmt_t(0.0)},{_fmt_t(end_t)},HookCard,,0,0,0,,{{\\an8\\move({cx},{y_start},{cx},{y_target},0,280)\\fad(180,260)}}{rich}",
    ]

    bar_w = int(card_w * 0.72)
    bar_h = max(4, int(5 * k))
    bar_x = (width - bar_w) // 2
    bar_y = card_y_t + card_h - int(2 * k)
    events.append(
        f"Dialogue: 4,{_fmt_t(0.08)},{_fmt_t(end_t)},MotionVec,,0,0,0,,"
        f"{{\\an7\\pos({bar_x},{bar_y})\\fscx0\\t(80,380,\\fscx100)\\fad(120,250)\\p1}}"
        f"m 0 0 l {bar_w} 0 l {bar_w} {bar_h} l 0 {bar_h}{{\\p0}}"
    )
    return events


def build_lower_third_events(
    creator_name: Optional[str],
    topic_label: Optional[str],
    duration: float,
    width: int = 1080,
    height: int = 1920,
    y_pos: Optional[int] = None,
) -> list[str]:
    """Generates an animated slide-in glassmorphic lower-third badge for creator & topic tag."""
    if not creator_name and not topic_label:
        return []
    k = height / 1920.0
    start_t = 0.25
    end_t = min(duration, 4.8)
    if end_t - start_t < 1.2:
        return []

    y = y_pos if y_pos is not None else int(height * 0.72)
    x_target = int(58 * k)
    x_start = -int(420 * k)

    title_str = _esc(creator_name or "Marhaba ClipMind")
    # Smart whole-word truncation for topic badge
    raw_sub = _esc((topic_label or "HIGHLIGHT").upper())
    sub_words = raw_sub.split()
    sub_str = ""
    for sw in sub_words:
        cand = f"{sub_str} {sw}".strip()
        if len(cand) <= 26:
            sub_str = cand
        else:
            break
    sub_str = sub_str or raw_sub[:26]
    if is_urdu_text(title_str):
        title_str = format_urdu_rtl_ass(title_str)
    if is_urdu_text(sub_str):
        sub_str = format_urdu_rtl_ass(sub_str)

    pill_w = max(int(360 * k), min(int(width * 0.76), max(len(title_str), len(sub_str)) * int(22 * k) + int(70 * k)))
    pill_h = int(96 * k)
    accent_w = max(6, int(10 * k))
    pill_vec = _rounded_rect_path(pill_w, pill_h, r=max(8, int(12 * k)))
    tag_style = "TopicTagUrdu" if is_urdu_text(sub_str) else "TopicTag"

    events = [
        # Dark translucent rounded backdrop box sliding in
        f"Dialogue: 2,{_fmt_t(start_t)},{_fmt_t(end_t)},LowerThirdBg,,0,0,0,,"
        f"{{\\move({x_start},{y},{x_target},{y},0,300)\\fad(120,280)\\alpha&H25&\\p1}}{pill_vec}{{\\p0}}",
        # Left bright accent stripe
        f"Dialogue: 3,{_fmt_t(start_t)},{_fmt_t(end_t)},MotionVec,,0,0,0,,"
        f"{{\\move({x_start},{y},{x_target},{y},0,300)\\fad(120,280)\\p1}}"
        f"m 0 0 l {accent_w} 0 l {accent_w} {pill_h} l 0 {pill_h}{{\\p0}}",
        # Topic tag (crisp Inter for English uppercase, Noto Nastaliq Urdu for Urdu)
        f"Dialogue: 4,{_fmt_t(start_t)},{_fmt_t(end_t)},{tag_style},,0,0,0,,"
        f"{{\\move({x_start + int(24*k)},{y + int(12*k)},{x_target + int(24*k)},{y + int(12*k)},0,300)\\fad(150,280)}}{sub_str}",
        # Creator name (main line inside pill)
        f"Dialogue: 4,{_fmt_t(start_t)},{_fmt_t(end_t)},CreatorPill,,0,0,0,,"
        f"{{\\move({x_start + int(24*k)},{y + int(44*k)},{x_target + int(24*k)},{y + int(44*k)},0,300)\\fad(150,280)}}{title_str}",
    ]
    return events


def build_stat_callout_events(
    words: Sequence,
    clip_start: float,
    clip_end: float,
    width: int = 1080,
    height: int = 1920,
    y_pos: Optional[int] = None,
) -> list[str]:
    """Extracts spoken key figures (%, years, currency) and pops kinetic callout pills in the clear lower-third band."""
    if not words:
        return []
    k = height / 1920.0
    duration = max(1.0, clip_end - clip_start)
    if duration < 7.5:
        return []

    tokens = [w for w in words if w.e > clip_start and w.s < clip_end]
    stats = []
    for idx, w in enumerate(tokens):
        raw = w.w.strip()
        nxt = tokens[idx + 1].w.strip() if idx + 1 < len(tokens) else ""
        # Match percentages ("94%"), separated percent ("94" "%"), years ("2026"), or significant values (>= 10)
        m_pct = re.fullmatch(r"(\d+(?:\.\d+)?)%", raw)
        m_num = re.fullmatch(r"\d+(?:\.\d+)?", raw)
        label = None
        val = None
        if m_pct:
            label, val = "KEY STAT", raw
        elif m_num and nxt == "%":
            label, val = "KEY STAT", f"{raw}%"
        elif m_num and re.fullmatch(r"20[2-3]\d", raw):
            label, val = "YEAR", raw
        elif m_num and float(raw) >= 50:
            label, val = "KEY FIGURE", raw
        if not val:
            continue

        rel_t = max(5.1, (w.s - clip_start) - 0.15)  # starts strictly after lower-third badge fades out at 4.8s
        if rel_t + 2.4 > duration - 0.6:
            continue
        if any(abs(rel_t - prev[0]) < 4.5 for prev in stats):
            continue
        if any(val == prev[2] for prev in stats):
            continue
        stats.append((rel_t, label, val))
        if len(stats) >= 3:
            break

    if not stats:
        return []

    y = y_pos if y_pos is not None else int(height * 0.66)
    x_t = int(58 * k)
    x_s = -int(340 * k)
    events: list[str] = []
    for st_t, lbl, val in stats:
        en_t = min(duration - 0.2, st_t + 2.6)
        disp = f"{lbl}  •  {val}"
        bw = max(int(240 * k), min(int(width * 0.58), len(disp) * int(20 * k) + int(64 * k)))
        bh = int(64 * k)
        accent_w = max(6, int(8 * k))
        vec = _rounded_rect_path(bw, bh, r=max(8, int(10 * k)))
        events.extend([
            f"Dialogue: 2,{_fmt_t(st_t)},{_fmt_t(en_t)},LowerThirdBg,,0,0,0,,"
            f"{{\\move({x_s},{y},{x_t},{y},0,240)\\fad(120,240)\\alpha&H20&\\p1}}{vec}{{\\p0}}",
            f"Dialogue: 3,{_fmt_t(st_t)},{_fmt_t(en_t)},MotionVec,,0,0,0,,"
            f"{{\\move({x_s},{y},{x_t},{y},0,240)\\fad(120,240)\\p1}}m 0 0 l {accent_w} 0 l {accent_w} {bh} l 0 {bh}{{\\p0}}",
            f"Dialogue: 4,{_fmt_t(st_t)},{_fmt_t(en_t)},StatValue,,0,0,0,,"
            f"{{\\move({x_s + int(22*k)},{y + int(16*k)},{x_t + int(22*k)},{y + int(16*k)},0,240)\\fad(140,240)}}{disp}",
        ])
    return events


def spring_caption_tag(is_animated: bool = True) -> str:
    """Micro-spring pop-in ASS override safe for both English LTR and Urdu RTL lines."""
    if not is_animated:
        return ""
    return "{\\fscx86\\fscy86\\t(0,95,\\fscx104\\fscy104)\\t(95,165,\\fscx100\\fscy100)}"
