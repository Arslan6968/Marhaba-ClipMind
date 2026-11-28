from pathlib import Path
import pytest
from clipmind.schemas import WordToken, ClipCandidate, ContentClassification, EvaluationDimensions
from clipmind.production.captions import generate_ass_subtitles
from clipmind.production.layout import build_ffmpeg_filtergraph, build_gliding_crop_expr
from clipmind.production.audio_fx import build_sfx_track
from clipmind.production.publishing import generate_publishing_package
from clipmind.production.presets import EDITING_PRESETS, get_preset
from clipmind.config import get_settings


def test_ass_subtitles_english_karaoke_colors():
    words = [
        WordToken(w="This", s=1.0, e=1.3),
        WordToken(w="is", s=1.35, e=1.5),
        WordToken(w="amazing!", s=1.55, e=2.1),
    ]
    ass = generate_ass_subtitles(
        words, 1.0, 2.5, preset_name="dynamic",
        hook_card_text="Watch this *insane* moment",
        creator_name="Marhaba Studio", topic_label="AI Demo", motion_graphics=True,
    )
    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "Style: MotionVec" in ass
    assert "Dialogue: 0," in ass
    assert "{\\k" in ass
    assert "{\\fscx86" in ass
    assert "Marhaba Studio" in ass
    assert "{\\c&H003B3BFF&}insane{\\c&H00FFFFFF&}" in ass or "insane" in ass


def test_ass_subtitles_urdu_rtl_and_roman_urdu():
    words = [
        WordToken(w="یہ", s=0.5, e=0.9),
        WordToken(w="BS", s=0.95, e=1.4),
        WordToken(w="ضروری", s=1.45, e=1.9),
        WordToken(w="ہے۔", s=1.95, e=2.3),
    ]
    ass_ur = generate_ass_subtitles(
        words, 0.5, 2.5, preset_name="dynamic", caption_lang="original",
        hook_card_text="انتظار کی *گھڑیاں* ختم! گورنمنٹ BS نرسنگ ایڈمیشن اوپن",
        motion_graphics=True,
    )
    assert "Noto Nastaliq Urdu" in ass_ur
    # Karaoke PrimaryColour != SecondaryColour so style supports animation
    style_line = next(l for l in ass_ur.splitlines() if l.startswith("Style: Default,"))
    assert "&H0000FFFF,&H00FFFFFF" in style_line
    # Native Urdu caption dialogue must wrap with RLE/PDF & RLM and must NOT insert mid-line {\k tags that flip RTL to LTR
    cap_line = next(l for l in ass_ur.splitlines() if l.startswith("Dialogue: 0,"))
    assert "\u202b" in cap_line and "\u202c" in cap_line
    assert "\u200fBS\u200f" in cap_line
    assert "{\\k" not in cap_line
    hook_line = next(l for l in ass_ur.splitlines() if ",HookCard," in l)
    assert "\u202b" in hook_line and "\u200fBS\u200f" in hook_line
    assert "{\\c&H" not in hook_line

    ass_roman = generate_ass_subtitles(
        [
            WordToken(w="مارننگ", s=0.5, e=0.9),
            WordToken(w="94", s=0.95, e=1.2),
            WordToken(w="%", s=1.25, e=1.4),
            WordToken(w="ضروری", s=1.45, e=1.9),
            WordToken(w="ہے۔", s=1.95, e=2.3),
        ],
        0.5, 2.5, preset_name="dynamic", caption_lang="roman_urdu",
        hook_card_text="گورنمنٹ BS نرسنگ ایڈمیشن",
    )
    assert "94%" in ass_roman
    assert "zaroori" in ass_roman.lower()
    assert "Govt BS Nursing Admission" in ass_roman
    assert "{\\k" in ass_roman


def test_all_6_presets_and_4_aspect_ratios_filtergraphs():
    fonts_dir = get_settings().fonts_dir
    for name in EDITING_PRESETS:
        p = get_preset(name)
        assert p["id"] == name

    for ar, dims in [("9:16", (1080, 1920)), ("16:9", (1920, 1080)), ("1:1", (1080, 1080)), ("original", (1280, 720))]:
        fg, pad = build_ffmpeg_filtergraph(
            layout_mode="fit_blur" if ar == "9:16" else ar,
            src_width=1280,
            src_height=720,
            target_width=dims[0],
            target_height=dims[1],
            ass_path="C:/tmp/test.ass",
            fonts_dir=fonts_dir,
            aspect_ratio=ar,
        )
        assert pad == "[vout]"
        assert "fontsdir=" in fg


def test_gliding_crop_expression_and_sfx(tmp_path):
    expr = build_gliding_crop_expr([(0.0, 0.25), (4.0, 0.72), (8.0, 0.45)], sw=1920, cw=608, clip_duration=10.0)
    assert "if(lt(t," in expr

    wav_out = build_sfx_track(tmp_path / "sfx.wav", clip_duration=10.0, punch_times=[(2.0, 3.2), (6.5, 7.8)], include_hook_whoosh=True)
    assert wav_out is not None and wav_out.is_file() and wav_out.stat().st_size > 10_000


def test_publishing_package():
    clip = ClipCandidate(
        id="clip_1",
        start=10.0,
        end=35.0,
        duration=25.0,
        hook_time=12.0,
        dimensions=EvaluationDimensions(),
        overall_score=8.2,
        title="Server Architecture Secret",
        hook_text="What is the real secret behind scalable systems?",
        event_text="The real secret is immutable infrastructure and decoupled queues.",
        transcript_text="What is the real secret behind scalable systems? The real secret is immutable infrastructure.",
    )
    genre = ContentClassification(genre="podcast", confidence=0.85, recommended_preset="podcast")
    pub = generate_publishing_package(clip, genre, creator_name="Marhaba Tech")

    assert pub.title
    assert pub.quality_score == 8.2
    assert pub.category == "podcast"
    assert "#marhabatech" in pub.hashtags
    assert len(pub.ab_hooks) == 2
    assert "immutable infrastructure" in pub.description.lower()

    # Validate rounded vector glass backdrop + English Inter TopicTag + spoken StatValue popup
    long_words = [
        WordToken(w="Morning", s=1.0, e=1.4),
        WordToken(w="merit", s=1.45, e=1.9),
        WordToken(w="closed", s=5.2, e=5.7),
        WordToken(w="at", s=5.75, e=5.9),
        WordToken(w="94", s=5.95, e=6.3),
        WordToken(w="%", s=6.35, e=6.6),
    ]
    ass_stats = generate_ass_subtitles(
        long_words, 1.0, 11.0, preset_name="educational", caption_lang="roman_urdu",
        hook_card_text="Merit List 2026 Explained", creator_name="Shahzada", topic_label="NURSING MERIT", motion_graphics=True,
    )
    assert "Style: TopicTag,Inter," in ass_stats
    assert "Style: StatValue,Inter," in ass_stats
    assert "KEY STAT  •  94%" in ass_stats
    assert ",LowerThirdBg," in ass_stats and " b " in ass_stats

    pub_rom = generate_publishing_package(clip, genre, creator_name="Marhaba Tech", caption_lang="roman_urdu")
    assert "reference" in pub_rom.caption.lower() or "perspective" in pub_rom.caption.lower() or "thoughts" in pub_rom.caption.lower()
    assert clip.title[:6].lower() in pub_rom.title.lower()

    clip_slash = ClipCandidate(
        id="clip_2", start=0.0, end=20.0, duration=20.0, hook_time=2.0,
        dimensions=EvaluationDimensions(), overall_score=8.0,
        title="Male/Female Seats & 94% Merit", hook_text="کون اپلائی کر سکتا ہے؟",
        event_text="Morning 94% closing merit hai.", transcript_text="Morning 94% closing merit hai.",
        ab_hooks=["کون اپلائی کر سکتا ہے؟"],
    )
    pub_slash = generate_publishing_package(clip_slash, ContentClassification(genre="lecture_education", confidence=0.8, recommended_preset="educational"), caption_lang="roman_urdu")
    assert "Male & Female" in pub_slash.title
    assert "kaun apply kar sakta" in pub_slash.ab_hooks[0].lower()

    pub_isl = generate_publishing_package(clip, ContentClassification(genre="islamic", confidence=0.9, recommended_preset="islamic"))
    assert "#islamic" in pub_isl.hashtags and "JazakAllah" in pub_isl.caption
    pub_mov = generate_publishing_package(clip, ContentClassification(genre="movie_shot", confidence=0.9, recommended_preset="cinematic"))
    assert "#cinema" in pub_mov.hashtags and "scene" in pub_mov.caption.lower()

    # Qur'anic Ayat words must be detected and suppressed from burned-in ASS captions in both original & roman_urdu
    bayan_words = [
        WordToken(w="قُلْ", s=0.5, e=0.9),
        WordToken(w="هُوَ", s=0.95, e=1.3),
        WordToken(w="اللَّهُ", s=1.35, e=1.8),
        WordToken(w="أَحَدٌ", s=1.85, e=2.4),
        WordToken(w="یہ", s=3.2, e=3.5),
        WordToken(w="توحید", s=3.55, e=4.0),
        WordToken(w="کا", s=4.05, e=4.3),
        WordToken(w="خلاصہ", s=4.35, e=4.8),
        WordToken(w="ہے۔", s=4.85, e=5.2),
    ]
    for mode in ("original", "roman_urdu"):
        ass_bayan = generate_ass_subtitles(
            bayan_words, 0.5, 5.5, preset_name="islamic", caption_lang=mode,
            hook_card_text="قُلْ هُوَ اللَّهُ أَحَدٌ",
        )
        assert "اللَّهُ" not in ass_bayan
        assert "ql ho" not in ass_bayan.lower()
        assert "Dialogue: 0," in ass_bayan
