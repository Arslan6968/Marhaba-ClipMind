import pytest
from clipmind.understand.intent import (
    parse_user_query,
    get_intent_weights,
    calculate_topic_relevance,
    calculate_speaker_relevance,
    generate_adaptive_questions,
)
from clipmind.understand.classifier import classify_content
from clipmind.understand.urdu import (
    format_code_switching,
    transliterate_urdu_to_roman,
    is_urdu_text,
    build_domain_asr_prompt,
)
from clipmind.understand.brains import resolve_brain_mode, is_grounded, extract_json_object
from clipmind.schemas import Transcript, TranscriptSegment


def test_parse_user_query_objectives():
    i1 = parse_user_query("find the funniest moments in this stream")
    assert i1.objective == "funny"
    assert i1.target_duration == "medium"

    i2 = parse_user_query("extract educational takeaways about artificial intelligence")
    assert i2.objective == "educational"
    assert i2.topic.lower() == "artificial intelligence"

    i3 = parse_user_query("give me quick 15s clips with powerful statements")
    assert i3.objective == "powerful"
    assert i3.target_duration == "short"

    i4 = parse_user_query("cut square 1:1 reels in roman urdu using podcast style")
    assert i4.aspect_ratio == "1:1"
    assert i4.caption_lang == "roman_urdu"
    assert i4.preset == "podcast"

    i5 = parse_user_query("find moments featuring Elon Musk talks about rockets")
    assert i5.speaker == "Elon Musk"


def test_intent_weights_boost():
    funny_w = get_intent_weights(parse_user_query("funny"))
    edu_w = get_intent_weights(parse_user_query("educational"))
    assert funny_w["emotional_value"] > edu_w["emotional_value"]
    assert edu_w["content_value"] > funny_w["content_value"]
    assert abs(sum(funny_w.values()) - 1.0) < 0.01


def test_semantic_bilingual_and_fuzzy_topic_relevance():
    text_ai = "We built a neural network and machine learning model to automate speech."
    assert calculate_topic_relevance(text_ai, "artificial intelligence") >= 0.8

    # Urdu concept matching for English topic query
    text_urdu_biz = "کاروبار میں پیسہ اور منافع بڑھانے کے لیے صحیح حکمت عملی ضروری ہے۔"
    assert calculate_topic_relevance(text_urdu_biz, "finance") >= 0.6

    # Unseen morphological stem / fuzzy match
    text_stem = "They discussed climate warming and renewable solar technologies."
    assert calculate_topic_relevance(text_stem, "climatic warming") >= 0.5

    assert calculate_topic_relevance(text_ai, "gardening") <= 0.2
    assert calculate_topic_relevance(text_ai, None) == 1.0


def test_speaker_relevance():
    assert calculate_speaker_relevance("Here Sam Altman explains the roadmap.", "Sam Altman") == 1.0
    assert calculate_speaker_relevance("Nothing related here.", "Sam Altman") <= 0.25


def test_adaptive_questions_skipped_when_intent_clear():
    assert len(generate_adaptive_questions({}, None, "find funny gaming moments")) == 0
    qs_empty = generate_adaptive_questions({"duration": 600}, None, "")
    assert len(qs_empty) == 2


def test_urdu_code_switching_and_roman():
    mixed = "یہ ویڈیو ایڈیٹنگ سافٹ ویئر بہت زبردست ہے"
    norm = format_code_switching(mixed)
    assert "Video" in norm
    assert "Software" in norm
    assert is_urdu_text(norm)

    roman = transliterate_urdu_to_roman("یہ بہت اچھا ہے")
    assert "yeh" in roman.lower()
    assert "bohot" in roman.lower()
    assert "acha" in roman.lower()

    roman_mixed = transliterate_urdu_to_roman("بیس نرسین کا میرٹلس کب لگے گا؟")
    assert "BS Nursing" in roman_mixed
    assert "Merit List" in roman_mixed
    assert "kab lage ga" in roman_mixed.lower()


def test_classifier_12_genres():
    seg_pod = TranscriptSegment(id=0, start=0.0, end=4.0, text="Welcome back to the podcast. My guest today is a founder.")
    res_pod = classify_content(Transcript(language="en", segments=[seg_pod], full_text=seg_pod.text), cuts=[], duration=60.0)
    assert res_pod.genre in ("podcast", "interview")
    assert res_pod.recommended_preset == "podcast"

    seg_tut = TranscriptSegment(id=0, start=0.0, end=5.0, text="Step by step tutorial: click the timeline tab and add a keyframe then export render settings.")
    res_tut = classify_content(Transcript(language="en", segments=[seg_tut], full_text=seg_tut.text), cuts=[], duration=120.0)
    assert res_tut.genre == "tutorial"
    assert res_tut.recommended_preset == "educational"

    seg_adm = TranscriptSegment(id=0, start=0.0, end=5.0, text="BS Nursing admission 2026 closing merit eligibility stipend seats university.")
    res_adm = classify_content(Transcript(language="ur", segments=[seg_adm], full_text=seg_adm.text), cuts=[], duration=180.0)
    assert res_adm.genre == "lecture_education"
    assert res_adm.recommended_preset == "educational"

    prm = build_domain_asr_prompt(title="BS Nursing Admission 2026", topic="merit stipend", language="ur")
    assert "BS Nursing Admission 2026" in prm
    assert "merit stipend" in prm

    seg_isl = TranscriptSegment(id=0, start=0.0, end=5.0, text="Allah ki rehmat, Quran aur Hadith ki roshni mein sabr aur tawakkul ka bayan.")
    res_isl = classify_content(Transcript(language="ur", segments=[seg_isl], full_text=seg_isl.text), cuts=[], duration=300.0)
    assert res_isl.genre == "islamic"
    assert res_isl.recommended_preset == "islamic"

    seg_inf = TranscriptSegment(id=0, start=0.0, end=5.0, text="Stop doing this mistake if you want discipline, habits, confidence, and personal brand growth.")
    res_inf = classify_content(Transcript(language="en", segments=[seg_inf], full_text=seg_inf.text), cuts=[], duration=180.0)
    assert res_inf.genre == "influencing"
    assert res_inf.recommended_preset == "social_media"

    seg_mov = TranscriptSegment(id=0, start=0.0, end=5.0, text="That cinematic movie scene and film dialogue twist at the ending had incredible cinematography.")
    res_mov = classify_content(Transcript(language="en", segments=[seg_mov], full_text=seg_mov.text), cuts=[float(i * 2) for i in range(40)], duration=120.0)
    assert res_mov.genre == "movie_shot"
    assert res_mov.recommended_preset == "cinematic"

    assert parse_user_query("extract islamic bayan reminders").preset == "islamic"
    assert parse_user_query("clip the best movie shot scene").preset == "cinematic"


def test_brain_helpers():
    mode, _ = resolve_brain_mode("rules")
    assert mode == "rules"
    assert is_grounded("Server Crashed Today", "Oh no the production server crashed today in seconds") is True
    assert is_grounded("Aliens Landed Yesterday", "The production server crashed today in seconds") is False
    assert extract_json_object("```json\n{\"ok\": true}\n```") == {"ok": True}
