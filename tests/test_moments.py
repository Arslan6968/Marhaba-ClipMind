import pytest
from clipmind.schemas import Transcript, TranscriptSegment, WordToken, UserIntent
from clipmind.media.audio import compute_db_spikes
from clipmind.moments.units import build_thought_units, annotate_units
from clipmind.moments.scoring import evaluate_span
from clipmind.moments.verify import verify_and_adjust_clip
from clipmind.moments.dedupe import deduplicate_candidates


def _make_transcript():
    words = [
        WordToken(w="Why", s=1.0, e=1.4),
        WordToken(w="did", s=1.45, e=1.7),
        WordToken(w="the", s=1.75, e=1.9),
        WordToken(w="server", s=1.95, e=2.4),
        WordToken(w="crash?", s=2.45, e=3.0),
        WordToken(w="Because", s=3.8, e=4.2),
        WordToken(w="someone", s=4.25, e=4.7),
        WordToken(w="deleted", s=4.75, e=5.2),
        WordToken(w="the", s=5.25, e=5.4),
        WordToken(w="database.", s=5.45, e=6.1),
        WordToken(w="Oh", s=6.8, e=7.0),
        WordToken(w="my", s=7.05, e=7.2),
        WordToken(w="god!", s=7.25, e=7.8),
    ]
    seg1 = TranscriptSegment(id=0, start=1.0, end=3.0, text="Why did the server crash?", words=words[:5])
    seg2 = TranscriptSegment(id=1, start=3.8, end=6.1, text="Because someone deleted the database.", words=words[5:10])
    seg3 = TranscriptSegment(id=2, start=6.8, end=7.8, text="Oh my god!", words=words[10:])
    return Transcript(language="en", segments=[seg1, seg2, seg3], words=words,
                      full_text="Why did the server crash? Because someone deleted the database. Oh my god!")


def test_thought_units_split_on_punctuation():
    tr = _make_transcript()
    units = build_thought_units(tr)
    assert len(units) == 3
    assert units[0]["is_question"] is True
    assert units[1]["text"].startswith("Because")
    assert units[2]["text"].startswith("Oh")


def test_verify_snaps_dependent_openers_and_preserves_words():
    tr = _make_transcript()
    units = build_thought_units(tr)
    annotate_units(units, energy_profile=[-30.0] * 10, scene_cuts=[])

    cand = {
        "start": units[1]["start"],
        "end": units[1]["end"],
        "dimensions": evaluate_span(units, 1, 1, 1, [], [], False, UserIntent())["dimensions"],
        "overall_score": 8.0,
        "anchor_idx": 1,
        "hook_unit": units[1],
        "peak_unit": units[1],
    }

    clip = verify_and_adjust_clip(cand, units, video_path=None, intent=UserIntent())
    assert clip.boundary_adjusted is True
    assert clip.start <= units[0]["start"]
    assert "Why did the server crash" in clip.transcript_text
    assert len(clip.words) >= 10
    assert clip.words[0].w == "Why"


def test_verify_trims_filler_opener():
    words = [
        WordToken(w="Um,", s=1.0, e=1.3),
        WordToken(w="this", s=1.4, e=1.7),
        WordToken(w="database", s=1.75, e=2.4),
        WordToken(w="architecture", s=2.45, e=3.4),
        WordToken(w="scales", s=3.45, e=4.1),
        WordToken(w="millions", s=4.15, e=5.0),
        WordToken(w="of", s=5.05, e=5.3),
        WordToken(w="requests", s=5.35, e=6.2),
        WordToken(w="every", s=6.25, e=7.0),
        WordToken(w="single", s=7.05, e=8.2),
        WordToken(w="second.", s=8.25, e=10.0),
    ]
    seg = TranscriptSegment(id=0, start=1.0, end=10.0, text=" ".join(w.w for w in words), words=words)
    tr = Transcript(language="en", segments=[seg], words=words, full_text=seg.text)
    units = build_thought_units(tr)
    annotate_units(units, energy_profile=[-28.0] * 12, scene_cuts=[])
    cand = evaluate_span(units, 0, 0, 0, [], [], False, UserIntent())
    cand["anchor_idx"] = 0
    clip = verify_and_adjust_clip(cand, units, video_path=None, intent=UserIntent())
    assert clip.boundary_adjusted is True
    assert clip.start >= 1.25


def test_deduplicate_temporal_and_text_similarity():
    cands = [
        {"start": 10.0, "end": 40.0, "overall_score": 8.5, "transcript_text": "alpha beta gamma delta epsilon"},
        {"start": 12.0, "end": 42.0, "overall_score": 7.0, "transcript_text": "something else entirely"},
        {"start": 100.0, "end": 130.0, "overall_score": 7.5, "transcript_text": "alpha beta gamma delta epsilon"},
        {"start": 200.0, "end": 230.0, "overall_score": 8.0, "transcript_text": "unique different topic discussion"},
    ]
    deduped = deduplicate_candidates(cands, max_overlap=0.35, max_text_sim=0.65)
    assert len(deduped) == 2
    assert deduped[0]["overall_score"] == 8.5
    assert deduped[1]["overall_score"] == 8.0


def test_compute_db_spikes():
    curve = [-30.0] * 25 + [-15.0] * 3 + [-30.0] * 25
    spikes = compute_db_spikes(curve)
    assert len(spikes) == len(curve)
    assert max(spikes[25:28]) >= 10.0
    assert max(spikes[:20]) < 1.0


def test_teaser_guard_prefers_concrete_facts():
    w_teaser = [
        WordToken(w="Aaj", s=2.0, e=2.3),
        WordToken(w="ki", s=2.3, e=2.5),
        WordToken(w="video", s=2.5, e=2.9),
        WordToken(w="mein", s=2.9, e=3.2),
        WordToken(w="kya", s=3.2, e=3.5),
        WordToken(w="hoga?", s=3.5, e=4.0),
        WordToken(w="Kaun", s=4.5, e=4.9),
        WordToken(w="apply", s=4.9, e=5.4),
        WordToken(w="kare?", s=5.4, e=6.0),
    ]
    w_fact = [
        WordToken(w="Morning", s=70.0, e=70.5),
        WordToken(w="closing", s=70.5, e=71.0),
        WordToken(w="merit", s=71.0, e=71.4),
        WordToken(w="94%", s=71.4, e=72.0),
        WordToken(w="aur", s=72.0, e=72.3),
        WordToken(w="2026", s=72.3, e=72.9),
        WordToken(w="stipend", s=72.9, e=73.4),
        WordToken(w="31000", s=73.4, e=74.0),
        WordToken(w="hai.", s=74.0, e=74.5),
    ]
    tr = Transcript(
        language="ur",
        segments=[
            TranscriptSegment(id=0, start=2.0, end=6.0, text="Aaj ki video mein kya hoga? Kaun apply kare?", words=w_teaser),
            TranscriptSegment(id=1, start=70.0, end=74.5, text="Morning closing merit 94% aur 2026 stipend 31000 hai.", words=w_fact),
        ],
        words=w_teaser + w_fact,
        full_text="Aaj ki video mein kya hoga? Kaun apply kare? Morning closing merit 94% aur 2026 stipend 31000 hai.",
    )
    units = build_thought_units(tr)
    annotate_units(units, energy_profile=[-28.0] * 80, scene_cuts=[], intent=UserIntent(objective="educational"))
    teaser_res = evaluate_span(units, 0, 1, 0, [], [], False, UserIntent(objective="educational"))
    fact_res = evaluate_span(units, 2, 2, 2, [], [], False, UserIntent(objective="educational"))
    assert fact_res["overall_score"] > teaser_res["overall_score"]
