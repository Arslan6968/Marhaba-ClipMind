import json
from pathlib import Path
import subprocess
import pytest

from clipmind.pipeline.runner import ClipMindPipeline, remake_clips
from clipmind.schemas import UserIntent, Transcript, TranscriptSegment, WordToken
from clipmind.understand.brains import NeedAgentInput
from clipmind.media.probe import probe_media


def create_synthetic_test_video(path: Path, duration: int = 18) -> Path:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


def _mock_ai_transcript():
    words = [
        WordToken(w="What", s=1.0, e=1.4),
        WordToken(w="is", s=1.45, e=1.7),
        WordToken(w="the", s=1.75, e=1.9),
        WordToken(w="secret", s=1.95, e=2.4),
        WordToken(w="behind", s=2.45, e=2.8),
        WordToken(w="AI?", s=2.85, e=3.4),
        WordToken(w="The", s=4.5, e=4.8),
        WordToken(w="secret", s=4.85, e=5.3),
        WordToken(w="is", s=5.35, e=5.6),
        WordToken(w="massive", s=5.65, e=6.1),
        WordToken(w="data", s=6.15, e=6.6),
        WordToken(w="and", s=6.65, e=6.9),
        WordToken(w="neural", s=6.95, e=7.4),
        WordToken(w="networks.", s=7.45, e=8.2),
        WordToken(w="It", s=10.0, e=10.3),
        WordToken(w="is", s=10.35, e=10.6),
        WordToken(w="revolutionizing", s=10.65, e=11.6),
        WordToken(w="technology", s=11.65, e=12.4),
        WordToken(w="and", s=12.45, e=12.7),
        WordToken(w="human", s=12.75, e=13.2),
        WordToken(w="progress.", s=13.25, e=14.1),
        WordToken(w="That", s=15.0, e=15.3),
        WordToken(w="is", s=15.35, e=15.6),
        WordToken(w="truly", s=15.65, e=16.1),
        WordToken(w="incredible!", s=16.15, e=17.2),
    ]
    s1 = TranscriptSegment(id=0, start=1.0, end=3.4, text="What is the secret behind AI?", words=words[:6])
    s2 = TranscriptSegment(id=1, start=4.5, end=8.2, text="The secret is massive data and neural networks.", words=words[6:14])
    s3 = TranscriptSegment(id=2, start=10.0, end=14.1, text="It is revolutionizing technology and human progress.", words=words[14:21])
    s4 = TranscriptSegment(id=3, start=15.0, end=17.2, text="That is truly incredible!", words=words[21:])
    return Transcript(
        language="en",
        segments=[s1, s2, s3, s4],
        words=words,
        full_text="What is the secret behind AI? The secret is massive data and neural networks. It is revolutionizing technology and human progress. That is truly incredible!",
    )


def test_pipeline_synthetic_video(tmp_path):
    video_path = create_synthetic_test_video(tmp_path / "test_input.mp4", duration=16)
    out_dir = tmp_path / "output_clips"
    pipeline = ClipMindPipeline(work_root=tmp_path / "work")
    res = pipeline.run(str(video_path), output_dir=out_dir)
    assert res["status"] == "completed"
    assert res["clips_count"] == 0
    assert "No spoken words" in res["message"]


def test_pipeline_end_to_end_and_remake(tmp_path, monkeypatch):
    video_path = create_synthetic_test_video(tmp_path / "test_speech.mp4", duration=22)
    from clipmind.understand.asr import SpeechTranscriber
    monkeypatch.setattr(SpeechTranscriber, "transcribe_chunks", lambda *a, **k: _mock_ai_transcript())

    out_dir = tmp_path / "output_clips"
    pipeline = ClipMindPipeline(work_root=tmp_path / "work")
    intent = UserIntent(objective="educational", topic="artificial intelligence", max_clips=2, sfx=True)
    res = pipeline.run(str(video_path), user_intent=intent, output_dir=out_dir)

    assert res["status"] == "completed"
    assert res["clips_count"] >= 1
    assert res["peak_rss_mb"] > 0

    clip0 = res["rendered_clips"][0]
    clip_file = Path(clip0.output_path)
    assert clip_file.is_file() and clip0.verification["ok"] is True
    assert clip0.verification["checks"]["width"] == 1080
    assert clip0.verification["checks"]["height"] == 1920
    assert clip0.verification["checks"]["decodes"] is True
    assert Path(clip0.check_strip_path).is_file()

    # Verify slim manifest (no inline render_meta/words bloat), sidecar, check_all.jpg, and POSTING.md
    manifest_data = json.loads(Path(res["manifest_file"]).read_text(encoding="utf-8"))
    assert len(manifest_data["rendered_clips"]) >= 1
    assert "render_meta" not in manifest_data["rendered_clips"][0]
    sidecar_rel = manifest_data["rendered_clips"][0]["render_meta_file"]
    assert (out_dir / sidecar_rel).is_file()
    assert Path(res["composite_check_strip"]).is_file()
    assert Path(manifest_data["composite_check_strip"]).is_file()
    assert "Publishing Pack" in Path(res["posting_pack"]).read_text(encoding="utf-8")

    # Test instant remake from sidecar (switch preset to social_media + custom hook)
    rm_res = remake_clips(out_dir, clip_indices=[1], preset="social_media", hook="What is the *secret* behind AI?")
    assert rm_res["remade_count"] == 1
    assert Path(rm_res["composite_check_strip"]).is_file()
    updated_manifest = json.loads(Path(res["manifest_file"]).read_text(encoding="utf-8"))
    assert updated_manifest["rendered_clips"][0]["preset"] == "social_media"
    assert updated_manifest["rendered_clips"][0]["verification"]["ok"] is True


def test_pipeline_agent_mode_roundtrip(tmp_path, monkeypatch):
    video_path = create_synthetic_test_video(tmp_path / "test_agent.mp4", duration=22)
    from clipmind.understand.asr import SpeechTranscriber
    monkeypatch.setattr(SpeechTranscriber, "transcribe_chunks", lambda *a, **k: _mock_ai_transcript())

    out_dir = tmp_path / "agent_clips"
    work_dir = tmp_path / "agent_work"
    pipeline = ClipMindPipeline(work_root=work_dir)
    intent = UserIntent(objective="educational", topic="artificial intelligence", brain="agent", max_clips=1)

    with pytest.raises(NeedAgentInput) as excinfo:
        pipeline.run(str(video_path), user_intent=intent, output_dir=out_dir, keep_temp=True)

    need = excinfo.value
    assert need.task_file.is_file()
    task_json_p = need.task_file.parent / "STORY_TASK.json"
    assert task_json_p.is_file()
    task_obj = json.loads(task_json_p.read_text(encoding="utf-8"))
    assert len(task_obj["candidates"]) >= 1
    batch_answers = {
        c["id"]: {
            "verdict": "fix",
            "start": 0.9,
            "end": 17.5,
            "title": "Secret Behind AI Models",
            "hook": "The real *secret* behind AI",
            "reason": "Agent verified complete explanation",
        }
        for c in task_obj["candidates"]
    }
    Path(task_obj["answer_file"]).write_text(json.dumps(batch_answers), encoding="utf-8")

    # Re-run and verify answers are consumed without needing re-transcription
    res2 = pipeline.run(str(video_path), user_intent=intent, output_dir=out_dir)
    assert res2["status"] == "completed"
    assert res2["clips_count"] == 1
    assert res2["rendered_clips"][0].publishing.title == "Secret Behind AI Models"
