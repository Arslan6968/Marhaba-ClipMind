import pytest
from pathlib import Path

from clipmind.ingest.local import LocalFileAdapter
from clipmind.ingest.url import DirectUrlAdapter, validate_public_ip, SSRFError
from clipmind.ingest.ytdlp import YtDlpAdapter, friendly_ytdlp_error
from clipmind.ingest import probe_source, get_adapter
from clipmind.ingest.trends import extract_seed_topics
from clipmind.schemas import ChannelVideo

def test_local_file_adapter_can_handle(tmp_path):
    f = tmp_path / "sample.mp4"
    f.write_text("data")
    adapter = LocalFileAdapter()
    assert adapter.can_handle(str(f)) is True
    assert adapter.can_handle("https://youtube.com/watch?v=123") is False

def test_ytdlp_adapter_can_handle():
    adapter = YtDlpAdapter()
    assert adapter.can_handle("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert adapter.can_handle("https://twitch.tv/videos/12345") is True
    assert adapter.can_handle("local_file.mp4") is False

def test_direct_url_adapter():
    adapter = DirectUrlAdapter()
    assert adapter.can_handle("https://example.com/video.mp4") is True
    assert adapter.can_handle("https://example.com/stream.webm?token=abc") is True
    assert adapter.can_handle("not_a_url") is False

def test_ssrf_rejects_private_ips():
    with pytest.raises(SSRFError):
        validate_public_ip("127.0.0.1")
    with pytest.raises(SSRFError):
        validate_public_ip("192.168.1.1")
    with pytest.raises(SSRFError):
        validate_public_ip("10.0.0.5")

def test_probe_source_nonexistent():
    info = probe_source("nonexistent_video.mp4")
    assert info.is_supported is False
    assert "not found" in info.error_message.lower()


def test_extract_seed_topics():
    vids = [
        ChannelVideo(video_id="1", title="BS Nursing Admission 2026-27 Pakistan | Last Date & Stipend", url="https://youtu.be/1"),
        ChannelVideo(video_id="2", title="MBBS Closing Merit Session 2026-27 | Private Colleges", url="https://youtu.be/2"),
    ]
    seeds = extract_seed_topics(vids)
    assert any("bs nursing" in s.lower() for s in seeds)
    assert any("mbbs closing" in s.lower() for s in seeds)


def test_friendly_ytdlp_error_classifies_bot_before_age():
    bot_msg = friendly_ytdlp_error("ERROR: [youtube] xyz: Sign in to confirm you’re not a bot.")
    age_msg = friendly_ytdlp_error("ERROR: [youtube] xyz: Sign in to confirm your age.")
    assert "anti-bot" in bot_msg.lower() or "not a bot" in bot_msg.lower()
    assert "age-restricted" not in bot_msg.lower()
    assert "age-restricted" in age_msg.lower()
