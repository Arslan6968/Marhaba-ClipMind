"""Audio extraction, speech enhancement, rolling spike curves, and envelope sync.
Streams PCM in bounded buffers for 8 GB RAM compatibility.
"""
import math
from pathlib import Path
import re
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from ..config import get_settings


def decode_pcm(media_path: Path, rate: int = 16000, start: Optional[float] = None, duration: Optional[float] = None) -> np.ndarray:
    """Decodes audio segment to 1D float32 numpy array [-1, 1] at `rate` Hz."""
    settings = get_settings()
    cmd = [settings.ffmpeg_bin, "-v", "error", "-nostdin"]
    if start is not None and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(media_path), "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if not res.stdout:
        return np.zeros(0, dtype=np.float32)
    n = len(res.stdout) // 2
    samples = np.frombuffer(res.stdout[:n * 2], dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def extract_audio_chunks(media_path: Path, work_dir: Path, chunk_sec: int = 300,
                         start_at: float = 0.0, end_at: Optional[float] = None,
                         preprocess: bool = False) -> List[Dict[str, Any]]:
    """Splits media audio into 16 kHz mono FLAC chunks (reusing cached chunks if present)."""
    settings = get_settings()
    work_dir.mkdir(parents=True, exist_ok=True)

    from .probe import probe_media
    meta = probe_media(media_path)
    total_dur = meta["duration"]
    if total_dur <= 0:
        return []

    hi = min(total_dur, end_at) if end_at is not None else total_dur
    lo = max(0.0, start_at)
    if hi - lo < 0.5:
        return []

    af_chain = "highpass=f=80,lowpass=f=8000,dynaudnorm=f=120:g=15:p=0.95" if preprocess else None

    chunks = []
    t = lo
    idx = int(lo // chunk_sec)
    while t < hi:
        chunk_len = min(float(chunk_sec), hi - t)
        if chunk_len < 0.5:
            break
        out_file = work_dir / f"chunk_{idx:04d}.flac"
        if out_file.is_file() and out_file.stat().st_size > 1000:
            chunks.append({
                "index": idx,
                "file": str(out_file),
                "offset": round(t, 2),
                "duration": round(chunk_len, 2),
            })
            t += chunk_len
            idx += 1
            continue

        cmd = [
            settings.ffmpeg_bin,
            "-y", "-v", "error", "-nostdin",
            "-ss", f"{t:.2f}",
            "-t", f"{chunk_len:.2f}",
            "-i", str(media_path),
            "-vn", "-ar", "16000", "-ac", "1",
        ]
        if af_chain:
            cmd += ["-af", af_chain]
        cmd += ["-c:a", "flac", str(out_file)]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 and af_chain:
            # Clean fallback without filter chain
            fb = [
                settings.ffmpeg_bin, "-y", "-v", "error", "-nostdin",
                "-ss", f"{t:.2f}", "-t", f"{chunk_len:.2f}",
                "-i", str(media_path), "-vn", "-ar", "16000", "-ac", "1", "-c:a", "flac", str(out_file),
            ]
            res = subprocess.run(fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if res.returncode == 0 and out_file.is_file():
            chunks.append({
                "index": idx,
                "file": str(out_file),
                "offset": round(t, 2),
                "duration": round(chunk_len, 2),
            })
        t += chunk_len
        idx += 1

    return chunks


def calculate_loudness_profile(media_path: Path) -> List[float]:
    """Streams 16-bit mono PCM audio from ffmpeg to compute per-second RMS dBFS."""
    settings = get_settings()
    cmd = [
        settings.ffmpeg_bin,
        "-v", "error", "-nostdin",
        "-i", str(media_path),
        "-vn", "-ar", "16000", "-ac", "1", "-f", "s16le", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=64000)
    bytes_per_sec = 16000 * 2
    energies: List[float] = []

    try:
        while True:
            raw = proc.stdout.read(bytes_per_sec)
            if not raw or len(raw) < 100:
                break
            num_samples = len(raw) // 2
            samples = np.frombuffer(raw[:num_samples * 2], dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            db = 20.0 * math.log10(rms) if rms > 1e-4 else -80.0
            energies.append(round(max(-80.0, min(0.0, db)), 2))
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    return energies


def compute_db_spikes(energy_curve: List[float], window_sec: int = 180) -> List[float]:
    """Measures how many dB each second rises above its local 3-minute 60th-percentile baseline."""
    v = np.asarray(energy_curve, dtype=np.float32)
    if len(v) < 10:
        return [0.0] * len(v)
    step = 5
    coarse = v[::step]
    half = max(1, window_sec // step // 2)
    pad = np.pad(coarse, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    base = np.repeat(np.percentile(windows, 60, axis=1), step)[:len(v)]
    spikes = np.clip(v - base, 0.0, None)
    return [round(float(x), 2) for x in spikes]


def _envelope(samples: np.ndarray, rate: int = 8000, env_rate: int = 100) -> np.ndarray:
    hop = rate // env_rate
    n = len(samples) // hop
    if not n:
        return np.zeros(0, dtype=np.float32)
    e = np.log(np.sqrt(np.mean(samples[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-4)
    k = env_rate
    trend = np.convolve(np.pad(e, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:len(e)]
    return (e - trend).astype(np.float32)


def sync_clip_to_audio(clip_path: Path, full_audio_path: Path, guess_start: float, reach: float = 25.0) -> Tuple[float, float]:
    """Aligns a downloaded video section against the master audio using cross-correlation.
    Returns (aligned_stream_start_sec, correlation_confidence_0_to_1).
    """
    env_rate = 100
    probe_len = 18.0
    a = _envelope(decode_pcm(clip_path, 8000, 0.5, probe_len), 8000, env_rate)
    lo = max(0.0, guess_start + 0.5 - reach)
    b = _envelope(decode_pcm(full_audio_path, 8000, lo, probe_len + 2 * reach), 8000, env_rate)
    if len(a) < env_rate * 4 or len(b) <= len(a):
        return guess_start, 0.0
    a_norm = (a - a.mean()) / (a.std() + 1e-6)
    win = np.lib.stride_tricks.sliding_window_view(b, len(a_norm))
    mu = win.mean(axis=1)
    sd = win.std(axis=1) + 1e-6
    score = (win @ a_norm - mu * a_norm.sum()) / (len(a_norm) * sd)
    k = int(np.argmax(score))
    return round(lo + k / env_rate - 0.5, 3), round(float(score[k]), 3)


def parse_silence_intervals(video_path: Path, noise_db: int = -35, min_dur: float = 0.5) -> List[List[float]]:
    """Runs ffmpeg silencedetect to identify natural pauses."""
    settings = get_settings()
    cmd = [
        settings.ffmpeg_bin,
        "-nostdin", "-i", str(video_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    intervals = []
    start = None
    for line in res.stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m_start:
            start = float(m_start.group(1))
        m_end = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m_end and start is not None:
            intervals.append([round(start, 2), round(float(m_end.group(1)), 2)])
            start = None
    return intervals
