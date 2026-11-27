"""Procedural Sound Design for ClipMind (100% numpy-synthesized, zero external asset dependencies)."""
from pathlib import Path
from typing import List, Tuple, Optional
import wave
import numpy as np


def _write_wav_16bit(path: Path, samples: np.ndarray, rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def generate_whoosh(duration: float = 0.42, rate: int = 48000) -> np.ndarray:
    """Soft low-passed airy whoosh for hook/card reveal."""
    n = int(duration * rate)
    t = np.linspace(0.0, duration, n, dtype=np.float32)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, n).astype(np.float32)
    # Smooth moving-average lowpass sweep
    k = 18
    kernel = np.ones(k, dtype=np.float32) / k
    filtered = np.convolve(noise, kernel, mode="same")
    # Bell envelope peaking early
    env = np.sin(np.pi * (t / duration) ** 0.7) ** 2
    sub = np.sin(2.0 * np.pi * (95.0 - 45.0 * (t / duration)) * t) * 0.25
    return ((filtered * 0.55 + sub) * env * 0.35).astype(np.float32)


def generate_pop(duration: float = 0.14, rate: int = 48000) -> np.ndarray:
    """Subtle crisp pop for punch-in zoom transitions."""
    n = int(duration * rate)
    t = np.linspace(0.0, duration, n, dtype=np.float32)
    freq = 280.0 + 420.0 * np.exp(-t * 35.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / rate
    env = np.exp(-t * 28.0) * np.minimum(1.0, t / 0.008)
    return (np.sin(phase) * env * 0.30).astype(np.float32)


def build_sfx_track(dest_wav: Path, clip_duration: float, punch_times: List[Tuple[float, float]],
                    include_hook_whoosh: bool = True, rate: int = 48000) -> Optional[Path]:
    """Generates a single mixed WAV track synced to clip timestamps."""
    if not include_hook_whoosh and not punch_times:
        return None
    total_samples = int((clip_duration + 0.5) * rate)
    track = np.zeros(total_samples, dtype=np.float32)

    if include_hook_whoosh:
        w = generate_whoosh(rate=rate)
        n = min(len(w), total_samples)
        track[:n] += w[:n] * 0.60

    if punch_times:
        p = generate_pop(rate=rate)
        for s_t, _ in punch_times:
            idx = int(max(0.0, s_t) * rate)
            if idx < total_samples:
                seg_n = min(len(p), total_samples - idx)
                track[idx:idx + seg_n] += p[:seg_n] * 0.55

    if float(np.max(np.abs(track))) < 1e-4:
        return None

    _write_wav_16bit(dest_wav, track, rate=rate)
    return dest_wav
