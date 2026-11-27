"""Speech transcription engine using faster-whisper with chunk caching, Urdu code-switching,
hallucination filtering, and memory guards for 8 GB RAM systems.
"""
from collections import Counter
import hashlib
import gc
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import psutil

# Compatibility shim for newer PyAV which removed metadata_errors parameter
try:
    import av
    _orig_av_open = av.open

    def _compat_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _compat_av_open
except Exception:
    pass

from ..config import get_settings
from ..schemas import Transcript, TranscriptSegment, WordToken
from .urdu import CODE_SWITCHING_PROMPT, format_code_switching, is_urdu_text, sanitize_transcript_text


class InsufficientMemoryError(Exception):
    pass


class SpeechTranscriber:
    def __init__(self, model_size_or_path: Optional[str] = None):
        settings = get_settings()
        self.model_size = model_size_or_path or settings.default_whisper_size
        self.model_path = settings.resolve_whisper_model(self.model_size)
        self._model = None

    def _check_memory(self):
        settings = get_settings()
        mem = psutil.virtual_memory()
        free_mb = mem.available / (1024 * 1024)
        if free_mb < settings.min_free_ram_mb:
            raise InsufficientMemoryError(
                f"Free RAM ({free_mb:.0f} MB) is below safe threshold ({settings.min_free_ram_mb} MB) for Whisper."
            )

    def load_model(self):
        if self._model is not None:
            return
        self._check_memory()
        from faster_whisper import WhisperModel
        settings = get_settings()
        download_root = str(settings.models_dir / "whisper_cache")
        self._model = WhisperModel(
            self.model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=settings.cpu_threads,
            download_root=download_root,
        )

    def unload_model(self):
        self._model = None
        gc.collect()

    def transcribe_chunk(self, audio_path: Path, offset_sec: float = 0.0,
                         language: Optional[str] = None,
                         initial_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Transcribes one chunk with hallucination filtering and word timestamps."""
        self.load_model()
        prompt = initial_prompt
        if prompt is None and language in ("ur", "hi"):
            prompt = CODE_SWITCHING_PROMPT

        beam = 1 if self.model_size == "tiny" else 3
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            word_timestamps=True,
            beam_size=beam,
            condition_on_previous_text=False,
        )

        segments: List[TranscriptSegment] = []
        words: List[WordToken] = []
        full_text_parts: List[str] = []
        last_text = ""

        for seg in segments_gen:
            text = (seg.text or "").strip()
            if not text:
                continue

            lp = getattr(seg, "avg_logprob", None)
            ns = getattr(seg, "no_speech_prob", None)
            # Filter non-speech hallucinations and consecutive repeat loops
            if ns is not None and ns > 0.8 and lp is not None and lp < -1.0:
                continue
            if text.lower() == last_text.lower() and (seg.end - seg.start) < 2.0:
                continue
            last_text = text

            if is_urdu_text(text) or info.language in ("ur", "hi"):
                text = format_code_switching(text)
            else:
                text = sanitize_transcript_text(text)
            if not text:
                continue

            s_start = round(seg.start + offset_sec, 2)
            s_end = round(seg.end + offset_sec, 2)
            full_text_parts.append(text)

            seg_words: List[WordToken] = []
            raw_words = [w for w in (seg.words or []) if (w.word or "").strip()]
            if raw_words:
                for w in raw_words:
                    wt = w.word.strip()
                    wt = format_code_switching(wt) if is_urdu_text(wt) else sanitize_transcript_text(wt)
                    if not wt:
                        continue
                    w_token = WordToken(
                        w=wt,
                        s=round(w.start + offset_sec, 2),
                        e=round(w.end + offset_sec, 2),
                        p=round(float(getattr(w, "probability", 0.8)), 2),
                    )
                    seg_words.append(w_token)
                    words.append(w_token)
            else:
                toks = text.split()
                if toks:
                    step = max(0.05, (s_end - s_start) / len(toks))
                    for i, t in enumerate(toks):
                        w_token = WordToken(
                            w=t,
                            s=round(s_start + i * step, 2),
                            e=round(s_start + (i + 1) * step, 2),
                            p=0.7,
                        )
                        seg_words.append(w_token)
                        words.append(w_token)

            mean_p = sum(w.p for w in seg_words) / len(seg_words) if seg_words else 0.7
            uncertain = bool(
                (lp is not None and lp < -1.0)
                or (ns is not None and ns > 0.6)
                or mean_p < 0.5
            )
            speaker_turn = bool(segments and (s_start - segments[-1].end >= 0.85 or segments[-1].text.endswith(("?", "؟"))))

            segments.append(TranscriptSegment(
                id=len(segments),
                start=s_start,
                end=s_end,
                text=text,
                avg_logprob=round(lp, 3) if lp is not None else None,
                no_speech_prob=round(ns, 3) if ns is not None else None,
                uncertain=uncertain,
                speaker_turn=speaker_turn,
                words=seg_words,
            ))

        return {
            "language": info.language or (language or "en"),
            "language_probability": round(float(info.language_probability or 0.0), 2),
            "segments": segments,
            "words": words,
            "full_text": " ".join(full_text_parts),
        }

    def transcribe_chunks(self, chunks: List[Dict[str, Any]], language: Optional[str] = None,
                          cache_dir: Optional[Path] = None,
                          progress_cb: Optional[Callable[[float, str], None]] = None,
                          initial_prompt: Optional[str] = None) -> Transcript:
        """Transcribes chunks sequentially with per-chunk JSON caching."""
        all_segments: List[TranscriptSegment] = []
        all_words: List[WordToken] = []
        lang_counts: Counter = Counter()
        lang_probs: List[float] = []

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        p_tag = f"_{hashlib.sha1(initial_prompt.encode('utf-8')).hexdigest()[:6]}" if initial_prompt else ""
        total_chunks = max(1, len(chunks))
        try:
            for idx, c in enumerate(chunks):
                chunk_idx = c.get("index", idx)
                cache_file = (cache_dir / f"asr_{chunk_idx:04d}_{self.model_size}_{language or 'auto'}{p_tag}.json") if cache_dir else None

                res = None
                if cache_file and cache_file.is_file():
                    try:
                        cached = json.loads(cache_file.read_text(encoding="utf-8"))
                        res = {
                            "language": cached["language"],
                            "language_probability": cached["language_probability"],
                            "segments": [TranscriptSegment.model_validate(s) for s in cached["segments"]],
                            "words": [WordToken.model_validate(w) for w in cached["words"]],
                        }
                    except Exception:
                        res = None

                if res is None:
                    res = self.transcribe_chunk(
                        Path(c["file"]), offset_sec=c["offset"], language=language, initial_prompt=initial_prompt
                    )
                    if cache_file:
                        try:
                            payload = {
                                "language": res["language"],
                                "language_probability": res["language_probability"],
                                "segments": [s.model_dump() for s in res["segments"]],
                                "words": [w.model_dump() for w in res["words"]],
                            }
                            cache_file.write_text(json.dumps(payload), encoding="utf-8")
                        except Exception:
                            pass

                lang_counts[res["language"]] += max(1, len(res["segments"]))
                lang_probs.append(res["language_probability"])

                for s in res["segments"]:
                    s.id = len(all_segments)
                    all_segments.append(s)
                all_words.extend(res["words"])

                if progress_cb:
                    progress_cb((idx + 1) / total_chunks, f"Transcribed chunk {idx + 1}/{total_chunks} ({res['language']})")
        finally:
            self.unload_model()

        detected_lang = lang_counts.most_common(1)[0][0] if lang_counts else (language or "en")
        avg_prob = round(sum(lang_probs) / len(lang_probs), 2) if lang_probs else 1.0
        full_text = " ".join(s.text for s in all_segments)

        return Transcript(
            language=detected_lang,
            language_probability=avg_prob,
            segments=all_segments,
            words=all_words,
            full_text=full_text,
        )
