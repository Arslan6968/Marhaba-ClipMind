# Marhaba ClipMind v4 — Universal AI Video Clipping Skill & Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Local-First](https://img.shields.io/badge/AI-100%25%20Local--First-10B981.svg)](#2-requirements--setup)
[![Zero API Keys](https://img.shields.io/badge/Paid%20APIs-Zero%20Required-EF4444.svg)](#)
[![Author](https://img.shields.io/badge/Creator-Arslan%20Rasheed%20(@Arslan6968)-111827.svg)](https://github.com/Arslan6968)

**Universal Intelligence Behind Every Clip — built by [Arslan Rasheed (@Arslan6968)](https://github.com/Arslan6968).**

Marhaba ClipMind is a general-purpose, **100% local-first AI video clipping system & agent skill** (compatible with Antigravity, Claude Code, Codex, Gemini CLI, Cursor) for podcasts, interviews, tutorials, lectures, gaming streams, speeches, news, debates, documentaries, vlogs, and bilingual **Urdu RTL / Roman Urdu / English** videos.

Engineered specifically for **8 GB RAM CPU-first laptops** with **zero paid cloud AI APIs**.

### Why Marhaba ClipMind Stands Out
- **Audio-First & Section-Only Download:** Downloads only audio first for Whisper + loudness analysis, then fetches **only winning clip timestamps** in 1080p.
- **Context-Primed Urdu + English ASR:** Dynamically primes Whisper (`tiny` / `base` / `small`) using title, channel, and topic context.
- **Teaser Guard + 9-Dimension Scoring:** Skips opening agenda/promo teasers, favors concrete facts (numbers, dates, Q&A answers), and enforces strict non-overlap (`<= 12%`).
- **Studio Vector Motion Graphics:** Rounded bezier glassmorphic Hook Cards, adaptive `Inter` / `Noto Nastaliq Urdu` badges, kinetic **Key Stat** callout pills, true **R-to-L Urdu BiDi**, and word-by-word karaoke in **Roman Urdu** & English.
- **Trend & Demand Radar (`trend`):** Ranks creator videos against live YouTube search autocomplete signals and hourly velocity.

### Marhaba Open-Source Ecosystem
- **[Marhaba-ClipMind](https://github.com/Arslan6968/Marhaba-ClipMind)** — Universal Local AI Video Clipping Skill & Engine
- **[Marhaba-Vox](https://github.com/Arslan6968/Marhaba-Vox)** — Local Voice / Audio Stack
- **[Marhaba-Project-Continuity](https://github.com/Arslan6968/Marhaba-Project-Continuity)** — Agent Workflow Continuity

### 30-Second Quickstart
```bash
git clone https://github.com/Arslan6968/Marhaba-ClipMind.git
cd Marhaba-ClipMind
pip install -r requirements.txt
python clipmind.py doctor
python clipmind.py clip "https://www.youtube.com/watch?v=VIDEO_ID" --intent "educational takeaways" --caption-lang roman_urdu
```

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AI AGENT SKILL LAYER                            │
│  SKILL.md / AGENTS.md / GEMINI.md  (Claude, Codex, Antigravity, Gemini)│
│  - Adaptive Questioning (asks <=2 questions only when intent unclear)  │
│  - Workflow A: Single Video URL / Local File                           │
│  - Workflow B: Creator & Channel Discovery                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       CLIPMIND CLI RUNNER                              │
│  python clipmind.py [doctor | probe | find | trend | clip | remake]    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    DECOUPLED CORE ENGINE                               │
│                                                                        │
│  Ingest Adapters:                                                      │
│  - LocalFileAdapter (zero-copy in-place reading)                       │
│  - YtDlpAdapter (audio-first, storyboard, live chat, section download) │
│  - DirectUrlAdapter (strict SSRF-validated redirect pinning)           │
│  - Channel & Trend Radar (autocomplete demand, velocity, lazy meta)    │
│                                                                        │
│  Understanding & Discovery:                                            │
│  - SpeechTranscriber (faster-whisper INT8, dynamic domain prompting)   │
│  - 12-Genre Content Classifier + Bilingual Semantic Topic Matcher      │
│  - Multi-Tier Brains (rules | local llama.cpp GGUF | --brain agent)    │
│  - Complete-Thought Units + Teaser Guard + 9D Scoring + Dedupe (0.12)  │
│  - Story Verification (setup snap, payoff extend, 4x4 contact sheets)  │
│                                                                        │
│  Production Studio:                                                    │
│  - Edge-aware Facecam Detection + Cosine-Eased Gliding Speaker Crop    │
│  - 6 Distinct Presets (clean, dynamic, podcast, gaming, edu, social)   │
│  - Vector Glass Hook Cards, Key-Stat Popups & Adaptive Topic Badges    │
│  - True R-to-L Urdu BiDi (Noto Nastaliq) + Roman Urdu + Karaoke ASS    │
│  - Procedural Sound Design (whoosh/pop), -14 LUFS loudnorm, 4 ratios   │
│  - Strict Decode Verification (verify_export) + instant `remake`       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                 FUTURE WEB APP COMPATIBILITY LAYER                     │
│  - ClipMindPipeline(work_root).run(..., progress_cb, cancel_token)     │
│  - Checkpoints (`job_state.json`), JSON manifests, zero UI coupling    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Requirements & Setup

### Hardware Requirements (Verified on Target Hardware)
- **OS:** Windows 10/11, Linux, or macOS.
- **RAM:** **8 GB total RAM** (600 MB minimum free RAM guard for Whisper Base; `tiny` option for tighter systems).
- **CPU:** 4+ logical cores (100% CPU-first via CTranslate2 INT8; GPU optional via `--fast`).
- **Storage:** SSD recommended; temporary audio/frame artifacts clean up automatically.

### Required External Tools
- **FFmpeg + FFprobe** compiled with `libass`, `libfribidi`, and `libharfbuzz` (`winget install Gyan.FFmpeg` on Windows).
- **yt-dlp** (`pip install yt-dlp`) + **Node.js or Deno** for YouTube extraction.
- **Python 3.11+** with packages in `requirements.txt`.

### Model Locations
- `clipmind/models/whisper-base/` (145 MB CTranslate2 INT8 — downloaded automatically on first run via `faster-whisper`; excluded from Git due to GitHub's 100 MB file limit)
- `clipmind/models/face_detection_yunet_2023mar.onnx` (232 KB YuNet — included)
- `clipmind/fonts/` (Noto Nastaliq Urdu, Gulzar, Nastaliq, Inter, Poppins, Manrope — included)
- Optional local LLM GGUF (`qwen2.5-1.5b-instruct-q4_k_m.gguf`) placed in `clipmind/models/` if using `--brain local`.

---

## 3. CLI Reference & Example Workflows

```powershell
# 1. Verify system readiness
python clipmind.py doctor

# 2. Inspect URL metadata & adaptive questions before processing
python clipmind.py probe "https://www.youtube.com/watch?v=..."

# 3. Workflow B: Discover creator videos by topic and heat
python clipmind.py find "Lex Fridman" --topic "artificial intelligence" --sort hot --within month --count 6

# 4. Workflow A: Clip a remote video (audio-first + targeted section download)
python clipmind.py clip "https://www.youtube.com/watch?v=..." --intent "educational takeaways about AI" --preset educational --max-clips 4

# 5. Workflow A: Clip an Urdu / mixed-language local tutorial into Roman Urdu or Nastaliq RTL
python clipmind.py clip "lecture.mp4" --language ur --caption-lang original --preset social_media --sfx

# 6. Interactive Agent Mode (writes STORY_TASK.md + 4x4 contact sheets for agent vision review)
python clipmind.py clip "video.mp4" --brain agent

# 7. Instantly change preset, hook, or aspect ratio on finished clips
python clipmind.py remake "clipmind_workspace/my_job/clips" --clip 1 --preset gaming --hook "This *one setting* changes everything"

# 8. Prune caches older than 7 days
python clipmind.py prune --days 7
```

---

## 4. Troubleshooting

- **`YouTube JS Runtime` warning in `doctor`:** Install Node.js (`winget install OpenJS.NodeJS.LTS`) or Deno (`winget install DenoLand.Deno`) so `yt-dlp` reliably resolves YouTube formats.
- **Low RAM errors:** Pass `--whisper-model tiny` (75 MB) and `--brain rules`, and close heavy browser tabs during encoding.
- **Platform restrictions:** Age-restricted, subscriber-only, DRM-protected, and currently live streams cannot be clipped until public/finished. ClipMind detects these and reports clearly without bypassing access controls.
- **Urdu text rendering:** ClipMind passes `fontsdir` directly to FFmpeg's `libass` filter so bundled `Noto Nastaliq Urdu` renders right-to-left connected script without installing system fonts.

---

## 5. Future Web App Integration

Import `ClipMindPipeline` and `remake_clips` directly inside FastAPI, Celery, or RQ workers:

```python
from pathlib import Path
from clipmind.pipeline.runner import ClipMindPipeline
from clipmind.pipeline.state import CancellationToken
from clipmind.schemas import UserIntent

token = CancellationToken()
pipeline = ClipMindPipeline(work_root=Path("/data/jobs"))
result = pipeline.run(
    source_input="/data/uploads/podcast.mp4",
    user_intent=UserIntent(objective="educational", topic="startup finance", aspect_ratio="9:16"),
    progress_cb=lambda stage, pct, msg: print(f"{stage}: {pct:.0%} {msg}"),
    cancel_token=token,
)
```

---

## 6. Testing

Run the comprehensive automated suite:

```powershell
python -m pytest
```
