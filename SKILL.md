---
name: marhaba-clipmind
description: Universal AI Video Clipping Skill by Marhaba. Understands user intent, transcribes locally with faster-whisper, classifies 12 content genres (podcasts, interviews, education, tutorials, gaming, commentary, speeches, vlogs, news, debates, documentaries, general), discovers contextually complete moments, verifies story completeness, and produces verified 9:16, 16:9, or 1:1 clips with native Urdu RTL, Roman Urdu, and English captions plus publishing packs. 100% local-first, 8 GB RAM optimized, zero paid AI APIs. Supports single video URLs (audio-first + targeted section fetch), local files, and creator/channel discovery.
---

# Marhaba ClipMind v4 — Universal AI Video Clipping Studio

**Intelligent, general-purpose video clipping system.** Long videos in, story-verified clips, kinetic motion graphics, & publishing packs out.
**You run everything; the user only says what they want.**

### v4 Highlights
- **Context-Primed Whisper ASR:** Dynamically primes Whisper `initial_prompt` from the video title, channel, and topic for cleaner code-switched Urdu/English terminology.
- **Teaser-Guarded Answer Discovery:** Penalizes opening agenda/question teasers without answers, rewards concrete factual density (%, dates, amounts, steps), and enforces strict non-overlapping separation (`max_overlap <= 0.12`).
- **Studio Motion Graphics & Typography:** Rounded bezier glassmorphic Hook Cards with docked accent bars, adaptive `Inter` / `Noto Nastaliq Urdu` lower-third badges, kinetic **Key Stat** callout pills (`KEY STAT • 94%`), strict **R-to-L Urdu BiDi** (`RLE/RLM`), and word-by-word karaoke in Roman Urdu (`--caption-lang roman_urdu`) & English.
- **Rate-Limit-Friendly Discovery:** Lightweight `--flat-playlist` probing avoids unnecessary metadata bursts on platforms.

Compatible with **Antigravity, Claude Code, OpenAI Codex, Gemini CLI, Cursor, and OpenCode**.
Zero paid cloud AI APIs, zero API keys, 100% private and local-first (8 GB RAM CPU-first safe).

---

## 0. Invocation & Diagnostics

`SKILL_DIR` = the folder containing this `SKILL.md`.
Run every command with any Python 3 on the system (`clipmind.py` auto-selects the working environment):

```powershell
# Windows:
python "SKILL_DIR\clipmind.py" <command> ...

# macOS / Linux:
python3 "SKILL_DIR/clipmind.py" <command> ...
```

First-time check on a machine:

```powershell
python "SKILL_DIR\clipmind.py" doctor
```

Verifies FFmpeg (`libass` + `libharfbuzz` RTL), `yt-dlp`, JS runtime (`deno`/`node`), available RAM, Whisper models (`tiny`/`base`/`small`), YuNet face detector, and bundled Urdu/Latin fonts.

---

## 1. Guided Workflows

### Workflow A: Single Video URL or Local Video File
Use when the user provides a URL (YouTube, Twitch, Kick, Vimeo, direct `.mp4` link) or a local file path.

1. **Check Metadata & Intent (Adaptive Questioning):**
   - If the user **already specified** what they want (e.g., *"find funny moments"*, *"extract AI explanations"*, *"3 short Urdu clips"*, *"educational clips"*): **do NOT ask questionnaires** — start clipping immediately!
   - If the user **only pasted a bare URL or file path** with no instructions:
     First inspect the video silently:
     ```powershell
     python "SKILL_DIR\clipmind.py" probe "<URL_OR_PATH>" --json
     ```
     Then ask **at most 2 short, high-value questions** in ONE message based on what you learned (duration, creator, title):
     > 1. **What kind of clips do you want** from this video? *(Best moments overall, educational takeaways, funny reactions, or moments about a specific topic/person?)*
     > 2. **What format and length?** *(Default: 9:16 vertical Reels/Shorts 30–60s; or short 15–30s, long 60–90s, 1:1 square, or original 16:9?)*

2. **Run Clipping (in background; report progress plainly):**
   ```powershell
   python "SKILL_DIR\clipmind.py" clip "<URL_OR_PATH>" --intent "<USER_GOAL>" --max-clips 4
   ```
   - For platform URLs (YouTube/Twitch/Kick): ClipMind automatically downloads **audio only first**, scans loudness + storyboard + chat bursts, transcribes with cached chunked Whisper, discovers & verifies moments, and then downloads **only the winning clip seconds** in 1080p with envelope audio sync!
   - For local files: reads **in-place** with zero multi-GB SSD copying!

---

### Workflow B: Creator or Channel Discovery + Trend Demand Radar
Use when the user gives a creator name (`"Lex Fridman"`, `"IShowSpeed"`, `"Dr. Israr Ahmed"`), `@handle`, or channel URL.

1. **Search Available Videos & YouTube Search Demand (never download an entire channel automatically):**
   ```powershell
   python "SKILL_DIR\clipmind.py" find "<CREATOR_OR_URL>" --sort hot --within week --count 6 --topic "<OPTIONAL_TOPIC>"
   # Or run YouTube Autocomplete + Market Demand Analysis to rank the most searched topics & videos:
   python "SKILL_DIR\clipmind.py" trend "<CREATOR_OR_URL>" --count 8
   ```
   Options: `--sort hot|views|latest`, `--within 24h|48h|week|month|all`, `--include-shorts`, `--all` (include previously clipped URLs).

2. **Present Clean Options to the User:**
   List title, creator, duration, view metrics, and URL. Ask which video(s) to process (latest, top N, or specific picks).

3. **Process Selected Video(s):**
   Run Workflow A on the chosen URL(s).

---

## 2. Intent, Topics, Languages & Brain Modes

### Intent & Topic Steering
- `--intent "find funny reactions"` $\rightarrow$ Boosts emotional energy spikes, laughter, and comedic pacing.
- `--intent "educational takeaways"` $\rightarrow$ Boosts dense explanations, Q&A answers, and standalone clarity.
- `--topic "artificial intelligence"` $\rightarrow$ Semantically matches discussions (English + Urdu + Roman Urdu concepts, stems, and fuzzy n-grams) even when exact words are not spoken.
- `--speaker "Elon Musk"` $\rightarrow$ Prioritizes moments where that person speaks or is discussed.
- `--section "1:15:00-2:45:00"` $\rightarrow$ Clips only a target time range inside long streams.

### Language & Subtitles
- `--language ur` (or auto-detect): Enables Urdu + English code-switching prompt & normalization (`سافٹ ویئر` $\rightarrow$ `Software`, `ویڈیو` $\rightarrow$ `Video`).
- `--caption-lang original`: Native RTL `Noto Nastaliq Urdu` / `Inter` subtitles with word-level karaoke highlights.
- `--caption-lang roman_urdu`: Automatic Urdu-to-Roman-Urdu transliteration.
- `--caption-lang none`: Clean video without burned-in subtitles.
- `--whisper-model tiny|base|small`: Use `tiny` on ultra-low RAM (< 1.5 GB free) or `small` for higher Urdu accuracy.

### Reasoning Brain (`--brain`)
| Mode | Behavior |
| :--- | :--- |
| `auto` *(default)* | Uses local GGUF reviewer (`llama.cpp`) if installed and RAM $\ge 1.6\text{ GB}$ free; otherwise uses deterministic rules (0 MB extra RAM). |
| `agent` | **Host Agent Vision Review:** Pauses with **exit code 3** and writes `STORY_TASK.md` + 4x4 timestamped contact sheets (`story_sheet_1.jpg`) + `.story.prompt.txt`. **Open the contact sheets**, inspect what happens visually and in the transcript, write JSON verdicts (`keep`/`fix`/`drop`, refined `start`/`end`, `hook`, `title`) into `.story.answer.json`, and re-run the exact same command! |
| `local` | Forces local `llama.cpp` GGUF review (falls back safely if low RAM). |
| `rules` / `none` | Fast deterministic 9-dimension scoring & story boundary snapping. |

---

## 3. Six Editing Presets & Aspect Ratios

Aspect ratios: `--aspect 9:16` *(default)*, `16:9`, `1:1`, `original`.
Presets (`python "SKILL_DIR\clipmind.py" styles` for details):

| Preset | Visual Behavior & Layout | Best For |
| :--- | :--- | :--- |
| `clean` | Steady framing, subtle fade-in/out, minimal crisp captions | Business, keynotes, documentaries |
| `dynamic` | Loudness-triggered 1.14x punch-in zooms + vibrant gold karaoke pop | Commentary, high-energy vlogs |
| `podcast` | Cosine-eased gliding speaker tracking + emerald lower-third captions | Podcasts, interviews, debates |
| `gaming` | Canny+YuNet stacked corner facecam + uncropped gameplay + punch-ins | Gaming streams, live reactions |
| `educational` | Top topic banner card + uncropped `fit_blur` clarity for slides/UI | Tutorials, lectures, screen shares |
| `social_media` | Clean dark 5:4 canvas + top `*accent*` crimson hook card + mobile captions | Viral hooks, stories, general reels |

Add `--sfx` for subtle procedural whoosh/pop sound design, and `--fast` to auto-probe GPU encoders (`h264_nvenc`, `h264_qsv`, `h264_amf`).

---

## 4. Mandatory Visual Check & Instant `remake`

Every exported clip produces a **3-frame visual verification strip**: `check_<clip_id>.jpg` (Start, Middle, End of the actual rendered video).

1. **Always inspect `check_<clip_id>.jpg`** (using your image viewer tool or opening it) before handing off!
2. Verify:
   - Faces and UI/slides are not cut off.
   - Hook text and captions are crisp, readable, properly positioned, and Urdu RTL letters are connected cleanly.
   - No awkwardly cropped frame or empty black bar glitch.
3. **Fix or change style in seconds without re-downloading:**
   ```powershell
   python "SKILL_DIR\clipmind.py" remake "<OUTPUT_DIR>" --clip 1 --preset social_media --hook "Wait until you see this *crazy* moment"
   ```

---

## 5. Handover

Present to the user:
- Output folder path and list of verified `.mp4` files (`duration`, `preset`, `quality_score`).
- Point directly to `POSTING.md` (grounded titles, descriptions, hashtags, A/B hooks, and honest selection rationale) and `manifest.json`.
- **Never promise virality, guaranteed views, or guaranteed engagement.**
- Clean old caches periodically with `python "SKILL_DIR\clipmind.py" prune --days 7`.
