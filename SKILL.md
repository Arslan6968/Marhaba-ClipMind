---
name: marhaba-clipmind
description: Universal local-first AI video clipping skill (8 GB RAM safe, zero paid APIs). Supports YouTube/Twitch/Kick/local video, creator & trend discovery, Urdu RTL / Roman Urdu / English captions, rounded vector motion graphics, and instant remake.
---

# Marhaba ClipMind v4 — Token-Lean Agent Guide

Run commands via: `python "SKILL_DIR\clipmind.py" <cmd> ...` (use `--compact` on CLI calls to save tokens).
Read `references/cli-reference.md` only if you need detailed flag tables or JSON schemas.

## 1. Workflows
- **First use:** Run `doctor` once per machine.
- **Workflow A (URL / Local File):**
  - If user intent is clear, run `clip` directly. If bare URL/file only, run `probe "<URL>" --compact --json` and ask <=2 short questions.
  - Command: `python "SKILL_DIR\clipmind.py" clip "<URL_OR_PATH>" --intent "<GOAL>" --max-clips 4 --compact`
  - Remote URLs download audio first, then fetch only winning 1080p clip sections.
- **Workflow B (Creator / Trend Discovery):**
  - Never download a full channel. Run `find "<CREATOR>" --sort hot --within week --count 6 --compact` or `trend "<CREATOR>" --count 8 --compact`, present picks, then run Workflow A.

## 2. Key Options
- **Steering:** `--topic "..."`, `--speaker "..."`, `--section "1:00:00-2:30:00"`, `--duration short|medium|long`
- **Language & Captions:** `--language ur|en`, `--caption-lang original|roman_urdu|none`, `--whisper-model tiny|base|small` (Urdu uses strict R-to-L BiDi; `roman_urdu` uses `Inter` karaoke).
- **Styles (`--preset`):** `clean` | `dynamic` | `podcast` | `gaming` | `educational` | `social_media` + `--aspect 9:16|16:9|1:1|original` + `--sfx` + `--fast`.
- **Brain (`--brain`):** `auto` (local GGUF or deterministic rules) | `agent` (exit code 3 -> read `STORY_TASK.json` + sheets, write `STORY_ANSWERS.json`, re-run) | `rules`.

## 3. Visual Verification & `remake`
1. Inspect **`check_all.jpg`** (1 image covering all clips) — open individual `check_<clip_id>.jpg` only if zooming into a specific clip.
2. Fix hook/preset/language offline without re-downloading:
   `python "SKILL_DIR\clipmind.py" remake "<OUT_DIR>" --clip 1 --preset social_media --hook "Your *accent* hook" --compact`

## 4. Rules & Handover
- Zero paid cloud AI APIs; never bypass age-gates, logins, DRM, paywalls, or anti-bot checks.
- Report output folder, verified `.mp4` files, `POSTING.md`, and `manifest.json` honestly without promising virality.
